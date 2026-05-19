import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import numpy as np
import torchmetrics

batch_size = 64
n_epochs = 300
device = "cuda" if torch.cuda.is_available() else "cpu"
global_size = 224
local_size = 112
n_local_views = 4


def load_img(fp):
    img = Image.open(fp).convert("RGB")
    w, h = img.size
    if w < h:
        new_w = global_size
        new_h = int(h * (global_size / w))
    else:
        new_h = global_size
        new_w = int(w * (global_size / h))
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = (new_w - global_size) // 2
    top = (new_h - global_size) // 2
    return np.array(img.crop((left, top, left + global_size, top + global_size)))


class VRAMGlobalManifold:
    def __init__(self, path, device):
        self.path = Path(path)
        files = sorted([x for x in self.path.iterdir()
                       if x.suffix.lower() == ".jpg"])
        print(
            f"[IO] Parallel loading {len(files)} images from {self.path.name}...")
        with ThreadPoolExecutor(max_workers=64) as executor:
            imgs = list(tqdm(executor.map(load_img, files), total=len(files)))
        self.data = torch.from_numpy(np.stack(imgs)).permute(
            0, 3, 1, 2).to(device, dtype=torch.uint8)
        labels = [int(f.stem.split("_")[0]) if "test" not in str(
            path) else -1 for f in files]
        self.labels = torch.tensor(labels, device=device).long()
        print(
            f"[System] {self.path.name} resident in VRAM. Shape: {self.data.shape}")


class VectorizedGPUAugmentor:

    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def __call__(self, x):
        B, C, H, W = x.shape
        scale = torch.empty(B, device=self.device).uniform_(0.6, 1.0)
        theta = torch.zeros(B, 2, 3, device=self.device)
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = torch.empty(
            B, device=self.device).uniform_(-0.15, 0.15)
        theta[:, 1, 2] = torch.empty(
            B, device=self.device).uniform_(-0.15, 0.15)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear',
                          padding_mode='reflection', align_corners=False)
        flip_mask = torch.rand(B, 1, 1, 1, device=self.device) > 0.5
        x = torch.where(flip_mask, torch.flip(x, dims=[3]), x)
        b_factor = torch.empty(
            B, 1, 1, 1, device=self.device).uniform_(0.8, 1.2)
        x = (x * b_factor).clamp(0, 1)
        c_mask = torch.rand(B, 1, 1, 1, device=self.device) > 0.5
        mean_val = x.mean(dim=[2, 3], keepdim=True)
        c_factor = torch.empty(
            B, 1, 1, 1, device=self.device).uniform_(0.8, 1.2)
        x = torch.where(c_mask, (x - mean_val) *
                        c_factor + mean_val, x).clamp(0, 1)
        s_mask = torch.rand(B, 1, 1, 1, device=self.device) > 0.5
        if s_mask.any():
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            s_factor = torch.empty(
                B, 1, 1, 1, device=self.device).uniform_(0.5, 1.5)
            x = torch.where(s_mask, (x - gray) *
                            s_factor + gray, x).clamp(0, 1)
        gray_mask = torch.rand(B, 1, 1, 1, device=self.device) < 0.10
        if gray_mask.any():
            gray_all = 0.299 * x[:, 0:1] + 0.587 * \
                x[:, 1:2] + 0.114 * x[:, 2:3]
            x = torch.where(gray_mask, gray_all.repeat(1, 3, 1, 1), x)
        erasing_mask = torch.rand(B, device=self.device) < 0.3
        if erasing_mask.any():
            valid_idx = erasing_mask.nonzero(as_tuple=True)[0]
            valid_B = len(valid_idx)
            s_e = torch.empty(valid_B, device=self.device).uniform_(
                0.02, 0.20) * (H * W)
            r_e = torch.empty(valid_B, device=self.device).uniform_(0.3, 3.3)
            h_e = torch.sqrt(s_e * r_e).clamp(max=H-1).long()
            w_e = torch.sqrt(s_e / r_e).clamp(max=W-1).long()
            y_e = (torch.rand(valid_B, device=self.device)
                   * (H - h_e + 1)).long()
            x_e = (torch.rand(valid_B, device=self.device)
                   * (W - w_e + 1)).long()
            for i, v_idx in enumerate(valid_idx):
                x[v_idx, :, y_e[i]:y_e[i]+h_e[i], x_e[i]:x_e[i]+w_e[i]] = 0.0
        return x


class MultiViewGPUAugmentor:

    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def _apply_pipeline(self, x, target_h, target_w, scale_min, scale_max, erasing=False):
        B, C, H, W = x.shape
        scale = torch.empty(B, device=self.device).uniform_(
            scale_min, scale_max)
        theta = torch.zeros(B, 2, 3, device=self.device)
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = torch.empty(
            B, device=self.device).uniform_(-0.18, 0.18)
        theta[:, 1, 2] = torch.empty(
            B, device=self.device).uniform_(-0.18, 0.18)

        grid = F.affine_grid(
            theta, [B, C, target_h, target_w], align_corners=False)
        x_aug = F.grid_sample(x, grid, mode='bilinear',
                              padding_mode='reflection', align_corners=False)

        flip_mask = torch.rand(B, 1, 1, 1, device=self.device) > 0.5
        x_aug = torch.where(flip_mask, torch.flip(x_aug, dims=[3]), x_aug)

        b_factor = torch.empty(
            B, 1, 1, 1, device=self.device).uniform_(0.7, 1.3)
        x_aug = (x_aug * b_factor).clamp(0, 1)
        c_mask = torch.rand(B, 1, 1, 1, device=self.device) > 0.5
        mean_val = x_aug.mean(dim=[2, 3], keepdim=True)
        c_factor = torch.empty(
            B, 1, 1, 1, device=self.device).uniform_(0.7, 1.3)
        x_aug = torch.where(c_mask, (x_aug - mean_val) *
                            c_factor + mean_val, x_aug).clamp(0, 1)

        gray_mask = torch.rand(B, 1, 1, 1, device=self.device) < 0.20
        if gray_mask.any():
            gray_all = 0.299 * x_aug[:, 0:1] + 0.587 * \
                x_aug[:, 1:2] + 0.114 * x_aug[:, 2:3]
            x_aug = torch.where(gray_mask, gray_all.repeat(1, 3, 1, 1), x_aug)

        if erasing and torch.rand(1).item() < 0.3:
            s_e = torch.empty(B, device=self.device).uniform_(
                0.02, 0.15) * (target_h * target_w)
            r_e = torch.empty(B, device=self.device).uniform_(0.3, 3.3)
            h_e = torch.sqrt(s_e * r_e).clamp(max=target_h-1).long()
            w_e = torch.sqrt(s_e / r_e).clamp(max=target_w-1).long()
            y_e = (torch.rand(B, device=self.device)
                   * (target_h - h_e + 1)).long()
            x_e = (torch.rand(B, device=self.device)
                   * (target_w - w_e + 1)).long()
            for i in range(B):
                x_aug[i, :, y_e[i]:y_e[i]+h_e[i], x_e[i]:x_e[i]+w_e[i]] = 0.0
        return x_aug

    def __call__(self, x):
        g1 = self._apply_pipeline(
            x, global_size, global_size, 0.6, 1.0, erasing=True)
        g2 = self._apply_pipeline(
            x, global_size, global_size, 0.6, 1.0, erasing=True)
        local_views = [self._apply_pipeline(
            x, local_size, local_size, 0.25, 0.45, erasing=False) for _ in range(n_local_views)]
        return g1, g2, local_views


def rand_bbox(size, lam):
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def gpu_mixed_strategy(imgs, labels):
    mix_prob = 0.80
    if np.random.rand() >= mix_prob:
        return imgs, labels, labels, 1.0
    if np.random.rand() < 0.5:
        lam = np.random.beta(0.4, 0.4)
        idx = torch.randperm(imgs.size(0), device=imgs.device)
        return lam * imgs + (1 - lam) * imgs[idx], labels, labels[idx], lam
    else:
        lam = np.random.beta(1.0, 1.0)
        idx = torch.randperm(imgs.size(0), device=imgs.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
        imgs[:, :, bby1:bby2, bbx1:bbx2] = imgs[idx, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.size()[-1] * imgs.size()[-2]))
        return imgs, labels, labels[idx], lam


def split_weights(model):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias") or "BatchNorm" in n or "GroupNorm" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': 0.05}, {'params': no_decay, 'weight_decay': 0.0}]


class RegNetYBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.regnet_y_3_2gf(weights=None)
        self.stem = base.stem
        self.trunk_output = base.trunk_output
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.in_features = base.fc.in_features

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk_output(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class BYOLMLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class RegNetYMultiViewEngine(nn.Module):
    def __init__(self, num_classes=11, hidden_dim=4096, proj_dim=256):
        super().__init__()
        self.backbone = RegNetYBackbone()
        backbone_dim = self.backbone.in_features

        self.fc = nn.Sequential(nn.Dropout(
            0.5), nn.Linear(backbone_dim, num_classes))
        self.online_projector = BYOLMLPHead(backbone_dim, hidden_dim, proj_dim)
        self.online_predictor = BYOLMLPHead(proj_dim, hidden_dim, proj_dim)

        self.target_backbone = copy.deepcopy(self.backbone)
        self.target_projector = copy.deepcopy(self.online_projector)
        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_network(self, tau):
        for online_p, target_p in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            target_p.data = target_p.data * tau + online_p.data * (1. - tau)
        for online_p, target_p in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_p.data = target_p.data * tau + online_p.data * (1. - tau)

    def forward(self, x):
        return self.fc(self.backbone(x))

    def compute_multiview_byol_loss(self, g1, g2, local_views):
        B = g1.size(0)

        with torch.no_grad():
            z_g1 = F.normalize(self.target_projector(
                self.target_backbone(g1)), dim=-1)
            z_g2 = F.normalize(self.target_projector(
                self.target_backbone(g2)), dim=-1)

        globals_all = torch.cat([g1, g2], dim=0)
        feats_globals = self.backbone(globals_all)
        projs_globals = F.normalize(self.online_predictor(
            self.online_projector(feats_globals)), dim=-1)
        p_g1 = projs_globals[0: B]
        p_g2 = projs_globals[B: 2*B]

        locals_all = torch.cat(local_views, dim=0)
        feats_locals = self.backbone(locals_all)
        projs_locals = F.normalize(self.online_predictor(
            self.online_projector(feats_locals)), dim=-1)
        p_locals = projs_locals.chunk(len(local_views), dim=0)

        loss_global = (2 - 2 * (p_g1 * z_g2).sum(dim=-1).mean()) + \
            (2 - 2 * (p_g2 * z_g1).sum(dim=-1).mean())

        loss_local = 0.0
        for p_local in p_locals:
            loss_local += (2 - 2 * (p_local * z_g1).sum(dim=-1).mean())
            loss_local += (2 - 2 * (p_local * z_g2).sum(dim=-1).mean())

        total_byol_loss = (loss_global + loss_local) / \
            (2 + 2 * len(local_views))

        return 0.40 * total_byol_loss


def main():
    train_manifold = VRAMGlobalManifold("food11/training", device)
    val_manifold = VRAMGlobalManifold("food11/validation", device)

    float_data = train_manifold.data.float() / 255.0
    mean = float_data.mean([0, 2, 3], keepdim=True)
    std = float_data.std([0, 2, 3], keepdim=True) + 1e-6
    del float_data

    cls_augmentor = VectorizedGPUAugmentor(device)
    multiview_augmentor = MultiViewGPUAugmentor(device)

    model = RegNetYMultiViewEngine(num_classes=11).to(device)
    ema = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(0.999), use_buffers=True)

    optimizer = torch.optim.AdamW(split_weights(model), lr=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    steps_per_epoch = len(train_manifold.labels) // batch_size
    global_step = 0
    total_steps = steps_per_epoch * n_epochs

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=steps_per_epoch,
        epochs=n_epochs, pct_start=0.1, anneal_strategy='cos'
    )

    scaler = torch.cuda.amp.GradScaler()

    train_loss_cls = torchmetrics.MeanMetric().to(device)
    train_loss_byol = torchmetrics.MeanMetric().to(device)
    train_acc = torchmetrics.MeanMetric().to(device)

    val_acc_metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=11).to(device)
    val_ema_acc_metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=11).to(device)

    base_tau, max_tau = 0.99, 1.0
    print(f"[System] Start training ...")
    best_ema_acc = 0

    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(len(train_manifold.labels), device=device)
        train_loss_cls.reset()
        train_loss_byol.reset()
        train_acc.reset()

        for s in range(steps_per_epoch):
            batch_idx = indices[s*batch_size: (s+1)*batch_size]
            labels = train_manifold.labels[batch_idx]
            raw_imgs = train_manifold.data[batch_idx].float() / 255.0

            imgs_cls = cls_augmentor(raw_imgs)
            imgs_cls, l_a, l_b, lam = gpu_mixed_strategy(imgs_cls, labels)
            imgs_cls = (imgs_cls - mean) / std

            g1, g2, local_views = multiview_augmentor(raw_imgs)
            g1, g2 = (g1 - mean) / std, (g2 - mean) / std
            local_views = [(l - mean) / std for l in local_views]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(imgs_cls)
                logits_fp32 = logits.float()
                loss_cls = lam * \
                    criterion(logits_fp32, l_a) + (1 - lam) * \
                    criterion(logits_fp32, l_b)

                loss_byol = model.compute_multiview_byol_loss(
                    g1, g2, local_views)

                total_loss = loss_cls + loss_byol

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update_parameters(model)

            tau = max_tau - (max_tau - base_tau) * 0.5 * \
                (1. + math.cos(math.pi * global_step / total_steps))
            model.update_target_network(tau)
            global_step += 1

            if not torch.isnan(total_loss):
                train_loss_cls.update(loss_cls.detach())
                train_loss_byol.update(loss_byol.detach())
                preds = logits_fp32.detach()
                acc_a = (preds.argmax(dim=1) == l_a).float().mean()
                acc_b = (preds.argmax(dim=1) == l_b).float().mean()
                train_acc.update(lam * acc_a + (1.0 - lam) * acc_b)

        epoch_loss_cls = train_loss_cls.compute().item()
        epoch_loss_byol = train_loss_byol.compute().item()
        epoch_train_acc = train_acc.compute().item()

        model.eval()
        ema.eval()
        val_acc_metric.reset()
        val_ema_acc_metric.reset()

        with torch.no_grad():
            for i in range(0, len(val_manifold.labels), batch_size):
                v_imgs = val_manifold.data[i:i+batch_size].float() / 255.0
                v_imgs = (v_imgs - mean) / std
                v_labels = val_manifold.labels[i:i+batch_size]

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    v_logits = model(v_imgs)
                    v_logits_ema = ema(v_imgs)

                val_acc_metric.update(v_logits.detach(), v_labels)
                val_ema_acc_metric.update(v_logits_ema.detach(), v_labels)

        epoch_val_acc = val_acc_metric.compute().item()
        epoch_val_ema_acc = val_ema_acc_metric.compute().item()

        print(f"Epoch {epoch+1:03d} | Loss CE: {epoch_loss_cls:.4f} | Loss MV-BYOL: {epoch_loss_byol:.4f} | Train Mix ACC: {epoch_train_acc:.4f} || "
              f"Val ACC: {epoch_val_acc:.4f} | EMA Val ACC: {epoch_val_ema_acc:.4f} (Peak: {best_ema_acc:.4f}) || "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | BYOL Tau: {tau:.5f}")

        if epoch_val_ema_acc > best_ema_acc:
            best_ema_acc = epoch_val_ema_acc
            torch.save(ema.state_dict(),
                       "best_food11_regnety_multiview_byol.pt")

    print(
        f"[Execution Complete] Historical Peak Validation Accuracy: {best_ema_acc:.4f}")


if __name__ == "__main__":
    main()
