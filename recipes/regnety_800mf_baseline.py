import os
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
img_size = 224


def load_img(fp):
    img = Image.open(fp).convert("RGB")
    w, h = img.size
    if w < h:
        new_w = img_size
        new_h = int(h * (img_size / w))
    else:
        new_h = img_size
        new_w = int(w * (img_size / h))

    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    left = (new_w - img_size) // 2
    top = (new_h - img_size) // 2
    img = img.crop((left, top, left + img_size, top + img_size))
    return np.array(img)


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

        scale = torch.empty(B, device=self.device).uniform_(0.7, 1.0)
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


def gpu_mixed_strategy(imgs, labels):
    mix_prob = 0.80
    r = np.random.rand()
    if r >= mix_prob:
        return imgs, labels, labels, 1.0

    if np.random.rand() < 0.5:
        lam = np.random.beta(0.4, 0.4)
        idx = torch.randperm(imgs.size(0), device=imgs.device)
        mixed_imgs = lam * imgs + (1 - lam) * imgs[idx]
        return mixed_imgs, labels, labels[idx], lam
    else:
        lam = np.random.beta(1.0, 1.0)
        idx = torch.randperm(imgs.size(0), device=imgs.device)

        H, W = imgs.size(2), imgs.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        actual_lams = []
        total_area = H * W

        imgs_orig = imgs.clone()

        for i in range(imgs.size(0)):
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            imgs[i, :, bby1:bby2, bbx1:bbx2] = imgs_orig[idx[i],
                                                         :, bby1:bby2, bbx1:bbx2]
            actual_lams.append(
                1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / total_area))

        effective_lam = np.mean(actual_lams)
        return imgs, labels, labels[idx], effective_lam


def split_weights(model):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': 0.05}, {'params': no_decay, 'weight_decay': 0.0}]


def main():
    train_manifold = VRAMGlobalManifold("food11/training", device)
    val_manifold = VRAMGlobalManifold("food11/validation", device)

    float_data = train_manifold.data.float() / 255.0
    mean = float_data.mean([0, 2, 3], keepdim=True)
    std = float_data.std([0, 2, 3], keepdim=True) + 1e-6
    del float_data

    augmentor = VectorizedGPUAugmentor(device)

    model = models.regnet_y_800mf(num_classes=11).to(device)
    ema = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(0.999), use_buffers=True)

    optimizer = torch.optim.AdamW(split_weights(model), lr=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    steps_per_epoch = len(train_manifold.labels) // batch_size

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=steps_per_epoch,
        epochs=n_epochs, pct_start=0.1, anneal_strategy='cos'
    )

    scaler = torch.cuda.amp.GradScaler()

    train_loss = torchmetrics.MeanMetric().to(device)
    train_acc = torchmetrics.MeanMetric().to(device)
    train_grad_norm = torchmetrics.MeanMetric().to(device)

    val_acc_metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=11).to(device)
    val_ema_acc_metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=11).to(device)

    print(f"[System] Start training ...")
    best_ema_acc = 0

    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(len(train_manifold.labels), device=device)

        train_loss.reset()
        train_acc.reset()
        train_grad_norm.reset()

        for s in range(steps_per_epoch):
            batch_idx = indices[s*batch_size: (s+1)*batch_size]

            imgs = train_manifold.data[batch_idx].float() / 255.0
            imgs = augmentor(imgs)
            imgs = (imgs - mean) / std
            labels = train_manifold.labels[batch_idx]

            imgs, l_a, l_b, lam = gpu_mixed_strategy(imgs, labels)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(imgs)
                logits_fp32 = logits.float()
                loss = lam * criterion(logits_fp32, l_a) + \
                    (1 - lam) * criterion(logits_fp32, l_b)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            if torch.isfinite(torch.as_tensor(grad_norm)):
                scaler.step(optimizer)
                ema.update_parameters(model)
                scheduler.step()
                train_grad_norm.update(grad_norm.detach())

            scaler.update()

            if not torch.isnan(loss):
                train_loss.update(loss.detach())
                preds = logits_fp32.detach()
                acc_a = (preds.argmax(dim=1) == l_a).float().mean()
                acc_b = (preds.argmax(dim=1) == l_b).float().mean()
                train_acc.update(lam * acc_a + (1.0 - lam) * acc_b)

        epoch_train_loss = train_loss.compute().item()
        epoch_train_acc = train_acc.compute().item()

        try:
            epoch_grad_norm = train_grad_norm.compute().item()
        except:
            epoch_grad_norm = 0.0

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

        print(f"Epoch {epoch+1:03d} | Train Loss: {epoch_train_loss:.4f} | Mix ACC: {epoch_train_acc:.4f} || "
              f"Val ACC: {epoch_val_acc:.4f} | EMA Val ACC: {epoch_val_ema_acc:.4f} (Peak: {best_ema_acc:.4f}) || "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | GradNorm: {epoch_grad_norm:.4f}")

        if epoch_val_ema_acc > best_ema_acc:
            best_ema_acc = epoch_val_ema_acc
            torch.save(ema.state_dict(), "best_food11_regnety.pt")

    print(
        f"[Execution Complete] Historical Peak Validation Accuracy: {best_ema_acc:.4f}")


if __name__ == "__main__":
    main()
