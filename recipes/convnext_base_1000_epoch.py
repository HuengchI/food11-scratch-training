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
n_epochs = 1000
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
        if not self.path.exists():
            print(f"[Warning] Path {path} not found. Skipping.")
            self.data = torch.empty(
                0, 3, img_size, img_size, device=device, dtype=torch.uint8)
            self.labels = torch.empty(0, device=device, dtype=torch.long)
            return

        files = sorted([x for x in self.path.iterdir()
                       if x.suffix.lower() == ".jpg"])
        print(
            f"[IO] Parallel loading {len(files)} images from {self.path.name}...")
        with ThreadPoolExecutor(max_workers=64) as executor:
            imgs = list(tqdm(executor.map(load_img, files), total=len(files)))

        self.data = torch.from_numpy(np.stack(imgs)).permute(
            0, 3, 1, 2).to(device, dtype=torch.uint8)
        labels = [int(f.stem.split("_")[0]) if "test" not in str(
            path).lower() else -1 for f in files]
        self.labels = torch.tensor(labels, device=device).long()
        print(
            f"[System] {self.path.name} resident in VRAM. Shape: {self.data.shape}")


class ConvNeXtGPUAugmentor:

    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def __call__(self, x):
        B, C, H, W = x.shape

        scale = torch.empty(B, device=self.device).uniform_(0.5, 1.0)

        theta = torch.zeros(B, 2, 3, device=self.device)
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = torch.empty(
            B, device=self.device).uniform_(-0.25, 0.25) * (1.0 - scale)
        theta[:, 1, 2] = torch.empty(
            B, device=self.device).uniform_(-0.25, 0.25) * (1.0 - scale)

        grid = F.affine_grid(
            theta, [B, C, img_size, img_size], align_corners=False)
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

        erasing_mask = torch.rand(B, device=self.device) < 0.35
        if erasing_mask.any():
            valid_idx = erasing_mask.nonzero(as_tuple=True)[0]
            for idx in valid_idx:

                s_e = np.random.uniform(0.02, 0.20) * (H * W)
                r_e = np.random.uniform(0.3, 3.3)
                h_e = int(math.sqrt(s_e * r_e))
                w_e = int(math.sqrt(s_e / r_e))

                h_e = max(1, min(h_e, H - 1))
                w_e = max(1, min(w_e, W - 1))

                y_e = np.random.randint(0, H - h_e)
                x_e = np.random.randint(0, W - w_e)

                x_aug[idx, :, y_e:y_e+h_e, x_e:x_e+w_e] = 0.0

        return x_aug


def gpu_mixed_strategy(imgs, labels):

    mix_prob = 0.80
    r = np.random.rand()
    B, C, H, W = imgs.size()

    if r >= mix_prob:
        ones_lam = torch.ones(B, 1, device=imgs.device, dtype=torch.float32)
        return imgs, labels, labels, ones_lam

    if np.random.rand() < 0.5:
        lam_val = np.random.beta(0.8, 0.8)
        idx = torch.randperm(B, device=imgs.device)
        mixed_imgs = lam_val * imgs + (1 - lam_val) * imgs[idx]
        lam_tensor = torch.full(
            (B, 1), lam_val, device=imgs.device, dtype=torch.float32)
        return mixed_imgs, labels, labels[idx], lam_tensor
    else:
        idx = torch.randperm(B, device=imgs.device)
        lam_values = torch.from_numpy(np.random.beta(
            1.0, 1.0, size=(B,))).to(imgs.device).float()
        cut_rat = torch.sqrt(1. - lam_values)
        cut_w = (W * cut_rat).long()
        cut_h = (H * cut_rat).long()

        cx = torch.randint(0, W, (B,), device=imgs.device)
        cy = torch.randint(0, H, (B,), device=imgs.device)

        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)

        r_grid = torch.arange(H, device=imgs.device).view(1, 1, H, 1)
        c_grid = torch.arange(W, device=imgs.device).view(1, 1, 1, W)

        mask = (r_grid >= bby1.view(B, 1, 1, 1)) & (r_grid < bby2.view(B, 1, 1, 1)) & \
               (c_grid >= bbx1.view(B, 1, 1, 1)) & (
                   c_grid < bbx2.view(B, 1, 1, 1))

        imgs = torch.where(mask, imgs[idx], imgs)
        actual_lam = 1.0 - \
            (mask[:, 0:1, :, :].float().sum(dim=[2, 3]) / (H * W))
        return imgs, labels, labels[idx], actual_lam


def split_weights_convnext(model):

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias") or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': 0.05}, {'params': no_decay, 'weight_decay': 0.0}]


def main():
    train_manifold = VRAMGlobalManifold("food11/training", device)
    val_manifold = VRAMGlobalManifold("food11/validation", device)

    print("[System] Calculating dataset statistics based strictly on Train set...")
    sum_, sq_sum_ = torch.zeros(
        3, device=device), torch.zeros(3, device=device)
    total_pixels = len(train_manifold.data) * img_size * img_size
    chunk_size = 128

    for i in range(0, len(train_manifold.data), chunk_size):
        batch = train_manifold.data[i:i+chunk_size].float() / 255.0
        sum_ += batch.sum(dim=[0, 2, 3])
        sq_sum_ += (batch ** 2).sum(dim=[0, 2, 3])

    mean_vec = sum_ / total_pixels
    std_vec = torch.sqrt(torch.clamp(
        sq_sum_ / total_pixels - mean_vec ** 2, min=0.0)) + 1e-6
    mean, std = mean_vec.view(1, 3, 1, 1), std_vec.view(1, 3, 1, 1)

    augmentor = ConvNeXtGPUAugmentor(device)

    model = models.convnext_base(
        weights=None, num_classes=11, stochastic_depth_prob=0.5).to(device)

    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Dropout(0.5),
        nn.Linear(in_features, 11)
    ).to(device)

    ema = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(0.9995), use_buffers=True)

    optimizer = torch.optim.AdamW(split_weights_convnext(model), lr=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')

    steps_per_epoch = len(train_manifold.labels) // batch_size

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=steps_per_epoch,
        epochs=n_epochs, pct_start=0.05, anneal_strategy='cos'
    )
    scaler = torch.amp.GradScaler('cuda')

    train_loss_cls = torchmetrics.MeanMetric().to(device)
    train_acc = torchmetrics.MeanMetric().to(device)
    val_acc_metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=11).to(device)
    val_ema_acc_metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=11).to(device)

    print(f"[System] Start training ...")
    best_ema_acc = 0

    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(len(train_manifold.labels), device=device)

        train_loss_cls.reset()
        train_acc.reset()

        for s in range(steps_per_epoch):
            batch_idx = indices[s*batch_size: (s+1)*batch_size]
            labels = train_manifold.labels[batch_idx]
            raw_imgs = train_manifold.data[batch_idx].float() / 255.0

            imgs_aug = augmentor(raw_imgs)
            imgs_mixed, l_a, l_b, lam = gpu_mixed_strategy(imgs_aug, labels)
            imgs_mixed = (imgs_mixed - mean) / std

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(imgs_mixed)
                logits_fp32 = logits.float()
                loss_a = criterion(logits_fp32, l_a)
                loss_b = criterion(logits_fp32, l_b)
                loss = (lam.view(-1) * loss_a +
                        (1.0 - lam.view(-1)) * loss_b).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            if torch.isfinite(torch.as_tensor(grad_norm)):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                ema.update_parameters(model)
            else:
                scaler.update()

            if not torch.isnan(loss):
                train_loss_cls.update(loss.detach())
                preds = logits_fp32.detach()
                acc_a = (preds.argmax(dim=1) == l_a).float()
                acc_b = (preds.argmax(dim=1) == l_b).float()
                train_acc.update(
                    (lam.view(-1) * acc_a + (1.0 - lam.view(-1)) * acc_b).mean())

        epoch_loss_cls = train_loss_cls.compute().item()
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

        print(f"Epoch {epoch+1:04d} | CE Loss: {epoch_loss_cls:.4f} | "
              f"Tr ACC: {epoch_train_acc:.4f} || Val ACC: {epoch_val_acc:.4f} | EMA: {epoch_val_ema_acc:.4f} (Peak: {best_ema_acc:.4f}) | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | Grad Norm: {grad_norm:.4f}")

        if epoch_val_ema_acc > best_ema_acc:
            best_ema_acc = epoch_val_ema_acc
            torch.save(ema.state_dict(), "best_convnext_base_1000ep.pth")

    print(
        f"[Execution Complete] Historical Peak Validation Accuracy: {best_ema_acc:.4f}")


if __name__ == "__main__":
    main()
