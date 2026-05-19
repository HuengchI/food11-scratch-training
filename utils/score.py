import copy
import argparse
import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import numpy as np
import torchmetrics

img_size = 224
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"


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


class VRAMInferManifold:

    def __init__(self, path, device):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset path {self.path} does not exist.")

        self.files = sorted([x for x in self.path.iterdir()
                            if x.suffix.lower() == ".jpg"])

        print(
            f"[IO] Parallel loading {len(self.files)} images from {self.path.name}...")
        with ThreadPoolExecutor(max_workers=64) as executor:
            imgs = list(
                tqdm(executor.map(load_img, self.files), total=len(self.files)))

        self.data = torch.from_numpy(np.stack(imgs)).permute(
            0, 3, 1, 2).to(device, dtype=torch.uint8)

        labels = [int(f.stem.split("_")[0])
                  if "_" in f.stem else -1 for f in self.files]
        self.labels = torch.tensor(labels, device=device).long()

        self.ids = [f.stem.split("_")[-1] for f in self.files]
        print(
            f"[System] {self.path.name} resident in VRAM. Shape: {self.data.shape}")


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

# Model Definition here

def get_model():
    model = RegNetYMultiViewEngine(num_classes=11).to(device)

    return model


def main():
    """
    Usage:
        `python score.py --ckpt checkpoint_path.pt --split val`: Calculate the val ACC score.
        `python score.py --ckpt best_food11_regnety.pt --split test --output`: Inference on the test set and generate the submission.csv file.
    """
    parser = argparse.ArgumentParser(
        description="Food11 Inference & Scoring Engine")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the saved EMA .pt or .pth file")
    parser.add_argument(
        "--split", type=str, choices=["val", "test"], required=True, help="Which split to evaluate on")
    parser.add_argument("--output", action="store_true",
                        help="Flag to generate submission.csv")
    args = parser.parse_args()

    print("[System] Calculating Train Set statistics...")
    train_manifold = VRAMInferManifold("food11/training", device)
    float_data = train_manifold.data.float() / 255.0
    mean = float_data.mean([0, 2, 3], keepdim=True)
    std = float_data.std([0, 2, 3], keepdim=True) + 1e-6
    del float_data, train_manifold
    torch.cuda.empty_cache()

    target_path = "food11/validation" if args.split == "val" else "food11/test"
    target_manifold = VRAMInferManifold(target_path, device)

    print(
        f"[System] Initializing model and loading checkpoint from {args.ckpt}...")
    base_model = get_model().to(device)

    ema = AveragedModel(
        base_model, multi_avg_fn=get_ema_multi_avg_fn(0.999), use_buffers=True)

    state_dict = torch.load(args.ckpt, map_location=device)
    ema.load_state_dict(state_dict)
    ema.eval()

    acc_metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=11).to(device)

    all_preds = []

    print(f"[Inference] Running forward passes on {args.split} set...")
    with torch.no_grad():
        for i in tqdm(range(0, len(target_manifold.data), batch_size), desc="Inferring"):
            v_imgs = target_manifold.data[i:i+batch_size].float() / 255.0
            v_imgs = (v_imgs - mean) / std
            v_labels = target_manifold.labels[i:i+batch_size]

            with torch.amp.autocast('cuda', dtype=torch.float16):

                v_logits = ema(v_imgs)

            preds = v_logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())

            if (v_labels != -1).all():
                acc_metric.update(v_logits.detach(), v_labels)

    if (target_manifold.labels != -1).all():
        final_acc = acc_metric.compute().item()
        print(f"\n[Result] {args.split.upper()} Set Accuracy: {final_acc:.4f}")
    else:
        print(
            f"\n[Result] {args.split.upper()} Set evaluated. Labels are hidden/missing (ACC not calculated).")

    if args.output:
        csv_filename = "submission.csv"
        print(f"[IO] Generating {csv_filename}...")
        with open(csv_filename, "w") as f:
            f.write("Id,Category\n")
            for img_id, pred in zip(target_manifold.ids, all_preds):

                f.write(f"{img_id},{pred}\n")
        print(f"[Success] Output successfully written to {csv_filename}")


if __name__ == "__main__":
    main()
