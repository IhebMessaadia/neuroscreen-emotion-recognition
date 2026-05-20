import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm


CLASS_NAMES = ["happy", "sad", "stressed", "neutral"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for name, idx in CLASS_TO_ID.items()}


@dataclass
class V3Config:
    cache_dir: str = "cache/dinov2_base_16f_features"
    v2_output_dir: str = "runs/dinov2_base_cached_temporal"
    output_dir: str = "runs/dinov2_base_v3_ensemble"
    batch_size: int = 64
    epochs: int = 80
    early_stopping_patience: int = 12
    seeds: List[int] = None
    heads: List[str] = None
    learning_rates: List[float] = None
    dropouts: List[float] = None
    weight_decays: List[float] = None
    label_smoothing: float = 0.05
    feature_noise_std: float = 0.01
    feature_dropout_p: float = 0.05
    mixup_alpha: float = 0.0
    top_k_ensemble: int = 5
    hidden_dim: int = 384
    transformer_layers: int = 1
    transformer_heads: int = 4
    num_workers: int = 0
    mixed_precision: bool = True
    debug_max_videos: int = 0
    debug_epochs: int = 2

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [11, 22, 33, 44, 55]
        if self.heads is None:
            self.heads = ["transformer_light", "attn_mlp", "bigru_attn"]
        if self.learning_rates is None:
            self.learning_rates = [0.0001, 0.0002]
        if self.dropouts is None:
            self.dropouts = [0.35, 0.45]
        if self.weight_decays is None:
            self.weight_decays = [0.001, 0.003]


def load_config(path: Path) -> V3Config:
    cfg = V3Config()
    data = json.loads(path.read_text(encoding="utf-8"))
    valid = set(asdict(cfg).keys())
    unknown = sorted(set(data) - valid)
    if unknown:
        print(f"Warning: unknown config keys ignored: {unknown}")
    for key, value in data.items():
        if key in valid:
            setattr(cfg, key, value)
    cfg.__post_init__()
    return cfg


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_v2_artifacts(cfg: V3Config) -> Dict[str, List[Dict[str, object]]]:
    split_path = Path(cfg.v2_output_dir) / "actor_group_split.json"
    manifest_path = Path(cfg.v2_output_dir) / "cache_manifest.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing V2 split: {split_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing V2 cache manifest: {manifest_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    manifest = {}
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            manifest[row["video_id"]] = row

    result = {}
    for split_name in ["train", "val", "test"]:
        rows = []
        for item in split[split_name]:
            video_id = str(item["video_id"])
            row = dict(manifest.get(video_id, {}))
            cache_path = Path(row.get("cache_path", item.get("cache_path", "")))
            if not cache_path.is_absolute():
                cache_path = Path(cache_path)
            if not cache_path.exists():
                raise FileNotFoundError(f"Missing cached feature for {video_id}: {cache_path}")
            rows.append(
                {
                    "split": split_name,
                    "video_id": video_id,
                    "actor_id": str(item["actor_id"]),
                    "class_name": str(item["class_name"]),
                    "label": int(item["label"]),
                    "video_path": str(item.get("path", row.get("video_path", ""))),
                    "cache_path": str(cache_path),
                }
            )
        result[split_name] = rows

    actor_overlap = (
        set(x["actor_id"] for x in result["train"]) & set(x["actor_id"] for x in result["val"])
        or set(x["actor_id"] for x in result["train"]) & set(x["actor_id"] for x in result["test"])
        or set(x["actor_id"] for x in result["val"]) & set(x["actor_id"] for x in result["test"])
    )
    if actor_overlap:
        raise RuntimeError("Actor leakage detected in V2 split.")
    return result


def limit_debug_samples(samples_by_split: Dict[str, List[Dict[str, object]]], max_videos: int) -> Dict[str, List[Dict[str, object]]]:
    if max_videos <= 0:
        return samples_by_split
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    limited = {}
    for split_name, samples in samples_by_split.items():
        target = max(4, int(round(max_videos * ratios[split_name])))
        by_class = defaultdict(list)
        for sample in samples:
            by_class[sample["class_name"]].append(sample)
        selected = []
        pos = 0
        while len(selected) < target:
            added = False
            for class_name in CLASS_NAMES:
                items = by_class[class_name]
                if pos < len(items):
                    selected.append(items[pos])
                    added = True
                    if len(selected) >= target:
                        break
            if not added:
                break
            pos += 1
        limited[split_name] = selected
    return limited


def label_counts(samples: Sequence[Dict[str, object]]) -> Dict[str, int]:
    counts = Counter(int(s["label"]) for s in samples)
    return {name: int(counts.get(i, 0)) for i, name in enumerate(CLASS_NAMES)}


def print_split_report(samples_by_split: Dict[str, List[Dict[str, object]]]) -> None:
    for split_name in ["train", "val", "test"]:
        samples = samples_by_split[split_name]
        actors = sorted(set(str(s["actor_id"]) for s in samples))
        print(f"{split_name:5s}: videos={len(samples):4d} | actors={len(actors):2d} | counts={label_counts(samples)}")


class CachedFeatureDataset(Dataset):
    def __init__(self, samples: Sequence[Dict[str, object]], train: bool, noise_std: float, feature_dropout_p: float):
        self.samples = list(samples)
        self.train = train
        self.noise_std = float(noise_std)
        self.feature_dropout_p = float(feature_dropout_p)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        data = torch.load(sample["cache_path"], map_location="cpu")
        features = data["features"].float()
        if self.train:
            if self.noise_std > 0:
                features = features + torch.randn_like(features) * self.noise_std
            if self.feature_dropout_p > 0:
                mask = torch.rand(features.shape[-1]) >= self.feature_dropout_p
                features = features * mask.float().view(1, -1)
        return {
            "features": features,
            "labels": torch.tensor(int(sample["label"]), dtype=torch.long),
            "path": str(sample["video_path"]),
            "video_id": str(sample["video_id"]),
        }


def collate_batch(batch):
    return {
        "features": torch.stack([x["features"] for x in batch], dim=0),
        "labels": torch.stack([x["labels"] for x in batch], dim=0),
        "paths": [x["path"] for x in batch],
        "video_ids": [x["video_id"] for x in batch],
    }


class AttentionPool(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, max(64, dim // 2)), nn.Tanh(), nn.Dropout(dropout), nn.Linear(max(64, dim // 2), 1))

    def forward(self, x):
        weights = torch.softmax(self.net(x).squeeze(-1), dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class TransformerLight(nn.Module):
    def __init__(self, input_dim: int, cfg: V3Config, dropout: float):
        super().__init__()
        h = int(cfg.hidden_dim)
        self.in_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, h)
        self.pos = nn.Parameter(torch.zeros(1, 16, h))
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=int(cfg.transformer_heads),
            dim_feedforward=h * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(cfg.transformer_layers))
        self.pool = AttentionPool(h, dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, len(CLASS_NAMES)))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        x = self.proj(self.in_norm(x))
        x = x + self.pos[:, : x.size(1), :]
        x = self.encoder(x)
        return self.classifier(self.pool(x))


class AttnMlpHead(nn.Module):
    def __init__(self, input_dim: int, cfg: V3Config, dropout: float):
        super().__init__()
        h = int(cfg.hidden_dim)
        self.in_norm = nn.LayerNorm(input_dim)
        self.frame_mlp = nn.Sequential(nn.Linear(input_dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, h), nn.GELU())
        self.pool = AttentionPool(h, dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(h * 3), nn.Linear(h * 3, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, len(CLASS_NAMES)))

    def forward(self, x):
        x = self.frame_mlp(self.in_norm(x))
        attn = self.pool(x)
        mean = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        return self.classifier(torch.cat([attn, mean, max_pool], dim=1))


class BiGruAttnHead(nn.Module):
    def __init__(self, input_dim: int, cfg: V3Config, dropout: float):
        super().__init__()
        h = int(cfg.hidden_dim)
        self.in_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, h)
        self.gru = nn.GRU(h, h // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pool = AttentionPool(h, dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, len(CLASS_NAMES)))

    def forward(self, x):
        x = self.proj(self.in_norm(x))
        x, _ = self.gru(x)
        return self.classifier(self.pool(x))


def build_model(head: str, input_dim: int, cfg: V3Config, dropout: float) -> nn.Module:
    if head == "transformer_light":
        return TransformerLight(input_dim, cfg, dropout)
    if head == "attn_mlp":
        return AttnMlpHead(input_dim, cfg, dropout)
    if head == "bigru_attn":
        return BiGruAttnHead(input_dim, cfg, dropout)
    raise ValueError(f"Unknown head: {head}")


def compact_plan(cfg: V3Config) -> List[Dict[str, object]]:
    plan = []
    for seed_index, seed in enumerate(cfg.seeds):
        for head_index, head in enumerate(cfg.heads):
            lr = cfg.learning_rates[(seed_index + head_index) % len(cfg.learning_rates)]
            dropout = cfg.dropouts[(seed_index + head_index) % len(cfg.dropouts)]
            wd = cfg.weight_decays[(seed_index + 2 * head_index) % len(cfg.weight_decays)]
            plan.append({"seed": int(seed), "head": head, "learning_rate": float(lr), "dropout": float(dropout), "weight_decay": float(wd)})
    return plan


def class_weights(samples, device):
    y = np.array([int(s["label"]) for s in samples])
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(CLASS_NAMES)), y=y).astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def sampler_for(samples):
    counts = Counter(int(s["label"]) for s in samples)
    weights = [1.0 / max(1, counts[int(s["label"])]) for s in samples]
    return WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)


def maybe_mixup(features, labels, alpha: float):
    if alpha <= 0:
        return features, labels, None
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(features.size(0), device=features.device)
    mixed = lam * features + (1.0 - lam) * features[perm]
    return mixed, labels, (labels[perm], float(lam))


def loss_fn(logits, labels, criterion, mixup_info):
    if mixup_info is None:
        return criterion(logits, labels)
    labels_b, lam = mixup_info
    return lam * criterion(logits, labels) + (1.0 - lam) * criterion(logits, labels_b)


def summarize(labels, preds, loss: float) -> Dict[str, float]:
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def train_epoch(model, loader, criterion, optimizer, device, cfg):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.mixed_precision) and device.type == "cuda")
    total_loss, total_count = 0.0, 0
    labels_all, preds_all = [], []
    for batch in tqdm(loader, desc="train", leave=False, disable=True):
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        features, labels_a, mix = maybe_mixup(features, labels, float(cfg.mixup_alpha))
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=bool(cfg.mixed_precision) and device.type == "cuda"):
            logits = model(features)
            loss = loss_fn(logits, labels_a, criterion, mix)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            total_count += bs
            labels_all.extend(labels.cpu().tolist())
            preds_all.extend(preds.cpu().tolist())
    return summarize(labels_all, preds_all, total_loss / max(1, total_count))


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg, pred_path: Optional[Path] = None) -> Dict[str, object]:
    model.eval()
    labels_all, preds_all, probs_all, paths_all = [], [], [], []
    total_loss, total_count = 0.0, 0
    for batch in tqdm(loader, desc="eval", leave=False, disable=True):
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=bool(cfg.mixed_precision) and device.type == "cuda"):
            logits = model(features)
            loss = criterion(logits, labels)
        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)
        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs
        labels_all.extend(labels.cpu().tolist())
        preds_all.extend(preds.cpu().tolist())
        probs_all.extend(probs.cpu().tolist())
        paths_all.extend(batch["paths"])
    metrics = summarize(labels_all, preds_all, total_loss / max(1, total_count))
    metrics["labels"] = labels_all
    metrics["preds"] = preds_all
    metrics["probs"] = probs_all
    if pred_path is not None:
        write_predictions(pred_path, paths_all, labels_all, preds_all, probs_all)
    return metrics


def write_predictions(path: Path, paths, labels, preds, probs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "pred_label"] + [f"prob_{c}" for c in CLASS_NAMES])
        for p, y, pred, prob in zip(paths, labels, preds, probs):
            writer.writerow([p, ID_TO_CLASS[int(y)], ID_TO_CLASS[int(pred)]] + [f"{float(x):.8f}" for x in prob])


def report_dict(labels, preds) -> Dict[str, object]:
    return classification_report(labels, preds, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, zero_division=0, output_dict=True)


def save_confusion(labels, preds, path: Path, title: str) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def first_feature_dim(samples):
    data = torch.load(samples[0]["cache_path"], map_location="cpu")
    return int(data["features"].shape[-1])


def make_loaders(samples_by_split, cfg, run_train_aug: bool = True):
    train_ds = CachedFeatureDataset(samples_by_split["train"], train=run_train_aug, noise_std=cfg.feature_noise_std, feature_dropout_p=cfg.feature_dropout_p)
    val_ds = CachedFeatureDataset(samples_by_split["val"], train=False, noise_std=0, feature_dropout_p=0)
    test_ds = CachedFeatureDataset(samples_by_split["test"], train=False, noise_std=0, feature_dropout_p=0)
    kwargs = {"batch_size": int(cfg.batch_size), "num_workers": int(cfg.num_workers), "pin_memory": torch.cuda.is_available(), "collate_fn": collate_batch}
    train_loader = DataLoader(train_ds, sampler=sampler_for(samples_by_split["train"]), **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


def train_single_run(run_cfg, run_index: int, cfg: V3Config, samples_by_split, device, epochs: int) -> Dict[str, object]:
    seed_everything(int(run_cfg["seed"]))
    run_name = f"run_{run_index:02d}_{run_cfg['head']}_seed{run_cfg['seed']}"
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    input_dim = first_feature_dim(samples_by_split["train"])
    train_loader, val_loader, test_loader = make_loaders(samples_by_split, cfg)
    model = build_model(str(run_cfg["head"]), input_dim, cfg, float(run_cfg["dropout"])).to(device)
    weights = class_weights(samples_by_split["train"], device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(cfg.label_smoothing))
    optimizer = AdamW(model.parameters(), lr=float(run_cfg["learning_rate"]), weight_decay=float(run_cfg["weight_decay"]))
    best_val, best_epoch, bad_epochs = -1.0, 0, 0
    history = []
    start = time.time()
    for epoch in range(1, int(epochs) + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, cfg)
        val_metrics = evaluate(model, val_loader, criterion, device, cfg)
        history.append({"epoch": epoch, "train": {k: v for k, v in train_metrics.items() if k not in {"labels", "preds", "probs"}}, "val": {k: v for k, v in val_metrics.items() if k not in {"labels", "preds", "probs"}}})
        print(f"{run_name} epoch {epoch}/{epochs}: train_mF1={train_metrics['macro_f1']:.4f} val_mF1={val_metrics['macro_f1']:.4f}")
        if float(val_metrics["macro_f1"]) > best_val:
            best_val = float(val_metrics["macro_f1"])
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "run_cfg": run_cfg, "input_dim": input_dim, "best_epoch": best_epoch, "best_val_macro_f1": best_val}, run_dir / "best_model.pt")
        else:
            bad_epochs += 1
        if bad_epochs >= int(cfg.early_stopping_patience):
            break
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_final = evaluate(model, val_loader, criterion, device, cfg, run_dir / "val_predictions.csv")
    test_final = evaluate(model, test_loader, criterion, device, cfg, run_dir / "test_predictions.csv")
    metrics = {
        "run_name": run_name,
        "run_cfg": run_cfg,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_macro_f1": float(checkpoint["best_val_macro_f1"]),
        "elapsed_seconds": time.time() - start,
        "val_metrics": {k: v for k, v in val_final.items() if k not in {"labels", "preds", "probs"}},
        "test_metrics": {k: v for k, v in test_final.items() if k not in {"labels", "preds", "probs"}},
        "val_report": report_dict(val_final["labels"], val_final["preds"]),
        "test_report": report_dict(test_final["labels"], test_final["preds"]),
        "confusion_matrix_val": confusion_matrix(val_final["labels"], val_final["preds"], labels=list(range(len(CLASS_NAMES)))).tolist(),
        "confusion_matrix_test": confusion_matrix(test_final["labels"], test_final["preds"], labels=list(range(len(CLASS_NAMES)))).tolist(),
        "history": history,
    }
    save_json(metrics, run_dir / "metrics.json")
    return metrics


def read_pred_csv(path: Path):
    paths, labels, probs = [], [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            paths.append(row["path"])
            labels.append(CLASS_TO_ID[row["true_label"]])
            probs.append([float(row[f"prob_{c}"]) for c in CLASS_NAMES])
    return paths, np.array(labels, dtype=np.int64), np.array(probs, dtype=np.float64)


def metrics_from_probs(labels, probs):
    preds = probs.argmax(axis=1)
    return summarize(labels.tolist(), preds.tolist(), 0.0), preds


def ensemble_from_runs(cfg: V3Config) -> Optional[Dict[str, object]]:
    out = Path(cfg.output_dir)
    run_metrics = []
    for metrics_path in sorted(out.glob("run_*/metrics.json")):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (metrics_path.parent / "val_predictions.csv").exists() and (metrics_path.parent / "test_predictions.csv").exists():
            run_metrics.append((float(data["best_val_macro_f1"]), metrics_path.parent, data))
    if not run_metrics:
        print("No V3 run predictions available for ensemble.")
        return None
    run_metrics.sort(key=lambda x: x[0], reverse=True)
    selected = run_metrics[: max(1, int(cfg.top_k_ensemble))]
    print("Selected ensemble members:")
    for score, path, _ in selected:
        print(f"  {path.name}: val_macroF1={score:.4f}")

    val_paths, val_labels, val_probs = read_pred_csv(selected[0][1] / "val_predictions.csv")
    test_paths, test_labels, test_probs = read_pred_csv(selected[0][1] / "test_predictions.csv")
    val_stack = [val_probs]
    test_stack = [test_probs]
    for _, path, _ in selected[1:]:
        _, labels_check, probs = read_pred_csv(path / "val_predictions.csv")
        if not np.array_equal(labels_check, val_labels):
            raise RuntimeError(f"Validation label order mismatch: {path}")
        val_stack.append(probs)
        _, labels_check, probs = read_pred_csv(path / "test_predictions.csv")
        if not np.array_equal(labels_check, test_labels):
            raise RuntimeError(f"Test label order mismatch: {path}")
        test_stack.append(probs)

    val_avg = np.mean(np.stack(val_stack, axis=0), axis=0)
    test_avg = np.mean(np.stack(test_stack, axis=0), axis=0)
    val_metrics, val_preds = metrics_from_probs(val_labels, val_avg)
    test_metrics, test_preds = metrics_from_probs(test_labels, test_avg)
    write_predictions(out / "ensemble_val_predictions.csv", val_paths, val_labels, val_preds, val_avg)
    write_predictions(out / "ensemble_test_predictions.csv", test_paths, test_labels, test_preds, test_avg)
    save_confusion(val_labels, val_preds, out / "ensemble_confusion_matrix_val.png", "V3 Ensemble Validation")
    save_confusion(test_labels, test_preds, out / "ensemble_confusion_matrix_test.png", "V3 Ensemble Test")
    payload = {
        "selected_runs": [path.name for _, path, _ in selected],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_report": report_dict(val_labels, val_preds),
        "test_report": report_dict(test_labels, test_preds),
        "confusion_matrix_val": confusion_matrix(val_labels, val_preds, labels=list(range(len(CLASS_NAMES)))).tolist(),
        "confusion_matrix_test": confusion_matrix(test_labels, test_preds, labels=list(range(len(CLASS_NAMES)))).tolist(),
    }
    save_json(payload, out / "ensemble_metrics.json")
    calibrate_ensemble(out, val_paths, val_labels, val_avg, test_paths, test_labels, test_avg)
    return payload


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def calibrate_ensemble(out: Path, val_paths, val_labels, val_probs, test_paths, test_labels, test_probs):
    logits_val = np.log(np.clip(val_probs, 1e-8, 1.0))
    logits_test = np.log(np.clip(test_probs, 1e-8, 1.0))
    candidates = [-0.5, -0.25, 0.0, 0.25, 0.5]
    best_bias = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    best_score = -1.0
    for b0 in candidates:
        for b1 in candidates:
            for b2 in candidates:
                for b3 in candidates:
                    bias = np.array([b0, b1, b2, b3], dtype=np.float64)
                    probs = softmax_np(logits_val + bias.reshape(1, -1))
                    preds = probs.argmax(axis=1)
                    score = f1_score(val_labels, preds, average="macro", zero_division=0)
                    if score > best_score:
                        best_score = float(score)
                        best_bias = bias
    val_cal = softmax_np(logits_val + best_bias.reshape(1, -1))
    test_cal = softmax_np(logits_test + best_bias.reshape(1, -1))
    val_metrics, val_preds = metrics_from_probs(val_labels, val_cal)
    test_metrics, test_preds = metrics_from_probs(test_labels, test_cal)
    write_predictions(out / "calibrated_ensemble_val_predictions.csv", val_paths, val_labels, val_preds, val_cal)
    write_predictions(out / "calibrated_ensemble_test_predictions.csv", test_paths, test_labels, test_preds, test_cal)
    payload = {
        "bias": {name: float(v) for name, v in zip(CLASS_NAMES, best_bias.tolist())},
        "selection_metric": "validation_macro_f1",
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_report": report_dict(val_labels, val_preds),
        "test_report": report_dict(test_labels, test_preds),
        "confusion_matrix_val": confusion_matrix(val_labels, val_preds, labels=list(range(len(CLASS_NAMES)))).tolist(),
        "confusion_matrix_test": confusion_matrix(test_labels, test_preds, labels=list(range(len(CLASS_NAMES)))).tolist(),
    }
    save_json(payload, out / "calibrated_ensemble_metrics.json")


def run_analyzer() -> None:
    import analyze_v2_v3_results

    analyze_v2_v3_results.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config_v3_ensemble.json"))
    parser.add_argument("--debug-max-videos", type=int, default=None)
    parser.add_argument("--debug-epochs", type=int, default=None)
    parser.add_argument("--run-debug-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--ensemble-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.debug_max_videos is not None:
        cfg.debug_max_videos = int(args.debug_max_videos)
    if args.debug_epochs is not None:
        cfg.debug_epochs = int(args.debug_epochs)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(asdict(cfg), output_dir / "used_config.json")

    if args.analyze_only:
        run_analyzer()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.run_debug_only:
        device = torch.device("cpu")
    print("=" * 90)
    print("CREMA-D V3 Cached Feature Ensemble")
    print("=" * 90)
    print(f"Python executable: {sys.executable}")
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'}")
    print(f"Training device for this command: {device}")
    print(f"Output directory: {cfg.output_dir}")
    if device.type != "cuda" and not args.run_debug_only:
        raise RuntimeError("CUDA is required for V3 training.")

    samples_by_split = load_v2_artifacts(cfg)
    if args.run_debug_only or int(cfg.debug_max_videos) > 0:
        samples_by_split = limit_debug_samples(samples_by_split, int(cfg.debug_max_videos))
    print_split_report(samples_by_split)

    if args.ensemble_only:
        ensemble_from_runs(cfg)
        return
    if not (args.train or args.run_debug_only):
        print("No training started. Use --train for full V3 or --run-debug-only for a smoke test.")
        return

    epochs = int(cfg.debug_epochs) if args.run_debug_only else int(cfg.epochs)
    plan = compact_plan(cfg)
    save_json({"plan": plan, "epochs": epochs, "debug": bool(args.run_debug_only)}, output_dir / "run_plan.json")
    print(f"Run plan: {len(plan)} runs | epochs per run: {epochs}")
    completed = []
    for idx, run_cfg in enumerate(plan, start=1):
        print("=" * 90)
        print(f"Starting run {idx}/{len(plan)}: {run_cfg}")
        completed.append(train_single_run(run_cfg, idx, cfg, samples_by_split, device, epochs))
    save_json({"completed_runs": len(completed), "runs": completed}, output_dir / "debug_summary.json" if args.run_debug_only else output_dir / "training_summary.json")
    ensemble_from_runs(cfg)
    print(f"Completed runs: {len(completed)}")
    if args.run_debug_only:
        print("Debug-only run complete. Full V3 training was not started.")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
