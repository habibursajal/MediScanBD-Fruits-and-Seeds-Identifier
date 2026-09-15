# ==============================================================================
# RESEARCH PROJECT: BDMediHerb Triple-Stream Hybrid Model (TSHE Framework)
# DATASET: ORIGINAL IMAGES ONLY (19 Classes, 3,800 Images)
# ARCHITECTURE: MobileNetV3 + ResNet50 + ViT-B16 + Learned Meta-Learner
# EVALUATION: 4-PASS FLIP-BASED TTA WITH LOGIT-LEVEL AVERAGING
# REPRODUCIBILITY: Fixed Seed + Shared Train/Val/Test Split
# ==============================================================================

# ==============================================================================
# PHASE 0: SYSTEM IMPORTS & ENVIRONMENT SETUP
# ==============================================================================
import os
import sys
import time
import gc
import random
import json
import hashlib
import pathlib
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

try:
    from torchinfo import summary
except ImportError:
    import subprocess
    print("[INFO] Installing torchinfo for detailed model analysis...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchinfo", "-q"])
    from torchinfo import summary


# ==============================================================================
# PHASE 1: GLOBAL CONFIGURATION & ENVIRONMENT DETECTOR
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

LOCAL_DATASET = pathlib.Path(r"E:\Research Project\BDMediHerb\Original Dataset")
LOCAL_PROJECT_ROOT = pathlib.Path(r"E:\Research Project\BDMediHerb\BDMediHerb_Reviewer_Revision")

if LOCAL_DATASET.exists():
    BASE_DIR = str(LOCAL_DATASET)
    PROJECT_ROOT = str(LOCAL_PROJECT_ROOT)
    print("[ENV] Local Windows PC detected")
elif os.path.exists("/kaggle/input/datasets/habibursojol/bdmediherb-original-final/Original Dataset"):
    BASE_DIR = "/kaggle/input/datasets/habibursojol/bdmediherb-original-final/Original Dataset"
    PROJECT_ROOT = "/kaggle/working/BDMediHerb_Reviewer_Revision"
    print("[ENV] Kaggle detected")
else:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/BDMediHerb/Original Dataset"
    PROJECT_ROOT = "/content/drive/MyDrive/BDMediHerb/BDMediHerb_Reviewer_Revision"
    print("[ENV] Google Colab detected")

SPLIT_FILE = os.path.join(PROJECT_ROOT, "shared_stratified_split.json")
CONTROLLED_INIT_DIR = os.path.join(PROJECT_ROOT, "Controlled_Baselines")
NUM_WORKERS = 0 if os.name == 'nt' else 2

BEST_3_NAMES = [
    "MobileNet_V3_Large",
    "ResNet50",
    "ViT_B16"
]

CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "num_classes": 19,
    "patience": 8,

    # Explicit hybrid optimizer settings
    "backbone_lr": 1e-6,
    "head_lr": 1e-4,
    "weight_decay": 1e-2,

    # Explicit focal-loss settings
    "focal_alpha": 1.0,
    "focal_gamma": 2.0,
    "focal_label_smoothing": 0.05,

    # Scheduler settings
    "scheduler_T0": 10,
    "scheduler_Tmult": 2,
    "scheduler_eta_min": 1e-7,

    # Dataset checks
    "expected_total_images": 3800,
    "expected_images_per_class": 200,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15
}

os.makedirs(PROJECT_ROOT, exist_ok=True)


# ==============================================================================
# PHASE 1A: REPRODUCIBILITY & UTILITIES
# ==============================================================================
def seed_everything(seed=42):
    """
    Ensures complete reproducibility across Python, NumPy, and PyTorch execution environments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

SHOW_FIGURES = True
FIG_DPI = 500


def show_saved_figure(path):
    """
    Displays a saved figure inline if running within IPython/Colab environments.
    """
    if not SHOW_FIGURES:
        return
    try:
        from IPython.display import display, Image as IPImage
        display(IPImage(filename=str(path)))
    except Exception as exc:
        print(f"[INFO] Figure saved but inline display was unavailable: {exc}")


def save_figure(path):
    """
    Saves a Matplotlib figure with high DPI for publication quality.
    """
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    show_saved_figure(path)
    plt.close()


def dataset_signature(ds):
    """
    Calculates a SHA-256 hash over image paths and targets to guarantee dataset integrity.
    """
    h = hashlib.sha256()
    for path, target in ds.samples:
        h.update(str(pathlib.Path(path).name).encode("utf-8"))
        h.update(str(int(target)).encode("utf-8"))
    return h.hexdigest()


def synchronize_cuda():
    """
    Synchronizes CUDA execution for exact latency and time benchmarking.
    """
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ==============================================================================
# PHASE 2: DATASET + SHARED STRATIFIED PARTITION
# ==============================================================================
class MapDataset(torch.utils.data.Dataset):
    """
    Wraps a subset to apply dynamic transformations during data loading.
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


def create_or_load_shared_split(full_ds):
    """
    Creates or loads the exact stratified 70:15:15 partition shared across all experiments.
    """
    labels = np.array(full_ds.targets)
    indices = np.arange(len(labels))

    if len(full_ds) != CONFIG["expected_total_images"]:
        raise ValueError(
            f"Expected {CONFIG['expected_total_images']} original images, "
            f"but found {len(full_ds)} in BASE_DIR."
        )

    class_counts = np.bincount(
        labels,
        minlength=CONFIG["num_classes"]
    )

    if not np.all(class_counts == CONFIG["expected_images_per_class"]):
        raise ValueError(
            f"Expected 200 images per class. Observed counts: {class_counts.tolist()}"
        )

    current_signature = dataset_signature(full_ds)

    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE, "r") as f:
            split = json.load(f)

        saved_signature = split.get("dataset_signature")
        if saved_signature is not None and saved_signature != current_signature:
            print("[WARN] Shared split signature differs. Recreating split with seed 42.")
            os.remove(SPLIT_FILE)
            return create_or_load_shared_split(full_ds)

        train_idx = np.array(split["train_idx"], dtype=int)
        val_idx = np.array(split["val_idx"], dtype=int)
        test_idx = np.array(split["test_idx"], dtype=int)

        if (len(train_idx), len(val_idx), len(test_idx)) != (2660, 570, 570):
            raise RuntimeError("Invalid split sizes. Expected 2660/570/570.")

        print(f"[INFO] Loaded verified shared split from: {SPLIT_FILE}")
    else:
        tr_val_idx, test_idx = train_test_split(
            indices,
            test_size=CONFIG["test_ratio"],
            stratify=labels,
            random_state=SEED
        )

        val_fraction_of_remaining = CONFIG["val_ratio"] / (
            CONFIG["train_ratio"] + CONFIG["val_ratio"]
        )

        train_idx, val_idx = train_test_split(
            tr_val_idx,
            test_size=val_fraction_of_remaining,
            stratify=labels[tr_val_idx],
            random_state=SEED
        )

        split = {
            "seed": SEED,
            "train_ratio": CONFIG["train_ratio"],
            "val_ratio": CONFIG["val_ratio"],
            "test_ratio": CONFIG["test_ratio"],
            "dataset_signature": current_signature,
            "num_classes": len(full_ds.classes),
            "total_images": len(full_ds),
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist()
        }

        os.makedirs(os.path.dirname(SPLIT_FILE), exist_ok=True)

        with open(SPLIT_FILE, "w") as f:
            json.dump(split, f, indent=2)

        print(f"[INFO] Created shared split: {SPLIT_FILE}")

    print(
        f"[INFO] Split Partition Complete -> "
        f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}"
    )

    return train_idx, val_idx, test_idx


def get_hybrid_loaders(train_idx, val_idx, test_idx, img_size, full_ds):
    """
    Constructs PyTorch DataLoaders with resolution-aware augmentations for training
    and deterministic transformations for validation and testing.
    """
    train_trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    eval_trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_loader = DataLoader(
        MapDataset(Subset(full_ds, train_idx), train_trans),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        MapDataset(Subset(full_ds, val_idx), eval_trans),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        MapDataset(Subset(full_ds, test_idx), eval_trans),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


# ==============================================================================
# PHASE 3: ARCHITECTURAL FACTORY & LOSS FUNCTIONS
# ==============================================================================
def initialize_architecture(name):
    """
    Recreates individual architectures to match saved weights exactly.
    """
    if name == "MobileNet_V3_Large":
        model = models.mobilenet_v3_large(weights="DEFAULT")
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, CONFIG["num_classes"])
        )

    elif name == "ResNet50":
        model = models.resnet50(weights="DEFAULT")
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, CONFIG["num_classes"])

    elif name == "ViT_B16":
        model = models.vit_b_16(weights="DEFAULT")
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, CONFIG["num_classes"])

    else:
        raise ValueError(f"Unsupported backbone: {name}")

    return model.to(DEVICE)


class FocalLoss(nn.Module):
    """
    Implements Focal Loss with label smoothing to handle difficult samples.
    """
    def __init__(
        self,
        alpha=1.0,
        gamma=2.0,
        label_smoothing=0.05
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal = (
            self.alpha *
            ((1 - pt) ** self.gamma) *
            ce_loss
        )
        return focal.mean()


class FeatureExtractor(nn.Module):
    """
    Loads controlled baseline checkpoints and removes final classification heads.
    """
    def __init__(self, model_name):
        super().__init__()
        base_model = initialize_architecture(model_name)
        weights_path = os.path.join(CONTROLLED_INIT_DIR, model_name, "best_model.pth")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Controlled checkpoint not found:\n{weights_path}\n"
                f"Please execute 02_controlled_individual_baselines.py first."
            )

        base_model.load_state_dict(
            torch.load(weights_path, map_location=DEVICE)
        )

        if "ViT" in model_name:
            self.feat_dim = base_model.heads.head.in_features
            base_model.heads = nn.Identity()

        elif "ResNet" in model_name:
            self.feat_dim = base_model.fc.in_features
            base_model.fc = nn.Identity()

        elif "MobileNet" in model_name:
            self.feat_dim = base_model.classifier[0].in_features
            base_model.classifier = nn.Identity()

        else:
            raise ValueError(f"Unsupported feature extractor: {model_name}")

        self.backbone = base_model

    def forward(self, x):
        x = self.backbone(x)
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        return x


class EnsembleStackingNet(nn.Module):
    """
    Triple-Stream Stacking Architecture with GELU Meta-Learner.
    Converts 3x19 streams into 57 logits, mapping to 256 -> 19 outputs.
    """
    def __init__(self, model_names, num_classes=19):
        super().__init__()

        self.stream1 = FeatureExtractor(model_names[0])
        self.stream2 = FeatureExtractor(model_names[1])
        self.stream3 = FeatureExtractor(model_names[2])

        self.head1 = nn.Linear(self.stream1.feat_dim, num_classes)
        self.head2 = nn.Linear(self.stream2.feat_dim, num_classes)
        self.head3 = nn.Linear(self.stream3.feat_dim, num_classes)

        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * 3, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f1 = self.stream1(x)
        f2 = self.stream2(x)
        f3 = self.stream3(x)

        out1 = self.head1(f1)
        out2 = self.head2(f2)
        out3 = self.head3(f3)

        stacked_logits = torch.cat([out1, out2, out3], dim=-1)
        return self.meta_learner(stacked_logits)


# ==============================================================================
# PHASE 4: DIAGNOSTIC & ARTIFACT REPORTING UTILITIES
# ==============================================================================
def export_model_summary_csv(model, export_path, img_size=224):
    """
    Saves layer-wise architectural metadata using torchinfo for journal review.
    """
    try:
        stats = summary(
            model,
            input_size=(CONFIG["batch_size"], 3, img_size, img_size),
            verbose=0
        )

        summary_list = []
        for layer in stats.summary_list:
            summary_list.append({
                "Layer Name": layer.class_name,
                "Input Shape": str(layer.input_size),
                "Output Shape": str(layer.output_size),
                "Parameters": layer.num_params,
                "Trainable": layer.trainable
            })

        pd.DataFrame(summary_list).to_csv(
            os.path.join(export_path, "architecture_summary.csv"),
            index=False
        )
    except Exception as exc:
        print(f"[INFO] Architecture summary skipped: {exc}")


def run_failure_analysis(model, loader, class_names, export_path, num_visuals=20):
    """
    Exports misclassified samples and detailed failure CSV logs.
    """
    model.eval()
    error_dir = os.path.join(export_path, "Failure_Analysis")
    os.makedirs(error_dir, exist_ok=True)

    error_log = []
    visual_count = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs_dev = inputs.to(DEVICE, non_blocking=True)

            # Use 4-pass TTA logits for failure identification
            o1 = model(inputs_dev)
            o2 = model(torch.flip(inputs_dev, dims=[3]))
            o3 = model(torch.flip(inputs_dev, dims=[2]))
            o4 = model(torch.flip(inputs_dev, dims=[2, 3]))

            avg_logits = (o1 + o2 + o3 + o4) / 4.0
            probs = F.softmax(avg_logits, dim=1)
            confs, preds = torch.max(probs, 1)

            wrong = (preds.cpu() != labels).nonzero(as_tuple=True)[0]

            for idx in wrong:
                act_nm = class_names[labels[idx].item()]
                prd_nm = class_names[preds[idx].item()]
                conf = round(confs[idx].item() * 100, 4)

                error_log.append({
                    "Actual": act_nm,
                    "Predicted": prd_nm,
                    "Confidence (%)": conf
                })

                if visual_count < num_visuals:
                    img = inputs[idx].numpy().transpose((1, 2, 0))
                    img = np.clip(
                        img * np.array([0.229, 0.224, 0.225]) +
                        np.array([0.485, 0.456, 0.406]),
                        0, 1
                    )

                    plt.figure()
                    plt.imshow(img)
                    plt.axis("off")
                    plt.title(
                        f"Actual: {act_nm} | Pred: {prd_nm}\nConf: {conf}%",
                        color="red"
                    )
                    save_figure(
                        os.path.join(error_dir, f"fail_{visual_count + 1}.png")
                    )
                    visual_count += 1

    pd.DataFrame(error_log).to_csv(
        os.path.join(export_path, "detailed_misclassifications.csv"),
        index=False
    )


# ==============================================================================
# PHASE 5: MAIN TRAINING EXECUTION
# ==============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("BDMEDIHERB TRIPLE-STREAM HYBRID MODEL TRAINING (TSHE PIPELINE)")
    print("=" * 80)

    hybrid_ds_raw = datasets.ImageFolder(BASE_DIR)
    DATASET_SIGNATURE = dataset_signature(hybrid_ds_raw)

    print(f"[INFO] Dataset Signature: {DATASET_SIGNATURE}")
    print(f"[INFO] Total Images Loaded: {len(hybrid_ds_raw)}")
    print(f"[INFO] Classes Count: {len(hybrid_ds_raw.classes)}")

    idx_train_hybrid, hybrid_val_idx, hybrid_test_idx = create_or_load_shared_split(
        hybrid_ds_raw
    )

    hybrid_loaders = get_hybrid_loaders(
        idx_train_hybrid,
        hybrid_val_idx,
        hybrid_test_idx,
        224,
        hybrid_ds_raw
    )

    model_hybrid = EnsembleStackingNet(
        BEST_3_NAMES,
        num_classes=CONFIG["num_classes"]
    ).to(DEVICE)

    EXPORT_DIR_HYB = os.path.join(PROJECT_ROOT, "Hybrid_TripleStream")
    os.makedirs(EXPORT_DIR_HYB, exist_ok=True)

    export_model_summary_csv(model_hybrid, EXPORT_DIR_HYB, 224)

    optimizer = optim.AdamW([
        {"params": model_hybrid.stream1.parameters(), "lr": CONFIG["backbone_lr"]},
        {"params": model_hybrid.stream2.parameters(), "lr": CONFIG["backbone_lr"]},
        {"params": model_hybrid.stream3.parameters(), "lr": CONFIG["backbone_lr"]},
        {"params": model_hybrid.head1.parameters(), "lr": CONFIG["head_lr"]},
        {"params": model_hybrid.head2.parameters(), "lr": CONFIG["head_lr"]},
        {"params": model_hybrid.head3.parameters(), "lr": CONFIG["head_lr"]},
        {"params": model_hybrid.meta_learner.parameters(), "lr": CONFIG["head_lr"]}
    ], weight_decay=CONFIG["weight_decay"])

    criterion = FocalLoss(
        alpha=CONFIG["focal_alpha"],
        gamma=CONFIG["focal_gamma"],
        label_smoothing=CONFIG["focal_label_smoothing"]
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CONFIG["scheduler_T0"],
        T_mult=CONFIG["scheduler_Tmult"],
        eta_min=CONFIG["scheduler_eta_min"]
    )

    best_model_path = os.path.join(EXPORT_DIR_HYB, "best_hybrid_model.pth")

    best_hyb_acc = 0.0
    early_stop_counter = 0

    len_train = len(idx_train_hybrid)
    len_val = len(hybrid_val_idx)

    history_hyb = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "inf_ms": [],
        "time_sec": [],
        "backbone_lr": [],
        "head_lr": []
    }

    print(
        f"\n{'Epoch':<8} | {'Tr. Loss':<10} | {'Tr. Acc':<10} | "
        f"{'Val Loss':<10} | {'Val Acc':<10} | {'Inference':<12} | {'Time'}"
    )
    print("-" * 115)

    # Training Loop
    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.perf_counter()

        model_hybrid.train()
        tr_loss, tr_correct = 0.0, 0

        for imgs, lbls in hybrid_loaders["train"]:
            imgs = imgs.to(DEVICE, non_blocking=True)
            lbls = lbls.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            out = model_hybrid(imgs)
            loss = criterion(out, lbls)

            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * imgs.size(0)
            tr_correct += (out.max(1)[1] == lbls).sum().item()

        model_hybrid.eval()
        v_loss, v_correct, latencies = 0.0, 0, []

        with torch.no_grad():
            for imgs, lbls in hybrid_loaders["val"]:
                imgs = imgs.to(DEVICE, non_blocking=True)
                lbls = lbls.to(DEVICE, non_blocking=True)

                synchronize_cuda()
                tic = time.perf_counter()

                out = model_hybrid(imgs)

                synchronize_cuda()
                elapsed = time.perf_counter() - tic

                latencies.append(elapsed / imgs.size(0))

                v_loss += criterion(out, lbls).item() * imgs.size(0)
                v_correct += (out.max(1)[1] == lbls).sum().item()

        cur_t_acc = round(tr_correct / len_train, 4)
        cur_v_acc = round(v_correct / len_val, 4)
        cur_t_loss = round(tr_loss / len_train, 4)
        cur_v_loss = round(v_loss / len_val, 4)
        cur_inf_ms = round(np.mean(latencies) * 1000, 4)
        cur_time = round(time.perf_counter() - epoch_start, 2)

        history_hyb["epoch"].append(epoch + 1)
        history_hyb["train_loss"].append(cur_t_loss)
        history_hyb["train_acc"].append(cur_t_acc)
        history_hyb["val_loss"].append(cur_v_loss)
        history_hyb["val_acc"].append(cur_v_acc)
        history_hyb["inf_ms"].append(cur_inf_ms)
        history_hyb["time_sec"].append(cur_time)
        history_hyb["backbone_lr"].append(optimizer.param_groups[0]["lr"])
        history_hyb["head_lr"].append(optimizer.param_groups[3]["lr"])

        print(
            f"{epoch+1:<8} | {cur_t_loss:<10.4f} | {cur_t_acc:<10.4f} | "
            f"{cur_v_loss:<10.4f} | {cur_v_acc:<10.4f} | "
            f"{cur_inf_ms:<10.4f} ms | {cur_time:<8}s"
        )

        if cur_v_acc > best_hyb_acc:
            best_hyb_acc = cur_v_acc

            torch.save(model_hybrid.state_dict(), best_model_path)

            print(
                f" ---> [SAVED] Model improved to {best_hyb_acc:.4f}. Checkpoint synced."
            )

            early_stop_counter = 0
        else:
            early_stop_counter += 1

            if early_stop_counter >= CONFIG["patience"]:
                print(f"\n[INFO] Early stopping triggered at epoch {epoch + 1}")
                break

        scheduler.step(epoch + 1)

    pd.DataFrame(history_hyb).to_csv(
        os.path.join(EXPORT_DIR_HYB, "hybrid_training_history.csv"),
        index=False
    )

    metadata = {
        "seed": SEED,
        "total_images": len(hybrid_ds_raw),
        "num_classes": len(hybrid_ds_raw.classes),
        "train_images": len(idx_train_hybrid),
        "validation_images": len(hybrid_val_idx),
        "test_images": len(hybrid_test_idx),
        "best_validation_accuracy": best_hyb_acc,
        "initialization": "Controlled_Baselines checkpoints required",
        "backbone_lr": CONFIG["backbone_lr"],
        "head_lr": CONFIG["head_lr"],
        "focal_gamma": CONFIG["focal_gamma"],
        "focal_label_smoothing": CONFIG["focal_label_smoothing"],
        "weight_decay": CONFIG["weight_decay"],
        "scheduler": "CosineAnnealingWarmRestarts",
        "scheduler_T0": CONFIG["scheduler_T0"],
        "scheduler_Tmult": CONFIG["scheduler_Tmult"],
        "scheduler_eta_min": CONFIG["scheduler_eta_min"],
        "tta": "4-pass flip-based TTA; logits averaged before softmax",
        "dataset_signature": DATASET_SIGNATURE
    }

    with open(
        os.path.join(EXPORT_DIR_HYB, "experiment_metadata.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=2)

    del model_hybrid, optimizer
    gc.collect()
    torch.cuda.empty_cache()


    # ==========================================================================
    # PHASE 6: FINAL EVALUATION WITH 4-PASS FLIP-BASED TTA
    # ==========================================================================
    print("\n" + "=" * 80)
    print("[INFO] Finalizing Hybrid Model Evaluation with 4-Pass TTA...")
    print("=" * 80)

    eval_model = EnsembleStackingNet(
        BEST_3_NAMES,
        num_classes=CONFIG["num_classes"]
    ).to(DEVICE)

    eval_model.load_state_dict(
        torch.load(best_model_path, map_location=DEVICE)
    )
    eval_model.eval()

    # Warmup CUDA
    with torch.inference_mode():
        warm = next(iter(hybrid_loaders["test"]))[0].to(DEVICE)
        for _ in range(10):
            _ = eval_model(warm)
        synchronize_cuda()

    y_true, y_pred, y_score = [], [], []
    tta_latencies = []
    single_pass_latencies = []

    with torch.no_grad():
        for imgs, lbls in hybrid_loaders["test"]:
            imgs = imgs.to(DEVICE, non_blocking=True)

            synchronize_cuda()
            single_t0 = time.perf_counter()
            _single_logits = eval_model(imgs)
            synchronize_cuda()
            single_pass_latencies.append(
                (time.perf_counter() - single_t0) / imgs.size(0)
            )

            synchronize_cuda()
            tic = time.perf_counter()

            o1 = _single_logits
            o2 = eval_model(torch.flip(imgs, dims=[3]))
            o3 = eval_model(torch.flip(imgs, dims=[2]))
            o4 = eval_model(torch.flip(imgs, dims=[2, 3]))

            synchronize_cuda()
            tta_latencies.append((time.perf_counter() - tic) / imgs.size(0))

            averaged_logits = (o1 + o2 + o3 + o4) / 4.0
            probabilities = F.softmax(averaged_logits, dim=1)
            _, preds = torch.max(averaged_logits, 1)

            y_true.extend(lbls.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probabilities.cpu().numpy())

    pd.DataFrame({
        "metric": ["mean_tta_latency_ms_per_image", "std_tta_latency_ms_per_image"],
        "value": [np.mean(tta_latencies) * 1000, np.std(tta_latencies) * 1000]
    }).to_csv(
        os.path.join(EXPORT_DIR_HYB, "tta_latency_summary.csv"),
        index=False
    )

    pd.DataFrame({
        "metric": ["mean_single_pass_latency_ms_per_image", "std_single_pass_latency_ms_per_image"],
        "value": [np.mean(single_pass_latencies) * 1000, np.std(single_pass_latencies) * 1000]
    }).to_csv(
        os.path.join(EXPORT_DIR_HYB, "single_pass_latency_summary.csv"),
        index=False
    )

    print(f"[LATENCY] TSHE Single-pass: {np.mean(single_pass_latencies)*1000:.4f} ms/image")
    print(f"[LATENCY] TSHE 4-Pass TTA:   {np.mean(tta_latencies)*1000:.4f} ms/image")


    # ==========================================================================
    # PHASE 7: CLASSIFICATION REPORT & CSV METRICS
    # ==========================================================================
    report_hyb = classification_report(
        y_true,
        y_pred,
        target_names=hybrid_ds_raw.classes,
        output_dict=True,
        digits=4
    )

    pd.DataFrame(report_hyb).transpose().to_csv(
        os.path.join(EXPORT_DIR_HYB, "hybrid_final_metrics_report.csv"),
        float_format="%.4f"
    )

    print(
        "\n[RESULT] Hybrid Model Test Accuracy:",
        round(report_hyb["accuracy"] * 100, 2),
        "%"
    )


    # ==========================================================================
    # PHASE 8: CONFUSION MATRIX GENERATION
    # ==========================================================================
    plt.figure(figsize=(16, 14))

    cm_hyb = confusion_matrix(y_true, y_pred)

    sns.heatmap(
        cm_hyb,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=hybrid_ds_raw.classes,
        yticklabels=hybrid_ds_raw.classes
    )

    plt.title(
        "Confusion Matrix: Proposed Triple-Stream Hybrid",
        fontsize=18,
        fontweight="bold"
    )
    plt.ylabel("Ground Truth", fontsize=14)
    plt.xlabel("Predicted Class", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_figure(os.path.join(EXPORT_DIR_HYB, "hybrid_confusion_matrix.png"))


    # ==========================================================================
    # PHASE 9: MULTI-CLASS ROC-AUC CURVES
    # ==========================================================================
    y_true_bin = label_binarize(
        y_true,
        classes=range(CONFIG["num_classes"])
    )

    plt.figure(figsize=(12, 10))

    for i in range(CONFIG["num_classes"]):
        fpr, tpr, _ = roc_curve(
            y_true_bin[:, i],
            np.array(y_score)[:, i]
        )
        plt.plot(
            fpr,
            tpr,
            label=f"{hybrid_ds_raw.classes[i]} (AUC={auc(fpr, tpr):.4f})"
        )

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=9
    )

    plt.title("Multi-class ROC: Hybrid Model", fontsize=16)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    save_figure(os.path.join(EXPORT_DIR_HYB, "hybrid_roc_auc.png"))


    # ==========================================================================
    # PHASE 10: PER-CLASS MACRO F1-SCORE BAR CHART
    # ==========================================================================
    f1_vals = [
        report_hyb[cls]["f1-score"] * 100
        for cls in hybrid_ds_raw.classes
    ]

    fig, ax = plt.subplots(figsize=(16, 8))
    x_bar = np.arange(len(f1_vals))
    bars = ax.bar(x_bar, f1_vals, width=0.7)

    ax.set_xticks(x_bar)
    ax.set_xticklabels(
        hybrid_ds_raw.classes,
        rotation=45,
        ha="right"
    )
    ax.set_ylabel("Macro F1-score (%)")
    ax.set_title(
        "Per-Class F1-score: Proposed Triple-Stream Hybrid",
        fontsize=16,
        fontweight="bold"
    )
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)

    for bar, v in zip(bars, f1_vals):
        ax.annotate(
            f"{v:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold"
        )

    fig.tight_layout()
    save_figure(os.path.join(EXPORT_DIR_HYB, "hybrid_f1_score_performance.png"))


    # ==========================================================================
    # PHASE 11: LEARNING CURVES PLOTTING
    # ==========================================================================
    df_hist_hyb = pd.read_csv(
        os.path.join(EXPORT_DIR_HYB, "hybrid_training_history.csv")
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        df_hist_hyb["train_acc"],
        label="Training Accuracy",
        marker="o",
        alpha=0.75
    )
    plt.plot(
        df_hist_hyb["val_acc"],
        label="Validation Accuracy",
        marker="s",
        alpha=0.75
    )
    plt.title("Hybrid Model: Accuracy Curve", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_figure(os.path.join(EXPORT_DIR_HYB, "hybrid_accuracy_curve.png"))

    plt.figure(figsize=(10, 6))
    plt.plot(
        df_hist_hyb["train_loss"],
        label="Training Loss",
        marker="o",
        alpha=0.75
    )
    plt.plot(
        df_hist_hyb["val_loss"],
        label="Validation Loss",
        marker="s",
        alpha=0.75
    )
    plt.title("Hybrid Model: Loss Curve", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_figure(os.path.join(EXPORT_DIR_HYB, "hybrid_loss_curve.png"))


    # ==========================================================================
    # PHASE 12: FAILURE ANALYSIS & ERROR LOGGING
    # ==========================================================================
    run_failure_analysis(
        eval_model,
        hybrid_loaders["test"],
        hybrid_ds_raw.classes,
        EXPORT_DIR_HYB
    )

    print(
        f"\n[SUCCESS] ALL Hybrid Evaluation Artifacts & Plots Saved to: "
        f"{EXPORT_DIR_HYB}"
    )