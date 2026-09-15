# ==============================================================================
# RESEARCH PROJECT: Automated Classification of Bangladeshi Medicinal Fruits/Seeds
# DATASET: BDMediHerb - ORIGINAL IMAGES ONLY (19 Classes, 3,800 Images)
# PIPELINE: Sequential Multi-Model Training & Comparative Analysis
# STRATEGY: Stratified 70:15:15 Partitioning | Dynamic Resolution | Partial Fine-tuning
# BASELINE PROTOCOL: Original/Native Individual-Model Training Configuration
# REPRODUCIBILITY: Fixed Seed + Shared Train/Val/Test Split
# ==============================================================================

# PHASE 0: SYSTEM IMPORTS & ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
import os
import time
import gc
import random
import json
import hashlib
import pathlib
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

try:
    from torchinfo import summary
except ImportError:
    print("[INFO] Installing torchinfo for detailed model analysis...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchinfo", "-q"])
    from torchinfo import summary



# PHASE 1: GLOBAL CONFIGURATIONS & HYPERPARAMETERS
# ------------------------------------------------------------------------------
# --------------------------- ENVIRONMENT -----------------------------------
# Works in both Kaggle and Google Colab. Kaggle paths are preferred when the
# dataset is available there; Colab Drive is used otherwise.
# PHASE 1: GLOBAL CONFIGURATIONS & HYPERPARAMETERS
# ------------------------------------------------------------------------------
LOCAL_DATASET = r"E:\Research Project\BDMediHerb\Original Dataset"
LOCAL_PROJECT_ROOT = r"E:\Research Project\BDMediHerb\BDMediHerb_Reviewer_Revision"

if os.path.exists(LOCAL_DATASET):
    BASE_DIR = LOCAL_DATASET
    PROJECT_ROOT = LOCAL_PROJECT_ROOT
    print("[ENV] Local Windows PC detected")

elif os.path.exists("/kaggle/input/datasets/habibursojol/bdmediherb/Original Dataset - Copy"):
    BASE_DIR = "/kaggle/input/datasets/habibursojol/bdmediherb/Original Dataset - Copy"
    PROJECT_ROOT = "/kaggle/working/BDMediHerb_Reviewer_Revision"
    print("[ENV] Kaggle detected")
    
else:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/BDMediHerb/Original Dataset"
    PROJECT_ROOT = "/content/drive/MyDrive/BDMediHerb/BDMediHerb_Reviewer_Revision"
    print("[ENV] Google Colab detected")

SPLIT_FILE = os.path.join(PROJECT_ROOT, "shared_stratified_split.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "lr": 5e-6,
    "num_classes": 19,
    "patience": 8,
    "label_smoothing": 0.3,
    "weight_decay": 1e-2,
    "expected_total_images": 3800,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15
}

MODEL_LIST = [
    "MobileNet_V3_Large",
    "EfficientNetV2_S",
    "ResNet50",
    "DenseNet121",
    "ViT_B16",
    "Inception_V3",
    "ConvNeXt_Tiny"
]

os.makedirs(PROJECT_ROOT, exist_ok=True)


# PHASE 1A: REPRODUCIBILITY UTILITIES
# ------------------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic settings improve reproducibility.
    # benchmark=False is intentionally used because input sizes are fixed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ==============================================================================
# REVIEWER-SAFE RESEARCH UTILITIES
# ==============================================================================
SHOW_FIGURES = True
FIG_DPI = 500


def show_saved_figure(path):
    """Display a saved publication figure in Colab while keeping the file on Drive."""
    if not SHOW_FIGURES:
        return
    try:
        from IPython.display import display, Image as IPImage
        display(IPImage(filename=str(path)))
    except Exception as exc:
        print(f"[INFO] Figure saved but inline display was unavailable: {exc}")


def save_figure(path):
    """Save a figure at publication resolution and immediately display it."""
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    show_saved_figure(path)
    plt.close()


def dataset_signature(ds):
    """Hash the ordered relative image paths and targets for reproducibility."""
    h = hashlib.sha256()
    for path, target in ds.samples:
        h.update(str(pathlib.Path(path).name).encode("utf-8"))
        h.update(str(int(target)).encode("utf-8"))
    return h.hexdigest()


def exact_latency_ms(model, loader, warmup_batches=10):
    """Measure complete model forward latency, synchronized for CUDA."""
    model.eval()
    times = []
    with torch.inference_mode():
        for batch_id, (imgs, _) in enumerate(loader):
            imgs = imgs.to(DEVICE, non_blocking=True)
            if batch_id < warmup_batches:
                _ = model(imgs)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                continue
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(imgs)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / imgs.size(0) * 1000.0)
    return float(np.mean(times)) if times else float("nan"), float(np.std(times)) if times else float("nan")


# PHASE 2: DATA ENGINEERING UTILITIES
# ------------------------------------------------------------------------------
class MapDataset(torch.utils.data.Dataset):
    """Wraps a subset to apply model-specific image augmentations."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


def create_or_load_shared_split(full_ds):
    """
    Creates the exact stratified 70:15:15 image-level split once and
    reuses it for all baseline models and the hybrid model.

    IMPORTANT:
    BASE_DIR must contain ONLY the 3,800 original images.
    The public dataset's 15,200 pre-augmented images are not used.
    """
    labels = np.array(full_ds.targets)
    indices = np.arange(len(labels))

    if len(full_ds) != CONFIG["expected_total_images"]:
        raise ValueError(
            f"Expected {CONFIG['expected_total_images']} original images, "
            f"but found {len(full_ds)} in:\n{BASE_DIR}\n"
            "Please make sure BASE_DIR points to the Original Dataset folder only."
        )

    class_counts = np.bincount(labels, minlength=CONFIG["num_classes"])
    if len(class_counts) != CONFIG["num_classes"] or not np.all(class_counts == 200):
        raise ValueError(
            f"Expected 19 classes with 200 original images each. "
            f"Observed class counts: {class_counts.tolist()}"
        )

    current_signature = dataset_signature(full_ds)

    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE, "r") as f:
            split = json.load(f)

        saved_signature = split.get("dataset_signature")
        if saved_signature is not None and saved_signature != current_signature:
            raise RuntimeError(
                "Existing shared split belongs to a different dataset version. "
                "Delete/rename shared_stratified_split.json and rerun the split. "
                f"Saved signature={saved_signature[:16]}..., current={current_signature[:16]}..."
            )

        train_idx = np.array(split["train_idx"], dtype=int)
        val_idx = np.array(split["val_idx"], dtype=int)
        test_idx = np.array(split["test_idx"], dtype=int)

        if len(train_idx) != 2660 or len(val_idx) != 570 or len(test_idx) != 570:
            raise RuntimeError(
                f"Invalid shared split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}. "
                "Expected 2660/570/570."
            )
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

        with open(SPLIT_FILE, "w") as f:
            json.dump(split, f, indent=2)

        print(f"[INFO] New shared split saved to: {SPLIT_FILE}")

    print(
        f"[INFO] Data Split Complete -> "
        f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}"
    )

    return train_idx, val_idx, test_idx


def get_loaders_for_resolution(train_idx, val_idx, test_idx, img_size, full_ds):
    """Generates resolution-aware DataLoaders and Weighted Samplers."""
    train_trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    eval_trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_labels = np.array(full_ds.targets)[train_idx]
    class_weights = 1. / np.bincount(
        train_labels, minlength=CONFIG["num_classes"]
    )
    sample_weights = np.array([class_weights[t] for t in train_labels])

    sampler = WeightedRandomSampler(
        torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    loaders = {
        "train": DataLoader(
            MapDataset(Subset(full_ds, train_idx), train_trans),
            batch_size=CONFIG["batch_size"],
            sampler=sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        ),
        "val": DataLoader(
            MapDataset(Subset(full_ds, val_idx), eval_trans),
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        ),
        "test": DataLoader(
            MapDataset(Subset(full_ds, test_idx), eval_trans),
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
    }

    return loaders


# PHASE 3: MODEL FACTORY & UNFREEZING STRATEGIES
# ------------------------------------------------------------------------------
def initialize_architecture(name):
    """Configures models with customized heads and selective backbone unfreezing."""
    if name == "MobileNet_V3_Large":
        model = models.mobilenet_v3_large(weights="DEFAULT")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.features[13:].parameters():
            p.requires_grad = True

        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, CONFIG["num_classes"])
        )

    elif name == "EfficientNetV2_S":
        model = models.efficientnet_v2_s(weights="DEFAULT")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.features[6:].parameters():
            p.requires_grad = True

        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, CONFIG["num_classes"])
        )

    elif name == "ResNet50":
        model = models.resnet50(weights="DEFAULT")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, CONFIG["num_classes"])

    elif name == "DenseNet121":
        model = models.densenet121(weights="DEFAULT")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.features.denseblock4.parameters():
            p.requires_grad = True

        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, CONFIG["num_classes"])

    elif name == "ViT_B16":
        model = models.vit_b_16(weights="DEFAULT")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.encoder.layers[10:].parameters():
            p.requires_grad = True

        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, CONFIG["num_classes"])

    elif name == "Inception_V3":
        model = models.inception_v3(weights="DEFAULT", aux_logits=True)
        model.aux_logits = False
        model.AuxLogits = None

        for p in model.parameters():
            p.requires_grad = False
        for p in model.Mixed_7c.parameters():
            p.requires_grad = True

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, CONFIG["num_classes"])

    elif name == "ConvNeXt_Tiny":
        model = models.convnext_tiny(weights="DEFAULT")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.features[7].parameters():
            p.requires_grad = True

        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, CONFIG["num_classes"])

    else:
        raise ValueError(f"Unknown model name: {name}")

    return model.to(DEVICE)


# PHASE 4: DIAGNOSTIC & ARTIFACT EXPORT FUNCTIONS
# ------------------------------------------------------------------------------
def export_model_summary_csv(model, export_path, img_size):
    """Saves layer-wise architectural metadata for publication methodology."""
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


def synchronize_cuda():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def measure_batch_latency(model, imgs):
    """
    Measures GPU inference latency correctly by synchronizing CUDA before
    and after the forward pass. Returns seconds per image.
    """
    synchronize_cuda()
    tic = time.perf_counter()
    _ = model(imgs)
    synchronize_cuda()
    elapsed = time.perf_counter() - tic
    return elapsed / imgs.size(0)


def run_failure_analysis(model, loader, class_names, export_path, num_visuals=20):
    """Analyzes model errors by saving misclassified images and a detailed CSV."""
    model.eval()
    error_dir = os.path.join(export_path, "Failure_Analysis")
    os.makedirs(error_dir, exist_ok=True)

    error_log = []
    visual_count = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs_dev = inputs.to(DEVICE, non_blocking=True)
            outputs = model(inputs_dev)

            probs = F.softmax(outputs, dim=1)
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
                        f"A: {act_nm} | P: {prd_nm}\nConf: {conf}%",
                        color="red"
                    )
                    plt.savefig(
                        os.path.join(error_dir, f"fail_{visual_count+1}.png"),
                        bbox_inches="tight",
                        dpi=200
                    )
                    plt.close()
                    visual_count += 1

    pd.DataFrame(error_log).to_csv(
        os.path.join(export_path, "detailed_misclassifications.csv"),
        index=False
    )


# PHASE 5: GLOBAL DATA PARTITIONING
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[INFO] Loading ORIGINAL BDMediHerb Dataset...")
print("=" * 80)

full_ds_raw = datasets.ImageFolder(BASE_DIR)
DATASET_SIGNATURE = dataset_signature(full_ds_raw)
print(f"[INFO] Dataset signature: {DATASET_SIGNATURE}")

# Exact duplicate audit: catches byte-identical images crossing partitions.
# This cannot prove specimen-level separation because the public dataset does
# not expose specimen IDs; that limitation is recorded in the metadata.
def exact_image_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

image_hashes = [exact_image_hash(path) for path, _ in full_ds_raw.samples]
print(f"[AUDIT] Unique exact image hashes: {len(set(image_hashes))}/{len(image_hashes)}")

print(f"[INFO] Total images found: {len(full_ds_raw)}")
print(f"[INFO] Number of classes: {len(full_ds_raw.classes)}")
print(f"[INFO] Classes: {full_ds_raw.classes}")

train_idx, val_idx, test_idx = create_or_load_shared_split(full_ds_raw)

print(
    "[INFO] IMPORTANT: Only original images are used. "
    "Training-time augmentation is generated dynamically and is NOT "
    "the 15,200 pre-augmented images from the public dataset."
)


# PHASE 6: MASTER EXECUTION LOOP
# ------------------------------------------------------------------------------
for MODEL_NAME in MODEL_LIST:
    current_res = 299 if MODEL_NAME == "Inception_V3" else 224

    print(
        f"\n\n{'=' * 80}\n"
        f"[JOURNAL PIPELINE] PROCESSING: {MODEL_NAME} | RESOLUTION: {current_res}\n"
        f"{'=' * 80}"
    )

    EXPORT_DIR = os.path.join(PROJECT_ROOT, MODEL_NAME)
    os.makedirs(EXPORT_DIR, exist_ok=True)

    loaders = get_loaders_for_resolution(
        train_idx, val_idx, test_idx, current_res, full_ds_raw
    )

    model = initialize_architecture(MODEL_NAME)
    export_model_summary_csv(model, EXPORT_DIR, current_res)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=CONFIG["label_smoothing"]
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=3,
        factor=0.5
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "inf_ms": [],
        "time_sec": [],
        "lr": []
    }

    best_acc = 0.0
    early_stop_counter = 0

    print(
        f"\n{'Epoch':<8} | {'Tr. Loss':<10} | {'Tr. Acc':<10} | "
        f"{'Val Loss':<10} | {'Val Acc':<10} | {'Inference':<12} | {'Time'}"
    )
    print("-" * 105)

    # --- CORE RESEARCH LOOP ---
    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.perf_counter()

        # 1. Training Phase
        model.train()
        tr_loss, tr_correct = 0.0, 0

        for imgs, lbls in loaders["train"]:
            imgs = imgs.to(DEVICE, non_blocking=True)
            lbls = lbls.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * imgs.size(0)
            tr_correct += (out.max(1)[1] == lbls).sum().item()

        # 2. Validation Phase
        model.eval()
        v_loss, v_correct, latencies = 0.0, 0, []

        with torch.no_grad():
            for imgs, lbls in loaders["val"]:
                imgs = imgs.to(DEVICE, non_blocking=True)
                lbls = lbls.to(DEVICE, non_blocking=True)

                synchronize_cuda()
                tic = time.perf_counter()
                out = model(imgs)
                synchronize_cuda()
                latencies.append(
                    (time.perf_counter() - tic) / imgs.size(0)
                )

                v_loss += criterion(out, lbls).item() * imgs.size(0)
                v_correct += (out.max(1)[1] == lbls).sum().item()

        # 3. Metrics
        cur_t_acc = round(tr_correct / len(train_idx), 4)
        cur_v_acc = round(v_correct / len(val_idx), 4)
        cur_t_loss = round(tr_loss / len(train_idx), 4)
        cur_v_loss = round(v_loss / len(val_idx), 4)
        cur_inf_ms = round(np.mean(latencies) * 1000, 4)
        cur_time = round(time.perf_counter() - epoch_start, 2)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(cur_t_loss)
        history["train_acc"].append(cur_t_acc)
        history["val_loss"].append(cur_v_loss)
        history["val_acc"].append(cur_v_acc)
        history["inf_ms"].append(cur_inf_ms)
        history["time_sec"].append(cur_time)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"{epoch+1:<8} | {cur_t_loss:<10.4f} | {cur_t_acc:<10.4f} | "
            f"{cur_v_loss:<10.4f} | {cur_v_acc:<10.4f} | "
            f"{cur_inf_ms:<10.4f} ms | {cur_time:<8}s"
        )

        # 4. Checkpoint + Early Stopping
        if cur_v_acc > best_acc:
            best_acc = cur_v_acc
            torch.save(
                model.state_dict(),
                os.path.join(EXPORT_DIR, "best_model.pth")
            )
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= CONFIG["patience"]:
                print(
                    f"[INFO] Early stopping triggered at epoch {epoch + 1}"
                )
                break

        scheduler.step(cur_v_acc)

    # PHASE 7: POST-TRAINING EVALUATION
    # ------------------------------------------------------------------------------
    print(f"[POST] Finalizing Artifacts for {MODEL_NAME}...")

    model.load_state_dict(
        torch.load(
            os.path.join(EXPORT_DIR, "best_model.pth"),
            map_location=DEVICE
        )
    )
    model.eval()

    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for imgs, lbls in loaders["test"]:
            imgs = imgs.to(DEVICE, non_blocking=True)
            outputs = model(imgs)

            y_true.extend(lbls.numpy())
            y_pred.extend(outputs.max(1)[1].cpu().numpy())
            y_score.extend(
                F.softmax(outputs, dim=1).cpu().numpy()
            )

    report = classification_report(
        y_true,
        y_pred,
        target_names=full_ds_raw.classes,
        output_dict=True,
        digits=4
    )

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(EXPORT_DIR, "final_metrics_report.csv"),
        float_format="%.4f"
    )

    # Confusion Matrix
    plt.figure(figsize=(16, 14))
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=full_ds_raw.classes,
        yticklabels=full_ds_raw.classes
    )

    plt.title(
        f"Confusion Matrix {MODEL_NAME}",
        fontsize=18,
        fontweight="bold"
    )
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_figure(os.path.join(EXPORT_DIR, "confusion_matrix.png"))

    # Multi-class ROC-AUC
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
            label=f"{full_ds_raw.classes[i]} (AUC={auc(fpr, tpr):.4f})"
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.title(f"Multi-class ROC: {MODEL_NAME}")
    save_figure(os.path.join(EXPORT_DIR, "roc_auc.png"))

    # F1 Score
    f1_vals = [
        report[cls]["f1-score"] * 100
        for cls in full_ds_raw.classes
    ]

    # Publication-safe F1 bar chart: larger canvas, extra headroom, and
    # labels positioned above each bar so high scores do not overlap.
    fig, ax = plt.subplots(figsize=(18, 9))
    x = np.arange(len(f1_vals))
    bars = ax.bar(x, f1_vals, width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(full_ds_raw.classes, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Macro F1-score (%)", fontsize=13)
    ax.set_xlabel("Class", fontsize=13)
    ax.set_title(f"Per-Class F1-score: {MODEL_NAME}", fontsize=17, fontweight="bold", pad=18)
    ymax = min(110, max(105, max(f1_vals) + 7))
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.25)
    for bar, v in zip(bars, f1_vals):
        ax.annotate(
            f"{v:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            clip_on=False
        )
    fig.tight_layout(pad=2.0)
    save_figure(os.path.join(EXPORT_DIR, "f1_score_performance.png"))

    # Learning Curves -- saved separately for clear manuscript presentation
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_acc"], label="Training Accuracy", marker="o", alpha=0.75)
    plt.plot(history["val_acc"], label="Validation Accuracy", marker="s", alpha=0.75)
    plt.title(f"{MODEL_NAME}: Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_figure(os.path.join(EXPORT_DIR, "accuracy_curve.png"))

    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss", marker="o", alpha=0.75)
    plt.plot(history["val_loss"], label="Validation Loss", marker="s", alpha=0.75)
    plt.title(f"{MODEL_NAME}: Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_figure(os.path.join(EXPORT_DIR, "loss_curve.png"))

    # Failure Analysis & Metadata
    run_failure_analysis(
        model,
        loaders["test"],
        full_ds_raw.classes,
        EXPORT_DIR
    )

    pd.DataFrame(history).to_csv(
        os.path.join(EXPORT_DIR, "training_history.csv"),
        index=False
    )

    # Save split metadata inside every model directory
    split_metadata = {
        "seed": SEED,
        "total_images": len(full_ds_raw),
        "num_classes": len(full_ds_raw.classes),
        "train_images": len(train_idx),
        "validation_images": len(val_idx),
        "test_images": len(test_idx),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "dataset_signature": DATASET_SIGNATURE,
        "specimen_level_ids_available": False,
        "specimen_level_note": "Public dataset documentation does not provide explicit specimen identifiers; image-level stratification is reproducible, but specimen-level grouping cannot be independently verified."
    }

    with open(
        os.path.join(EXPORT_DIR, "split_metadata.json"),
        "w"
    ) as f:
        json.dump(split_metadata, f, indent=2)

    del model, optimizer, loaders
    gc.collect()
    torch.cuda.empty_cache()

    print(f"✔ Pipeline finished for {MODEL_NAME}. Data archived.\n")


print(
    "\n" + "=" * 80 +
    "\n[FINAL] COMPREHENSIVE INDIVIDUAL-MODEL PIPELINE COMPLETED SUCCESSFULLY" +
    "\n" + "=" * 80
)
