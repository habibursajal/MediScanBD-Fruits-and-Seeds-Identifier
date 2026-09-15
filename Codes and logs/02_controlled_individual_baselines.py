# ============================================================================
# BDMediHerb REVIEWER-REQUIRED CONTROLLED INDIVIDUAL BASELINES (FULL REVISED)
# ============================================================================
# Purpose:
#   Address Reviewer Comments by evaluating individual backbones under
#   the SAME core training/evaluation protocol used by TSHE/Fusion Model:
#     - Same 3,800 original images and shared 70/15/15 split
#     - Advanced augmentation
#     - AdamW with differential backbone (1e-6) / head (1e-4) learning rates
#     - Focal Loss (gamma=2, label_smoothing=0.05)
#     - CosineAnnealingWarmRestarts
#     - 4-pass flip-based logit TTA for final test evaluation
#     - Complete Artifact Generation: Confusion Matrix, ROC-AUC, F1-Bar, 
#       Learning Curves, Architecture Summaries, and Failure Analysis.
# ============================================================================

import os
import json
import gc
import random
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

try:
    from torchinfo import summary
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchinfo", "-q"])
    from torchinfo import summary

# --------------------------- CONFIGURATION ---------------------------------
LOCAL_DATASET = Path(r"E:\Research Project\BDMediHerb\Original Dataset")
LOCAL_PROJECT_ROOT = Path(r"E:\Research Project\BDMediHerb\BDMediHerb_Reviewer_Revision")

if LOCAL_DATASET.exists():
    BASE_DIR = LOCAL_DATASET
    PROJECT_ROOT = LOCAL_PROJECT_ROOT
    print('[ENV] Local Windows PC detected')
elif Path('/kaggle/input/datasets/habibursojol/bdmediherb/Original Dataset - Copy').exists():
    BASE_DIR = Path('/kaggle/input/datasets/habibursojol/bdmediherb/Original Dataset - Copy')
    PROJECT_ROOT = Path('/kaggle/working/BDMediHerb_Reviewer_Revision')
    print('[ENV] Kaggle detected')
else:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = Path('/content/drive/MyDrive/BDMediHerb/Original Dataset')
    PROJECT_ROOT = Path('/content/drive/MyDrive/BDMediHerb/BDMediHerb_Reviewer_Revision')
    print('[ENV] Google Colab detected')

OUT_DIR = PROJECT_ROOT / 'Controlled_Baselines'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_FILE = PROJECT_ROOT / 'shared_stratified_split.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 8
NUM_CLASSES = 19
IMG_SIZE = 224
BACKBONE_LR = 1e-6
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-2
FOCAL_GAMMA = 2.0
FOCAL_SMOOTHING = 0.05
SCHEDULER_T0 = 10
SCHEDULER_TMULT = 2
SCHEDULER_ETA_MIN = 1e-7
WARMUP_BATCHES = 10
EXPECTED_TOTAL = 3800
EXPECTED_PER_CLASS = 200
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
NUM_WORKERS = 0 if os.name == 'nt' else 2  # Safe for Windows multiprocessing

MODEL_LIST = [
    'MobileNet_V3_Large', 'EfficientNetV2_S', 'ResNet50', 'DenseNet121',
    'ViT_B16', 'Inception_V3', 'ConvNeXt_Tiny'
]
TSHE_MODELS = ['MobileNet_V3_Large', 'ResNet50', 'ViT_B16']
FIG_DPI = 300

# --------------------------- REPRODUCIBILITY & UTILS -----------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

def sync_cuda():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

def save_figure(path):
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

# --------------------------- DATASET SIGNATURE & SPLIT ---------------------
def dataset_signature(ds):
    h = hashlib.sha256()
    for path, target in ds.samples:
        h.update(str(Path(path).name).encode('utf-8'))
        h.update(str(target).encode('utf-8'))
    return h.hexdigest()

def load_or_create_shared_split(ds):
    labels = np.array(ds.targets)
    indices = np.arange(len(ds))
    current_signature = DATASET_SIGNATURE
    if SPLIT_FILE.exists():
        try:
            with open(SPLIT_FILE) as f: 
                s = json.load(f)
            train_idx = np.array(s.get('train_idx', []), dtype=int)
            val_idx = np.array(s.get('val_idx', []), dtype=int)
            test_idx = np.array(s.get('test_idx', []), dtype=int)
            valid = (s.get('dataset_signature') == current_signature and
                     (len(train_idx), len(val_idx), len(test_idx)) == (2660, 570, 570) and
                     len(set(train_idx) & set(val_idx)) == 0 and 
                     len(set(train_idx) & set(test_idx)) == 0 and 
                     len(set(val_idx) & set(test_idx)) == 0)
            if valid:
                print(f"[INFO] Loaded verified shared split from: {SPLIT_FILE}")
                return train_idx, val_idx, test_idx
            print('[WARN] Existing shared split is stale/invalid; regenerating automatically.')
        except Exception as exc:
            print(f'[WARN] Could not read existing shared split ({exc}); regenerating automatically.')
    
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, stratify=labels, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=SEED)
    payload = {
        'seed': SEED, 'train_ratio': TRAIN_RATIO, 'val_ratio': VAL_RATIO, 'test_ratio': TEST_RATIO,
        'dataset_signature': DATASET_SIGNATURE, 'train_idx': train_idx.tolist(),
        'val_idx': val_idx.tolist(), 'test_idx': test_idx.tolist()
    }
    with open(SPLIT_FILE, 'w') as f: 
        json.dump(payload, f, indent=2)
    print(f'[INFO] Created/regenerated shared split: {SPLIT_FILE}')
    return train_idx, val_idx, test_idx

class MapDataset(Dataset):
    def __init__(self, base, indices, transform):
        self.base, self.indices, self.transform = base, list(indices), transform
    def __len__(self): 
        return len(self.indices)
    def __getitem__(self, i):
        img, y = self.base[self.indices[i]]
        return self.transform(img), y

def make_loaders(ds, train_idx, val_idx, test_idx, resolution=224):
    train_tf = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((resolution, resolution)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return (
        DataLoader(MapDataset(ds, train_idx, train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()),
        DataLoader(MapDataset(ds, val_idx, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()),
        DataLoader(MapDataset(ds, test_idx, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    )

# --------------------------- MODEL FACTORY & LOSS --------------------------
def initialize_model(name):
    if name == 'MobileNet_V3_Large':
        m = models.mobilenet_v3_large(weights='DEFAULT')
        f = m.classifier[3].in_features
        m.classifier[3] = nn.Sequential(nn.Linear(f, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, NUM_CLASSES))
        head = list(m.classifier.parameters())
    elif name == 'EfficientNetV2_S':
        m = models.efficientnet_v2_s(weights='DEFAULT')
        f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(f, NUM_CLASSES))
        head = list(m.classifier.parameters())
    elif name == 'ResNet50':
        m = models.resnet50(weights='DEFAULT')
        f = m.fc.in_features
        m.fc = nn.Linear(f, NUM_CLASSES)
        head = list(m.fc.parameters())
    elif name == 'DenseNet121':
        m = models.densenet121(weights='DEFAULT')
        f = m.classifier.in_features
        m.classifier = nn.Linear(f, NUM_CLASSES)
        head = list(m.classifier.parameters())
    elif name == 'ViT_B16':
        m = models.vit_b_16(weights='DEFAULT')
        f = m.heads.head.in_features
        m.heads.head = nn.Linear(f, NUM_CLASSES)
        head = list(m.heads.parameters())
    elif name == 'Inception_V3':
        m = models.inception_v3(weights='DEFAULT', aux_logits=True)
        m.aux_logits = False
        m.AuxLogits = None
        f = m.fc.in_features
        m.fc = nn.Linear(f, NUM_CLASSES)
        head = list(m.fc.parameters())
    elif name == 'ConvNeXt_Tiny':
        m = models.convnext_tiny(weights='DEFAULT')
        f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(f, NUM_CLASSES)
        head = list(m.classifier.parameters())
    else: 
        raise ValueError(name)
    return m.to(DEVICE), head

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none', label_smoothing=self.smoothing)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

criterion = FocalLoss(FOCAL_GAMMA, FOCAL_SMOOTHING)

def tta_logits(model, x):
    o1 = model(x)
    if isinstance(o1, tuple): o1 = o1[0]
    o2 = model(torch.flip(x, dims=[3]))
    o3 = model(torch.flip(x, dims=[2]))
    o4 = model(torch.flip(x, dims=[2, 3]))
    if isinstance(o2, tuple): o2 = o2[0]
    if isinstance(o3, tuple): o3 = o3[0]
    if isinstance(o4, tuple): o4 = o4[0]
    return (o1 + o2 + o3 + o4) / 4.0

def evaluate(model, loader, use_tta=False, measure_latency=False):
    model.eval()
    ys, preds, scores, losses, times = [], [], [], [], []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            if measure_latency:
                sync_cuda()
                t = time.perf_counter()
                out = tta_logits(model, x) if use_tta else model(x)
                sync_cuda()
                times.append((time.perf_counter() - t) / len(y) * 1000)
            else:
                out = tta_logits(model, x) if use_tta else model(x)
            
            if isinstance(out, tuple): out = out[0]
            losses.append(criterion(out, y).item() * len(y))
            prob = F.softmax(out, 1)
            p = prob.argmax(1)
            
            ys.extend(y.cpu().numpy())
            preds.extend(p.cpu().numpy())
            scores.extend(prob.cpu().numpy())
            
    return (sum(losses) / len(ys), accuracy_score(ys, preds), f1_score(ys, preds, average='macro'),
            np.array(ys), np.array(preds), np.array(scores), float(np.mean(times)) if times else np.nan)

# --------------------------- EXPORT & PLOTTING UTILITIES ------------------
def export_model_summary(model, out_dir, img_size):
    try:
        stats = summary(model, input_size=(BATCH_SIZE, 3, img_size, img_size), verbose=0)
        summary_list = []
        for layer in stats.summary_list:
            summary_list.append({
                "Layer Name": layer.class_name,
                "Input Shape": str(layer.input_size),
                "Output Shape": str(layer.output_size),
                "Parameters": layer.num_params,
                "Trainable": layer.trainable
            })
        pd.DataFrame(summary_list).to_csv(out_dir / "architecture_summary.csv", index=False)
    except Exception as e:
        print(f"[INFO] Architecture summary skipped: {e}")

def export_publication_plots(model_name, out_dir, hist, y_true, y_pred, y_score, classes, report):
    # 1. Learning Curves
    df_hist = pd.DataFrame(hist)
    plt.figure(figsize=(10, 5))
    plt.plot(df_hist['epoch'], df_hist['train_loss'], 'b-', label='Training Loss')
    plt.plot(df_hist['epoch'], df_hist['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'{model_name}: Loss Curve')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)
    save_figure(out_dir / 'loss_curve.png')

    plt.figure(figsize=(10, 5))
    plt.plot(df_hist['epoch'], df_hist['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(df_hist['epoch'], df_hist['val_acc'], 'r-', label='Validation Accuracy')
    plt.title(f'{model_name}: Accuracy Curve')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True, alpha=0.3)
    save_figure(out_dir / 'accuracy_curve.png')

    # 2. Confusion Matrix PNG
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix: {model_name}", fontsize=16, fontweight="bold")
    plt.ylabel("Ground Truth"); plt.xlabel("Predicted Class")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    save_figure(out_dir / "confusion_matrix.png")

    # 3. ROC-AUC Curves
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    plt.figure(figsize=(12, 10))
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f"{classes[i]} (AUC={auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(f"Multi-Class ROC Curve: {model_name}")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    save_figure(out_dir / "roc_auc.png")

    # 4. Per-Class F1 Score Bar Chart
    f1_vals = [report[cls]["f1-score"] * 100 for cls in classes]
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(f1_vals))
    bars = ax.bar(x, f1_vals, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Macro F1-score (%)")
    ax.set_title(f"Per-Class F1-score: {model_name}", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 115); ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, f1_vals):
        ax.annotate(f"{v:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    fig.tight_layout()
    save_figure(out_dir / "f1_score_performance.png")

def run_failure_analysis(model, loader, classes, out_dir, num_visuals=20):
    model.eval()
    error_dir = out_dir / "Failure_Analysis"
    error_dir.mkdir(parents=True, exist_ok=True)
    error_log = []
    visual_count = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs_dev = inputs.to(DEVICE, non_blocking=True)
            outputs = tta_logits(model, inputs_dev)
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            wrong = (preds.cpu() != labels).nonzero(as_tuple=True)[0]

            for idx in wrong:
                act_nm = classes[labels[idx].item()]
                prd_nm = classes[preds[idx].item()]
                conf = round(confs[idx].item() * 100, 4)

                error_log.append({"Actual": act_nm, "Predicted": prd_nm, "Confidence (%)": conf})

                if visual_count < num_visuals:
                    img = inputs[idx].numpy().transpose((1, 2, 0))
                    img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
                    plt.figure()
                    plt.imshow(img)
                    plt.axis("off")
                    plt.title(f"Actual: {act_nm} | Pred: {prd_nm}\nConf: {conf}%", color="red")
                    save_figure(error_dir / f"fail_{visual_count+1}.png")
                    visual_count += 1

    pd.DataFrame(error_log).to_csv(out_dir / "detailed_misclassifications.csv", index=False)

# --------------------------- CORE TRAINING FUNCTION -----------------------
def train_one(name, ds, train_idx, val_idx, test_idx):
    seed_everything(SEED)
    resolution = 299 if name == 'Inception_V3' else 224
    train_loader, val_loader, test_loader = make_loaders(ds, train_idx, val_idx, test_idx, resolution)
    model, head_params = initialize_model(name)
    
    out_dir = OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / 'best_model.pth'

    export_model_summary(model, out_dir, resolution)

    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': BACKBONE_LR},
        {'params': head_params, 'lr': HEAD_LR}
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SCHEDULER_T0, T_mult=SCHEDULER_TMULT, eta_min=SCHEDULER_ETA_MIN
    )
    
    best = -1.0; wait = 0; hist = []
    
    print(f"\n{'Epoch':<8} | {'Tr. Loss':<10} | {'Tr. Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Time'}")
    print("-" * 75)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        total = correct = 0
        loss_sum = 0.0
        
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            out = model(x)
            out = out[0] if isinstance(out, tuple) else out
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
            
        vl, va, vf, _, _, _, _ = evaluate(model, val_loader, False, False)
        ep_sec = time.perf_counter() - t0
        hist.append({
            'epoch': epoch, 'train_loss': loss_sum / total, 'train_acc': correct / total,
            'val_loss': vl, 'val_acc': va, 'val_macro_f1': vf, 'epoch_sec': ep_sec
        })
        
        print(f"{epoch:<8} | {loss_sum/total:<10.4f} | {correct/total:<10.4f} | {vl:<10.4f} | {va:<10.4f} | {ep_sec:<6.2f}s")

        if va > best:
            best = va; wait = 0; torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[INFO] Early stopping triggered at epoch {epoch}")
                break
        scheduler.step(epoch)
        
    pd.DataFrame(hist).to_csv(out_dir / 'training_history.csv', index=False)

    
    # Load Best Model for Testing
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    
    # Clear VRAM cache before test evaluation
    del x, y, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    # Re-create test loader with batch_size=16 for memory safety
    _, _, test_loader = make_loaders(ds, train_idx, val_idx, test_idx, resolution)
    
    # Warmup CUDA for precise latency test
    with torch.inference_mode():
        for x_batch, _ in test_loader:
            x_test = x_batch[:4].to(DEVICE) # Small batch for warmup
            for _ in range(WARMUP_BATCHES): 
                _ = tta_logits(model, x_test)
            sync_cuda()
            break
            
    tl, ta, tf, y, p, score, lat = evaluate(model, test_loader, use_tta=True, measure_latency=True)

    
    # Classification Metrics
    report = classification_report(y, p, target_names=ds.classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(out_dir / 'classification_report.csv')
    pd.DataFrame(confusion_matrix(y, p), index=ds.classes, columns=ds.classes).to_csv(out_dir / 'confusion_matrix.csv')
    np.savez_compressed(out_dir / 'predictions.npz', y_true=y, y_pred=p, y_score=score)
    
    # Generate Plots & Failure Analysis
    export_publication_plots(name, out_dir, hist, y, p, score, ds.classes, report)
    run_failure_analysis(model, test_loader, ds.classes, out_dir)
    
    # Comprehensive Result Record
    result = {
        'model': name,
        'protocol': 'controlled_TSHE_aligned',
        'seed': SEED,
        'test_accuracy': float(ta),
        'macro_f1': float(tf),
        'weighted_f1': float(f1_score(y, p, average='weighted')),
        'macro_precision': float(precision_score(y, p, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y, p, average='macro', zero_division=0)),
        'weighted_precision': float(precision_score(y, p, average='weighted', zero_division=0)),
        'weighted_recall': float(recall_score(y, p, average='weighted', zero_division=0)),
        'latency_ms_per_image_4pass_TTA': float(lat),
        'best_validation_accuracy': float(best),
        'loss': 'FocalLoss(gamma=2,label_smoothing=0.05)',
        'backbone_lr': BACKBONE_LR, 'head_lr': HEAD_LR, 'weight_decay': WEIGHT_DECAY,
        'scheduler': 'CosineAnnealingWarmRestarts', 'tta': '4-pass flip logit averaging',
        'dataset_signature': DATASET_SIGNATURE
    }
    
    with open(out_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
        
    print(f'[DONE] {name}: Test Acc={ta*100:.2f}% | Macro-F1={tf:.4f} | 4-Pass TTA={lat:.4f} ms/image')
    
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result

# --------------------------- MAIN EXECUTION --------------------------------
if __name__ == '__main__':
    print('=' * 80)
    print('CONTROLLED BASELINES — ORIGINAL DATASET ONLY (FULL REVISED)')
    print('=' * 80)
    
    ds = datasets.ImageFolder(str(BASE_DIR))
    if len(ds) != EXPECTED_TOTAL:
        raise ValueError(f'Expected {EXPECTED_TOTAL} original images, found {len(ds)}')
    counts = np.bincount(np.array(ds.targets), minlength=NUM_CLASSES)
    if not np.all(counts == EXPECTED_PER_CLASS):
        raise ValueError(f'Expected 200/class; observed {counts.tolist()}')
        
    DATASET_SIGNATURE = dataset_signature(ds)
    train_idx, val_idx, test_idx = load_or_create_shared_split(ds)
    
    print(f'Dataset: {len(ds)} images | 19 classes | Signature={DATASET_SIGNATURE[:16]}...')
    print(f'Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}')

    results = []
    for name in MODEL_LIST:
        out_dir = OUT_DIR / name
        
        if (out_dir / 'result.json').exists():
            print(f"\n[SKIP] {name} already completed. Loading saved results...")
            with open(out_dir / 'result.json') as f:
                results.append(json.load(f))
            continue

        print(f"\n{'=' * 80}\n[CONTROLLED BASELINE] RUNNING: {name}\n{'=' * 80}")
        results.append(train_one(name, ds, train_idx, val_idx, test_idx))


    # Save Master Summary CSV
    df_summary = pd.DataFrame(results)
    df_summary.to_csv(OUT_DIR / 'controlled_baseline_summary.csv', index=False)
    
    protocol_meta = {
        'dataset': 'BDMediHerb original images only',
        'total_images': 3800, 'classes': 19, 'images_per_class': 200,
        'split': '70/15/15 shared stratified', 'seed': SEED,
        'models': MODEL_LIST, 'TSHE_initialization_models': TSHE_MODELS,
        'loss': 'FocalLoss gamma=2, label_smoothing=0.05',
        'backbone_lr': BACKBONE_LR, 'head_lr': HEAD_LR, 'weight_decay': WEIGHT_DECAY,
        'scheduler': 'CosineAnnealingWarmRestarts T0=10 Tmult=2 eta_min=1e-7',
        'augmentation': 'Resize + HFlip(0.5) + VFlip(0.3) + Rotation(20) + Affine translate 0.1/scale 0.9-1.1 + ColorJitter 0.2',
        'tta': '4-pass flip-based logit averaging before softmax',
        'dataset_signature': DATASET_SIGNATURE
    }
    with open(OUT_DIR / 'protocol_metadata.json', 'w') as f:
        json.dump(protocol_meta, f, indent=2)
        
    print('\n' + '=' * 80)
    print(f'ALL CONTROLLED BASELINE RESULTS & ARTIFACTS SAVED TO: {OUT_DIR}')
    print('=' * 80)

    