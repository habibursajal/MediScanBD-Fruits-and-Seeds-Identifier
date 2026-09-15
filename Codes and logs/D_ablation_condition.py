# ==============================================================================
# ABLATION CONDITION D: Focal Loss + 4-Pass TTA (Multi-Seed Folder Save)
# Output: Reviewer_Ablation_Statistics/seed_/checkpoints/
# Features: Skip completed seeds, CSV Summary & Full Classification Report Save
# ==============================================================================

import os, json, gc, random, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. PATH & SYSTEM CONFIGURATION
# ------------------------------------------------------------------------------
LOCAL_DATASET = Path(r"E:\Research Project\BDMediHerb\Original Dataset")
LOCAL_PROJECT_ROOT = Path(r"E:\Research Project\BDMediHerb\BDMediHerb_Reviewer_Revision")

if LOCAL_DATASET.exists():
    BASE_DIR, PROJECT_ROOT = LOCAL_DATASET, LOCAL_PROJECT_ROOT
else:
    BASE_DIR = Path("/kaggle/input/datasets/habibursojol/bdmediherb-original-final/Original Dataset")
    PROJECT_ROOT = Path("/kaggle/working/BDMediHerb_Reviewer_Revision")

OUT_DIR = PROJECT_ROOT / 'Reviewer_Ablation_Statistics'
CONTROLLED_INIT_DIR = PROJECT_ROOT / 'Controlled_Baselines'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEEDS = [42, 43, 44, 45, 46]
EPOCHS, PATIENCE, BATCH_SIZE, NUM_CLASSES, IMG_SIZE = 50, 8, 32, 19, 224
HEAD_LR, WEIGHT_DECAY = 3e-4, 1e-2

# 2. REPRODUCIBILITY UTILITY
# ------------------------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 3. DATASET & ARCHITECTURE DEFINITIONS
# ------------------------------------------------------------------------------
class MapDataset(Dataset):
    def __init__(self, base, indices, transform):
        self.base, self.indices, self.transform = base, list(indices), transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): img, y = self.base[self.indices[i]]; return self.transform(img), y

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma, self.smoothing = gamma, label_smoothing
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none', label_smoothing=self.smoothing)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

def init_backbone(name):
    if name == 'MobileNet_V3_Large':
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Sequential(nn.Linear(m.classifier[3].in_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, NUM_CLASSES))
    elif name == 'ResNet50':
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif name == 'ViT_B16':
        m = models.vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, NUM_CLASSES)
    return m

def load_feature(name):
    m = init_backbone(name)
    ckpt_path = CONTROLLED_INIT_DIR / name / 'best_model.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[ERR] Pretrained baseline model not found at: {ckpt_path}")
        
    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    if name.startswith('MobileNet'): dim = m.classifier[0].in_features; m.classifier = nn.Identity()
    elif name.startswith('ResNet'): dim = m.fc.in_features; m.fc = nn.Identity()
    else: dim = m.heads.head.in_features; m.heads = nn.Identity()
    return m, dim

class TSHE(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1, d1 = load_feature('MobileNet_V3_Large')
        self.s2, d2 = load_feature('ResNet50')
        self.s3, d3 = load_feature('ViT_B16')
        self.h1, self.h2, self.h3 = nn.Linear(d1, NUM_CLASSES), nn.Linear(d2, NUM_CLASSES), nn.Linear(d3, NUM_CLASSES)
        self.meta = nn.Sequential(nn.Linear(NUM_CLASSES * 3, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, NUM_CLASSES))
    def forward(self, x):
        return self.meta(torch.cat([self.h1(self.s1(x)), self.h2(self.s2(x)), self.h3(self.s3(x))], dim=1))

def predict_probabilities_tta(model, imgs):
    p1 = F.softmax(model(imgs), dim=1)
    p2 = F.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
    p3 = F.softmax(model(torch.flip(imgs, dims=[2])), dim=1)
    p4 = F.softmax(model(torch.flip(imgs, dims=[2, 3])), dim=1)
    return (p1 + p2 + p3 + p4) / 4.0

# 4. SINGLE SEED EXECUTION PIPELINE
# ------------------------------------------------------------------------------
def run_seed(seed, full_ds, split):
    seed_dir = OUT_DIR / f'seed_{seed}' / 'checkpoints'
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    res_json_path = seed_dir / 'D_Focal_4PassTTA_result.json'
    pred_npz_path = seed_dir / 'D_Focal_4PassTTA_predictions.npz'
    report_csv_path = seed_dir / 'D_Focal_4PassTTA_classification_report.csv'
    best_path = seed_dir / 'D_Focal_4PassTTA.pth'
    
    # --------------------------------------------------------------------------
    # RESUME / SKIP LOGIC
    # --------------------------------------------------------------------------
    if res_json_path.exists() and pred_npz_path.exists() and report_csv_path.exists() and best_path.exists():
        try:
            res = json.loads(res_json_path.read_text())
            print(f"  [SKIP] Seed {seed} for Condition D already completed! (Accuracy: {res.get('test_accuracy', 0)*100:.2f}%)")
            return res
        except Exception:
            print(f"  [WARNING] Seed {seed} result files corrupted. Re-running...")

    seed_everything(seed)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    g = torch.Generator()
    g.manual_seed(seed)

    kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(
        MapDataset(full_ds, split['train_idx'], train_tf),
        batch_size=BATCH_SIZE, shuffle=True, generator=g, worker_init_fn=seed_worker, **kwargs
    )
    val_loader = DataLoader(MapDataset(full_ds, split['val_idx'], eval_tf), batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(MapDataset(full_ds, split['test_idx'], eval_tf), batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    model = TSHE().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    crit = FocalLoss(gamma=2.0, label_smoothing=0.05)
    
    best_acc, wait = -1, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for imgs, y in train_loader:
            imgs, y = imgs.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(imgs), y).backward(); opt.step()
        
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for imgs, y in val_loader:
                ys.extend(y.numpy()); ps.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
        val_acc = accuracy_score(ys, ps)
        if val_acc > best_acc:
            best_acc = val_acc; wait = 0; torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= PATIENCE: 
                break

    # Evaluation on Test Set using 4-Pass TTA
    model.load_state_dict(torch.load(best_path))
    model.eval()
    ys, ps, probs = [], [], []
    with torch.no_grad():
        for imgs, y in test_loader:
            p = predict_probabilities_tta(model, imgs.to(DEVICE))
            ys.extend(y.numpy()); ps.extend(p.argmax(1).cpu().numpy()); probs.extend(p.cpu().numpy())
            
    acc = accuracy_score(ys, ps)
    macro_f1 = f1_score(ys, ps, average='macro')
    res = {'seed': seed, 'condition': 'D_Focal_4PassTTA', 'test_accuracy': float(acc), 'macro_f1': float(macro_f1)}
    
    # --------------------------------------------------------------------------
    # SAVE CSV & CLASSIFICATION REPORT
    # --------------------------------------------------------------------------
    class_names = full_ds.classes if hasattr(full_ds, 'classes') else [str(i) for i in range(NUM_CLASSES)]
    
    # 1. Classification Report DataFrame (CSV Format)
    report_dict = classification_report(ys, ps, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(report_csv_path)

    # 2. Results JSON & Predictions NPZ Save
    res_json_path.write_text(json.dumps(res, indent=2))
    np.savez_compressed(pred_npz_path, y_true=ys, y_pred=ps, y_score=probs)

    print(f"   [Seed {seed} Complete] Accuracy: {acc*100:.2f}% | F1: {macro_f1:.4f}")
    print(f"   [INFO] Classification Report saved to: {report_csv_path}")

    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return res

# 5. ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    full_ds = datasets.ImageFolder(str(BASE_DIR))
    split = json.loads((PROJECT_ROOT / 'shared_stratified_split.json').read_text())
    print("=" * 80 + "\n[CONDITION D] Processing Seeds: " + str(SEEDS) + "\n" + "=" * 80)
    
    all_results = []
    for s in SEEDS:
        res = run_seed(s, full_ds, split)
        if res:
            all_results.append(res)

    # Save summary of all seeds into a single combined CSV
    if all_results:
        summary_csv = OUT_DIR / 'D_Focal_4PassTTA_all_seeds_summary.csv'
        df_summary = pd.DataFrame(all_results)
        df_summary.to_csv(summary_csv, index=False)
        print("=" * 80)
        print(f"[SUMMARY SAVED] Combined seed results saved to: {summary_csv}")
        print("=" * 80)