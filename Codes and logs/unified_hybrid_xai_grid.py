# ==============================================================================
# PHASE 19: MASTER UNIFIED HYBRID XAI VISUALIZATION GRID
# Strategy: 3-Stream Unified Fusion CAM (MobileNetV3 + ResNet50 + ViT-B16)
# Output: Hybrid_TripleStream/XAI_Deep_Analysis/ (600 DPI PNG & PDF Vector)
# ==============================================================================

import os
import sys
import json
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import cv2
import matplotlib.pyplot as plt

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
except ImportError:
    import subprocess
    print("[INFO] Installing LIME and scikit-image...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lime", "scikit-image", "-q"])
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

print("\n" + "=" * 80)
print("[INFO] Generating Master Unified Hybrid XAI Precision Grid (Large Font)...")
print("=" * 80)

# 1. PATH & CONFIGURATION
# ------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HYBRID_RES = 224
NUM_CLASSES = 19
BEST_3_NAMES = ["MobileNet_V3_Large", "ResNet50", "ViT_B16"]

LOCAL_DATASET = pathlib.Path(r"E:\Research Project\BDMediHerb\Original Dataset")
LOCAL_PROJECT_ROOT = pathlib.Path(r"E:\Research Project\BDMediHerb\BDMediHerb_Reviewer_Revision")

if LOCAL_DATASET.exists():
    BASE_DIR = LOCAL_DATASET
    PROJECT_ROOT = LOCAL_PROJECT_ROOT
    print("[ENV] Local Windows PC detected")
elif os.path.exists("/kaggle/working/BDMediHerb_Reviewer_Revision"):
    BASE_DIR = pathlib.Path("/kaggle/input/datasets/habibursojol/bdmediherb-original-final/Original Dataset")
    PROJECT_ROOT = pathlib.Path("/kaggle/working/BDMediHerb_Reviewer_Revision")
    print("[ENV] Kaggle detected")
else:
    BASE_DIR = pathlib.Path("/content/drive/MyDrive/BDMediHerb/Original Dataset")
    PROJECT_ROOT = pathlib.Path("/content/drive/MyDrive/BDMediHerb/BDMediHerb_Reviewer_Revision")
    print("[ENV] Google Colab detected")

EXPORT_DIR_HYB = PROJECT_ROOT / "Hybrid_TripleStream"
SPLIT_FILE = PROJECT_ROOT / "shared_stratified_split.json"
XAI_REPORT_DIR = EXPORT_DIR_HYB / "XAI_Deep_Analysis"
XAI_REPORT_DIR.mkdir(parents=True, exist_ok=True)

xai_transforms = transforms.Compose([
    transforms.Resize((HYBRID_RES, HYBRID_RES)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. ARCHITECTURE RE-CONSTRUCTION
# ------------------------------------------------------------------------------
def initialize_architecture(name):
    if name == "MobileNet_V3_Large":
        m = models.mobilenet_v3_large(weights=None)
        f = m.classifier[3].in_features
        m.classifier[3] = nn.Sequential(nn.Linear(f, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, NUM_CLASSES))
    elif name == "ResNet50":
        m = models.resnet50(weights=None)
        f = m.fc.in_features
        m.fc = nn.Linear(f, NUM_CLASSES)
    elif name == "ViT_B16":
        m = models.vit_b_16(weights=None)
        f = m.heads.head.in_features
        m.heads.head = nn.Linear(f, NUM_CLASSES)
    return m.to(DEVICE)

class FeatureExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        base_model = initialize_architecture(model_name)
        if "ViT" in model_name:
            self.feat_dim = base_model.heads.head.in_features
            base_model.heads = nn.Identity()
        elif "ResNet" in model_name:
            self.feat_dim = base_model.fc.in_features
            base_model.fc = nn.Identity()
        elif "MobileNet" in model_name:
            self.feat_dim = base_model.classifier[0].in_features
            base_model.classifier = nn.Identity()
        self.backbone = base_model

    def forward(self, x):
        x = self.backbone(x)
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        return x

class EnsembleStackingNet(nn.Module):
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
        return self.meta_learner(torch.cat([out1, out2, out3], dim=-1))

# 3. UNIFIED GRAD-CAM CORE ENGINE
# ------------------------------------------------------------------------------
def generate_unified_cam_map(model, input_tensor, target_class):
    blobs = {}

    def mobilenet_fwd(m, i, o): blobs['m_feat'] = o
    def mobilenet_bwd(m, gi, go): blobs['m_grad'] = go[0]
    h1_f = model.stream1.backbone.features[-1].register_forward_hook(mobilenet_fwd)
    h1_b = model.stream1.backbone.features[-1].register_full_backward_hook(mobilenet_bwd)

    def resnet_fwd(m, i, o): blobs['r_feat'] = o
    def resnet_bwd(m, gi, go): blobs['r_grad'] = go[0]
    h2_f = model.stream2.backbone.layer4.register_forward_hook(resnet_fwd)
    h2_b = model.stream2.backbone.layer4.register_full_backward_hook(resnet_bwd)

    def vit_fwd(m, i, o): blobs['v_feat'] = o
    def vit_bwd(m, gi, go): blobs['v_grad'] = go[0]
    h3_f = model.stream3.backbone.encoder.ln.register_forward_hook(vit_fwd)
    h3_b = model.stream3.backbone.encoder.ln.register_full_backward_hook(vit_bwd)

    model.zero_grad()
    logits = model(input_tensor)
    score = logits[0, target_class]
    score.backward()

    h1_f.remove(); h1_b.remove()
    h2_f.remove(); h2_b.remove()
    h3_f.remove(); h3_b.remove()

    # Stream 1 CAM
    w1 = blobs['m_grad'].mean(dim=(2, 3), keepdim=True)
    c1 = F.relu((w1 * blobs['m_feat']).sum(dim=1, keepdim=True))
    c1 = F.interpolate(c1, size=(HYBRID_RES, HYBRID_RES), mode='bilinear', align_corners=False)

    # Stream 2 CAM
    w2 = blobs['r_grad'].mean(dim=(2, 3), keepdim=True)
    c2 = F.relu((w2 * blobs['r_feat']).sum(dim=1, keepdim=True))
    c2 = F.interpolate(c2, size=(HYBRID_RES, HYBRID_RES), mode='bilinear', align_corners=False)

    # Stream 3 CAM
    v_feats = blobs['v_feat'][:, 1:, :].transpose(1, 2).reshape(-1, 768, 14, 14)
    v_grads = blobs['v_grad'][:, 1:, :].transpose(1, 2).reshape(-1, 768, 14, 14)
    w3 = v_grads.mean(dim=(2, 3), keepdim=True)
    c3 = F.relu((w3 * v_feats).sum(dim=1, keepdim=True))
    c3 = F.interpolate(c3, size=(HYBRID_RES, HYBRID_RES), mode='bilinear', align_corners=False)

    cam_fused = ((c1 + c2 + c3) / 3.0).squeeze().cpu().detach().numpy()
    cam_fused = np.maximum(cam_fused, 0)
    return (cam_fused - cam_fused.min()) / (cam_fused.max() - cam_fused.min() + 1e-8)

def overlay_heatmap(img, mask, colormap=cv2.COLORMAP_JET, alpha=0.6):
    img_np = np.array(img.resize((HYBRID_RES, HYBRID_RES)))
    mask_np = np.uint8(255 * mask)
    heatmap = cv2.applyColorMap(mask_np, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)

# 4. LOAD MODEL & DATASET
# ------------------------------------------------------------------------------
model_xai = EnsembleStackingNet(BEST_3_NAMES).to(DEVICE)
model_path = EXPORT_DIR_HYB / "best_hybrid_model.pth"

if not model_path.exists():
    raise FileNotFoundError(f"Model checkpoint missing at: {model_path}")

model_xai.load_state_dict(torch.load(model_path, map_location=DEVICE))
model_xai.eval()

hybrid_ds_raw = datasets.ImageFolder(str(BASE_DIR))

if SPLIT_FILE.exists():
    with open(SPLIT_FILE, 'r') as f:
        hybrid_test_idx = json.load(f)['test_idx']
else:
    hybrid_test_idx = list(range(len(hybrid_ds_raw)))

np.random.seed(42)
test_samples = np.random.choice(hybrid_test_idx, 4, replace=False)

# 5. VISUALIZATION GRID GENERATION (EXTRA LARGE FONT CONFIGURATION)
# ------------------------------------------------------------------------------
# 🔹 [পরিবর্তন ১] গ্লোবাল ফন্ট সাইজ ১৬ করা হয়েছে
plt.rcParams.update({
    'font.family': 'serif',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.size': 16  # <--- গ্লোবাল ফন্ট বাড়িয়ে ১৬ করা হলো
})

fig, axes = plt.subplots(4, 4, figsize=(24, 24), facecolor='white', dpi=600)
plt.subplots_adjust(wspace=0.15, hspace=0.38) # বড় ফন্টের জন্য হরাইজন্টাল গ্যাপ বাড়ানো হয়েছে

explainer = lime_image.LimeImageExplainer(random_state=42)

for row, idx in enumerate(test_samples):
    img_path, true_cls = hybrid_ds_raw.samples[idx]
    raw_pil = Image.open(img_path).convert("RGB")
    input_tensor = xai_transforms(raw_pil).unsqueeze(0).to(DEVICE)

    logits = model_xai(input_tensor)
    pred_cls = logits.argmax(1).item()

    unified_cam = generate_unified_cam_map(model_xai, input_tensor, pred_cls)

    # 🔹 [পরিবর্তন ২] Column 1 Title Font Size: 18 (Bold)
    axes[row, 0].imshow(np.array(raw_pil.resize((HYBRID_RES, HYBRID_RES))))
    axes[row, 0].set_title(
        f"Sample {row+1}\nGT: {hybrid_ds_raw.classes[true_cls]}\nPD: {hybrid_ds_raw.classes[pred_cls]}",
        fontsize=18, fontweight='bold', pad=14, color='#1A2238'  # <---১৮ করা হয়েছে
    )
    axes[row, 0].axis('off')

    # 🔹 [পরিবর্তন ৩] Column 2 Title Font Size: 18 (Bold)
    axes[row, 1].imshow(overlay_heatmap(raw_pil, unified_cam, colormap=cv2.COLORMAP_VIRIDIS))
    axes[row, 1].set_title("Unified Focus (Viridis)", fontsize=18, fontweight='bold', pad=14, color='#1A2238')  # <---১৮ করা হয়েছে
    axes[row, 1].axis('off')

    # 🔹 [পরিবর্তন ৪] Column 3 Title Font Size: 18 (Bold)
    axes[row, 2].imshow(overlay_heatmap(raw_pil, unified_cam, colormap=cv2.COLORMAP_JET))
    axes[row, 2].set_title("Unified Hybrid Grad-CAM", fontsize=18, fontweight='bold', pad=14, color='#1A2238')  # <---১৮ করা হয়েছে
    axes[row, 2].axis('off')

    # 🔹 [পরিবর্তন ৫] Column 4 Title Font Size: 18 (Bold)
    def lime_predict(imgs):
        batch = torch.stack([xai_transforms(Image.fromarray(i)) for i in imgs]).to(DEVICE)
        with torch.no_grad():
            preds = F.softmax(model_xai(batch), dim=1)
        return preds.cpu().numpy()

    explanation = explainer.explain_instance(
        np.array(raw_pil.resize((HYBRID_RES, HYBRID_RES))),
        lime_predict, top_labels=1, num_samples=300
    )
    temp, mask = explanation.get_image_and_mask(pred_cls, positive_only=True, num_features=5, hide_rest=False)
    axes[row, 3].imshow(mark_boundaries(temp / 255.0, mask))
    axes[row, 3].set_title("LIME Key Segments", fontsize=18, fontweight='bold', pad=14, color='#1A2238')  # <---১৮ করা হয়েছে
    axes[row, 3].axis('off')

plt.tight_layout()

# 6. SAVE ARTIFACTS
# ------------------------------------------------------------------------------
master_xai_png = XAI_REPORT_DIR / "Master_Unified_Hybrid_XAI_Grid.png"
master_xai_pdf = XAI_REPORT_DIR / "Master_Unified_Hybrid_XAI_Grid.pdf"

plt.savefig(master_xai_png, dpi=600, bbox_inches='tight')
plt.savefig(master_xai_pdf, format='pdf', bbox_inches='tight')

print(f"\n[SUCCESS] Large Text Unified Hybrid XAI Grid saved at: {master_xai_png}")
print(f"[SUCCESS] Publication Vector PDF saved at: {master_xai_pdf}")
plt.show()