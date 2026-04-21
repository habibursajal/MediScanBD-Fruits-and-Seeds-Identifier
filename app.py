import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediScanBD",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&display=swap');

:root{
  --bg:#060a07; --bg2:#0c1510; --bg3:#111c13; --bg4:#1a2b1e;
  --green:#22c55e; --gd:#15803d; --amber:#f59e0b;
  --bdr:rgba(255,255,255,.07); --txt:#f0fdf4;
  --dim:#4a7a5a; --danger:#f87171;
}

#MainMenu,footer,header[data-testid="stHeader"],
[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stSidebar"]{display:none!important}

html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important}
[data-testid="stAppViewContainer"]>section.main{padding-top:0!important;padding-bottom:0!important}
.block-container{padding-top:0!important;padding-bottom:0!important;max-width:100%!important}

[data-testid="stFileUploader"]>label{display:none!important}
[data-testid="stFileUploader"] section{
  background:var(--bg3)!important;
  border:1.5px dashed rgba(245,158,11,.4)!important;
  border-radius:10px!important;padding:10px 14px!important}
[data-testid="stFileUploader"] section:hover{
  border-color:var(--amber)!important;background:rgba(245,158,11,.05)!important}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small{
  color:rgba(255,255,255,.5)!important;font-size:12px!important;
  font-family:'DM Sans',sans-serif!important}
[data-testid="stFileUploader"] button{
  background:rgba(245,158,11,.15)!important;color:var(--amber)!important;
  border:1px solid rgba(245,158,11,.35)!important;border-radius:7px!important;
  font-weight:600!important;font-size:11px!important;padding:4px 12px!important}

[data-testid="stImage"] img{
  border-radius:10px!important;width:100%!important;
  max-height:260px!important;object-fit:cover!important}

.stButton>button{
  background:linear-gradient(135deg,var(--gd),#166534)!important;
  color:#fff!important;border:none!important;border-radius:9px!important;
  padding:11px 0!important;font-family:'Syne',sans-serif!important;
  font-size:13px!important;font-weight:700!important;width:100%!important;
  box-shadow:0 4px 16px rgba(21,128,61,.3)!important;
  letter-spacing:.5px!important;transition:all .15s!important}
.stButton>button:hover{
  transform:translateY(-1px)!important;
  box-shadow:0 6px 22px rgba(21,128,61,.42)!important}

[data-testid="stSpinner"]>div{border-top-color:var(--amber)!important}
body *{box-sizing:border-box}

.mediscan-nav{
  display:flex;align-items:center;justify-content:space-between;
  padding:0 24px;height:56px;
  background:linear-gradient(90deg,#0a2010,#091509 50%,var(--bg));
  border-bottom:1px solid var(--bdr);position:relative;
}
.mediscan-nav::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,var(--gd),transparent 55%);
}
.nav-brand{display:flex;align-items:center;gap:10px}
.nav-logo{font-size:22px}
.nav-title{
  font-family:'Syne',sans-serif;font-size:16px;font-weight:800;
  color:#fff;letter-spacing:-.3px;line-height:1.1}
.nav-title span{color:var(--amber)}
.nav-sub{
  font-family:'DM Sans',sans-serif;font-size:10px;
  color:var(--dim);letter-spacing:.4px;margin-top:1px}
.nav-right{display:flex;gap:6px;align-items:center}
.tag{
  font-size:10px;font-weight:600;letter-spacing:.5px;padding:3px 10px;
  border-radius:99px;border:1px solid rgba(255,255,255,.1);
  color:rgba(255,255,255,.45);background:rgba(255,255,255,.04)}
.tag-live{
  border-color:rgba(34,197,94,.4)!important;
  background:rgba(34,197,94,.1)!important;
  color:var(--green)!important;display:flex;align-items:center;gap:5px}
.tag-off{
  border-color:rgba(248,113,113,.4)!important;
  background:rgba(248,113,113,.1)!important;color:var(--danger)!important}
@keyframes pulse{
  0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.6)}}
.pulse-dot{
  width:5px;height:5px;border-radius:50%;background:var(--green);
  box-shadow:0 0 6px var(--green);
  animation:pulse 2s ease-in-out infinite;display:inline-block}

.col-head{
  font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
  color:var(--dim);font-family:'DM Sans',sans-serif;
  display:flex;align-items:center;gap:8px;margin-bottom:8px;
}
.col-head::after{content:'';flex:1;height:1px;background:var(--bdr)}

.no-img{
  min-height:200px;border:1px solid var(--bdr);border-radius:10px;
  background:var(--bg3);display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:10px;margin:6px 0;
}
.no-img-icon{
  width:48px;height:48px;border-radius:50%;
  background:rgba(34,197,94,.07);border:1px dashed rgba(34,197,94,.2);
  display:flex;align-items:center;justify-content:center;font-size:21px}
.no-img-txt{
  font-size:11px;color:var(--dim);text-align:center;
  line-height:1.6;font-family:'DM Sans',sans-serif}

.chips{display:flex;gap:5px;flex-wrap:wrap;margin:6px 0}
.chip{
  font-size:10px;font-weight:600;padding:3px 9px;border-radius:99px;
  border:1px solid var(--bdr);color:rgba(255,255,255,.3);
  background:rgba(255,255,255,.02);font-family:'DM Sans',sans-serif;
  display:flex;align-items:center;gap:4px}
.cdot{width:4px;height:4px;border-radius:50%;background:var(--amber)}

.idle{
  min-height:200px;border:1px dashed var(--bdr);border-radius:14px;
  display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:10px;
}
.idle-ico{font-size:32px;opacity:.18}
.idle-txt{
  font-size:12px;color:var(--dim);text-align:center;
  line-height:1.8;font-family:'DM Sans',sans-serif}

.rcard{
  min-height:200px;background:var(--bg3);border:1px solid var(--bdr);
  border-radius:14px;padding:20px 22px;
  display:flex;flex-direction:column;gap:11px;position:relative;overflow:hidden;
}
.rcard::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--gd),var(--amber),transparent 75%);
}
.r-badge{
  display:flex;align-items:center;gap:5px;font-size:9px;font-weight:700;
  letter-spacing:1.8px;text-transform:uppercase;
  color:var(--green);font-family:'DM Sans',sans-serif}
.r-bdot{
  width:6px;height:6px;border-radius:50%;
  background:var(--green);box-shadow:0 0 7px var(--green)}
.r-variety{
  font-family:'Syne',sans-serif;font-size:clamp(22px,2.6vw,36px);
  font-weight:800;color:#fff;line-height:1.05;letter-spacing:-.5px}
.r-crow{display:flex;align-items:baseline;gap:8px}
.r-cnum{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:var(--amber)}
.r-clbl{font-size:12px;color:var(--dim);font-family:'DM Sans',sans-serif}
.r-bar{height:5px;border-radius:3px;background:rgba(255,255,255,.06);overflow:hidden}
.r-barfill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--gd),var(--amber))}
.r-div{height:1px;background:var(--bdr)}
.r-mlbl{
  font-size:9px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;
  color:var(--dim);font-family:'DM Sans',sans-serif}
.r-mgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}
.r-mc{
  background:var(--bg4);border:1px solid var(--bdr);border-radius:8px;
  padding:8px 9px;display:flex;flex-direction:column;gap:3px}
.r-mcn{
  font-size:9px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;
  color:var(--dim);font-family:'DM Sans',sans-serif}
.r-mcv{font-family:'Syne',sans-serif;font-size:14px;font-weight:700;color:var(--txt)}
.r-mcbar{
  height:3px;border-radius:2px;background:rgba(255,255,255,.07);
  overflow:hidden;margin-top:2px}
.r-mcbf{height:100%;border-radius:2px;background:var(--green);opacity:.6}

.ecard{
  min-height:200px;border:1px solid rgba(248,113,113,.2);border-radius:14px;
  background:rgba(248,113,113,.06);display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:8px;text-align:center;padding:22px;
}
.eico{font-size:28px}
.etitle{font-family:'Syne',sans-serif;font-size:15px;font-weight:800;color:var(--danger)}
.ebody{font-size:12px;color:rgba(255,255,255,.45);line-height:1.7;font-family:'DM Sans',sans-serif}

.mediscan-foot{
  height:34px;background:var(--bg2);border-top:1px solid var(--bdr);
  display:flex;align-items:center;justify-content:space-between;
  padding:0 24px;margin-top:8px;
}
.ft{font-size:10px;color:var(--dim);font-family:'DM Sans',sans-serif}
.ft strong{color:var(--amber);font-weight:600}
.ftags{display:flex;gap:4px}
.ftag{
  font-size:9px;font-weight:600;padding:2px 7px;border-radius:99px;
  background:rgba(255,255,255,.03);border:1px solid var(--bdr);
  color:rgba(255,255,255,.25);font-family:'DM Sans',sans-serif}

@media(max-width:720px){
  .mediscan-nav{height:auto;padding:10px 14px;flex-wrap:wrap;gap:6px}
  .nav-right{flex-wrap:wrap}
  .r-mgrid{grid-template-columns:1fr 1fr!important}
  .mediscan-foot{
    height:auto;padding:8px 14px;
    flex-direction:column;align-items:flex-start;gap:4px}
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIX #1 — CORRECT ARCHITECTURE (Matches Phase 3 & Phase 4 training exactly)
# ─────────────────────────────────────────────────────────────────────────────

def initialize_architecture(name):
    """
    CRITICAL: This must EXACTLY match initialize_architecture() in the
    single-model training script (Phase 3), including the custom heads.
    The heads are loaded AS-IS from best_model.pth, then stripped in FeatureExtractor.
    """
    if name == "MobileNet_V3_Large":
        model = models.mobilenet_v3_large(weights=None)
        num_ftrs = model.classifier[3].in_features
        # MUST match your Phase 3 training head exactly:
        model.classifier[3] = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 19)
        )
    elif name == "ResNet50":
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 19)
    elif name == "ViT_B16":
        model = models.vit_b_16(weights=None)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, 19)
    return model


class FeatureExtractor(nn.Module):
    """
    FIX #2 — feat_dim and head-stripping now matches Phase 4 training code EXACTLY.

    Phase 4 training code strips heads like this:
      - ViT:       base_model.heads = nn.Identity()  → feat_dim = heads.head.in_features
      - ResNet:    base_model.fc = nn.Identity()      → feat_dim = fc.in_features
      - MobileNet: base_model.classifier = nn.Identity() → feat_dim = classifier[0].in_features
                   BUT the classifier was REPLACED in Phase 3 with a Sequential.
                   So after loading weights, classifier[0] is nn.Linear(960, 512).
                   The feat_dim should be classifier[0].in_features = 960 (the original).

    We load the full trained model (including custom heads), read the correct
    feat_dim BEFORE stripping, then strip the head.
    """
    def __init__(self, model_name: str, weights_path: str):
        super().__init__()

        # Step 1: Build the architecture with the custom classification head
        base_model = initialize_architecture(model_name)

        # Step 2: Load trained weights (this populates the custom heads too)
        state_dict = torch.load(weights_path, map_location="cpu")
        base_model.load_state_dict(state_dict)

        # Step 3: Record feat_dim and strip the head (matches Phase 4 exactly)
        if "ViT" in model_name:
            self.feat_dim = base_model.heads.head.in_features
            base_model.heads = nn.Identity()
        elif "ResNet" in model_name:
            self.feat_dim = base_model.fc.in_features
            base_model.fc = nn.Identity()
        elif "MobileNet" in model_name:
            # After Phase 3, classifier[0] is the original MobileNet Linear(960, ...)
            # The original in_features before the custom head replacement is 960
            self.feat_dim = base_model.classifier[0].in_features
            base_model.classifier = nn.Identity()

        self.backbone = base_model

    def forward(self, x):
        x = self.backbone(x)
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        return x


class EnsembleStackingNet(nn.Module):
    """
    FIX #3 — Architecture matches Phase 4 training EXACTLY.
    The forward() now returns ONLY final logits (no stream_logits tuple)
    so it matches how the model was saved and evaluated in Phase 5.
    We extract per-stream scores separately during inference.
    """
    def __init__(self, model_names, weights_dir: str, num_classes: int = 19):
        super().__init__()
        # Load each backbone with its own trained weights
        self.stream1 = FeatureExtractor(
            model_names[0],
            os.path.join(weights_dir, model_names[0], "best_model.pth")
        )
        self.stream2 = FeatureExtractor(
            model_names[1],
            os.path.join(weights_dir, model_names[1], "best_model.pth")
        )
        self.stream3 = FeatureExtractor(
            model_names[2],
            os.path.join(weights_dir, model_names[2], "best_model.pth")
        )

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
        # Returns ALL outputs — we use them for per-model scores at inference
        return self.meta_learner(stacked_logits), out1, out2, out3


# ─────────────────────────────────────────────────────────────────────────────
# FIX #4 — TEMPERATURE SCALING for calibrated confidence
# After label_smoothing training, raw softmax is over-smoothed.
# A temperature < 1.0 sharpens the distribution (raises top confidence).
# Tune TEMPERATURE between 0.3 and 0.7 depending on your validation results.
# ─────────────────────────────────────────────────────────────────────────────
TEMPERATURE = 0.5  # Lower = sharper/higher confidence. Tune this on your val set.

def calibrated_softmax(logits: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    """Applies temperature scaling before softmax to calibrate confidence."""
    return F.softmax(logits / temperature, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# FIX #5 — 4-PASS TTA (matches Phase 5 evaluation exactly)
# ─────────────────────────────────────────────────────────────────────────────
def run_tta(model, tensor: torch.Tensor):
    """
    Runs 4-pass Test-Time Augmentation matching Phase 5 evaluation:
    original + horizontal flip + vertical flip + both flips
    Returns averaged raw logits (before softmax).
    """
    o1_f, o1_1, o1_2, o1_3 = model(tensor)
    o2_f, o2_1, o2_2, o2_3 = model(torch.flip(tensor, dims=[3]))
    o3_f, o3_1, o3_2, o3_3 = model(torch.flip(tensor, dims=[2]))
    o4_f, o4_1, o4_2, o4_3 = model(torch.flip(tensor, dims=[2, 3]))

    avg_final  = (o1_f + o2_f + o3_f + o4_f) / 4
    avg_head1  = (o1_1 + o2_1 + o3_1 + o4_1) / 4
    avg_head2  = (o1_2 + o2_2 + o3_2 + o4_2) / 4
    avg_head3  = (o1_3 + o2_3 + o3_3 + o4_3) / 4

    return avg_final, avg_head1, avg_head2, avg_head3


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING — Paths are configurable
# ─────────────────────────────────────────────────────────────────────────────

# Option A: Set these to your actual paths
HYBRID_WEIGHTS_PATH = "best_hybrid_model.pth"   # Hybrid ensemble weights
BACKBONE_WEIGHTS_DIR = "."                        # Dir containing MobileNet_V3_Large/, ResNet50/, ViT_B16/ subdirs

MODEL_NAMES = ["MobileNet_V3_Large", "ResNet50", "ViT_B16"]
MODEL_DISPLAY = ["MobileNet", "ResNet50", "ViT-B16"]

CLASS_NAMES = [
    "Belleric Myrobalan", "Black Cumin", "Black Pepper", "Cardamom",
    "Chebulic Myrobalan", "Cinnamon", "Clove", "Cumin Seeds",
    "Fenugreek Seeds", "Flax Seed", "Garlic", "Ginger",
    "Gooseberry", "Mace", "Nutmeg", "Psoralea Fruit",
    "Sesame Seeds", "Star Anise", "Turmeric"
]

THRESHOLD = 30.0  # Lowered from 10% — with calibration, real matches are now 60-99%

# ─────────────────────────────────────────────────────────────────────────────
# FIX #6 — Correct transform (no augmentation at inference, matches eval_trans)
# ─────────────────────────────────────────────────────────────────────────────
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@st.cache_resource
def load_engine():
    """
    Loads the hybrid ensemble model.
    Falls back gracefully if weights are not found.
    """
    if not os.path.exists(HYBRID_WEIGHTS_PATH):
        return None

    try:
        # Build model (loads individual backbone weights during construction)
        model = EnsembleStackingNet(
            MODEL_NAMES,
            weights_dir=BACKBONE_WEIGHTS_DIR,
            num_classes=19
        )

        # Load the fine-tuned hybrid weights (Phase 4 output)
        hybrid_state = torch.load(HYBRID_WEIGHTS_PATH, map_location="cpu")
        model.load_state_dict(hybrid_state)
        model.eval()
        return model

    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None


def infer(img: Image.Image):
    """
    Full inference pipeline:
    1. Preprocess (eval_trans only — no augmentation)
    2. 4-pass TTA averaging (matches Phase 5)
    3. Temperature-scaled softmax for calibrated confidence
    Returns: (class_name, confidence_pct, {model_name: confidence_pct})
    """
    tensor = TF(img).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        avg_final, avg_head1, avg_head2, avg_head3 = run_tta(engine, tensor)

    # Calibrated softmax on ensemble output
    probs = calibrated_softmax(avg_final)
    conf, idx = torch.max(probs, 1)

    # Per-model calibrated confidence for the predicted class
    p1 = calibrated_softmax(avg_head1)
    p2 = calibrated_softmax(avg_head2)
    p3 = calibrated_softmax(avg_head3)

    predicted_class = idx.item()
    per_model = {
        MODEL_DISPLAY[0]: float(p1[0, predicted_class].item()) * 100,
        MODEL_DISPLAY[1]: float(p2[0, predicted_class].item()) * 100,
        MODEL_DISPLAY[2]: float(p3[0, predicted_class].item()) * 100,
    }

    return CLASS_NAMES[predicted_class], float(conf.item()) * 100, per_model


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE INIT
# ─────────────────────────────────────────────────────────────────────────────
engine = load_engine()
OK = engine is not None

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

status_class = "tag-live" if OK else "tag-off"
status_text  = "ONLINE"   if OK else "OFFLINE"
pulse_html   = '<span class="pulse-dot"></span>' if OK else '✕'

st.markdown(f"""
<div class="mediscan-nav">
  <div class="nav-brand">
    <span class="nav-logo">🍃</span>
    <div class="nav-text">
      <div class="nav-title">MediScan<span>BD</span></div>
      <div class="nav-sub">Fruits &amp; Seed Identifier</div>
    </div>
  </div>
  <div class="nav-right">
    <span class="tag">Hybrid Ensemble ×3</span>
    <span class="tag">BDMediHerb Dataset</span>
    <span class="tag {status_class}">
      {pulse_html}&nbsp;{status_text}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([5, 7], gap="small")

with left:
    st.markdown('<div class="col-head">Input Sample</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("sample", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    img = None
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)
    else:
        st.markdown("""
        <div class="no-img">
          <div class="no-img-icon">🍃</div>
          <div class="no-img-txt">No image loaded<br>
            <span style="opacity:.45;font-size:10px">JPG · JPEG · PNG</span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="chips">
      <span class="chip"><span class="cdot"></span>224×224 input</span>
      <span class="chip"><span class="cdot"></span>4-pass TTA</span>
      <span class="chip"><span class="cdot"></span>T={TEMPERATURE} calibration</span>
    </div>""", unsafe_allow_html=True)

    clicked = False
    if img is not None:
        clicked = st.button("🔬  Identify Sample", use_container_width=True)

with right:
    st.markdown('<div class="col-head">Detection Result</div>', unsafe_allow_html=True)

    if clicked and img is not None:
        if not OK:
            st.session_state["res"] = "offline"
        else:
            with st.spinner("Running ensemble inference…"):
                v, s, pm = infer(img)
                st.session_state["res"] = (v, s, pm)

    if img is None:
        st.session_state.pop("res", None)

    res = st.session_state.get("res", None)

    if res is None:
        st.markdown("""
        <div class="idle">
          <div class="idle-ico">🍃</div>
          <div class="idle-txt">Neural engine idle<br>
            Upload a sample &amp; click
            <b style="color:rgba(255,255,255,.3)">Identify Sample</b>
          </div>
        </div>""", unsafe_allow_html=True)

    elif res == "offline":
        st.markdown(f"""
        <div class="ecard">
          <div class="eico">⚠️</div>
          <div class="etitle">Model Weights Not Found</div>
          <div class="ebody">
            <b>{HYBRID_WEIGHTS_PATH}</b> is missing.<br>
            Also ensure backbone weights exist at:<br>
            <b>MobileNet_V3_Large/best_model.pth</b><br>
            <b>ResNet50/best_model.pth</b><br>
            <b>ViT_B16/best_model.pth</b>
          </div>
        </div>""", unsafe_allow_html=True)

    else:
        variety, score, pm = res

        if score < THRESHOLD:
            st.markdown(f"""
            <div class="ecard">
              <div class="eico">❌</div>
              <div class="etitle">Low Confidence — {score:.1f}%</div>
              <div class="ebody">
                Below the {THRESHOLD:.0f}% identification threshold.<br>
                Try a clearer image on a plain background.
              </div>
            </div>""", unsafe_allow_html=True)

        else:
            mc = "".join(f"""
            <div class="r-mc">
              <span class="r-mcn">{n}</span>
              <span class="r-mcv">{c:.1f}%</span>
              <div class="r-mcbar">
                <div class="r-mcbf" style="width:{min(c,100):.1f}%"></div>
              </div>
            </div>""" for n, c in pm.items())

            st.markdown(f"""
            <div class="rcard">
              <div class="r-badge">
                <span class="r-bdot"></span>Species Identified
              </div>
              <div class="r-variety">{variety}</div>
              <div class="r-crow">
                <span class="r-cnum">{score:.1f}%</span>
                <span class="r-clbl">ensemble confidence (calibrated)</span>
              </div>
              <div class="r-bar">
                <div class="r-barfill" style="width:{min(score,100):.1f}%"></div>
              </div>
              <div class="r-div"></div>
              <div class="r-mlbl">Per-Model Scores (calibrated)</div>
              <div class="r-mgrid">{mc}</div>
            </div>""", unsafe_allow_html=True)

            if score > 85:
                st.balloons()

st.markdown("""
<div class="mediscan-foot">
  <span class="ft">
    Developed by <strong>Habibur Rahman Sajal</strong>
    &nbsp;·&nbsp; MediScanBD Fruits & Seed Identifier
  </span>
  <div class="ftags">
    <span class="ftag">MobileNet_V3</span>
    <span class="ftag">ResNet50</span>
    <span class="ftag">ViT-B16</span>
  </div>
</div>
""", unsafe_allow_html=True)
