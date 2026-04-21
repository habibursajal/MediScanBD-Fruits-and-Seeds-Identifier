import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediScanBD",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CLASSES (EXACT MATCH WITH YOUR TRAINING)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        if model_name == "MobileNet_V3_Large":
            m = models.mobilenet_v3_large(weights=None)
            self.feat_dim = m.classifier[0].in_features
            m.classifier = nn.Identity()
        elif model_name == "ResNet50":
            m = models.resnet50(weights=None)
            self.feat_dim = m.fc.in_features
            m.fc = nn.Identity()
        elif model_name == "ViT_B16":
            m = models.vit_b_16(weights=None)
            self.feat_dim = m.heads.head.in_features
            m.heads = nn.Identity()
        self.backbone = m

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
        f1 = self.stream1(x); f2 = self.stream2(x); f3 = self.stream3(x)
        out1 = self.head1(f1); out2 = self.head2(f2); out3 = self.head3(f3)
        stacked_logits = torch.cat([out1, out2, out3], dim=-1)
        return self.meta_learner(stacked_logits), [out1, out2, out3]

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE LOADING & STATUS DEFINITION (CRITICAL ORDER)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_engine():
    p = "best_hybrid_model.pth"
    class_names = [
        "Belleric Myrobalan", "Black Cumin", "Black Pepper", "Cardamom",
        "Chebulic Myrobalan", "Cinnamon", "Clove", "Cumin Seeds",
        "Fenugreek Seeds", "Flax Seed", "Garlic", "Ginger",
        "Gooseberry", "Mace", "Nutmeg", "Psoralea Fruit",
        "Sesame Seeds", "Star Anise", "Turmeric"
    ]
    if not os.path.exists(p): return None, {"classes": class_names}
    
    try:
        model = EnsembleStackingNet(["MobileNet_V3_Large", "ResNet50", "ViT_B16"])
        state_dict = torch.load(p, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model, {"classes": class_names}
    except Exception as e:
        return None, {"classes": class_names}

# Initialize variables BEFORE they are used in the Navbar
engine, meta = load_engine()
OK = engine is not None
THRESHOLD = 80.0 # Set back to 80 for professional results

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Dark Green Theme (Design Maintained)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&display=swap');
:root{ --bg:#060a07; --bg2:#0c1510; --bg3:#111c13; --bg4:#1a2b1e; --green:#22c55e; --gd:#15803d; --amber:#f59e0b; --bdr:rgba(255,255,255,.07); --txt:#f0fdf4; --dim:#4a7a5a; --danger:#f87171; }
#MainMenu,footer,header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stSidebar"]{display:none!important}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important}
.block-container{padding-top:0!important;padding-bottom:0!important;max-width:100%!important}
[data-testid="stFileUploader"] section{ background:var(--bg3)!important; border:1.5px dashed rgba(245,158,11,.4)!important; border-radius:10px!important;padding:10px 14px!important}
[data-testid="stImage"] img{ border-radius:10px!important;width:100%!important; max-height:260px!important;object-fit:cover!important}
.stButton>button{ background:linear-gradient(135deg,var(--gd),#166534)!important; color:#fff!important;border:none!important;border-radius:9px!important; padding:11px 0!important;font-family:'Syne',sans-serif!important; font-size:13px!important;font-weight:700!important;width:100%!important; box-shadow:0 4px 16px rgba(21,128,61,.3)!important; letter-spacing:.5px!important;transition:all .15s!important}
.mediscan-nav{ display:flex;align-items:center;justify-content:space-between; padding:0 24px;height:56px; background:linear-gradient(90deg,#0a2010,#091509 50%,var(--bg)); border-bottom:1px solid var(--bdr);position:relative;}
.nav-title{ font-family:'Syne',sans-serif;font-size:16px;font-weight:800; color:#fff;letter-spacing:-.3px;line-height:1.1}
.nav-title span{color:var(--amber)}
.tag{ font-size:10px;font-weight:600;letter-spacing:.5px;padding:3px 10px; border-radius:99px;border:1px solid rgba(255,255,255,.1); color:rgba(255,255,255,.45);background:rgba(255,255,255,.04)}
.tag-live{ border-color:rgba(34,197,94,.4)!important; background:rgba(34,197,94,.1)!important; color:var(--green)!important;display:flex;align-items:center;gap:5px}
.tag-off{ border-color:rgba(248,113,113,.4)!important; background:rgba(248,113,113,.1)!important;color:var(--danger)!important}
.pulse-dot{ width:5px;height:5px;border-radius:50%;background:var(--green); box-shadow:0 0 6px var(--green); animation:pulse 2s ease-in-out infinite;display:inline-block}
@keyframes pulse{ 0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.6)}}
.rcard{ min-height:200px;background:var(--bg3);border:1px solid var(--bdr); border-radius:14px;padding:20px 22px; display:flex;flex-direction:column;gap:11px;position:relative;overflow:hidden;}
.r-variety{ font-family:'Syne',sans-serif;font-size:32px; font-weight:800;color:#fff;line-height:1.05;letter-spacing:-.5px}
.r-mgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}
.r-mc{ background:var(--bg4);border:1px solid var(--bdr);border-radius:8px; padding:8px 9px;display:flex;flex-direction:column;gap:3px}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# NAVBAR LOGIC (SYNCED WITH OK VARIABLE)
# ─────────────────────────────────────────────────────────────────────────────
status_class = "tag-live" if OK else "tag-off"
status_text = "ONLINE" if OK else "OFFLINE"
pulse_html = '<span class="pulse-dot"></span>' if OK else '✕'

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
    <span class="tag">MediScanBD Dataset</span>
    <span class="tag {status_class}">
      {pulse_html}&nbsp;{status_text}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def infer(img: Image.Image):
    t = TF(img).unsqueeze(0)
    with torch.no_grad():
        final_logits, stream_logits = engine(t)
        probs = F.softmax(final_logits, dim=1)
        conf, idx = torch.max(probs, 1)
        
        pm = {}
        MODEL_NAMES = ["MobileNet", "ResNet50", "ViT-B16"]
        for name, logit in zip(MODEL_NAMES, stream_logits):
            s_probs = F.softmax(logit, dim=1)
            s_conf, _ = torch.max(s_probs, 1)
            pm[name] = float(s_conf.item()) * 100
            
    return meta["classes"][idx.item()], float(conf.item()) * 100, pm

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([5, 7], gap="small")

with left:
    st.markdown('<div class="col-head">Input Sample</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("sample", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)
        st.markdown(f'<div class="chips"><span class="chip"><span class="cdot"></span>224×224 input</span><span class="chip"><span class="cdot"></span>3-model ensemble</span><span class="chip"><span class="cdot"></span>{THRESHOLD}% threshold</span></div>', unsafe_allow_html=True)
        clicked = st.button("🔬 Identify Sample", use_container_width=True)
    else:
        st.markdown('<div class="no-img"><div class="no-img-icon">🍃</div><div class="no-img-txt">No image loaded</div></div>', unsafe_allow_html=True)
        img, clicked = None, False

with right:
    st.markdown('<div class="col-head">Detection Result</div>', unsafe_allow_html=True)
    
    if clicked and img is not None:
        if not OK:
            st.markdown('<div class="ecard"><div class="etitle">System Offline</div></div>', unsafe_allow_html=True)
        else:
            with st.spinner("Running ensemble inference..."):
                v, s, pm = infer(img)
                
                if s < THRESHOLD:
                    st.markdown(f'<div class="ecard"><div class="eico">❌</div><div class="etitle">Low Confidence — {s:.1f}%</div><div class="ebody">Below identification threshold.</div></div>', unsafe_allow_html=True)
                else:
                    mc_html = "".join([f'<div class="r-mc"><span class="r-mcn">{n}</span><span class="r-mcv">{c:.1f}%</span><div class="r-mcbar"><div class="r-mcbf" style="width:{min(c,100):.1f}%"></div></div></div>' for n, c in pm.items()])
                    st.markdown(f"""
                    <div class="rcard">
                      <div class="r-badge"><span class="r-bdot"></span>Species Identified</div>
                      <div class="r-variety">{v}</div>
                      <div class="r-crow"><span class="r-cnum">{s:.1f}%</span><span class="r-clbl">ensemble confidence</span></div>
                      <div class="r-bar"><div class="r-barfill" style="width:{min(s,100):.1f}%"></div></div>
                      <div class="r-div"></div>
                      <div class="r-mlbl">Per-Model Scores</div>
                      <div class="r-mgrid">{mc_html}</div>
                    </div>""", unsafe_allow_html=True)
                    if s > 85: st.balloons()
    else:
        st.markdown('<div class="idle"><div class="idle-ico">🍃</div><div class="idle-txt">Neural engine idle</div></div>', unsafe_allow_html=True)

st.markdown("""<div class="mediscan-foot"><span class="ft">Developed by <strong>Habibur Rahman Sajal</strong> &nbsp;·&nbsp; MediScanBD Dataset</span></div>""", unsafe_allow_html=True)
