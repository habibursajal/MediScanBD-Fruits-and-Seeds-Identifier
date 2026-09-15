# ==============================================================================
# PHASE 11: COMPUTATIONAL COMPLEXITY ANALYSIS
# REAL INFERENCE LATENCY COMPARISON (MILLISECONDS)
#
# Purpose:
#   Compare the measured inference latency of:
#       1. Seven Control Baseline Architectures (From Controlled_Baselines)
#       2. Proposed TSHE Hybrid - Single-Pass
#       3. Proposed TSHE Hybrid - 4-Pass TTA
#
# Sources:
#   - Controlled_Baselines/{model_name}/result.json (latency_ms_per_image_4pass_TTA)
#   - Hybrid_TripleStream/single_pass_latency_summary.csv
#   - Hybrid_TripleStream/tta_latency_summary.csv
# ==============================================================================

import os
import json
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch

# ==============================================================================
# 1. INITIALIZATION & PATH CONFIGURATION
# ==============================================================================

print("\n" + "=" * 80)
print(
    "[INFO] PHASE 11: Computational Complexity Analysis "
    "- Real Inference Latency"
)
print("=" * 80)

LOCAL_PROJECT_ROOT = pathlib.Path(
    r"E:\Research Project\BDMediHerb\BDMediHerb_Reviewer_Revision"
)

if LOCAL_PROJECT_ROOT.exists():
    PROJECT_ROOT = LOCAL_PROJECT_ROOT
    print("[ENV] Local Windows PC detected")
    print(f"[PATH] Project root: {PROJECT_ROOT}")

elif os.path.exists("/kaggle/working/BDMediHerb_Reviewer_Revision"):
    PROJECT_ROOT = pathlib.Path("/kaggle/working/BDMediHerb_Reviewer_Revision")
    print("[ENV] Google Colab / Kaggle detected")
    print(f"[PATH] Project root: {PROJECT_ROOT}")

else:
    PROJECT_ROOT = pathlib.Path(
        "/content/drive/MyDrive/BDMediHerb/BDMediHerb_Reviewer_Revision"
    )
    print("[ENV] Colab / Google Drive detected")
    print(f"[PATH] Project root: {PROJECT_ROOT}")

CONTROLLED_BASELINES_DIR = PROJECT_ROOT / "Controlled_Baselines"

# ==============================================================================
# 2. CONTROL BASELINE ARCHITECTURES & LATENCY EXTRACTION
# ==============================================================================

BASE_MODELS = [
    "MobileNet_V3_Large",
    "EfficientNetV2_S",
    "ResNet50",
    "DenseNet121",
    "ViT_B16",
    "Inception_V3",
    "ConvNeXt_Tiny"
]

inf_summary = {}

def get_base_model_latency(model_name):
    """
    Extract the 4-Pass TTA inference latency for a control baseline model from:
    Controlled_Baselines/{model_name}/result.json using key:
    'latency_ms_per_image_4pass_TTA'
    """
    res_json = CONTROLLED_BASELINES_DIR / model_name / "result.json"

    if res_json.exists():
        try:
            with open(res_json, "r") as f:
                data = json.load(f)

            # Targeted Key Extraction
            if "latency_ms_per_image_4pass_TTA" in data:
                return float(data["latency_ms_per_image_4pass_TTA"])
            elif "latency_ms_per_image" in data:
                return float(data["latency_ms_per_image"])

        except Exception as exc:
            print(f"[WARN] Could not read {res_json}: {exc}")

    return None

print("\n" + "-" * 80)
print("[INFO] Loading Control Baseline Latencies from Controlled_Baselines")
print("-" * 80)

for model_name in BASE_MODELS:
    latency = get_base_model_latency(model_name)

    if latency is not None:
        inf_summary[model_name] = latency
        print(
            f"[FOUND Control Baseline] "
            f"{model_name:<25} : {latency:>10.3f} ms"
        )
    else:
        print(
            f"[WARN Control Baseline] "
            f"{model_name:<25} : LATENCY NOT FOUND"
        )

# ==============================================================================
# 3. PROPOSED TSHE HYBRID MODEL LATENCY EXTRACTION
# ==============================================================================

HYBRID_DIR = PROJECT_ROOT / "Hybrid_TripleStream"

print("\n" + "-" * 80)
print("[INFO] Loading Proposed TSHE Hybrid Latencies")
print("-" * 80)
print(f"[PATH] Hybrid directory: {HYBRID_DIR}")

# 3.1 Proposed TSHE - Single-Pass Latency
single_csv = HYBRID_DIR / "single_pass_latency_summary.csv"

if single_csv.exists():
    try:
        df_single = pd.read_csv(single_csv)
        val = df_single.loc[
            df_single["metric"] == "mean_single_pass_latency_ms_per_image", "value"
        ].values

        if len(val) > 0:
            single_pass_latency = float(val[0])
            inf_summary["Proposed TSHE (Single-Pass)"] = single_pass_latency
            print(
                "[FOUND Proposed Hybrid] "
                "Single-Pass Latency: "
                f"{single_pass_latency:.3f} ms"
            )

    except Exception as exc:
        print(f"[WARN] Could not read {single_csv}: {exc}")
else:
    print(f"[WARN] Single-pass latency summary not found: {single_csv}")

# 3.2 Proposed TSHE - 4-Pass TTA Latency
tta_csv = HYBRID_DIR / "tta_latency_summary.csv"

if tta_csv.exists():
    try:
        df_tta = pd.read_csv(tta_csv)
        val = df_tta.loc[
            df_tta["metric"] == "mean_tta_latency_ms_per_image", "value"
        ].values

        if len(val) > 0:
            tta_latency = float(val[0])
            inf_summary["Proposed TSHE (4-Pass TTA)"] = tta_latency
            print(
                "[FOUND Proposed Hybrid] "
                "4-Pass TTA Latency: "
                f"{tta_latency:.3f} ms"
            )

    except Exception as exc:
        print(f"[WARN] Could not read {tta_csv}: {exc}")
else:
    print(f"[WARN] TTA latency summary not found: {tta_csv}")

# Fallback for Single-Pass if missing
if (
    "Proposed TSHE (Single-Pass)" not in inf_summary
    and (HYBRID_DIR / "hybrid_training_history.csv").exists()
):
    try:
        df_hybrid = pd.read_csv(HYBRID_DIR / "hybrid_training_history.csv")
        if "inf_ms" in df_hybrid.columns:
            fallback_latency = float(df_hybrid["inf_ms"].mean())
            inf_summary["Proposed TSHE (Single-Pass)"] = fallback_latency
            print(
                "[FALLBACK Proposed Hybrid] "
                "Single-Pass Latency: "
                f"{fallback_latency:.3f} ms"
            )
    except Exception as exc:
        print(f"[WARN] Could not read fallback file: {exc}")

# ==============================================================================
# 4. SORT ALL MODELS BY ACTUAL LATENCY (Fastest → Slowest)
# ==============================================================================

sorted_inf = dict(
    sorted(
        inf_summary.items(),
        key=lambda item: item[1]
    )
)

model_names = list(sorted_inf.keys())
latency_values = list(sorted_inf.values())

if latency_values:
    print("\n" + "=" * 80)
    print("[INFO] Final Bar-Chart Order (Fastest → Slowest)")
    print("=" * 80)
    for rank, (model_name, latency) in enumerate(sorted_inf.items(), start=1):
        print(f"{rank:02d}. {model_name:<35} {latency:.3f} ms")

# ==============================================================================
# 5. VISUALIZATION GENERATION
# ==============================================================================

if latency_values:
    plt.rcParams.update({
        "font.family": "serif",
        "font.weight": "bold"
    })

    fig, ax = plt.subplots(
        figsize=(16, 9),
        facecolor="white",
        dpi=600
    )

    color_hybrid_tta = "#2E7D32"     # Forest Green
    color_hybrid_single = "#1B5E20"  # Dark Green
    color_base = "#455A64"           # Slate Grey

    colors = []
    for model_name in model_names:
        if "4-Pass TTA" in model_name:
            colors.append(color_hybrid_tta)
        elif "Single-Pass" in model_name:
            colors.append(color_hybrid_single)
        else:
            colors.append(color_base)

    bars = ax.bar(
        model_names,
        latency_values,
        color=colors,
        edgecolor="black",
        linewidth=1.1,
        width=0.55,
        zorder=3
    )

    ax.set_title(
        "Computational Efficiency Analysis: "
        "Control Baselines vs Proposed Hybrid",
        fontsize=20,
        fontweight="bold",
        pad=30,
        color="#1A2238"
    )

    ax.set_ylabel(
        "Avg. Inference Latency per Sample (Milliseconds)",
        fontsize=13,
        fontweight="bold",
        labelpad=15
    )

    ax.set_xlabel(
        "Model Architectures",
        fontsize=13,
        fontweight="bold",
        labelpad=20
    )

    plt.xticks(
        rotation=20,
        ha="right",
        fontsize=11,
        fontweight="bold",
        color="#37474F"
    )
    plt.yticks(fontsize=11)

    max_lat = max(latency_values)
    ax.set_ylim(0, max_lat * 1.18)

    ax.yaxis.grid(
        True,
        linestyle="--",
        alpha=0.4,
        color="#B0BEC5",
        zorder=0
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_color("#78909C")
    ax.spines["bottom"].set_color("#78909C")

    # Value Annotations
    for bar, model_name in zip(bars, model_names):
        height = bar.get_height()
        is_hybrid = "Proposed" in model_name or "TSHE" in model_name
        text_color = "#1B5E20" if is_hybrid else "#1A2238"

        ax.annotate(
            f"{height:.3f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="black",
            color=text_color
        )

    legend_elements = [
        Patch(facecolor=color_hybrid_tta, edgecolor="black", label="Proposed Hybrid (4-Pass TTA)"),
        Patch(facecolor=color_hybrid_single, edgecolor="black", label="Proposed Hybrid (Single-Pass)"),
        Patch(facecolor=color_base, edgecolor="black", label="Control Baseline Architectures")
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=12,
        frameon=True,
        shadow=False,
        facecolor="#FAFAFA"
    )

    plt.tight_layout()

    complexity_save_png = PROJECT_ROOT / "Hybrid_Latency_ms_Analysis.png"
    complexity_save_pdf = PROJECT_ROOT / "Hybrid_Latency_ms_Analysis.pdf"

    plt.savefig(complexity_save_png, dpi=600, bbox_inches="tight")
    plt.savefig(complexity_save_pdf, format="pdf", bbox_inches="tight")

    print("\n" + "=" * 80)
    print("[SUCCESS] Publication-ready latency chart generated.")
    print("=" * 80)
    print(f"[PNG] {complexity_save_png}")
    print(f"[PDF] {complexity_save_pdf}")
    print("=" * 80)

    plt.show()

else:
    print("\n" + "=" * 80)
    print("[ERROR] No valid latency records found across project directories.")
    print("=" * 80)
    