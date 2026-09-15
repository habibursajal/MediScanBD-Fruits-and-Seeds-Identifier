# ==============================================================================
# PHASE 12: OPTIMIZATION DYNAMICS
# MASTER LEARNING RATE CONVERGENCE ANALYSIS
#
# CONTROL BASELINES vs PROPOSED TSHE HYBRID
#
# Purpose:
#   Publication-ready visualization of learning-rate evolution for:
#
#       1. MobileNet_V3_Large
#       2. EfficientNetV2_S
#       3. ResNet50
#       4. DenseNet121
#       5. ViT_B16
#       6. Inception_V3
#       7. ConvNeXt_Tiny
#       8. Proposed TSHE Hybrid
#
# Important Visualization Principle:
#
#   All controlled baselines follow the same LR optimization schedule.
#   Therefore, their underlying LR trajectories are identical.
#
#   To avoid misleading artificial LR offsets, the baseline models are
#   represented using distinct colored markers on BOTH:
#
#       - Head LR trajectory
#       - Backbone LR trajectory
#
#   Proposed TSHE is highlighted using thick green lines.
#
# Optimization Configuration:
#
#   Backbone LR : 1e-6
#   Head LR     : 1e-4
#
#   Optimizer   : AdamW
#   Scheduler   : CosineAnnealingWarmRestarts
#   T0          : 10
#   Tmult       : 2
#   eta_min     : 1e-7
#
# Outputs:
#
#   Master_LR_Convergence_Analysis.png
#   Master_LR_Convergence_Analysis.pdf
#   Master_LR_Convergence_Summary.csv
#
# Publication Target:
#   Q1 Journal / IEEE / SAGE
# ==============================================================================


# ==============================================================================
# 0. IMPORT REQUIRED LIBRARIES
# ==============================================================================

import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================================
# 1. SYSTEM HEADER
# ==============================================================================

print("\n" + "=" * 100)

print(
    f"{'[SYSTEM] MASTER LEARNING RATE CONVERGENCE ANALYSIS':^100}"
)

print("=" * 100)


# ==============================================================================
# 2. PATH CONFIGURATION
# ==============================================================================

LOCAL_PROJECT_ROOT = pathlib.Path(
    r"E:\Research Project\BDMediHerb\BDMediHerb_Reviewer_Revision"
)


# ------------------------------------------------------------------------------
# Automatic environment detection
# ------------------------------------------------------------------------------

if LOCAL_PROJECT_ROOT.exists():

    PROJECT_ROOT = LOCAL_PROJECT_ROOT

    print("[ENV] Local Windows PC detected")

elif os.path.exists(
    "/kaggle/working/BDMediHerb_Reviewer_Revision"
):

    PROJECT_ROOT = pathlib.Path(
        "/kaggle/working/BDMediHerb_Reviewer_Revision"
    )

    print("[ENV] Kaggle detected")

else:

    PROJECT_ROOT = pathlib.Path(
        "/content/drive/MyDrive/BDMediHerb/"
        "BDMediHerb_Reviewer_Revision"
    )

    print("[ENV] Google Colab detected")


print(
    f"[PATH] Project Root: {PROJECT_ROOT}"
)


# ==============================================================================
# 3. MODEL CONFIGURATION
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


HYBRID_MODEL_FOLDER = "Hybrid_TripleStream"


# ==============================================================================
# 4. EXACT OPTIMIZATION CONFIGURATION
# ==============================================================================

# ------------------------------------------------------------------------------
# Differential Learning Rates
# ------------------------------------------------------------------------------

BACKBONE_LR = 1e-6

HEAD_LR = 1e-4


# ------------------------------------------------------------------------------
# Cosine Annealing Warm Restarts
# ------------------------------------------------------------------------------

T0 = 10

T_MULT = 2

ETA_MIN = 1e-7


# ------------------------------------------------------------------------------
# Default number of epochs
# ------------------------------------------------------------------------------

DEFAULT_EPOCHS = 50


# ==============================================================================
# 5. PROFESSIONAL Q1 JOURNAL FIGURE STYLE
# ==============================================================================

plt.rcParams.update({

    "font.family": "serif",

    "font.weight": "bold",

    "axes.labelweight": "bold",

    "axes.titleweight": "bold",

    "xtick.labelsize": 12,

    "ytick.labelsize": 12,

    "legend.fontsize": 10.5,

    "axes.linewidth": 1.0,

    "figure.autolayout": False
})


# ==============================================================================
# 6. FIND BASELINE TRAINING HISTORY
# ==============================================================================

def find_base_history(model_name):

    possible_paths = [

        PROJECT_ROOT /
        model_name /
        "training_history.csv",

        PROJECT_ROOT /
        "Controlled_Baselines" /
        model_name /
        "training_history.csv",

        PROJECT_ROOT /
        "Individual_Models" /
        model_name /
        "training_history.csv"
    ]


    for path in possible_paths:

        if path.exists():

            return path


    return None


# ==============================================================================
# 7. FIND HYBRID TRAINING HISTORY
# ==============================================================================

hybrid_path = (

    PROJECT_ROOT /

    HYBRID_MODEL_FOLDER /

    "hybrid_training_history.csv"
)


# ==============================================================================
# 8. DETERMINE MAXIMUM TRAINING EPOCHS
# ==============================================================================

detected_epochs = []


# ------------------------------------------------------------------------------
# Check baseline histories
# ------------------------------------------------------------------------------

for model_name in BASE_MODELS:

    history_path = find_base_history(
        model_name
    )


    if history_path is not None:

        try:

            df_temp = pd.read_csv(
                history_path
            )


            if "epoch" in df_temp.columns:

                epoch_values = pd.to_numeric(
                    df_temp["epoch"],
                    errors="coerce"
                ).dropna()


                if len(epoch_values) > 0:

                    detected_epochs.append(
                        int(epoch_values.max())
                    )

            else:

                detected_epochs.append(
                    len(df_temp)
                )


        except Exception as e:

            print(
                f"[WARN] Could not determine epochs "
                f"for {model_name}: {e}"
            )


# ------------------------------------------------------------------------------
# Check hybrid history
# ------------------------------------------------------------------------------

if hybrid_path.exists():

    try:

        df_hybrid_temp = pd.read_csv(
            hybrid_path
        )


        if "epoch" in df_hybrid_temp.columns:

            epoch_values = pd.to_numeric(
                df_hybrid_temp["epoch"],
                errors="coerce"
            ).dropna()


            if len(epoch_values) > 0:

                detected_epochs.append(
                    int(epoch_values.max())
                )

        else:

            detected_epochs.append(
                len(df_hybrid_temp)
            )


    except Exception as e:

        print(
            f"[WARN] Could not determine hybrid epochs: {e}"
        )


# ------------------------------------------------------------------------------
# Final epoch count
# ------------------------------------------------------------------------------

if detected_epochs:

    MAX_EPOCHS = max(
        detected_epochs
    )

else:

    MAX_EPOCHS = DEFAULT_EPOCHS


print(
    f"[INFO] Maximum detected training epochs: "
    f"{MAX_EPOCHS}"
)


# ==============================================================================
# 9. COSINE ANNEALING WARM RESTARTS FUNCTION
# ==============================================================================

def generate_cosine_warm_restart_lr(

    base_lr,

    epochs,

    T0=10,

    T_mult=2,

    eta_min=1e-7

):

    """
    Generate epoch-wise nominal LR trajectory for
    CosineAnnealingWarmRestarts.

    This function does not artificially alter any model's LR.
    It reconstructs the common scheduled trajectory when actual
    epoch-wise LR values are not directly available.
    """

    lr_values = []


    cycle_length = T0

    cycle_start = 0


    for epoch in range(epochs):

        t_cur = epoch - cycle_start


        # ----------------------------------------------------------------------
        # Warm restart handling
        # ----------------------------------------------------------------------

        while t_cur >= cycle_length:

            cycle_start += cycle_length

            cycle_length *= T_mult

            t_cur = epoch - cycle_start


        # ----------------------------------------------------------------------
        # Cosine annealing equation
        # ----------------------------------------------------------------------

        cosine_factor = (

            0.5
            *
            (
                1.0
                +
                np.cos(
                    np.pi
                    *
                    t_cur
                    /
                    cycle_length
                )
            )
        )


        lr = (

            eta_min
            +
            (
                base_lr - eta_min
            )
            *
            cosine_factor
        )


        lr_values.append(
            lr
        )


    return np.asarray(
        lr_values,
        dtype=float
    )


# ==============================================================================
# 10. EPOCH VECTOR
# ==============================================================================

epochs = np.arange(

    1,

    MAX_EPOCHS + 1
)


# ==============================================================================
# 11. GENERATE BACKBONE LR TRAJECTORY
# ==============================================================================

backbone_lr_curve = generate_cosine_warm_restart_lr(

    base_lr=BACKBONE_LR,

    epochs=MAX_EPOCHS,

    T0=T0,

    T_mult=T_MULT,

    eta_min=ETA_MIN
)


# ==============================================================================
# 12. GENERATE HEAD LR TRAJECTORY
# ==============================================================================

head_lr_curve = generate_cosine_warm_restart_lr(

    base_lr=HEAD_LR,

    epochs=MAX_EPOCHS,

    T0=T0,

    T_mult=T_MULT,

    eta_min=ETA_MIN
)


# ==============================================================================
# 13. CREATE FIGURE
# ==============================================================================

fig, ax = plt.subplots(

    figsize=(18, 10),

    facecolor="white",

    dpi=600
)


# ==============================================================================
# 14. PROFESSIONAL BASELINE COLOR PALETTE
# ==============================================================================

base_colors = plt.cm.Blues(

    np.linspace(

        0.40,

        0.95,

        len(BASE_MODELS)
    )
)


# ==============================================================================
# 15. DISTINCT MARKER STYLES
# ==============================================================================

marker_styles = [

    "o",     # MobileNet_V3_Large

    "s",     # EfficientNetV2_S

    "^",     # ResNet50

    "D",     # DenseNet121

    "v",     # ViT_B16

    "P",     # Inception_V3

    "X"      # ConvNeXt_Tiny
]


# ==============================================================================
# 16. MARKER POSITIONS
#
# Each model receives different epoch positions.
# This prevents complete marker overlap while preserving exactly the same
# underlying LR values.
# ==============================================================================

marker_positions = [

    np.arange(1, MAX_EPOCHS + 1, 7),

    np.arange(2, MAX_EPOCHS + 1, 7),

    np.arange(3, MAX_EPOCHS + 1, 7),

    np.arange(4, MAX_EPOCHS + 1, 7),

    np.arange(5, MAX_EPOCHS + 1, 7),

    np.arange(6, MAX_EPOCHS + 1, 7),

    np.arange(7, MAX_EPOCHS + 1, 7)
]


# ==============================================================================
# 17. CONTROL BASELINE — SHARED HEAD LR
# ==============================================================================

ax.plot(

    epochs,

    head_lr_curve,

    color="#90A4AE",

    linewidth=2.2,

    linestyle="--",

    alpha=0.60,

    zorder=2,

    label="Control Baselines — Shared LR Schedule"
)


# ==============================================================================
# 18. CONTROL BASELINE — SHARED BACKBONE LR
# ==============================================================================

ax.plot(

    epochs,

    backbone_lr_curve,

    color="#90A4AE",

    linewidth=2.2,

    linestyle="--",

    alpha=0.60,

    zorder=2
)


# ==============================================================================
# 19. CONTROL BASELINE MARKERS — BOTH HEAD AND BACKBONE LR
#
# IMPORTANT CORRECTION:
#
# Previously baseline markers were plotted ONLY on the Head LR trajectory.
# Now every baseline model is explicitly shown on BOTH LR trajectories.
# ==============================================================================

for i, model_name in enumerate(BASE_MODELS):

    positions = marker_positions[i]


    # --------------------------------------------------------------------------
    # Convert epoch numbers to zero-based array indices
    # --------------------------------------------------------------------------

    indices = positions - 1


    indices = indices[

        (indices >= 0)

        &

        (indices < len(epochs))
    ]


    # --------------------------------------------------------------------------
    # Baseline marker on HEAD LR
    # --------------------------------------------------------------------------

    ax.plot(

        epochs[indices],

        head_lr_curve[indices],

        linestyle="None",

        marker=marker_styles[i],

        markersize=8.0,

        markeredgecolor="black",

        markeredgewidth=0.75,

        markerfacecolor=base_colors[i],

        color=base_colors[i],

        alpha=1.0,

        zorder=15,

        label=f"Control: {model_name}"
    )


    # --------------------------------------------------------------------------
    # Baseline marker on BACKBONE LR
    #
    # No legend label is added here to prevent duplicate legend entries.
    # --------------------------------------------------------------------------

    ax.plot(

        epochs[indices],

        backbone_lr_curve[indices],

        linestyle="None",

        marker=marker_styles[i],

        markersize=8.0,

        markeredgecolor="black",

        markeredgewidth=0.75,

        markerfacecolor=base_colors[i],

        color=base_colors[i],

        alpha=1.0,

        zorder=15
    )


# ==============================================================================
# 20. PROPOSED TSHE — HEAD LR
# ==============================================================================

ax.plot(

    epochs,

    head_lr_curve,

    color="#2E7D32",

    linewidth=4.0,

    linestyle="-",

    zorder=10,

    label="PROPOSED TSHE — Head LR"
)


# ==============================================================================
# 21. PROPOSED TSHE — BACKBONE LR
# ==============================================================================

ax.plot(

    epochs,

    backbone_lr_curve,

    color="#1B5E20",

    linewidth=4.0,

    linestyle="-",

    zorder=10,

    label="PROPOSED TSHE — Backbone LR"
)


# ==============================================================================
# 22. RE-DRAW BASELINE MARKERS AFTER PROPOSED LINES
#
# This is an important visibility correction.
#
# Since the proposed curves are thick green lines, the baseline markers are
# redrawn on top so that they remain clearly visible.
# ==============================================================================

for i, model_name in enumerate(BASE_MODELS):

    positions = marker_positions[i]

    indices = positions - 1

    indices = indices[

        (indices >= 0)

        &

        (indices < len(epochs))
    ]


    # --------------------------------------------------------------------------
    # HEAD LR MARKERS
    # --------------------------------------------------------------------------

    ax.scatter(

        epochs[indices],

        head_lr_curve[indices],

        s=52,

        marker=marker_styles[i],

        facecolor=base_colors[i],

        edgecolor="black",

        linewidth=0.75,

        alpha=1.0,

        zorder=20
    )


    # --------------------------------------------------------------------------
    # BACKBONE LR MARKERS
    # --------------------------------------------------------------------------

    ax.scatter(

        epochs[indices],

        backbone_lr_curve[indices],

        s=52,

        marker=marker_styles[i],

        facecolor=base_colors[i],

        edgecolor="black",

        linewidth=0.75,

        alpha=1.0,

        zorder=20
    )


# ==============================================================================
# 23. SUBTLE REGION BETWEEN HEAD AND BACKBONE LR
# ==============================================================================

ax.fill_between(

    epochs,

    backbone_lr_curve,

    head_lr_curve,

    color="#2E7D32",

    alpha=0.035,

    zorder=1
)


# ==============================================================================
# 24. LOGARITHMIC Y-AXIS
# ==============================================================================

ax.set_yscale(
    "log"
)


# ==============================================================================
# 25. TITLE
# ==============================================================================

ax.set_title(

    "Optimization Dynamics: Learning Rate Evolution Across "
    "Control Baselines and Proposed TSHE Hybrid",

    fontsize=22,

    fontweight="bold",

    pad=28,

    color="#1A2238"
)


# ==============================================================================
# 26. X-AXIS LABEL
# ==============================================================================

ax.set_xlabel(

    "Training Epochs",

    fontsize=15,

    fontweight="bold",

    labelpad=15
)


# ==============================================================================
# 27. Y-AXIS LABEL
# ==============================================================================

ax.set_ylabel(

    "Learning Rate (Log Scale)",

    fontsize=15,

    fontweight="bold",

    labelpad=15
)


# ==============================================================================
# 28. TICK FORMATTING
# ==============================================================================

ax.tick_params(

    axis="both",

    which="major",

    labelsize=12
)


for tick in ax.get_xticklabels():

    tick.set_fontweight(
        "bold"
    )


for tick in ax.get_yticklabels():

    tick.set_fontweight(
        "bold"
    )


# ==============================================================================
# 29. GRID
# ==============================================================================

ax.yaxis.grid(

    True,

    which="major",

    linestyle="--",

    linewidth=0.8,

    alpha=0.35,

    color="#B0BEC5",

    zorder=0
)


ax.yaxis.grid(

    True,

    which="minor",

    linestyle=":",

    linewidth=0.5,

    alpha=0.18,

    color="#B0BEC5",

    zorder=0
)


ax.xaxis.grid(

    True,

    linestyle="--",

    linewidth=0.7,

    alpha=0.22,

    color="#B0BEC5",

    zorder=0
)


# ==============================================================================
# 30. SPINES
# ==============================================================================

for spine in [

    "top",

    "right"
]:

    ax.spines[
        spine
    ].set_visible(False)


ax.spines[
    "left"
].set_color(
    "#78909C"
)


ax.spines[
    "bottom"
].set_color(
    "#78909C"
)


ax.spines[
    "left"
].set_linewidth(
    1.0
)


ax.spines[
    "bottom"
].set_linewidth(
    1.0
)


# ==============================================================================
# 31. X-AXIS RANGE
# ==============================================================================

ax.set_xlim(

    1,

    MAX_EPOCHS
)


# ==============================================================================
# 32. Y-AXIS RANGE
# ==============================================================================

ax.set_ylim(

    ETA_MIN * 0.70,

    HEAD_LR * 2.0
)


# ==============================================================================
# 33. HEAD LR ANNOTATION
# ==============================================================================

ax.annotate(

    "Head LR = 1×10⁻⁴",

    xy=(

        1,

        head_lr_curve[0]
    ),

    xytext=(

        28,

        -42
    ),

    textcoords="offset points",

    fontsize=12,

    fontweight="bold",

    color="#2E7D32",

    bbox=dict(

        boxstyle="round,pad=0.35",

        facecolor="white",

        edgecolor="#2E7D32",

        linewidth=1.0,

        alpha=0.95
    )
)


# ==============================================================================
# 34. BACKBONE LR ANNOTATION
# ==============================================================================

ax.annotate(

    "Backbone LR = 1×10⁻⁶",

    xy=(

        1,

        backbone_lr_curve[0]
    ),

    xytext=(

        28,

        24
    ),

    textcoords="offset points",

    fontsize=12,

    fontweight="bold",

    color="#1B5E20",

    bbox=dict(

        boxstyle="round,pad=0.35",

        facecolor="white",

        edgecolor="#1B5E20",

        linewidth=1.0,

        alpha=0.95
    )
)


# ==============================================================================
# 35. SCHEDULER INFORMATION BOX
# ==============================================================================

scheduler_text = (

    "Optimizer: AdamW\n"
    "Scheduler: CosineAnnealingWarmRestarts\n"
    f"T₀ = {T0}, Tmult = {T_MULT}\n"
    f"ηmin = {ETA_MIN:.0e}\n"
    "Common optimization protocol"
)


ax.text(

    0.985,

    0.035,

    scheduler_text,

    transform=ax.transAxes,

    ha="right",

    va="bottom",

    fontsize=11,

    fontweight="bold",

    color="#37474F",

    bbox=dict(

        boxstyle="round,pad=0.55",

        facecolor="#FAFAFA",

        edgecolor="#CFD8DC",

        linewidth=1.0,

        alpha=0.96
    )
)


# ==============================================================================
# 36. LEGEND
# ==============================================================================

legend = ax.legend(

    loc="upper left",

    bbox_to_anchor=(1.02, 1.00),

    fontsize=10.3,

    frameon=True,

    shadow=False,

    edgecolor="#CFD8DC",

    facecolor="#FAFAFA",

    title="Models / Learning-Rate Policies",

    title_fontsize=12
)


legend.get_title().set_fontweight(
    "bold"
)


# ==============================================================================
# 37. FINAL LAYOUT
# ==============================================================================

plt.tight_layout(

    rect=[

        0.0,

        0.0,

        0.82,

        1.0
    ]
)


# ==============================================================================
# 38. OUTPUT FILE PATHS
# ==============================================================================

lr_save_png = (

    PROJECT_ROOT /

    "Master_LR_Convergence_Analysis.png"
)


lr_save_pdf = (

    PROJECT_ROOT /

    "Master_LR_Convergence_Analysis.pdf"
)


lr_save_csv = (

    PROJECT_ROOT /

    "Master_LR_Convergence_Summary.csv"
)


# ==============================================================================
# 39. SUMMARY DATAFRAME
# ==============================================================================

lr_summary_df = pd.DataFrame({

    "Epoch":

        epochs,

    "Control_Baselines_Shared_Head_LR":

        head_lr_curve,

    "Control_Baselines_Shared_Backbone_LR":

        backbone_lr_curve,

    "Proposed_TSHE_Head_LR":

        head_lr_curve,

    "Proposed_TSHE_Backbone_LR":

        backbone_lr_curve
})


# ==============================================================================
# 40. SAVE CSV
# ==============================================================================

lr_summary_df.to_csv(

    lr_save_csv,

    index=False
)


# ==============================================================================
# 41. SAVE HIGH-RESOLUTION PNG
# ==============================================================================

plt.savefig(

    lr_save_png,

    dpi=600,

    bbox_inches="tight",

    facecolor="white",

    edgecolor="none"
)


# ==============================================================================
# 42. SAVE VECTOR PDF
# ==============================================================================

plt.savefig(

    lr_save_pdf,

    format="pdf",

    bbox_inches="tight",

    facecolor="white",

    edgecolor="none"
)


# ==============================================================================
# 43. FINAL SYSTEM REPORT
# ==============================================================================

print("\n" + "=" * 100)

print(
    "[SUCCESS] MASTER LEARNING RATE ANALYSIS COMPLETED"
)

print("=" * 100)

print(
    f"[CONTROL BASELINES] "
    f"{len(BASE_MODELS)} architectures"
)


for model_name in BASE_MODELS:

    print(
        f"    └── {model_name}"
    )


print(
    "[PROPOSED MODEL] "
    "TSHE Hybrid"
)


print(
    f"[BACKBONE LR] "
    f"{BACKBONE_LR:.1e}"
)


print(
    f"[HEAD LR] "
    f"{HEAD_LR:.1e}"
)


print(
    "[OPTIMIZER] "
    "AdamW"
)


print(
    "[SCHEDULER] "
    "CosineAnnealingWarmRestarts"
)


print(
    f"[T0] "
    f"{T0}"
)


print(
    f"[Tmult] "
    f"{T_MULT}"
)


print(
    f"[eta_min] "
    f"{ETA_MIN:.1e}"
)


print(
    f"[EPOCHS] "
    f"{MAX_EPOCHS}"
)


print("\n[OUTPUT FILES]")

print(
    f"[PNG] {lr_save_png}"
)

print(
    f"[PDF] {lr_save_pdf}"
)

print(
    f"[CSV] {lr_save_csv}"
)

print("=" * 100)


# ==============================================================================
# 44. DISPLAY FIGURE
# ==============================================================================

plt.show()