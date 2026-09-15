# ==============================================================================
# PHASE 10: COMPARATIVE PERFORMANCE ANALYSIS VISUALIZATION
# CONTROLLED BASELINES vs PROPOSED TSHE HYBRID
#
# Source: 02_controlled_individual_baselines.py & 03_hybrid_triple_stream.py
#
# IMPORTANT:
#   - Only CONTROLLED BASELINES are included.
#   - Proposed Hybrid (TSHE) is included.
#   - All plotting, colors, fonts, annotations, sorting, Y-ticks,
#     and export settings remain the same as the original code.
# ==============================================================================

import os
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


print("\n" + "=" * 80)
print("[INFO] Generating Clean High-Resolution Comparative Performance Chart...")
print("=" * 80)


# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================

LOCAL_PROJECT_ROOT = pathlib.Path(
    r"E:\Research Project\BDMediHerb\BDMediHerb_Reviewer_Revision"
)


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


print(f"[PATH] Project Root: {PROJECT_ROOT}")


# ==============================================================================
# 2. CONTROLLED BASELINE MODELS
# ==============================================================================

BASE_MODELS = [

    'Inception_V3',

    'ConvNeXt_Tiny',

    'DenseNet121',

    'ViT_B16',

    'ResNet50',

    'EfficientNetV2_S',

    'MobileNet_V3_Large'

]


comparison_metrics = []


# ==============================================================================
# 3. FIND CONTROLLED BASELINE MODEL FOLDER
#
# IMPORTANT CORRECTION:
# Only Controlled_Baselines is searched.
#
# This prevents Individual_Models or other uncontrolled results from
# accidentally entering the comparative analysis.
# ==============================================================================

def find_model_folder(model_name):

    controlled_path = (

        PROJECT_ROOT
        / "Controlled_Baselines"
        / model_name
    )


    if controlled_path.exists() and (

        (controlled_path / "result.json").exists()

        or

        (controlled_path / "classification_report.csv").exists()

        or

        (controlled_path / "final_metrics_report.csv").exists()

    ):

        return controlled_path


    return None


# ==============================================================================
# 4. FIND PROPOSED HYBRID MODEL FOLDER
# ==============================================================================

def find_hybrid_folder():

    possible_paths = [

        PROJECT_ROOT / "Hybrid_TripleStream",

        PROJECT_ROOT / "Hybrid_Model",

        PROJECT_ROOT / "TSHE_Hybrid"

    ]


    for path in possible_paths:

        if path.exists() and (

            (path / "result.json").exists()

            or

            (path / "hybrid_final_metrics_report.csv").exists()

            or

            (path / "classification_report.csv").exists()

        ):

            return path


    return None


# ==============================================================================
# 5. EXTRACT MODEL PERFORMANCE METRICS
# ==============================================================================

def extract_model_metrics(folder_path, display_name):

    if not folder_path:

        return None


    # --------------------------------------------------------------------------
    # JSON RESULT FILE
    # --------------------------------------------------------------------------

    json_path = folder_path / 'result.json'


    # --------------------------------------------------------------------------
    # POSSIBLE CSV REPORTS
    # --------------------------------------------------------------------------

    csv_paths = [

        folder_path / 'hybrid_final_metrics_report.csv',

        folder_path / 'classification_report.csv',

        folder_path / 'final_metrics_report.csv'

    ]


    # ==========================================================================
    # 5A. EXTRACT FROM result.json
    # ==========================================================================

    if json_path.exists():

        try:

            with open(
                json_path,
                'r'
            ) as f:

                data = json.load(f)


            acc = data.get(
                'test_accuracy'
            )


            prec = data.get(

                'weighted_precision',

                data.get(
                    'macro_precision'
                )

            )


            rec = data.get(

                'weighted_recall',

                data.get(
                    'macro_recall'
                )

            )


            f1 = data.get(

                'weighted_f1',

                data.get(
                    'macro_f1'
                )

            )


            if all(

                v is not None

                for v in [
                    acc,
                    prec,
                    rec,
                    f1
                ]

            ):

                return {

                    'Model':
                        display_name,

                    'Accuracy':
                        float(acc),

                    'Precision':
                        float(prec),

                    'Recall':
                        float(rec),

                    'F1-Score':
                        float(f1)

                }


        except Exception as e:

            print(
                f"[WARN] JSON read failed for "
                f"{display_name}: {e}"
            )


    # ==========================================================================
    # 5B. EXTRACT FROM CSV REPORT
    # ==========================================================================

    for cpath in csv_paths:

        if cpath.exists():

            try:

                df_rep = pd.read_csv(
                    cpath,
                    index_col=0
                )


                # ------------------------------------------------------------------
                # Accuracy
                # ------------------------------------------------------------------

                if 'accuracy' in df_rep.index:

                    if 'f1-score' in df_rep.columns:

                        acc = df_rep.loc[
                            'accuracy',
                            'f1-score'
                        ]

                    else:

                        acc = df_rep.loc[
                            'accuracy',
                            'precision'
                        ]

                else:

                    continue


                # ------------------------------------------------------------------
                # Precision
                # ------------------------------------------------------------------

                if 'weighted avg' in df_rep.index:

                    prec = df_rep.loc[
                        'weighted avg',
                        'precision'
                    ]

                elif 'macro avg' in df_rep.index:

                    prec = df_rep.loc[
                        'macro avg',
                        'precision'
                    ]

                else:

                    continue


                # ------------------------------------------------------------------
                # Recall
                # ------------------------------------------------------------------

                if 'weighted avg' in df_rep.index:

                    rec = df_rep.loc[
                        'weighted avg',
                        'recall'
                    ]

                elif 'macro avg' in df_rep.index:

                    rec = df_rep.loc[
                        'macro avg',
                        'recall'
                    ]

                else:

                    continue


                # ------------------------------------------------------------------
                # F1-Score
                # ------------------------------------------------------------------

                if 'weighted avg' in df_rep.index:

                    f1 = df_rep.loc[
                        'weighted avg',
                        'f1-score'
                    ]

                elif 'macro avg' in df_rep.index:

                    f1 = df_rep.loc[
                        'macro avg',
                        'f1-score'
                    ]

                else:

                    continue


                return {

                    'Model':
                        display_name,

                    'Accuracy':
                        float(acc),

                    'Precision':
                        float(prec),

                    'Recall':
                        float(rec),

                    'F1-Score':
                        float(f1)

                }


            except Exception as e:

                print(
                    f"[WARN] CSV read failed for "
                    f"{display_name}: {e}"
                )

            continue


    return None


# ==============================================================================
# 6. LOAD ONLY CONTROLLED BASELINE RESULTS
# ==============================================================================

print("\n" + "-" * 80)
print("[INFO] Loading Controlled Baseline Performance Metrics")
print("-" * 80)


for name in BASE_MODELS:

    folder = find_model_folder(name)


    if folder is None:

        print(
            f"[SKIP] Controlled baseline not found: "
            f"{name}"
        )

        continue


    print(
        f"[FOUND] Controlled Baseline: "
        f"{name}"
    )

    print(
        f"        Path: {folder}"
    )


    res = extract_model_metrics(
        folder,
        name
    )


    if res:

        comparison_metrics.append(
            res
        )

        print(
            f"        Accuracy : "
            f"{res['Accuracy']:.4f}"
        )

        print(
            f"        Precision: "
            f"{res['Precision']:.4f}"
        )

        print(
            f"        Recall   : "
            f"{res['Recall']:.4f}"
        )

        print(
            f"        F1-Score : "
            f"{res['F1-Score']:.4f}"
        )

    else:

        print(
            f"[WARN] Metrics could not be extracted "
            f"for {name}"
        )


# ==============================================================================
# 7. LOAD PROPOSED HYBRID MODEL
# ==============================================================================

print("\n" + "-" * 80)
print("[INFO] Loading Proposed TSHE Hybrid Performance Metrics")
print("-" * 80)


hybrid_folder = find_hybrid_folder()


if hybrid_folder:

    print(
        f"[FOUND] Proposed Hybrid Folder: "
        f"{hybrid_folder}"
    )


    hybrid_res = extract_model_metrics(

        hybrid_folder,

        "Proposed Hybrid (TSHE)"

    )


    if hybrid_res:

        comparison_metrics.append(
            hybrid_res
        )


        print(
            f"        Accuracy : "
            f"{hybrid_res['Accuracy']:.4f}"
        )

        print(
            f"        Precision: "
            f"{hybrid_res['Precision']:.4f}"
        )

        print(
            f"        Recall   : "
            f"{hybrid_res['Recall']:.4f}"
        )

        print(
            f"        F1-Score : "
            f"{hybrid_res['F1-Score']:.4f}"
        )


    else:

        print(
            "[WARN] Proposed Hybrid metrics "
            "could not be extracted."
        )


else:

    print(
        "[WARNING] Proposed Hybrid folder "
        "not found."
    )


# ==============================================================================
# 8. PLOTTING
# ==============================================================================

if comparison_metrics:

    df_metrics = pd.DataFrame(
        comparison_metrics
    )


    # ==========================================================================
    # SORT MODELS BY F1-SCORE
    #
    # Low → High
    #
    # This keeps the exact logic from your original code.
    # ==========================================================================

    df_metrics = (

        df_metrics

        .sort_values(
            by='F1-Score',
            ascending=True
        )

        .reset_index(
            drop=True
        )

    )


    # ==========================================================================
    # PRINT FINAL ORDER
    # ==========================================================================

    print("\n" + "=" * 80)
    print("[INFO] FINAL MODEL ORDER — LOW F1 → HIGH F1")
    print("=" * 80)


    for i, row in df_metrics.iterrows():

        print(

            f"{i + 1:02d}. "
            f"{row['Model']:<30} "
            f"F1 = {row['F1-Score']:.4f}"

        )


    # ==========================================================================
    # PROFESSIONAL FONT
    # ==========================================================================

    plt.rcParams['font.family'] = 'serif'


    # ==========================================================================
    # ORIGINAL COLOR PALETTE
    # ==========================================================================

    modern_colors = [

        '#1A237E',

        '#0288D1',

        '#00897B',

        '#D81B60'

    ]


    # ==========================================================================
    # HIGH-RESOLUTION FIGURE
    # ==========================================================================

    fig, ax = plt.subplots(

        figsize=(20, 11),

        facecolor='white',

        dpi=600

    )


    # ==========================================================================
    # BAR CHART
    # ==========================================================================

    bars_plot = df_metrics.plot(

        x='Model',

        kind='bar',

        y=[

            'Accuracy',

            'Precision',

            'Recall',

            'F1-Score'

        ],

        width=0.82,

        color=modern_colors,

        ax=ax,

        edgecolor='white',

        linewidth=0.6,

        zorder=3

    )


    # ==========================================================================
    # EXACT Y-TICKS CONFIGURATION
    # ==========================================================================

    ax.set_ylim(
        0.89,
        1.01
    )


    explicit_ticks = [

        0.90,

        0.92,

        0.94,

        0.96,

        0.98,

        1.00

    ]


    ax.set_yticks(
        explicit_ticks
    )


    ax.set_yticklabels(

        [
            f"{t:.2f}"
            for t in explicit_ticks
        ],

        fontsize=14,

        fontweight='bold'

    )


    # ==========================================================================
    # GRID
    # ==========================================================================

    ax.yaxis.grid(

        True,

        linestyle='--',

        color='#ECEFF1',

        alpha=0.8,

        zorder=0

    )


    ax.set_axisbelow(
        True
    )


    # ==========================================================================
    # SPINES
    # ==========================================================================

    for spine in [

        'top',

        'right'

    ]:

        ax.spines[
            spine
        ].set_visible(
            False
        )


    ax.spines[
        'left'
    ].set_color(
        '#78909C'
    )


    ax.spines[
        'bottom'
    ].set_color(
        '#78909C'
    )


    # ==========================================================================
    # TITLE
    # ==========================================================================

    ax.set_title(

        "Comprehensive Performance Benchmark: "
        "BDMediHerb Classification",

        fontsize=22,

        fontweight='bold',

        pad=30,

        color='#1A237E'

    )


    # ==========================================================================
    # X-AXIS LABEL
    # ==========================================================================

    ax.set_xlabel(

        "Model Architectures "
        "(Controlled Baselines vs. Proposed Hybrid)",

        fontsize=14,

        labelpad=25,

        fontweight='bold'

    )


    # ==========================================================================
    # Y-AXIS LABEL
    # ==========================================================================

    ax.set_ylabel(

        "Metric Scores "
        "(Normalized Scale 0.0 - 1.0)",

        fontsize=14,

        labelpad=12,

        fontweight='bold'

    )


    # ==========================================================================
    # X-TICKS
    # ==========================================================================

    plt.xticks(

        rotation=15,

        ha='right',

        fontsize=14,

        fontweight='bold',

        color='#263238'

    )


    # ==========================================================================
    # LEGEND
    # ==========================================================================

    ax.legend(

        loc='upper center',

        bbox_to_anchor=(
            0.5,
            -0.18
        ),

        ncol=4,

        frameon=True,

        fontsize=14,

        shadow=False,

        edgecolor='#CFD8DC',

        facecolor='#FAFAFA'

    )


    # ==========================================================================
    # BAR VALUE ANNOTATIONS
    # ==========================================================================

    for p in ax.patches:

        val = p.get_height()


        if val > 0:

            ax.annotate(

                f'{val:.4f}',

                (

                    p.get_x()
                    +
                    p.get_width() / 2.,

                    val

                ),

                ha='center',

                va='bottom',

                xytext=(

                    0,

                    5

                ),

                textcoords='offset points',

                fontsize=14,

                fontweight='bold',

                rotation=90,

                color='#263238'

            )


    # ==========================================================================
    # FINAL LAYOUT
    # ==========================================================================

    plt.tight_layout()


    # ==============================================================================
    # 9. EXPORT PATHS
    # ==============================================================================

    save_png_path = (

        PROJECT_ROOT /

        'Global_Research_Comparison_Viz.png'

    )


    save_pdf_path = (

        PROJECT_ROOT /

        'Global_Research_Comparison_Viz.pdf'

    )


    # ==============================================================================
    # 10. SAVE HIGH-RESOLUTION PNG
    # ==============================================================================

    plt.savefig(

        save_png_path,

        dpi=600,

        bbox_inches='tight'

    )


    # ==============================================================================
    # 11. SAVE VECTOR PDF
    # ==============================================================================

    plt.savefig(

        save_pdf_path,

        format='pdf',

        bbox_inches='tight'

    )


    # ==============================================================================
    # 12. SUCCESS REPORT
    # ==============================================================================

    print("\n" + "=" * 80)

    print(
        "[SUCCESS] Comparative Performance Chart Generated"
    )

    print("=" * 80)


    print(
        f"[MODELS INCLUDED] "
        f"{len(df_metrics)}"
    )


    print(
        "[CONTROL GROUP] "
        "Controlled_Baselines only"
    )


    print(
        "[PROPOSED GROUP] "
        "Proposed Hybrid (TSHE)"
    )


    print(
        f"[PNG] {save_png_path}"
    )


    print(
        f"[PDF] {save_pdf_path}"
    )


    print("=" * 80)


    # ==========================================================================
    # DISPLAY
    # ==========================================================================

    plt.show()


else:

    print(
        "[ERROR] No metric files found."
    )

    print(
        "[ERROR] Please verify the Controlled_Baselines "
        "and Hybrid_TripleStream directories."
    )