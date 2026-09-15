# ==============================================================================
# PHASE 13: GLOBAL MISCLASSIFICATION ANALYSIS - ERROR REDUCTION VISUALIZATION
# Strategy: Dynamic File Resolution | High-Resolution Research Graphics (600 DPI)
# Focus: Controlled Baselines vs. Proposed Hybrid TripleStream
# ==============================================================================

import os
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

print("\n" + "=" * 80)
print(f"{'[SYSTEM] INITIATING GLOBAL ERROR ANALYSIS PIPELINE':^80}")
print("=" * 80)

# 1. SETUP & PATH CONFIGURATION
# ------------------------------------------------------------------------------
LOCAL_PROJECT_ROOT = pathlib.Path(r"E:\Research Project\BDMediHerb\BDMediHerb_Reviewer_Revision")

if LOCAL_PROJECT_ROOT.exists():
    PROJECT_ROOT = LOCAL_PROJECT_ROOT
    print("[ENV] Local Windows PC detected")
elif os.path.exists("/kaggle/working/BDMediHerb_Reviewer_Revision"):
    PROJECT_ROOT = pathlib.Path("/kaggle/working/BDMediHerb_Reviewer_Revision")
    print("[ENV] Kaggle detected")
else:
    PROJECT_ROOT = pathlib.Path("/content/drive/MyDrive/BDMediHerb/BDMediHerb_Reviewer_Revision")
    print("[ENV] Google Colab detected")

error_viz_dir = PROJECT_ROOT / "Global_Error_Analysis"
error_viz_dir.mkdir(parents=True, exist_ok=True)

# Updated for Controlled Baselines
BASE_MODELS = [
    "MobileNet_V3_Large", "EfficientNetV2_S", "ResNet50", 
    "DenseNet121", "ViT_B16", "Inception_V3", "ConvNeXt_Tiny"
]
MODELS_TO_ANALYZE = BASE_MODELS + ["Hybrid_TripleStream"]

error_summary_data = []

# 2. DYNAMIC DATA AGGREGATION LOGIC (CONTROLLED BASELINES target)
# ------------------------------------------------------------------------------
def get_model_error_count(name):
    display_name = "Proposed Hybrid (TSHE)" if name == "Hybrid_TripleStream" else name
    is_hybrid = "Hybrid" in name
    
    # Priority given specifically to Controlled_Baselines for base models
    possible_folders = [
        PROJECT_ROOT / "Controlled_Baselines" / name,
        PROJECT_ROOT / "Hybrid_TripleStream" if is_hybrid else None,
        PROJECT_ROOT / name
    ]
    
    for folder in possible_folders:
        if not folder or not folder.exists():
            continue
            
        # Check all possible misclassification CSV filenames
        csv_files = [
            folder / "misclassifications.csv",
            folder / "detailed_misclassifications.csv",
            folder / "Failure_Analysis" / "misclassifications.csv"
        ]
        
        for cfile in csv_files:
            if cfile.exists():
                try:
                    df = pd.read_csv(cfile)
                    error_cnt = len(df)
                    return display_name, error_cnt, is_hybrid
                except Exception:
                    pass
                    
        # Fallback via accuracy in result.json if CSV missing (Test set = 570 samples)
        json_file = folder / "result.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                acc = data.get('test_accuracy')
                if acc is not None:
                    total_test = data.get('test_images', 570)
                    error_cnt = int(round(total_test * (1.0 - float(acc))))
                    return display_name, error_cnt, is_hybrid
            except Exception:
                pass

    return display_name, None, is_hybrid

for name in MODELS_TO_ANALYZE:
    d_name, err_count, is_hyb = get_model_error_count(name)
    if err_count is not None:
        error_summary_data.append({
            'Model': d_name,
            'Total Errors': err_count,
            'IsHybrid': is_hyb
        })
        print(f"[FOUND] {d_name:<28} | Misclassifications: {err_count}")
    else:
        print(f"[WARN] No error logs found for {name} in Controlled Baselines")

# Sort by Total Errors descending (Worst to Best)
df_err_plot = pd.DataFrame(error_summary_data).sort_values('Total Errors', ascending=False).reset_index(drop=True)

# 3. HIGH-RESOLUTION VISUALIZATION
# ------------------------------------------------------------------------------
if not df_err_plot.empty:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })

    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white', dpi=600)

    base_color = '#546E7A'      # Slate Grey
    proposed_color = '#27AE60'  # Emerald Green

    colors = [proposed_color if row['IsHybrid'] else base_color for _, row in df_err_plot.iterrows()]

    bars = ax.bar(
        df_err_plot['Model'], 
        df_err_plot['Total Errors'],
        color=colors, 
        edgecolor='#1A2238', 
        linewidth=1.2, 
        width=0.58, 
        alpha=0.92, 
        zorder=3
    )

    # Titles & Labels
    ax.set_title(
        "Quantitative Error Reduction: Proposed Hybrid vs. Controlled Baselines",
        fontsize=22, pad=30, fontweight='black', color='#1A2238'
    )
    ax.set_ylabel("Number of Misclassified Samples (Test Set)", fontsize=14, labelpad=15, fontweight='bold')
    ax.set_xlabel("Controlled Baseline Architectures", fontsize=14, labelpad=15, fontweight='bold')

    plt.xticks(rotation=20, ha='right', fontsize=11, fontweight='bold', color='#263238')
    plt.yticks(fontsize=11, fontweight='bold', color='#263238')

    # Headroom on Y-axis so text annotations don't touch top border
    max_err = df_err_plot['Total Errors'].max()
    ax.set_ylim(0, max_err * 1.18)

    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#90A4AE', zorder=0)
    for spine in ['top', 'right']: 
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#78909C')
    ax.spines['bottom'].set_color('#78909C')

    # 4. VALUE ANNOTATIONS
    # ------------------------------------------------------------------------------
    for bar, (_, row) in zip(bars, df_err_plot.iterrows()):
        height = bar.get_height()
        txt_color = proposed_color if row['IsHybrid'] else '#1A2238'

        ax.annotate(
            f'{int(height)}',
            xy=(bar.get_x() + bar.get_width() / 2., height),
            xytext=(0, 6), 
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=13, fontweight='black', 
            color=txt_color
        )

    # Legend
    legend_elements = [
        Patch(facecolor=base_color, edgecolor='#1A2238', label='Controlled Baselines'),
        Patch(facecolor=proposed_color, edgecolor='#1A2238', label='Proposed Hybrid (Ours)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=13, frameon=True, shadow=False, facecolor='#FAFAFA', edgecolor='#CFD8DC')

    plt.tight_layout()

    # Save PNG + PDF Vector
    final_save_png = error_viz_dir / "Controlled_Baseline_Error_Reduction_Benchmark.png"
    final_save_pdf = error_viz_dir / "Controlled_Baseline_Error_Reduction_Benchmark.pdf"

    plt.savefig(final_save_png, dpi=600, bbox_inches='tight')
    plt.savefig(final_save_pdf, format='pdf', bbox_inches='tight')

    print(f"\n[SUCCESS] Global Error Comparison Chart saved at: {final_save_png}")
    print(f"[SUCCESS] Vector PDF saved at: {final_save_pdf}")
    plt.show()

else:
    print("[ERROR] No error log files found across Controlled_Baselines directories.")