"""Generates visualization of the full PFS dataset classification results (~200,241 targets)."""
import os
import sys
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR

def plot_full_summary():
    db_path = RESULTS_DIR / "pfs_classifications_216k.sqlite3"
    conn = sqlite3.connect(db_path)
    
    # Query data for visualization
    import pandas as pd
    df = pd.read_sql_query("""
    SELECT objId, db_class, db_redshift, pred_class, cluster_id, cluster_name,
           est_redshift, est_redshift_quality, anomaly_score, arm_mode
    FROM classifications;
    """, conn)
    conn.close()

    print(f"Loaded {len(df)} records for plotting.")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Broad Class Distribution (Bar plot)
    ax1 = axes[0, 0]
    classes = ['GALAXY', 'QSO', 'STAR']
    pred_counts = df['pred_class'].value_counts()
    db_counts = df['db_class'].value_counts()
    
    x = np.arange(len(classes))
    width = 0.35
    ax1.bar(x - width/2, [pred_counts.get(c, 0) for c in classes], width, label='SimCLR Prediction', color='#1f77b4', alpha=0.85)
    ax1.bar(x + width/2, [db_counts.get(c, 0) for c in classes], width, label='DB Label', color='#aec7e8', alpha=0.7, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Targets', fontsize=11)
    ax1.set_title('[1] Macro Class Distribution (Full 200,241 Targets)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4, axis='y')
    for i, c in enumerate(classes):
        cnt = pred_counts.get(c, 0)
        ax1.text(i - width/2, cnt + 2000, f"{cnt:,}\n({cnt/len(df)*100:.1f}%)", ha='center', fontsize=9)

    # Panel 2: Astrophysical Subtype Clusters (K=6)
    ax2 = axes[0, 1]
    cluster_counts = df['cluster_id'].value_counts().sort_index()
    c_names = [f"C{k}: {df[df['cluster_id'] == k]['cluster_name'].iloc[0][:22]}..." for k in range(6)]
    colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a', '#d62728', '#ff9896']
    bars = ax2.barh(range(6), [cluster_counts.get(k, 0) for k in range(6)], color=colors, alpha=0.85, edgecolor='black', lw=0.8)
    ax2.set_yticks(range(6))
    ax2.set_yticklabels(c_names, fontsize=9.5)
    ax2.set_xlabel('Target Count', fontsize=11)
    ax2.set_title('[2] Discovered Subtype Clusters (K=6 GMM)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4, axis='x')
    for i, bar in enumerate(bars):
        w = bar.get_width()
        ax2.text(w + 1000, bar.get_y() + bar.get_height()/2, f"{w:,} ({w/len(df)*100:.1f}%)", va='center', fontsize=8.5)

    # Panel 3: Redshift Estimation vs DB Redshift (Density Hexbin)
    ax3 = axes[1, 0]
    valid_z = df['db_redshift'].notnull() & np.isfinite(df['db_redshift']) & (df['db_redshift'] > 0) & (df['db_redshift'] < 3.5)
    df_z = df[valid_z]
    hb = ax3.hexbin(df_z['db_redshift'], df_z['est_redshift'], gridsize=60, cmap='viridis', mincnt=1, bins='log')
    ax3.plot([0, 3.5], [0, 3.5], 'r--', lw=1.5, label='1:1 Line')
    cb = plt.colorbar(hb, ax=ax3)
    cb.set_label('Log10(Count)', fontsize=10)
    ax3.set_xlabel('Pipeline DB Redshift (z_DB)', fontsize=11)
    ax3.set_ylabel('Template Matched Redshift (z_est)', fontsize=11)
    ax3.set_title(f'[3] Redshift Agreement Density (N={len(df_z):,})', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 3.5)
    ax3.set_ylim(0, 3.5)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.4)

    # Panel 4: Anomaly Score Distribution
    ax4 = axes[1, 1]
    anom = df['anomaly_score']
    ax4.hist(anom, bins=80, color='#9467bd', alpha=0.75, edgecolor='black', lw=0.5)
    p99 = np.percentile(anom, 99)
    ax4.axvline(p99, color='red', linestyle='--', label=f'99th Percentile Threshold ({p99:.1f})')
    ax4.set_xlabel('Anomaly Score (NLL / Distance)', fontsize=11)
    ax4.set_ylabel('Target Count', fontsize=11)
    ax4.set_title(f'[4] Anomaly Score Distribution (Rare Spectra & Pipelines Errors)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    out_fig = RESULTS_DIR / "full_dataset_summary_200k.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"Saved full summary figure to {out_fig}.")

if __name__ == "__main__":
    plot_full_summary()
