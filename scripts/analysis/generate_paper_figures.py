#!/usr/bin/env python3
"""Generate publication-quality figures for the paper."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})

output_dir = 'paper_draft/images'
os.makedirs(output_dir, exist_ok=True)


# --- Figure: Model Count Ablation ---
def plot_ablation():
    df = pd.read_csv('results/model_count_ablation.csv')
    test_df = df[df['subset'] == 'test']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for n in [1, 2, 3, 4]:
        sub = test_df[test_df['n_models'] == n]
        sens_vals = sub['sensitivity'].values
        prec_vals = sub['precision'].values

        # Jitter for visibility
        jitter = np.random.RandomState(42).uniform(-0.08, 0.08, len(sens_vals))

        ax1.scatter(np.full(len(sens_vals), n) + jitter, sens_vals * 100,
                   alpha=0.5, s=30, color='C0', zorder=3)
        ax1.errorbar(n, sens_vals.mean() * 100, yerr=sens_vals.std() * 100,
                    fmt='o', color='C0', markersize=8, capsize=5, capthick=2,
                    linewidth=2, zorder=4)

        ax2.scatter(np.full(len(prec_vals), n) + jitter, prec_vals * 100,
                   alpha=0.5, s=30, color='C1', zorder=3)
        ax2.errorbar(n, prec_vals.mean() * 100, yerr=prec_vals.std() * 100,
                    fmt='o', color='C1', markersize=8, capsize=5, capthick=2,
                    linewidth=2, zorder=4)

    for ax, ylabel in [(ax1, 'Sensitivity (%)'), (ax2, 'Precision (%)')]:
        ax.set_xlabel('Number of models in consensus')
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_ylim([70, 102])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color='gray', linestyle=':', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ablation.png'), bbox_inches='tight')
    plt.close(fig)
    print("Saved ablation.png")


# --- Figure: Performance Comparison Bar Chart ---
def plot_performance_comparison():
    """Compare old LOOCV T1 vs new production consensus."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    methods = [
        'Single T1\nmodel\n(LOOCV, n=26)',
        'Single T1\nmodel\n(production, n=25)',
        '4-model\nconsensus\n(production, n=25)',
    ]
    # Old LOOCV T1 performance from the preprint
    # New: single model mean from ablation, consensus from eval (with confidence floor)
    sensitivities = [79, 84.2, 97.3]
    precisions = [67, 92.0, 96.0]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, sensitivities, width, label='Sensitivity',
                   color='#4C72B0', edgecolor='white')
    bars2 = ax.bar(x + width/2, precisions, width, label='Precision',
                   color='#DD8452', edgecolor='white')

    ax.set_ylabel('Performance (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 108])
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'performance_comparison.png'), bbox_inches='tight')
    plt.close(fig)
    print("Saved performance_comparison.png")


# --- Figure: Per-subject results ---
def plot_per_subject():
    df = pd.read_csv('results/production_all_eval.csv')

    fig, ax = plt.subplots(figsize=(12, 3.5))

    subjects = df['subject_id'].values
    n = len(subjects)
    x = np.arange(n)

    # Color by perfect vs imperfect
    colors_sens = ['#4C72B0' if s == 1.0 else '#C44E52' for s in df['sensitivity']]
    colors_prec = ['#DD8452' if p == 1.0 else '#C44E52' for p in df['precision']]

    width = 0.35
    ax.bar(x - width/2, df['sensitivity'] * 100, width, color=colors_sens, alpha=0.8, label='Sensitivity')
    ax.bar(x + width/2, df['precision'] * 100, width, color=colors_prec, alpha=0.8, label='Precision')

    ax.set_ylabel('Performance (%)')
    ax.set_xlabel('Subject')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('z', '') for s in subjects], rotation=90, fontsize=6)
    ax.set_ylim([0, 110])
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
    ax.legend(loc='lower left')
    ax.grid(True, axis='y', alpha=0.3)

    # Mark val vs test
    val_subjects = ['S001', 'S014', 'S022', 'S025', 'S032',
                    'S047', 'S057', 'S059', 'S120']
    for i, s in enumerate(subjects):
        if s in val_subjects:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='green')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'per_subject.png'), bbox_inches='tight')
    plt.close(fig)
    print("Saved per_subject.png")


# --- Figure: Consensus Strategy Comparison ---
def plot_strategy_comparison():
    df = pd.read_csv('results/threshold_experiments.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Threshold sweep (strategy B)
    b_test = df[(df['strategy'].str.startswith('B_')) & (df['subset'] == 'test')]
    thresholds = b_test['det_threshold'].values
    ax1.plot(thresholds, b_test['sensitivity'] * 100, 'o-', color='C0', label='Sensitivity')
    ax1.plot(thresholds, b_test['precision'] * 100, 's-', color='C1', label='Precision')
    ax1.set_xlabel('Detection threshold')
    ax1.set_ylabel('Performance (%)')
    ax1.set_title('(a) Detection threshold sweep')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([85, 102])

    # Strategy comparison
    strategies = {
        'Baseline\n(top-3, t=0.1)': ('A_baseline_top3', 'test'),
        'Top-3\nt=0.05': ('B_top3_det0.05', 'test'),
        'Hybrid\n(top-3, floor=0.2)': ('D_hybrid_cf0.2', 'test'),
        'Adaptive\n(floor=0.2)': ('C_adaptive_cf0.2', 'test'),
    }
    x = np.arange(len(strategies))
    sens_vals = []
    prec_vals = []
    for label, (strat, subset) in strategies.items():
        row = df[(df['strategy'] == strat) & (df['subset'] == subset)]
        sens_vals.append(row['sensitivity'].values[0] * 100)
        prec_vals.append(row['precision'].values[0] * 100)

    width = 0.35
    ax2.bar(x - width/2, sens_vals, width, label='Sensitivity', color='#4C72B0')
    ax2.bar(x + width/2, prec_vals, width, label='Precision', color='#DD8452')
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(strategies.keys()), fontsize=9)
    ax2.set_ylabel('Performance (%)')
    ax2.set_title('(b) Strategy comparison (test set)')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim([70, 105])

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'strategies.png'), bbox_inches='tight')
    plt.close(fig)
    print("Saved strategies.png")


if __name__ == '__main__':
    plot_ablation()
    plot_performance_comparison()
    print("All paper figures generated.")
