"""
Plot ACC vs lead day (0-15) for each variable and region.
Overlays multiple models. Marks the 0.6 skill threshold.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

PLOT_DIR    = './plots_acc/'
THRESHOLD   = 0.6
MAX_LEAD    = 45
PLOT_MAX_LEAD = 15

MODELS = {
    'UC-S2S':    './results_ours_si_s2s/acc_summary.pkl',
    'UC-S2S v2': './results_ours_si_postprocess_v2/acc_summary.pkl',
    'SI-S2S':    './results_amip_s2s_train/acc_summary.pkl',
    'GenCast':   './results_gencast/acc_summary.pkl',
    'NGCM':      './results_ngcm/acc_summary.pkl',
}

MODEL_STYLES = {
    'UC-S2S':    {'color': '#1f77b4', 'linestyle': '-',  'linewidth': 2},
    'UC-S2S v2': {'color': '#17becf', 'linestyle': '-',  'linewidth': 2},
    'SI-S2S':    {'color': '#9467bd', 'linestyle': '-',  'linewidth': 2},
    'GenCast':   {'color': '#d62728', 'linestyle': '--', 'linewidth': 2},
    'NGCM':      {'color': '#2ca02c', 'linestyle': '-.',  'linewidth': 2},
}

VAR_LABELS = {
    't2m':  '2m Temperature',
    'z500': 'Geopotential 500 hPa',
    'tp':   'Total Precipitation 24hr',
    'u10':  '10m U-wind',
    'v10':  '10m V-wind',
}

REGIONS = ['global', 'tropics', 'NH', 'SH']
REGION_TITLES = {
    'global':  'Global',
    'tropics': 'Tropics (30S-30N)',
    'NH':      'Extratropics NH (30N-90N)',
    'SH':      'Extratropics SH (30S-90S)',
}

os.makedirs(PLOT_DIR, exist_ok=True)

# Load all model results
all_results = {}
for model_name, pkl_path in MODELS.items():
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            all_results[model_name] = pickle.load(f)['acc_summary']
        print(f"Loaded {model_name}: vars={list(all_results[model_name].keys())}")
    else:
        print(f"WARNING: {pkl_path} not found, skipping {model_name}")

leads = np.arange(0, MAX_LEAD + 1)   # 0..45
plot_leads = leads[:PLOT_MAX_LEAD + 1]  # 0..15

# Collect all variables across all models
all_vars = set()
for res in all_results.values():
    all_vars.update(res.keys())
all_vars = [v for v in VAR_LABELS if v in all_vars]  # preserve order

# ---- One figure per variable: subplots for each region ----
for label in all_vars:
    full_name = VAR_LABELS[label]
    n_regions = len(REGIONS)
    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 4), sharey=True)
    axes = np.array(axes).flatten()

    for ax, region in zip(axes, REGIONS):
        for model_name, acc_summary in all_results.items():
            if label not in acc_summary:
                continue
            arr = acc_summary[label].get(region)
            if arr is None:
                continue
            style = MODEL_STYLES[model_name]
            ax.plot(plot_leads, arr[:PLOT_MAX_LEAD + 1],
                    label=model_name, **style)

            # Mark last lead day >= 0.6
            above = np.where(arr[:PLOT_MAX_LEAD + 1] >= THRESHOLD)[0]
            if len(above) > 0:
                d = above[-1]
                ax.axvline(d, color=style['color'], linewidth=0.8,
                           linestyle=':', alpha=0.6)
                ax.text(d + 0.1, THRESHOLD + 0.01, str(d),
                        color=style['color'], fontsize=8, va='bottom')

        ax.axhline(THRESHOLD, color='black', linewidth=1.2,
                   linestyle='--', label='ACC=0.6')
        ax.set_xlim(-0.5, PLOT_MAX_LEAD + 0.5)
        ax.set_xticks(range(0, PLOT_MAX_LEAD + 1))
        ax.set_ylim(-0.1, 1.05)
        ax.set_xlabel('Lead day', fontsize=10)
        ax.set_title(REGION_TITLES[region], fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('ACC', fontsize=11)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=len(handles),
               fontsize=10, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f'ACC — {full_name}', fontsize=13)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f'acc_{label}_regions.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")

# ---- One combined figure: all variables x global region ----
if all_vars:
    ncols = min(3, len(all_vars))
    nrows = int(np.ceil(len(all_vars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, label in zip(axes, all_vars):
        for model_name, acc_summary in all_results.items():
            if label not in acc_summary:
                continue
            arr = acc_summary[label].get('global')
            if arr is None:
                continue
            style = MODEL_STYLES[model_name]
            ax.plot(plot_leads, arr[:PLOT_MAX_LEAD + 1],
                    label=model_name, **style)
        ax.axhline(THRESHOLD, color='black', linewidth=1, linestyle='--')
        ax.set_title(VAR_LABELS[label], fontsize=10)
        ax.set_xlim(-0.5, PLOT_MAX_LEAD + 0.5)
        ax.set_xticks(range(0, PLOT_MAX_LEAD + 1))
        ax.set_ylim(-0.1, 1.05)
        ax.set_xlabel('Lead day', fontsize=9)
        ax.set_ylabel('ACC', fontsize=9)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(all_vars):]:
        ax.set_visible(False)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower right', fontsize=10, ncol=2)
    fig.suptitle('ACC vs Lead Day — Global — Ensemble Mean', fontsize=13)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, 'acc_global_all_vars.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
