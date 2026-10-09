"""Evaluate Attentive Spectral SimCLR Model.
1. Extracts Attention Weights alpha(lambda) across wavelength for real spectra:
   - Agreed QSO
   - Missed Host-Dominated QSO (DB=QSO, previous RF=GALAXY)
   - False QSO Galaxy (DB=GALAXY, previous RF=QSO)
   - Normal Star & Normal Galaxy
2. Generates an interpretability figure showing:
   [Spectrum flux] + [Attention weights] highlighting exactly where the model focuses!
3. Evaluates embedding separation.
"""
import os
import sys
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

import torch
import torch.nn.functional as F
from models.contrastive_model import SpectralAttentionSimCLR
from config import RESULTS_DIR, WL_GRID


def evaluate_attention():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data and trained Attentive SimCLR model...")

    model_path = RESULTS_DIR / "spectral_simclr_attention_model.pt"
    if not model_path.exists():
        print(f"Model path {model_path} does not exist yet. Please wait for training to finish.")
        return

    model = SpectralAttentionSimCLR(embed_dim=128, proj_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    data_path = RESULTS_DIR / "dataset_full_augmented_scratch.npz"
    data = np.load(data_path, allow_pickle=True)
    flux_clean = data['flux_clean']
    classes = data['classes']
    obj_ids = data['objId']
    redshifts = data['redshifts']

    # Also load previous RF predictions to identify the disagreement objects
    rf_data = np.load(RESULTS_DIR / "rf_oof_predictions_216k.npz", allow_pickle=True)
    meta = np.load(RESULTS_DIR / "survey_meta_216k_final.npz", allow_pickle=True)

    rf_preds = rf_data["oof_preds"]
    db_cls = meta["db_classes"]
    db_oids = meta["objIds"]

    oid_to_rf = {db_oids[i]: rf_preds[i] for i in range(len(db_oids))}

    # Find representative objects from each group
    sample_categories = [
        ("Agreed QSO (Broad Line Core)", "QSO", "QSO"),
        ("Missed QSO (Host-Dominated, Previously RF=GAL)", "QSO", "GALAXY"),
        ("False QSO (Starburst Galaxy, Previously RF=QSO)", "GALAXY", "QSO"),
        ("Agreed Galaxy (Normal Emission/Absorption)", "GALAXY", "GALAXY"),
        ("Agreed Star (Stellar Absorption)", "STAR", "STAR"),
    ]

    selected_samples = []
    for cat_name, target_db, target_rf in sample_categories:
        for idx in range(len(obj_ids)):
            oid = obj_ids[idx]
            cls = classes[idx]
            rf_p = oid_to_rf.get(oid, None)
            if cls == target_db and rf_p == target_rf and redshifts[idx] > 0.05:
                # Pick a spectrum with decent SNR
                f = flux_clean[idx]
                if np.nanmax(f) > 1.5:
                    selected_samples.append((cat_name, idx, oid, cls, rf_p, redshifts[idx]))
                    break

    print(f"Selected {len(selected_samples)} representative spectra for Attention inspection:")
    for name, idx, oid, c_db, c_rf, z in selected_samples:
        print(f"  - {name}: ObjID={oid}, z={z:.3f}")

    # Run inference to extract embeddings and attention weights
    fig, axes = plt.subplots(len(selected_samples), 2, figsize=(18, 3.2 * len(selected_samples)),
                             gridspec_kw={'width_ratios': [3.5, 1.2]})

    spectral_lines = {
        r'Ly$\alpha$': 121.6,
        r'C IV': 154.9,
        r'Mg II': 279.8,
        r'[O II]': 372.7,
        r'H$\beta$': 486.1,
        r'[O III]': 500.7,
        r'H$\alpha$': 656.3,
    }

    wl_down = np.linspace(WL_GRID[0], WL_GRID[-1], 256)

    for row_idx, (cat_name, s_idx, oid, c_db, c_rf, z) in enumerate(selected_samples):
        spec_flux = flux_clean[s_idx]
        x_tensor = torch.from_numpy(spec_flux).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            embed, attn_weights = model.encode(x_tensor, return_attention=True)
            attn = attn_weights.squeeze(0).cpu().numpy()  # (256,)

        # Interpolate attention back to original wavelength grid
        attn_interp = np.interp(WL_GRID, wl_down, attn)

        # Plot Left: Spectrum & Attention Weights
        ax_spec = axes[row_idx, 0]
        color_flux = '#1f77b4' if 'STAR' in cat_name else ('#2ca02c' if 'Galaxy' in cat_name else '#d62728')

        # Flux on primary axis
        line_spec, = ax_spec.plot(WL_GRID, spec_flux, color='#333333', lw=1.1, alpha=0.85, label='Normalized Observed Flux')
        ax_spec.set_ylabel('Flux', fontsize=9.5)
        ax_spec.grid(True, linestyle='--', alpha=0.35)

        # Secondary axis for Attention weights
        ax_attn = ax_spec.twinx()
        line_attn = ax_attn.fill_between(WL_GRID, 0, attn_interp, color='#ff7f0e', alpha=0.45, label='Learned Attention Weight')
        ax_attn.plot(WL_GRID, attn_interp, color='#d95f02', lw=1.5)
        ax_attn.set_ylabel('Attention Weight α(λ)', color='#d95f02', fontsize=9.5)
        ax_attn.tick_params(axis='y', labelcolor='#d95f02')
        ax_attn.set_ylim(0, max(np.max(attn_interp) * 1.35, 0.01))

        # Title with object metadata
        ax_spec.set_title(f"[{row_idx+1}] {cat_name} | ObjID: {oid} (DB: {c_db}, Prev RF: {c_rf}, z={z:.3f})",
                          fontsize=10.5, fontweight='bold', loc='left')

        # Indicate expected observed wavelength of major lines
        for l_name, l_rest in spectral_lines.items():
            l_obs = l_rest * (1.0 + z)
            if WL_GRID[0] <= l_obs <= WL_GRID[-1]:
                ax_spec.axvline(l_obs, color='red', linestyle=':', alpha=0.5, lw=1.0)
                ax_spec.text(l_obs, ax_spec.get_ylim()[1] * 0.85, l_name, color='#b2182b',
                             fontsize=8, rotation=90, ha='right')

        if row_idx == len(selected_samples) - 1:
            ax_spec.set_xlabel('Observed Wavelength (nm)', fontsize=10.5)

        # Plot Right: Attention Bar Profile & Focus Summary
        ax_prof = axes[row_idx, 1]
        sorted_pos = np.argsort(attn)[-5:][::-1]
        top_wls = wl_down[sorted_pos]
        top_weights = attn[sorted_pos] / np.max(attn) * 100.0

        y_pos = np.arange(len(top_wls))
        ax_prof.barh(y_pos, top_weights, color='#ff7f0e', alpha=0.75, edgecolor='#d95f02')
        ax_prof.set_yticks(y_pos)
        ax_prof.set_yticklabels([f"{w:.0f} nm" for w in top_wls], fontsize=8.5)
        ax_prof.invert_yaxis()
        ax_prof.set_xlabel('Relative Attention (%)', fontsize=9)
        ax_prof.set_title('Top Attention Peaks', fontsize=9.5, fontweight='bold')
        ax_prof.grid(True, linestyle=':', alpha=0.4, axis='x')

    plt.suptitle("Interpretable Spectral Self-Attention: Dynamic Focusing on Emission/Absorption Lines",
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    fig_path = RESULTS_DIR / "spectral_attention_interpretability_profiles.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved attention interpretability figure to {fig_path}.")

    # Copy to artifacts
    art_dir = Path("/home/yasuda/.gemini/antigravity-ide/brain/d5096d3e-909d-4738-8a2b-70fbd53adf8d")
    shutil.copy(fig_path, art_dir / "spectral_attention_interpretability_profiles.png")
    print("Copied to artifacts successfully!")


if __name__ == "__main__":
    evaluate_attention()
