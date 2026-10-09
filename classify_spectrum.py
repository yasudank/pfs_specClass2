"""PFS Spectrum Classification & Inference API.

Provides standalone inference pipeline for new Subaru/PFS spectra:
1. Loads raw FITS file.
2. Applies adaptive masking (BRN vs BMN detection, detector edges, spike suppression).
3. Resamples to 4,096-pixel log-wavelength grid.
4. Extracts 128-dimensional latent representation using trained 1D-ResNet SimCLR encoder.
5. Predicts unsupervised cluster, broad astrophysical class (STAR/GALAXY/QSO), and anomaly score.
"""
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import RESULTS_DIR, WL_GRID, N_GRID, LOG_WL_GRID
from dataset import load_raw_spectrum, preprocess_spectrum
from models.contrastive_model import SpectralSimCLR
from models.clustering import SpectralClusterClassifier


class PFSClassifier:
    """Loaded inference engine for classifying PFS spectra."""

    def __init__(
        self,
        simclr_checkpoint_path: Optional[Union[str, Path]] = None,
        cluster_model_path: Optional[Union[str, Path]] = None,
        canonical_templates_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        if simclr_checkpoint_path is None:
            simclr_checkpoint_path = RESULTS_DIR / "spectral_simclr_clean_v2.pt"
        if cluster_model_path is None:
            cluster_model_path = RESULTS_DIR / "spectral_cluster_classifier.joblib"
        if canonical_templates_path is None:
            canonical_templates_path = RESULTS_DIR / "canonical_templates.npz"

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load SimCLR encoder
        state_dict = torch.load(simclr_checkpoint_path, map_location=self.device, weights_only=False)
        self.encoder = SpectralSimCLR(embed_dim=128, proj_dim=64).to(self.device)
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval()

        # Load Cluster Classifier
        self.cluster_clf = SpectralClusterClassifier.load(str(cluster_model_path))

        # Load Canonical Rest Templates for redshift cross-correlation
        if os.path.exists(canonical_templates_path):
            tmpl = np.load(canonical_templates_path)
            self.templates = {
                'rest_wl': tmpl['rest_wl'],
                'STAR': tmpl['STAR'],
                'GALAXY': tmpl['GALAXY'],
                'QSO': tmpl['QSO'],
            }
        else:
            self.templates = None

    def estimate_redshift_template(self, flux: np.ndarray, pred_class: str) -> Tuple[float, float]:
        """Estimates redshift via cross-correlation with canonical rest-frame template."""
        if self.templates is None or pred_class == 'STAR':
            return 0.0, 1.0

        rest_wl = self.templates['rest_wl']
        template = self.templates.get(pred_class, self.templates['GALAXY'])

        z_grid = np.linspace(0.0, 1.5, 300) if pred_class == 'GALAXY' else np.linspace(0.2, 3.8, 360)
        f_norm = flux - np.mean(flux)
        f_std = np.std(f_norm)
        if f_std < 1e-6:
            return 0.0, 0.0
        f_norm /= f_std

        best_z = 0.0
        best_corr = -1.0
        for z in z_grid:
            t_obs = np.interp(WL_GRID, rest_wl * (1.0 + z), template, left=0.0, right=0.0)
            t_norm = t_obs - np.mean(t_obs)
            t_std = np.std(t_norm)
            if t_std < 1e-6:
                continue
            t_norm /= t_std
            corr = float(np.mean(f_norm * t_norm))
            if corr > best_corr:
                best_corr = corr
                best_z = float(z)

        return round(best_z, 4), round(max(best_corr, 0.0), 3)

    def classify(self, fits_path_or_dict: Union[str, Path, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Classifies a single PFS spectrum from FITS file or raw array dict.
        Returns detailed classification result including cluster, broad class, and anomaly score.
        """
        if isinstance(fits_path_or_dict, (str, Path)):
            raw = load_raw_spectrum(str(fits_path_or_dict))
            fits_path = str(fits_path_or_dict)
        else:
            raw = fits_path_or_dict
            fits_path = "memory_buffer"

        if raw is None:
            return {
                "status": "error",
                "message": f"Could not load valid spectrum from {fits_path}"
            }

        proc = preprocess_spectrum(raw)
        if proc is None:
            return {
                "status": "error",
                "message": f"Preprocessing failed (insufficient valid pixels) for {fits_path}"
            }

        flux = proc["flux"]              # (4096,) normalized flux
        flux_clean = proc["flux_clean"]  # (4096,) sky-dampened flux
        weight = proc["weight"]          # (4096,)
        mask = proc["mask"]              # (4096,)

        # Determine arm mode
        in_gap = (WL_GRID >= 670.0) & (WL_GRID <= 690.0)
        is_bmn = bool(np.all(mask[in_gap] == 0))
        arm_mode = "BMN (Medium-Resolution)" if is_bmn else "BRN (Low-Resolution)"

        # Quality metrics
        valid_frac = float(np.mean(mask > 0))
        snr_est = float(np.median(flux[mask > 0] * np.sqrt(np.maximum(weight[mask > 0], 1e-12)))) if np.any(mask > 0) else 0.0

        # Feature extraction via clean SimCLR (sky line suppressed!)
        with torch.no_grad():
            x = torch.from_numpy(flux_clean[None, None, :]).to(self.device)
            embedding = self.encoder.encode(x).cpu().numpy()[0]

        # Clustering & classification
        pred = self.cluster_clf.predict(embedding)
        broad_class = pred["predicted_broad_class"]

        # Shift-correlation redshift estimation
        est_z, r_match = self.estimate_redshift_template(flux_clean, broad_class)

        return {
            "status": "success",
            "fits_path": fits_path,
            "arm_mode": arm_mode,
            "predicted_broad_class": broad_class,
            "broad_class_confidence": round(pred["broad_class_confidence"], 4),
            "cluster_id": pred["cluster_id"],
            "cluster_name": pred["cluster_name"],
            "cluster_description": pred["cluster_description"],
            "cluster_confidence": round(pred["cluster_confidence"], 4),
            "anomaly_score": round(pred["anomaly_score"], 2),
            "estimated_redshift": est_z,
            "redshift_match_quality": r_match,
            "estimated_typical_redshift": round(pred["median_redshift"], 3) if np.isfinite(pred["median_redshift"]) else None,
            "quality": {
                "valid_pixel_fraction": round(valid_frac, 4),
                "median_snr_estimate": round(snr_est, 2),
                "norm_scale": round(float(proc["norm_scale"]), 4),
            },
            "latent_embedding": embedding.tolist(),
            "preprocessed_flux": flux,
            "preprocessed_flux_clean": flux_clean,
            "preprocessed_mask": mask,
        }

    def plot_classification(self, result: Dict[str, Any], save_path: Optional[str] = None):
        """Generates a comprehensive diagnostic plot of the spectrum and classification."""
        if result.get("status") != "success":
            print(f"Cannot plot failed result: {result.get('message')}")
            return

        flux = result["preprocessed_flux"]
        mask = result["preprocessed_mask"]
        cluster_id = result["cluster_id"]
        proto = self.cluster_clf.prototype_spectra[cluster_id]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Panel 1: Observed spectrum vs Cluster prototype
        ax1.plot(WL_GRID, flux, color="#1f77b4", lw=1.2, label=f"Observed Spectrum ({result['arm_mode']})")
        ax1.plot(WL_GRID, proto, color="#d62728", lw=1.5, ls="--", alpha=0.8, label=f"Cluster {cluster_id} Prototype")

        # Shading masked regions
        unmasked = (mask > 0)
        # Find contiguous masked segments
        masked_idx = np.where(mask == 0)[0]
        if len(masked_idx) > 0:
            diffs = np.diff(masked_idx)
            splits = np.where(diffs > 1)[0] + 1
            segments = np.split(masked_idx, splits)
            for seg in segments:
                if len(seg) > 0:
                    ax1.axvspan(WL_GRID[seg[0]], WL_GRID[seg[-1]], color='gray', alpha=0.18)

        title = (
            f"Prediction: {result['predicted_broad_class']} (conf: {result['broad_class_confidence']*100:.1f}%) | "
            f"{result['cluster_name']}\n"
            f"Anomaly Score: {result['anomaly_score']:.1f} | SNR: {result['quality']['median_snr_estimate']:.1f} | "
            f"Valid Pixels: {result['quality']['valid_pixel_fraction']*100:.1f}%"
        )
        ax1.set_title(title, fontsize=12, fontweight="bold", loc="left")
        ax1.set_ylabel("Normalized Flux", fontsize=10)
        ax1.set_ylim(-0.5, max(4.0, float(np.percentile(flux[unmasked], 99) * 1.3) if np.any(unmasked) else 4.0))
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", fontsize=9)

        # Panel 2: Difference / Residual
        diff = flux - proto
        diff[mask == 0] = 0.0
        ax2.plot(WL_GRID, diff, color="#2ca02c", lw=1.0, label="Residual (Observed - Prototype)")
        ax2.axhline(0.0, color="black", lw=0.8, ls=":")
        ax2.set_xlabel("Observed Wavelength (nm)", fontsize=11)
        ax2.set_ylabel("Residual", fontsize=10)
        ax2.set_ylim(-2.0, 2.0)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
            print(f"Saved classification diagnostic plot to {save_path}")
        else:
            plt.show()
        plt.close()


def classify_pfs_spectrum(fits_path: str, plot: bool = False, output_plot_path: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to classify a single PFS FITS file."""
    classifier = PFSClassifier()
    result = classifier.classify(fits_path)
    if plot and result.get("status") == "success":
        classifier.plot_classification(result, save_path=output_plot_path)
    return result


def classify_batch(
    fits_paths: List[Union[str, Path]],
    output_csv_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Classifies a batch of PFS spectra and optionally writes a summary CSV."""
    import pandas as pd
    classifier = PFSClassifier()
    records = []

    print(f"\nProcessing batch of {len(fits_paths)} spectra...")
    for i, path in enumerate(fits_paths):
        res = classifier.classify(str(path))
        if res.get("status") == "success":
            rec = {
                "fits_path": res["fits_path"],
                "arm_mode": res["arm_mode"],
                "predicted_broad_class": res["predicted_broad_class"],
                "broad_class_confidence": res["broad_class_confidence"],
                "cluster_id": res["cluster_id"],
                "cluster_name": res["cluster_name"],
                "cluster_confidence": res["cluster_confidence"],
                "anomaly_score": res["anomaly_score"],
                "estimated_typical_redshift": res["estimated_typical_redshift"],
                "valid_pixel_fraction": res["quality"]["valid_pixel_fraction"],
                "snr_estimate": res["quality"]["median_snr_estimate"],
            }
            records.append(rec)
        else:
            records.append({
                "fits_path": str(path),
                "arm_mode": "UNKNOWN",
                "predicted_broad_class": "ERROR",
                "broad_class_confidence": 0.0,
                "cluster_id": -1,
                "cluster_name": "Failed",
                "cluster_confidence": 0.0,
                "anomaly_score": np.nan,
                "estimated_typical_redshift": None,
                "valid_pixel_fraction": 0.0,
                "snr_estimate": 0.0,
            })

        if (i + 1) % 50 == 0 or (i + 1) == len(fits_paths):
            print(f"  Processed {i + 1} / {len(fits_paths)} spectra")

    if output_csv_path:
        df = pd.DataFrame(records)
        df.to_csv(output_csv_path, index=False)
        print(f"Summary table saved to {output_csv_path}")

    return records


def main():
    parser = argparse.ArgumentParser(description="Subaru/PFS Spectrum Unsupervised Classifier CLI")
    parser.add_argument("fits_file", type=str, nargs="?", default=None, help="Path to single PFS target FITS file")
    parser.add_argument("--batch-list", type=str, default=None, help="Text file with list of FITS paths to classify in batch")
    parser.add_argument("--batch-output", type=str, default=None, help="Output CSV path for batch classification")
    parser.add_argument("--plot", action="store_true", help="Generate diagnostic classification plot")
    parser.add_argument("--plot-output", type=str, default=None, help="File path to save diagnostic plot")
    parser.add_argument("--json", action="store_true", help="Output results formatted as JSON")
    args = parser.parse_args()

    if args.batch_list:
        with open(args.batch_list, "r") as f:
            paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        classify_batch(paths, output_csv_path=args.batch_output)
        return

    if not args.fits_file:
        parser.error("Must specify either a fits_file or --batch-list.")

    clf = PFSClassifier()
    res = clf.classify(args.fits_file)

    if args.plot:
        out_p = args.plot_output or str(RESULTS_DIR / f"classify_{Path(args.fits_file).stem}.png")
        clf.plot_classification(res, save_path=out_p)

    # Clean array fields for terminal display / JSON output
    disp_res = {k: v for k, v in res.items() if not isinstance(v, (np.ndarray, list))}

    if args.json:
        print(json.dumps(disp_res, indent=2))
    else:
        print("\n" + "=" * 60)
        print("          PFS Spectrum Classification Report          ")
        print("=" * 60)
        print(f"File: {res.get('fits_path')}")
        print(f"Arm Mode: {res.get('arm_mode')}")
        print(f"Predicted Broad Class: {res.get('predicted_broad_class')} (Confidence: {res.get('broad_class_confidence', 0)*100:.1f}%)")
        print(f"Estimated Redshift (Shift Match): {res.get('estimated_redshift')} (quality: {res.get('redshift_match_quality')})")
        print(f"Cluster Assignment: #{res.get('cluster_id')} - {res.get('cluster_name')}")
        print(f"Description: {res.get('cluster_description')}")
        print(f"Cluster Confidence: {res.get('cluster_confidence', 0)*100:.1f}%")
        print(f"Typical Redshift: {res.get('estimated_typical_redshift')}")
        print(f"Anomaly Score (NLL): {res.get('anomaly_score')}")
        print(f"Quality: Valid Pixels={res['quality']['valid_pixel_fraction']*100:.1f}%, S/N={res['quality']['median_snr_estimate']:.1f}")
        print("=" * 60)


if __name__ == "__main__":
    main()

