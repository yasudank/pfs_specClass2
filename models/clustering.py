"""Unsupervised Clustering and Spectral Classification Module."""
import os
import joblib
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from config import RESULTS_DIR, WL_GRID, N_GRID


CLUSTER_DEFINITIONS_K6 = {
    0: {
        "name": "Intermediate-z Active Galaxy / QSO",
        "description": "Active galaxies and moderate-redshift QSOs with emission features (median z ~ 0.75).",
        "typical_class": "GALAXY/QSO",
    },
    1: {
        "name": "Cool Star (K/G type) & Intermediate Early-Type Galaxy",
        "description": "Red continuum with absorption signatures; predominantly cool stars and early-type galaxies.",
        "typical_class": "STAR",
    },
    2: {
        "name": "High-z Broad-Line QSO",
        "description": "High-redshift quasars (median z ~ 1.51) with broad emission lines (Mg II, C III], C IV) and blue continuum.",
        "typical_class": "QSO",
    },
    3: {
        "name": "Hot Star (O/B/A) & Flat-Continuum AGN",
        "description": "Very blue/flat continuum with UV/blue flux; hot early-type stars and flat-spectrum AGN.",
        "typical_class": "STAR/QSO",
    },
    4: {
        "name": "Late-Type Star (M-Dwarf) & Red Passive Galaxy",
        "description": "Steep red continuum with strong molecular absorption or 4000A break; high Red/Blue flux ratio.",
        "typical_class": "STAR",
    },
    5: {
        "name": "Star-Forming Galaxy (z ~ 0.7)",
        "description": "Galaxies with prominent star-formation signatures and moderate red continuum (median z ~ 0.71).",
        "typical_class": "GALAXY",
    },
}


class SpectralClusterClassifier:
    """
    Unsupervised cluster classifier operating on 128-dimensional SimCLR latent representations.
    Provides cluster assignments, physical descriptions, broad-class predictions,
    membership probabilities, and anomaly scores.
    """

    def __init__(self, n_clusters: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.gmm: Optional[GaussianMixture] = None
        self.kmeans: Optional[KMeans] = None
        self.cluster_info: Dict[int, Dict[str, Any]] = {}
        self.prototype_spectra: Optional[np.ndarray] = None  # (K, N_GRID)

    def fit(
        self,
        embeddings: np.ndarray,
        spectra: np.ndarray,
        classes: Optional[np.ndarray] = None,
        redshifts: Optional[np.ndarray] = None
    ) -> "SpectralClusterClassifier":
        """
        Fits GMM and KMeans on unit-normalized embeddings.
        Computes prototype median spectra and cluster statistics.
        """
        # Ensure embeddings are L2-normalized
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed_embeds = embeddings / np.maximum(norms, 1e-12)

        # 1. Fit GMM and KMeans
        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type="diag",
            random_state=self.random_state,
            n_init=5
        )
        self.gmm.fit(normed_embeds)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans.fit(normed_embeds)

        cluster_labels = self.gmm.predict(normed_embeds)

        # 2. Compute prototype spectra & characterization
        self.prototype_spectra = np.zeros((self.n_clusters, spectra.shape[1]), dtype=np.float32)
        self.cluster_info = {}

        for k in range(self.n_clusters):
            mask = (cluster_labels == k)
            n_members = int(np.sum(mask))

            if n_members > 0:
                self.prototype_spectra[k] = np.median(spectra[mask], axis=0).astype(np.float32)
            else:
                self.prototype_spectra[k] = np.zeros(spectra.shape[1], dtype=np.float32)

            info = CLUSTER_DEFINITIONS_K6.get(k, {
                "name": f"Spectral Cluster {k}",
                "description": f"Unsupervised cluster {k}",
                "typical_class": "UNKNOWN"
            }).copy()

            info["n_members"] = n_members

            if classes is not None and n_members > 0:
                from collections import Counter
                counts = Counter(classes[mask])
                info["class_breakdown"] = dict(counts)
                info["dominant_class"] = counts.most_common(1)[0][0]
                total = len(classes[mask])
                info["class_probs"] = {cls: cnt / total for cls, cnt in counts.items()}
            else:
                info["class_breakdown"] = {}
                info["dominant_class"] = info["typical_class"]
                info["class_probs"] = {}

            if redshifts is not None and n_members > 0:
                valid_z = redshifts[mask & np.isfinite(redshifts)]
                info["median_z"] = float(np.median(valid_z)) if len(valid_z) > 0 else np.nan
            else:
                info["median_z"] = np.nan

            self.cluster_info[k] = info

        return self

    def predict(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Classifies a single 128-dimensional embedding vector.
        Returns:
            - cluster_id (int)
            - cluster_name (str)
            - cluster_description (str)
            - cluster_probabilities (Dict[int, float])
            - predicted_broad_class (str: 'STAR', 'GALAXY', or 'QSO')
            - broad_class_confidence (float)
            - anomaly_score (float: higher indicates more atypical spectrum)
        """
        if self.gmm is None:
            raise RuntimeError("Classifier has not been fitted or loaded.")

        if embedding.ndim == 1:
            embedding = embedding[np.newaxis, :]

        # Normalize
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        normed = embedding / np.maximum(norm, 1e-12)

        # GMM prediction & probabilities
        probs = self.gmm.predict_proba(normed)[0]
        cluster_id = int(np.argmax(probs))
        cluster_conf = float(probs[cluster_id])

        # Anomaly score: negative log-likelihood (NLL)
        # Higher NLL = lower probability density in the learned latent distribution
        log_prob = float(self.gmm.score_samples(normed)[0])
        anomaly_score = float(-log_prob)

        # Broad class estimation from cluster's learned class distribution
        c_info = self.cluster_info[cluster_id]
        class_probs = c_info.get("class_probs", {})
        if class_probs:
            best_broad = max(class_probs.items(), key=lambda x: x[1])
            broad_class = best_broad[0]
            broad_conf = float(best_broad[1] * cluster_conf)
        else:
            broad_class = c_info.get("dominant_class", "UNKNOWN")
            broad_conf = cluster_conf

        return {
            "cluster_id": cluster_id,
            "cluster_name": c_info["name"],
            "cluster_description": c_info["description"],
            "cluster_confidence": cluster_conf,
            "cluster_probabilities": {k: float(probs[k]) for k in range(self.n_clusters)},
            "predicted_broad_class": broad_class,
            "broad_class_confidence": broad_conf,
            "anomaly_score": anomaly_score,
            "median_redshift": c_info.get("median_z", np.nan),
        }

    def save(self, filepath: str):
        """Saves fitted classifier to disk."""
        data = {
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "gmm": self.gmm,
            "kmeans": self.kmeans,
            "cluster_info": self.cluster_info,
            "prototype_spectra": self.prototype_spectra,
        }
        joblib.dump(data, filepath)
        print(f"SpectralClusterClassifier saved to {filepath}.")

    @classmethod
    def load(cls, filepath: str) -> "SpectralClusterClassifier":
        """Loads classifier from disk."""
        data = joblib.load(filepath)
        instance = cls(n_clusters=data["n_clusters"], random_state=data["random_state"])
        instance.gmm = data["gmm"]
        instance.kmeans = data["kmeans"]
        instance.cluster_info = data["cluster_info"]
        instance.prototype_spectra = data["prototype_spectra"]
        return instance
