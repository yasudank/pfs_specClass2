"""Weighted PCA / EMPCA implementation for spectral data with missing/masked pixels.
Based on Bailey (2012) PASP 124, 1015 (EMPCA).
"""
import numpy as np
from typing import Tuple, Optional


class EMPCA:
    """Expectation-Maximization PCA for spectra with weights/masks."""

    def __init__(self, n_components: int = 10, max_iter: int = 20, tol: float = 1e-4, reg: float = 1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.mean_ = None        # (D,)
        self.components_ = None  # (K, D)
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray, W: np.ndarray) -> "EMPCA":
        """
        Fits EMPCA model.
        Args:
            X: (N, D) spectral flux
            W: (N, D) weights (0 for masked pixels)
        """
        N, D = X.shape
        K = self.n_components

        # 1. Compute weighted mean
        sum_w = np.sum(W, axis=0) + 1e-8
        self.mean_ = np.sum(W * X, axis=0) / sum_w

        # Centered data
        X_cent = X - self.mean_

        # 2. Initialize components randomly with orthonormal rows
        rng = np.random.RandomState(42)
        C = rng.randn(K, D).astype(np.float32)
        # Orthonormalize
        C, _ = np.linalg.qr(C.T)
        C = C.T[:K]  # (K, D)

        coeffs = np.zeros((N, K), dtype=np.float32)

        prev_loss = float("inf")
        for iteration in range(self.max_iter):
            # E-step: solve for coefficients a_n for each spectrum
            # a_n = (C W_n C^T + reg * I)^(-1) C W_n x_n
            for n in range(N):
                w_n = W[n]  # (D,)
                # CW_n = C * w_n (broadcasting along D)
                CW = C * w_n[None, :]  # (K, D)
                # M = C W C^T (K, K)
                M = np.dot(CW, C.T) + self.reg * np.eye(K)
                rhs = np.dot(CW, X_cent[n])
                try:
                    coeffs[n] = np.linalg.solve(M, rhs)
                except np.linalg.LinAlgError:
                    coeffs[n] = np.linalg.lstsq(M, rhs, rcond=None)[0]

            # M-step: solve for C given coeffs
            # For each wavelength d, C[:, d] = (A^T W_d A)^(-1) A^T W_d X_cent[:, d]
            for d in range(D):
                w_d = W[:, d]  # (N,)
                AW = coeffs * w_d[:, None]  # (N, K)
                M_d = np.dot(AW.T, coeffs) + self.reg * np.eye(K)
                rhs_d = np.dot(AW.T, X_cent[:, d])
                try:
                    C[:, d] = np.linalg.solve(M_d, rhs_d)
                except np.linalg.LinAlgError:
                    C[:, d] = np.linalg.lstsq(M_d, rhs_d, rcond=None)[0]

            # Orthonormalize components C using QR decomposition
            Q, _ = np.linalg.qr(C.T)
            C = Q.T[:K]

            # Compute weighted reconstruction error
            recon = self.mean_ + np.dot(coeffs, C)
            loss = np.sum(W * (X - recon) ** 2) / (np.sum(W) + 1e-8)

            rel_diff = abs(prev_loss - loss) / (prev_loss + 1e-8)
            prev_loss = loss
            if rel_diff < self.tol and iteration >= 3:
                break

        self.components_ = C
        return self

    def transform(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Projects spectra onto components taking weights/masks into account."""
        N, D = X.shape
        K = self.n_components
        X_cent = X - self.mean_
        coeffs = np.zeros((N, K), dtype=np.float32)

        for n in range(N):
            w_n = W[n]
            CW = self.components_ * w_n[None, :]
            M = np.dot(CW, self.components_.T) + self.reg * np.eye(K)
            rhs = np.dot(CW, X_cent[n])
            try:
                coeffs[n] = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                coeffs[n] = np.linalg.lstsq(M, rhs, rcond=None)[0]

        return coeffs

    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        """Reconstructs spectra from coefficients."""
        return self.mean_ + np.dot(coeffs, self.components_)
