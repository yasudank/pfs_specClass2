"""Step 4: Train Spectroscopic Redshift Estimation Models & Infer Redshifts
1. GALAXY: Trains 1D Deep Spectral ResNet on agreed galaxies (z in [0.0, 2.5]).
2. QSO: Trains 1D Spectral ResNet on agreed QSOs (z in [0.0, 4.5]) and builds
   Canonical QSO Template Cross-Correlation Function (CCF) with PFS arm-gap masking.
3. STAR: Sets rest-frame reference redshift z = 0.0000.
4. Outputs:
   - Checkpoint: results/galaxy_redshift_resnet.pt
   - Checkpoint: results/qso_redshift_resnet.pt
   - Predictions cache: results/redshift_predictions_scratch.npz
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR, WL_GRID, ARM_MASK_BRN


# ==============================================================================
# 1D Deep Spectral ResNet Architecture
# ==============================================================================
class ConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=7, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + res)


class SpectralRedshiftResNet(nn.Module):
    def __init__(self, in_channels=2, z_min=0.0, z_max=4.5, num_bins=1500):
        super().__init__()
        self.z_min = z_min
        self.z_max = z_max
        self.num_bins = num_bins
        self.register_buffer('z_grid', torch.linspace(z_min, z_max, num_bins))

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        self.layer1 = ConvBlock1D(32, 64, kernel_size=7, stride=2)
        self.layer2 = ConvBlock1D(64, 128, kernel_size=7, stride=2)
        self.layer3 = ConvBlock1D(128, 256, kernel_size=5, stride=2)
        self.layer4 = ConvBlock1D(256, 256, kernel_size=5, stride=2)

        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_bins)
        )

    def forward(self, x):
        feat = self.stem(x)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)

        avg_pool = torch.mean(feat, dim=-1)
        max_pool = torch.amax(feat, dim=-1)
        pool = torch.cat([avg_pool, max_pool], dim=-1)

        logits = self.head(pool)
        probs = F.softmax(logits, dim=-1)
        z_pred = torch.sum(probs * self.z_grid, dim=-1)
        return z_pred, probs, logits


def compute_loss(logits, z_pred, z_true, z_grid, sigma=0.006):
    diff = z_grid.unsqueeze(0) - z_true.unsqueeze(1)
    target_dist = torch.exp(-0.5 * (diff / sigma) ** 2)
    target_dist = target_dist / torch.clamp(torch.sum(target_dist, dim=1, keepdim=True), min=1e-8)

    log_probs = F.log_softmax(logits, dim=-1)
    ce_loss = -torch.mean(torch.sum(target_dist * log_probs, dim=-1))
    l1_loss = F.smooth_l1_loss(z_pred, z_true, beta=0.01)
    return ce_loss + 1.0 * l1_loss


def _residual_worker(f):
    c = ndimage.median_filter(f, size=128)
    c = ndimage.gaussian_filter1d(c, sigma=32)
    r = f - c
    return (3.0 * r).astype(np.float32)


def compute_residuals_parallel(flux_arr, max_workers=16):
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        res = list(ex.map(_residual_worker, flux_arr, chunksize=256))
    return np.array(res, dtype=np.float32)


def main():
    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("Step 4: Training Spectroscopic Redshift Estimation Models (GALAXY & QSO)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # 1. Load Data
    data_path = RESULTS_DIR / "dataset_full_augmented_scratch.npz"
    pred_path = RESULTS_DIR / "rf_oof_boost3x_smote_representative.npz"
    tmpl_path = RESULTS_DIR / "canonical_templates_scratch.npz"

    print("\n--- [4.1] Loading Representative Dataset & Stage 1 Classification ---")
    data = np.load(data_path, allow_pickle=True)
    preds = np.load(pred_path, allow_pickle=True)
    tmpl_data = np.load(tmpl_path, allow_pickle=True)

    all_flux = data['flux_clean']
    all_obj_ids = data['objId']
    _, unique_indices = np.unique(all_obj_ids, return_index=True)

    obj_ids = all_obj_ids[unique_indices]
    flux = all_flux[unique_indices]
    n_total = len(obj_ids)
    print(f"Unique target spectra: N = {n_total:,}")

    oof_preds = preds['oof_preds']
    oof_probs = preds['oof_probs']
    classes_order = list(preds['classes'])
    db_classes = preds['db_classes']
    db_redshifts = preds['db_redshifts']

    idx_qso_cls = classes_order.index('QSO')
    prob_qso = oof_probs[:, idx_qso_cls]
    pred_confidence = np.max(oof_probs, axis=1)

    # 2. Compute 2-channel input tensors
    print("\n--- [4.2] Computing Boosted 2-Channel Inputs (F(λ), 3.0×R(λ)) ---")
    t0 = time.time()
    res_boosted = compute_residuals_parallel(flux, max_workers=16)
    flux_2ch = np.stack([flux, res_boosted], axis=1).astype(np.float32)
    print(f"Computed in {time.time() - t0:.1f}s. Input shape: {flux_2ch.shape}")

    # 3. Train Galaxy Redshift ResNet
    print("\n--- [4.3] Training Galaxy Redshift ResNet (z in [0.0, 2.5]) ---")
    m_gal_train = (db_classes == 'GALAXY') & (oof_preds == 'GALAXY') & (db_redshifts > 0.005) & (db_redshifts < 2.5)
    idx_gal_train = np.where(m_gal_train)[0]
    print(f"Agreed training galaxies: N = {len(idx_gal_train):,}")

    loader_tr_g = DataLoader(TensorDataset(torch.from_numpy(flux_2ch[idx_gal_train]),
                                           torch.from_numpy(db_redshifts[idx_gal_train].astype(np.float32))),
                             batch_size=128, shuffle=True, drop_last=True)

    model_gal = SpectralRedshiftResNet(in_channels=2, z_min=0.0, z_max=2.5, num_bins=1250).to(device)
    optimizer_g = torch.optim.AdamW(model_gal.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs_g = 15
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=epochs_g, eta_min=1e-5)

    t0 = time.time()
    for ep in range(1, epochs_g + 1):
        model_gal.train()
        total_loss = 0.0
        n_b = 0
        for bx, by in loader_tr_g:
            bx, by = bx.to(device), by.to(device)
            optimizer_g.zero_grad(set_to_none=True)
            z_pred, _, logits = model_gal(bx)
            loss = compute_loss(logits, z_pred, by, model_gal.z_grid, sigma=0.005)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_gal.parameters(), 1.0)
            optimizer_g.step()
            total_loss += loss.item()
            n_b += 1
        scheduler_g.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  Galaxy Epoch {ep:02d}/{epochs_g} | Loss: {total_loss / n_b:.4f}")
    print(f"Galaxy model trained in {time.time() - t0:.1f}s.")

    # Save Galaxy Model Checkpoint
    ckpt_gal = RESULTS_DIR / "galaxy_redshift_resnet.pt"
    torch.save(model_gal.state_dict(), ckpt_gal)
    print(f"Saved Galaxy ResNet checkpoint to {ckpt_gal}.")

    # 4. Train QSO Redshift ResNet
    print("\n--- [4.4] Training QSO Redshift ResNet (z in [0.0, 4.5]) ---")
    m_qso_train = (db_classes == 'QSO') & (oof_preds == 'QSO') & (db_redshifts > 0.05) & (db_redshifts < 4.5)
    idx_qso_train = np.where(m_qso_train)[0]
    print(f"Agreed training QSOs: N = {len(idx_qso_train):,}")

    # Augment 4x for training stability
    X_tr_q = flux_2ch[idx_qso_train]
    y_tr_q = db_redshifts[idx_qso_train].astype(np.float32)
    X_tr_q_aug = [X_tr_q]
    y_tr_q_aug = [y_tr_q]
    for _ in range(3):
        X_tr_q_aug.append(X_tr_q + np.random.normal(0, 0.02, size=X_tr_q.shape).astype(np.float32))
        y_tr_q_aug.append(y_tr_q)
    X_tr_q_all = np.concatenate(X_tr_q_aug, axis=0)
    y_tr_q_all = np.concatenate(y_tr_q_aug, axis=0)

    loader_tr_q = DataLoader(TensorDataset(torch.from_numpy(X_tr_q_all), torch.from_numpy(y_tr_q_all)),
                             batch_size=64, shuffle=True, drop_last=True)

    model_qso = SpectralRedshiftResNet(in_channels=2, z_min=0.0, z_max=4.5, num_bins=1500).to(device)
    optimizer_q = torch.optim.AdamW(model_qso.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs_q = 20
    scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=epochs_q, eta_min=1e-5)

    t0 = time.time()
    for ep in range(1, epochs_q + 1):
        model_qso.train()
        total_loss = 0.0
        n_b = 0
        for bx, by in loader_tr_q:
            bx, by = bx.to(device), by.to(device)
            optimizer_q.zero_grad(set_to_none=True)
            z_pred, _, logits = model_qso(bx)
            loss = compute_loss(logits, z_pred, by, model_qso.z_grid, sigma=0.008)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_qso.parameters(), 1.0)
            optimizer_q.step()
            total_loss += loss.item()
            n_b += 1
        scheduler_q.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  QSO Epoch {ep:02d}/{epochs_q} | Loss: {total_loss / n_b:.4f}")
    print(f"QSO model trained in {time.time() - t0:.1f}s.")

    # Save QSO Model Checkpoint
    ckpt_qso = RESULTS_DIR / "qso_redshift_resnet.pt"
    torch.save(model_qso.state_dict(), ckpt_qso)
    print(f"Saved QSO ResNet checkpoint to {ckpt_qso}.")

    # 5. Build Canonical QSO Template CCF Engine
    print("\n--- [4.5] Building Canonical QSO Template CCF Engine with Arm-Gap Masking ---")
    valid_mask = np.ones(len(WL_GRID), dtype=bool)
    for w_min, w_max in ARM_MASK_BRN:
        valid_mask &= ~((WL_GRID >= w_min) & (WL_GRID <= w_max))

    rest_wl = tmpl_data['rest_wl']
    qso_tpl = tmpl_data['QSO']
    cont_tpl = ndimage.median_filter(qso_tpl, size=128)
    cont_tpl = ndimage.gaussian_filter1d(cont_tpl, sigma=32)
    res_tpl = qso_tpl - cont_tpl

    z_qso_grid = model_qso.z_grid.cpu().numpy()
    M_tpl = np.zeros((len(z_qso_grid), len(WL_GRID)), dtype=np.float32)
    M_flux = np.zeros((len(z_qso_grid), len(WL_GRID)), dtype=np.float32)

    for i, z in enumerate(z_qso_grid):
        lam_rest = WL_GRID / (1.0 + z)
        f_int = np.interp(lam_rest, rest_wl, res_tpl, left=0.0, right=0.0)
        f_int[~valid_mask] = 0.0
        norm_r = np.linalg.norm(f_int)
        if norm_r > 1e-6:
            M_tpl[i] = f_int / norm_r

        fl_int = np.interp(lam_rest, rest_wl, qso_tpl, left=0.0, right=0.0)
        fl_int[~valid_mask] = 0.0
        norm_f = np.linalg.norm(fl_int)
        if norm_f > 1e-6:
            M_flux[i] = fl_int / norm_f

    # 6. Redshift Inference
    print("\n--- [4.6] Redshift Inference on Representative Dataset ---")
    model_gal.eval()
    model_qso.eval()

    loader_all = DataLoader(TensorDataset(torch.from_numpy(flux_2ch)), batch_size=256, shuffle=False)
    z_gal_all = []
    print("Running Galaxy Redshift ResNet inference...")
    with torch.no_grad():
        for (bx,) in loader_all:
            bx = bx.to(device)
            zp, _, _ = model_gal(bx)
            z_gal_all.append(zp.cpu().numpy())
    z_gal_preds = np.concatenate(z_gal_all, axis=0)

    # QSO Candidates
    qso_candidate_mask = (oof_preds == 'QSO') | (prob_qso > 0.05) | (db_classes == 'QSO')
    idx_qso_cand = np.where(qso_candidate_mask)[0]
    print(f"Running QSO Hybrid (ResNet + CCF) on N = {len(idx_qso_cand):,} QSO candidates...")

    z_qso_preds = np.zeros(n_total, dtype=np.float32)
    qso_ccf_quality = np.zeros(n_total, dtype=np.float32)

    loader_qso_cand = DataLoader(TensorDataset(torch.from_numpy(flux_2ch[idx_qso_cand])), batch_size=256, shuffle=False)
    probs_qso_cand = []
    with torch.no_grad():
        for (bx,) in loader_qso_cand:
            bx = bx.to(device)
            _, p, _ = model_qso(bx)
            probs_qso_cand.append(p.cpu().numpy())
    probs_qso_cand = np.concatenate(probs_qso_cand, axis=0)

    flux_qso_cand = flux[idx_qso_cand]
    res_qso_cand = res_boosted[idx_qso_cand] / 3.0

    test_res_list = []
    test_fl_list = []
    for f, r in zip(flux_qso_cand, res_qso_cand):
        f_m = f.copy(); f_m[~valid_mask] = 0.0
        test_fl_list.append(f_m / max(np.linalg.norm(f_m), 1e-6))
        r_m = r.copy(); r_m[~valid_mask] = 0.0
        test_res_list.append(r_m / max(np.linalg.norm(r_m), 1e-6))
    cand_res = np.array(test_res_list, dtype=np.float32)
    cand_fl = np.array(test_fl_list, dtype=np.float32)

    ccf_comb = 0.6 * np.matmul(cand_res, M_tpl.T) + 0.4 * np.matmul(cand_fl, M_flux.T)

    for i_cand, global_i in enumerate(idx_qso_cand):
        p_idx = np.argsort(ccf_comb[i_cand])[::-1]
        candidates = []
        for pi in p_idx:
            cand_z = z_qso_grid[pi]
            if not any(abs(cand_z - c['z']) < 0.08 for c in candidates):
                z_lo = max(0.0, cand_z - 0.06)
                z_hi = min(4.5, cand_z + 0.06)
                m_win = (z_qso_grid >= z_lo) & (z_qso_grid <= z_hi)
                res_prob = np.sum(probs_qso_cand[i_cand][m_win])
                ccf_val = ccf_comb[i_cand][pi]
                candidates.append({'z': cand_z, 'ccf': ccf_val, 'resnet_p': res_prob})
            if len(candidates) >= 3:
                break

        best_cand = max(candidates, key=lambda c: (max(0, c['ccf']) ** 1.0) * (c['resnet_p'] ** 0.5 + 1e-4))
        best_pi = np.argmin(np.abs(z_qso_grid - best_cand['z']))
        i_lo = max(0, best_pi - 8); i_hi = min(len(z_qso_grid), best_pi + 9)
        sub_c = np.clip(ccf_comb[i_cand][i_lo:i_hi], 0, None)
        sub_z = z_qso_grid[i_lo:i_hi]
        refined_z = np.sum(sub_c * sub_z) / max(np.sum(sub_c), 1e-8)

        z_qso_preds[global_i] = float(refined_z)
        qso_ccf_quality[global_i] = float(best_cand['ccf'])

    # 7. Assemble Redshift Predictions
    final_pred_z = np.zeros(n_total, dtype=np.float32)
    final_z_qual = np.zeros(n_total, dtype=np.float32)

    for i in range(n_total):
        cls = oof_preds[i]
        if cls == 'GALAXY':
            final_pred_z[i] = z_gal_preds[i]
            final_z_qual[i] = float(pred_confidence[i])
        elif cls == 'QSO':
            final_pred_z[i] = z_qso_preds[i]
            final_z_qual[i] = float(qso_ccf_quality[i])
        else:  # STAR
            final_pred_z[i] = 0.0
            final_z_qual[i] = 1.0

    # 8. Performance Evaluation Summary
    print("\n" + "=" * 80)
    print("                STAGE 2 REDSHIFT ACCURACY ON AGREED OBJECTS")
    print("=" * 80)
    m_gal_eval = (db_classes == 'GALAXY') & (oof_preds == 'GALAXY') & (db_redshifts > 0.005)
    dz_gal = (final_pred_z[m_gal_eval] - db_redshifts[m_gal_eval]) / (1.0 + db_redshifts[m_gal_eval])
    nmad_gal = 1.4826 * np.median(np.abs(dz_gal - np.median(dz_gal)))
    outl_gal = np.mean(np.abs(dz_gal) > 0.05) * 100
    print(f"  GALAXY (N={len(dz_gal):,}): sigma_NMAD = {nmad_gal:.4f}, Outliers (|dz|>0.05) = {outl_gal:.2f}%, Bias = {np.median(dz_gal):+.4f}")

    m_qso_eval = (db_classes == 'QSO') & (oof_preds == 'QSO') & (db_redshifts > 0.05)
    dz_qso = (final_pred_z[m_qso_eval] - db_redshifts[m_qso_eval]) / (1.0 + db_redshifts[m_qso_eval])
    nmad_qso = 1.4826 * np.median(np.abs(dz_qso - np.median(dz_qso)))
    outl_qso = np.mean(np.abs(dz_qso) > 0.05) * 100
    print(f"  QSO    (N={len(dz_qso):,}): sigma_NMAD = {nmad_qso:.4f}, Outliers (|dz|>0.05) = {outl_qso:.2f}%, Bias = {np.median(dz_qso):+.4f}")

    # 9. Save Intermediate Redshift Cache
    out_cache = RESULTS_DIR / "redshift_predictions_scratch.npz"
    print(f"\nSaving redshift predictions cache: {out_cache} ...")
    np.savez_compressed(
        out_cache,
        objIds=obj_ids,
        pred_redshift=final_pred_z,
        redshift_quality=final_z_qual,
        z_galaxy_candidate=z_gal_preds,
        z_qso_candidate=z_qso_preds
    )

    print(f"Step 4 finished in {time.time() - t_start:.1f}s.")
    print("=" * 80)


if __name__ == "__main__":
    main()
