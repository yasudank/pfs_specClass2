"""Extract embeddings for unboosted (alpha=1.0) on representative dataset and compare with boosted (alpha=3.0).
"""
import os
import sys
import numpy as np
import scipy.ndimage as ndimage
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from concurrent.futures import ProcessPoolExecutor
from config import RESULTS_DIR
from models.contrastive_model import SpectralSimCLR


def compute_res_fn(f):
    c = ndimage.median_filter(f, size=128)
    c = ndimage.gaussian_filter1d(c, sigma=32)
    return (f - c).astype(np.float32)


def main():
    data = np.load(RESULTS_DIR / 'dataset_full_augmented_scratch.npz', allow_pickle=True)
    all_flux = data['flux_clean']
    all_classes = data['classes']
    obj_ids = data['objId']
    _, u_idx = np.unique(obj_ids, return_index=True)
    flux = all_flux[u_idx]
    classes_true = all_classes[u_idx]
    class_names = ['GALAXY', 'QSO', 'STAR']
    cls_map = {c: i for i, c in enumerate(class_names)}
    y_true = np.array([cls_map[c] for c in classes_true])

    print("Computing unboosted residuals...")
    with ProcessPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(compute_res_fn, flux, chunksize=256))
    res = np.array(res, dtype=np.float32)
    flux_2ch_unboost = np.stack([flux, res], axis=1).astype(np.float32)

    device = torch.device('cuda')
    model_unboost = SpectralSimCLR(in_channels=2, embed_dim=128, proj_dim=64).to(device)
    model_unboost.load_state_dict(torch.load(RESULTS_DIR / 'spectral_simclr_2channel.pt', map_location=device, weights_only=True))
    model_unboost.eval()

    eval_loader = DataLoader(TensorDataset(torch.from_numpy(flux_2ch_unboost)), batch_size=256, shuffle=False)
    embeds_unboost = []
    with torch.no_grad():
        for (bx,) in eval_loader:
            embeds_unboost.append(model_unboost.encode(bx.to(device)).cpu().numpy())
    embeds_unboost = np.vstack(embeds_unboost)

    # Load boosted embeddings
    embeds_boost = np.load(RESULTS_DIR / 'embeddings_representative_boost3x.npy')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def run_eval(emb, name):
        preds = np.zeros(len(y_true), dtype=int)
        for tr, val in skf.split(emb, y_true):
            rf = RandomForestClassifier(n_estimators=100, max_depth=16, class_weight='balanced_subsample', random_state=42, n_jobs=16)
            rf.fit(emb[tr], y_true[tr])
            preds[val] = rf.predict(emb[val])
        print(f"\n================ {name} ================")
        print(classification_report(y_true, preds, target_names=class_names, digits=4))
        return preds

    run_eval(embeds_unboost, "Unboosted (alpha=1.0) 2-Channel SimCLR")
    run_eval(embeds_boost, "Boosted (alpha=3.0) 2-Channel SimCLR")


if __name__ == "__main__":
    main()
