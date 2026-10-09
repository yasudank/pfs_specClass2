import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import find_peaks

data = np.load("results/dataset_full_augmented_scratch.npz", allow_pickle=True)
flux = data["flux_clean"]
classes = data["classes"]
obj_ids = data["objId"]

rf_data = np.load("results/rf_oof_predictions_216k.npz", allow_pickle=True)
meta = np.load("results/survey_meta_216k_final.npz", allow_pickle=True)
rf_preds = rf_data["oof_preds"]
db_oids = meta["objIds"]

oid_to_rf = {db_oids[i]: rf_preds[i] for i in range(len(db_oids))}

def extract_line_features(f):
    # 1. Continuum estimation (median filter size 128 ~ 27 nm)
    cont = ndimage.median_filter(f, size=128)
    cont = ndimage.gaussian_filter1d(cont, sigma=32)
    res = f - cont

    # 2. Positive & Negative energy
    pos_res = np.maximum(0.0, res)
    neg_res = np.maximum(0.0, -res)
    pos_energy = np.sum(pos_res)
    neg_energy = np.sum(neg_res)
    energy_ratio = pos_energy / (neg_energy + 1e-4)

    # 3. Peak detection on positive residual
    peaks, props = find_peaks(res, height=0.3, distance=15, prominence=0.25, width=1)
    n_peaks = len(peaks)
    
    if n_peaks > 0:
        heights = props["peak_heights"]
        widths = props["widths"]
        order = np.argsort(heights)[::-1]
        top_h = heights[order][:3]
        top_w = widths[order][:3]
        max_h = float(top_h[0])
        max_w = float(top_w[0])
        mean_w = float(np.mean(top_w))
    else:
        max_h, max_w, mean_w = 0.0, 0.0, 0.0

    # 4. Multi-scale power: broad (smoothed by 12 bins ~ 2.5 nm) vs narrow
    broad_res = ndimage.gaussian_filter1d(pos_res, sigma=12)
    broad_power = np.sum(broad_res**2)
    narrow_power = np.sum(pos_res**2)
    broad_ratio = broad_power / (narrow_power + 1e-4)

    return {
        "pos_energy": pos_energy,
        "energy_ratio": energy_ratio,
        "max_h": max_h,
        "max_w": max_w,
        "mean_w": mean_w,
        "broad_ratio": broad_ratio,
        "n_peaks": n_peaks
    }

groups = {
    "Agreed QSO": (classes == "QSO") & np.array([oid_to_rf.get(oid) == "QSO" for oid in obj_ids]),
    "Missed QSO (DB=QSO, RF=GAL)": (classes == "QSO") & np.array([oid_to_rf.get(oid) == "GALAXY" for oid in obj_ids]),
    "False QSO (DB=GAL, RF=QSO)": (classes == "GALAXY") & np.array([oid_to_rf.get(oid) == "QSO" for oid in obj_ids]),
    "Agreed Galaxy": (classes == "GALAXY") & np.array([oid_to_rf.get(oid) == "GALAXY" for oid in obj_ids]),
    "Agreed Star": (classes == "STAR") & np.array([oid_to_rf.get(oid) == "STAR" for oid in obj_ids]),
}

header = f"{'Group':<30} | {'Pos Energy':>10} | {'Max Peak H':>10} | {'Peak Width':>10} | {'Broad Ratio':>11}"
print(header)
print("-" * len(header))
for name, m in groups.items():
    idx_list = np.where(m)[0][:150]
    feats = [extract_line_features(flux[i]) for i in idx_list]
    pe = np.median([x["pos_energy"] for x in feats])
    mh = np.median([x["max_h"] for x in feats])
    mw = np.median([x["mean_w"] for x in feats])
    br = np.median([x["broad_ratio"] for x in feats])
    print(f"{name:<30} | {pe:>10.1f} | {mh:>10.2f} | {mw:>10.1f} | {br:>11.3f}")
