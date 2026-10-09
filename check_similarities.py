import numpy as np
from config import RESULTS_DIR

pred_data = np.load(RESULTS_DIR / "rf_oof_boost3x_smote_representative.npz", allow_pickle=True)
embeds = np.load(RESULTS_DIR / "embeddings_representative_boost3x.npy")
db_classes = pred_data["db_classes"]
oof_preds = pred_data["oof_preds"]

def norm(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

embeds_norm = norm(embeds)

m_agreed_qso = (db_classes == "QSO") & (oof_preds == "QSO")
m_agreed_gal = (db_classes == "GALAXY") & (oof_preds == "GALAXY")
m_agreed_star = (db_classes == "STAR") & (oof_preds == "STAR")
m_missed_gal = (db_classes == "QSO") & (oof_preds == "GALAXY")
m_missed_star = (db_classes == "QSO") & (oof_preds == "STAR")
m_new_qso = (db_classes == "GALAXY") & (oof_preds == "QSO")

proto_qso = norm(np.mean(embeds_norm[m_agreed_qso], axis=0, keepdims=True))[0]
proto_gal = norm(np.mean(embeds_norm[m_agreed_gal], axis=0, keepdims=True))[0]
proto_star = norm(np.mean(embeds_norm[m_agreed_star], axis=0, keepdims=True))[0]

groups = {
    "Agreed QSOs": m_agreed_qso,
    "Remaining Missed QSOs -> GAL": m_missed_gal,
    "Remaining Missed QSOs -> STAR": m_missed_star,
    "New QSO Candidates": m_new_qso,
    "Agreed Galaxies": m_agreed_gal,
    "Agreed Stars": m_agreed_star
}

print("{:<32} | {:>6} | {:>9} | {:>9} | {:>9}".format("Group", "N", "Sim(QSO)", "Sim(GAL)", "Sim(STAR)"))
print("-" * 75)
for name, m in groups.items():
    sub_e = embeds_norm[m]
    s_q = np.mean(np.dot(sub_e, proto_qso))
    s_g = np.mean(np.dot(sub_e, proto_gal))
    s_s = np.mean(np.dot(sub_e, proto_star))
    print("{:<32} | {:>6} | {:>9.4f} | {:>9.4f} | {:>9.4f}".format(name, int(np.sum(m)), s_q, s_g, s_s))
