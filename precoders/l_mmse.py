# precoders/l_mmse.py

import numpy as np

def precoding_vector(serving_aps, H_hat, ue_id, rho_dist, D, lN, sigma2):
    W = {}
    for ap in serving_aps:
        ap_id = ap.id
        served_ues = np.where(D[ap_id, :] == 1)[0]
        interfering_ues = [k for k in served_ues if k != ue_id]

        C_tmp = np.zeros((lN, lN), dtype=complex)
        for k in interfering_ues:
            h = H_hat[ap_id, :, k]
            C_tmp += np.outer(h, h.conj())

        trace_C = np.trace(C_tmp) / lN if np.trace(C_tmp) > 0 else 1e-3
        eps = 1e-2 * trace_C
        C_total = C_tmp + (sigma2 + eps) * np.eye(lN)

        try:
            w = np.linalg.inv(C_total) @ H_hat[ap_id, :, ue_id]
            w *= np.sqrt(rho_dist[ap_id, ue_id])
        except np.linalg.LinAlgError:
            w = np.zeros(lN, dtype=complex)

        W[ap_id] = w
    return W
