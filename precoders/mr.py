# precoders/mr.py

import numpy as np

def precoding_vector(serving_aps, H_hat, ue_id, rho_dist, D, lN, sigma2):
    W = {}
    for ap in serving_aps:
        ap_id = ap.id
        h = H_hat[ap_id, :, ue_id]
        w = h * np.sqrt(rho_dist[ap_id, ue_id])
        W[ap_id] = w
    return W
