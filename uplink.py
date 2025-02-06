# uplink.py
import numpy as np
def compute_uplink_SE_advanced(serving_aps, H_hat, H_true, ue_id, all_ues, lN, p, sigma2, tau_c, tau_p):
    """计算考虑干扰和噪声的上行SE"""
    if len(serving_aps) == 0:
        return 0.0
    
    # 构建MR合并向量
    v = np.concatenate([H_hat[ap.id, :, ue_id] for ap in serving_aps])
    h_true = np.concatenate([H_true[ap.id, :, ue_id] for ap in serving_aps])
    
    # 信号功率
    signal = p * np.abs(np.vdot(v, h_true))**2
    
    # 干扰项（其他UE的贡献）
    interference = 0.0
    for other_ue in all_ues:
        if other_ue.id == ue_id:
            continue
        h_interf = np.concatenate([H_true[ap.id, :, other_ue.id] for ap in serving_aps])
        interference += p * np.abs(np.vdot(v, h_interf))**2
    
    # 噪声项
    noise = sigma2 * np.linalg.norm(v)**2
    
    # SINR计算
    SINR = signal / (interference + noise)
    prelog = (tau_c - tau_p) / tau_c
    SE = prelog * np.log2(1 + SINR)
    return SE