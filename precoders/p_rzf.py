# precoders/p_rzf.py
import numpy as np

# 可以根据需要从配置中获取正则化系数alpha，如果没有特殊设定，可将alpha设为sigma2（退化为MMSE）
alpha = None  # 可以在函数调用前由外部配置

def precoding_vector(serving_aps, H_hat, ue_id, rho_dist, D, lN, sigma2):
    W = {}
    cluster_aps = serving_aps
    # 构造和P-MMSE类似的集群信道矩阵（参见上面的p_mmse实现）
    # ... (略) 构造 h_ue_cluster 和 H_interf 矩阵 ...
    # 设置正则化系数
    reg_coeff = alpha if alpha is not None else sigma2
    # 计算RZF矩阵并求解
    R = H_interf @ H_interf.conj().T + reg_coeff * np.eye(len(cluster_aps)*lN)
    w_cluster = np.linalg.solve(R, h_ue_cluster)
    # 拆分为每个AP的向量并归一化
    for idx, ap in enumerate(cluster_aps):
        w_ap = w_cluster[idx*lN:(idx+1)*lN]
        if rho_dist is not None:
            coeff = rho_dist.get((ap, ue_id), 1.0) if isinstance(rho_dist, dict) else rho_dist[ap, ue_id]
            w_ap = w_ap / (np.linalg.norm(w_ap) + 1e-12)
            w_ap = np.sqrt(coeff) * w_ap
        else:
            w_ap = w_ap / (np.linalg.norm(w_ap) + 1e-12)
        W[ap] = w_ap.ravel()
    return W
