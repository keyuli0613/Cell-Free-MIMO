# precoders/p_mmse.py
import numpy as np

def precoding_vector(serving_aps, H_hat, ue_id, rho_dist, D, lN, sigma2):
    W = {}
    # 假设 serving_aps 列表就是用户ue_id的集群
    # 我们以该集群内所有AP的天线作为一个整体进行MMSE计算
    # 构造集群信道矩阵: 包含集群内每个AP对集群内相关用户的信道
    # (简单起见，只考虑对目标用户自身和可能干扰用户的信道)
    cluster_aps = serving_aps
    # 构造目标用户在所有集群AP的信道向量堆叠
    # 维度: (cluster_size * lN, 1)
    h_ue_cluster = np.vstack([H_hat[ap, ue_id, :].reshape(lN, 1) for ap in cluster_aps])
    # 构造干扰用户在集群AP的信道矩阵 (cluster_size * lN, num_interf_users)
    # 这里干扰用户可选：例如所有其他用户，或者与目标用户共享部分AP的用户
    num_UEs = H_hat.shape[1]
    interferers = [u for u in range(num_UEs) if u != ue_id]
    H_interf = np.hstack([
        np.vstack([H_hat[ap, other, :].reshape(lN, 1) for ap in cluster_aps])
        for other in interferers
    ])  # 维度: (cluster_size*lN, num_interf)
    # 组装MMSE矩阵并求解
    R = H_interf @ H_interf.conj().T + sigma2 * np.eye(len(cluster_aps)*lN)
    w_cluster = np.linalg.solve(R, h_ue_cluster)
    # 将联合预编码向量拆分回各AP对应的子向量，并归一化功率
    for idx, ap in enumerate(cluster_aps):
        w_ap = w_cluster[idx*lN:(idx+1)*lN]  # 取出该AP天线的权重
        # 应用功率分配系数
        if rho_dist is not None:
            coeff = rho_dist[ap, ue_id] if isinstance(rho_dist, np.ndarray) else rho_dist.get((ap, ue_id), 1.0)
            w_ap = w_ap / (np.linalg.norm(w_ap) + 1e-12)
            w_ap = np.sqrt(coeff) * w_ap
        else:
            w_ap = w_ap / (np.linalg.norm(w_ap) + 1e-12)
        W[ap] = w_ap.ravel()
    return W
