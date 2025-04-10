import numpy as np

def compute_downlink_SE(serving_aps, ue_id, all_ues,
                        lN, sigma2,
                        rho_dist, pilot_assignments,
                        beta_matrix, gamma_matrix, Delta, tau_sl,
                        tau_c, tau_p,
                        precoding_scheme='PPZF'):
    """
    根据 PPZF 方案及文献公式计算目标 UE 的下行频谱效率（SE）和能耗。
    """
    num_AP = rho_dist.shape[0]
    num_UE = rho_dist.shape[1]

    # 精简调试信息
    print("[DEBUG] Matrix Stats:")
    print(f"  beta_matrix:     min={beta_matrix.min():.2e}, max={beta_matrix.max():.2e}, mean={beta_matrix.mean():.2e}")
    print(f"  gamma_matrix:    min={gamma_matrix.min():.2e}, max={gamma_matrix.max():.2e}, mean={gamma_matrix.mean():.2e}")
    print(f"  rho_dist:        min={rho_dist.min():.2e}, max={rho_dist.max():.2e}, mean={rho_dist.mean():.2e}")

    # 确定目标 UE 所属导频组
    pilot_k = pilot_assignments[ue_id]
    pilot_group = [ue for ue in all_ues if pilot_assignments.get(ue.id) == pilot_k]

    # 信号项 S_k
    S_k = 0.0
    for l in range(num_AP):
        term = (lN - tau_sl[l]) * rho_dist[l, ue_id] * gamma_matrix[l, ue_id]
        S_k += np.sqrt(np.maximum(term, 0))

    # 干扰项 I1
    I1 = 0.0
    for ue in pilot_group:
        if ue.id == ue_id:
            continue
        temp = 0.0
        for l in range(num_AP):
            term = (lN - tau_sl[l]) * rho_dist[l, ue.id] * gamma_matrix[l, ue_id]
            temp += np.sqrt(np.maximum(term, 0))
        I1 += temp ** 2

    # 干扰项 I2
    I2 = 0.0
    for t in range(num_UE):
        for l in range(num_AP):
            term = beta_matrix[l, ue_id] - Delta[l, ue_id] * gamma_matrix[l, ue_id]
            I2 += rho_dist[l, t] * np.maximum(term, 0)

    SINR = (S_k ** 2) / (I1 + I2 + sigma2)
    prelog = (tau_c - tau_p) / tau_c
    SE = prelog * np.log2(1 + SINR)
    total_energy = np.sum(rho_dist[:, ue_id])

    # 精简但清晰的结果输出
    print(f"[DEBUG] S_k^2={S_k**2:.2e}, I1={I1:.2e}, I2={I2:.2e}, SINR={SINR:.2e}, SE={SE:.4f}, Energy={total_energy:.2e}")

    return SE, total_energy



def power_allocation(gain_matrix, D, RHO_TOT_PER_AP):
    """
    按式(6.36)实现功率分配：采用平方根加权分配方法。
    输入：
      gain_matrix: (L, K) 大尺度衰落系数矩阵（例如使用 trace(R) 作为 beta）
      D: (L, K) 服务矩阵，D[l,k]==1 表示 AP l 服务 UE k
      rho_tot: 每个 AP 的总下行功率约束（RHO_TOT）
    输出：
      rho: (L, K) 功率分配矩阵
    """
    L, K = gain_matrix.shape
    rho = np.zeros((L, K))
    for l in range(L):
        served_ues = np.where(D[l, :] == 1)[0]
        if len(served_ues) == 0:
            continue
        sqrt_gain = np.sqrt(np.maximum(gain_matrix[l, served_ues], 1e-10))
        total = np.sum(sqrt_gain)
        if total > 0:
            rho[l, served_ues] = RHO_TOT_PER_AP * sqrt_gain / total
    return rho