import numpy as np

def compute_downlink_SE(serving_aps, ue_id, all_ues,
                        lN, sigma2,
                        rho_dist, pilot_assignments,
                        beta_matrix, gamma_matrix, Delta, tau_sl,
                        tau_c, tau_p,
                        precoding_scheme='PPZF'):
    """
    根据 PPZF 方案及文献公式计算目标 UE 的下行频谱效率（SE）和能耗。

    输入参数：
      serving_aps: 所有 AP（列表，遍历所有 AP用于计算），注意 rho_dist 中非服务 UE的分量应为 0
      ue_id: 目标 UE 的 id
      all_ues: 所有 UE 对象列表
      lN: 每个 AP 的天线数（即 M_l）
      sigma2: 噪声功率
      rho_dist: 下行功率分配矩阵，形状 (num_AP, num_UE)
      pilot_assignments: 字典 {ue_id: pilot_index}，记录各 UE 的导频分配
      beta_matrix: (num_AP, num_UE) 大尺度衰落系数矩阵
      gamma_matrix: (num_AP, num_UE) 信道估计质量矩阵（线性量度）
      Delta: (num_AP, num_UE) 二值矩阵，指示 UE 是否为 AP 的强信道用户（1：强信道，0：弱信道）
      tau_sl: 长度为 num_AP 的数组，每个 AP 对应的强信道 UE 数（tau_{S_l}）
      tau_c: 系统载波块长度
      tau_p: 导频数
      precoding_scheme: 此处固定采用 'PPZF'
      
    输出：
      SE: 下行频谱效率，单位 bit/s/Hz
      total_energy: 目标 UE 的下行传输总能耗（所有 AP分配功率之和）
    """
    num_AP = rho_dist.shape[0]
    num_UE = rho_dist.shape[1]
    
    # 确定目标 UE所属的导频组 P_k
    pilot_k = pilot_assignments[ue_id]
    pilot_group = [ue for ue in all_ues if pilot_assignments.get(ue.id) == pilot_k]
    
    # 信号项 S_k：累加所有 AP 的贡献 (M_l - tau_sl[l]) * rho_dist[l, ue_id] * gamma_matrix[l, ue_id]
    S_k = 0.0
    for l in range(num_AP):
        S_k += (lN - tau_sl[l]) * rho_dist[l, ue_id] * gamma_matrix[l, ue_id]
    
    # 干扰项 I1：同一导频内除目标 UE 外，其它 UE 的干扰贡献（先在每个 AP 累加再平方求和）
    I1 = 0.0
    for ue in pilot_group:
        if ue.id == ue_id:
            continue
        temp = 0.0
        for l in range(num_AP):
            temp += (lN - tau_sl[l]) * rho_dist[l, ue.id] * gamma_matrix[l, ue_id]
        I1 += temp ** 2

    # 干扰项 I2：所有 UE 对目标 UE 的残余干扰
    I2 = 0.0
    for t in range(num_UE):
        for l in range(num_AP):
            I2 += rho_dist[l, t] * (beta_matrix[l, ue_id] - Delta[l, ue_id] * gamma_matrix[l, ue_id])
    
    SINR = (S_k ** 2) / (I1 + I2 + sigma2)
    prelog = (tau_c - tau_p) / tau_c
    SE = prelog * np.log2(1 + SINR)
    
    # 此处能耗定义为各 AP 对 UE 的分配功率之和
    total_energy = np.sum(rho_dist[:, ue_id])
    
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