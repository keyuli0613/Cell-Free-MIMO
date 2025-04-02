# downlink.py

import numpy as np
from scipy.linalg import inv, pinv
from config import NOISE_DL, RHO_TOT_PER_AP

def compute_downlink_SE(serving_aps, H_hat, H_true, ue_id, all_ues,
                        lN, p, sigma2, precoding_scheme='MR',
                        rho_dist=None, D=None, pilot_assignments=None, all_aps_global=None):
    """
    计算下行链路频谱效率（SE）和总能耗（各AP预编码向量能量总和）。
    预编码方案支持：
      - 'MR'：最大比预编码
      - 'L-MMSE'：本地 MMSE 预编码（排除目标 UE 贡献，用于干扰抑制）
      - 'P-MMSE'：部分 MMSE 预编码（低复杂度近似全局 MMSE）
      - 'ZF'：零逼近预编码（利用伪逆进行干扰消除）
      
    输入参数：
      serving_aps: 为目标 UE 服务的 AP 对象列表
      H_hat: 信道估计矩阵，形状 (NUM_AP, lN, NUM_UE)
      H_true: 真实信道矩阵，形状 (NUM_AP, lN, NUM_UE)
      ue_id: 目标 UE 的 id（整数）
      all_ues: 所有 UE 对象列表
      lN: 每个 AP 的天线数
      p: 发射功率（用于预编码计算）
      sigma2: 噪声功率（NOISE_DL）
      precoding_scheme: 预编码方案，取值 'MR'、'L-MMSE'、'P-MMSE' 或 'ZF'
      rho_dist: 下行功率分配矩阵，形状 (NUM_AP, NUM_UE)
      D: 服务矩阵 (NUM_AP, NUM_UE)，D[l, k]==1 表示 AP l 为 UE k 服务
      pilot_assignments: （暂未使用）
      all_aps_global: 若提供，则用于计算全局干扰，否则只用 serving_aps
      
    输出：
      SE: 下行频谱效率（bit/s/Hz）
      total_energy: 该 UE 下行传输所消耗的总能耗（所有服务AP预编码向量能量之和）
    """
    if len(serving_aps) == 0:
        return 0.0, 0.0

    # 干扰计算：如果提供全局AP列表，则使用之；否则仅使用 serving_aps
    if all_aps_global is not None:
        all_aps_for_interference = all_aps_global
    else:
        all_aps_for_interference = serving_aps

    # 预编码向量字典，W[ap_id] 保存该 AP 针对目标 UE 的预编码向量（shape: (lN,)）
    W = {}

    # 根据不同预编码方案分别计算预编码向量
    if precoding_scheme == 'MR':
        for ap in serving_aps:
            ap_id = ap.id
            # MR预编码：直接使用目标 UE 的信道估计向量，再乘以下行功率分配系数的平方根
            w = H_hat[ap_id, :, ue_id] * np.sqrt(rho_dist[ap_id, ue_id])
            W[ap_id] = w
    # elif precoding_scheme == 'L-MMSE':
    #     for ap in serving_aps:
    #         ap_id = ap.id
    #         # 得到 AP l 服务的所有 UE 索引
    #         served_ues = np.where(D[ap_id, :] == 1)[0]
    #     # 排除目标 UE 自身，构造干扰部分协方差矩阵
    #         interfering_ues = [k for k in served_ues if k != ue_id]
    #         if interfering_ues:
    #             C_tmp = sum(np.outer(H_hat[ap_id, :, k], H_hat[ap_id, :, k].conj())
    #                     for k in interfering_ues)
    #         else:
    #             C_tmp = np.zeros((lN, lN), dtype=complex)
    #     # 添加固定正则化项，避免矩阵奇异
    #         eps = 1e-3  # 固定值，替换动态计算的 eps
    #         C_total = C_tmp + (sigma2 + eps) * np.eye(lN)
    #         # 使用伪逆代替 inv
    #         w = np.linalg.pinv(C_total) @ H_hat[ap_id, :, ue_id]
    #         w *= np.sqrt(rho_dist[ap_id, ue_id])
    #         W[ap_id] = w
    #     # 调试信息
    #         print(f"AP {ap_id}: interfering_ues = {len(interfering_ues)}, trace_C_tmp = {np.trace(C_tmp):.4f}")
    elif precoding_scheme == 'L-MMSE-OLD':
        for ap in serving_aps:
            ap_id = ap.id
            # 得到 AP l 服务的所有 UE 索引
            served_ues = np.where(D[ap_id, :] == 1)[0]
            # 排除目标 UE 自身，构造干扰部分协方差矩阵
            interfering_ues = [k for k in served_ues if k != ue_id]
            if interfering_ues:
                C_tmp = sum(np.outer(H_hat[ap_id, :, k], H_hat[ap_id, :, k].conj())
                            for k in interfering_ues)
            else:
                C_tmp = np.zeros((lN, lN), dtype=complex)
            # 设置正则化参数，调低正则化强度
            trace_C = np.trace(C_tmp) / lN if np.trace(C_tmp) > 0 else 1e-3
            eps = 1e-2 * trace_C
            C_total = C_tmp + (sigma2 + eps) * np.eye(lN)
            try:
                w = inv(C_total) @ H_hat[ap_id, :, ue_id]
                w *= np.sqrt(rho_dist[ap_id, ue_id])
            except np.linalg.LinAlgError as e:
                print(f"AP {ap_id} inversion failed in L-MMSE: {str(e)}")
                w = np.zeros(lN, dtype=complex)
            W[ap_id] = w
    elif precoding_scheme == 'L-MMSE':
        for ap in serving_aps:
            ap_id = ap.id
        # 找出该 AP 服务的所有 UE 索引
            served_ues = np.where(D[ap_id, :] == 1)[0]
        # 提取 AP ap_id 对所有服务 UE 的信道估计，得到矩阵 Hhatallj，尺寸为 (lN, number_of_served_UEs)
            Hhatallj = H_hat[ap_id, :, :][:, served_ues]
        
        # 如果有估计误差协方差矩阵 C，则计算服务 UE 的协方差之和；否则设为零矩阵
            try:
            # 假设 C 的尺寸为 (num_AP, lN, lN, num_UE)
                Cserved = sum(C[ap_id, :, :, k] for k in served_ues)
            except NameError:
                Cserved = np.zeros((lN, lN), dtype=complex)
        
        # 构造总矩阵 A = p*(Hhatallj * Hhatallj^H) + p*(Cserved) + I
            A = p * (Hhatallj @ Hhatallj.conj().T) + p * Cserved + np.eye(lN)
        
            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError as e:
                print(f"AP {ap_id} inversion failed in L-MMSE-MATLAB: {str(e)}")
                A_inv = np.zeros((lN, lN), dtype=complex)
        
        # 计算联合 L-MMSE 预编码（或组合）矩阵：V_L_MMSE = p * A_inv * Hhatallj
            V_L_MMSE = p * (A_inv @ Hhatallj)
        
        # 存储 AP ap_id 对所有服务 UE 的联合预编码矩阵，每一列对应一个 UE 的组合向量
            W[ap_id] = V_L_MMSE


    elif precoding_scheme == 'P-MMSE':
        # 调用 P-MMSE 预编码函数（独立实现，参考书中公式6.33）
        num_AP = H_hat.shape[0]
        num_UE = H_hat.shape[2]
        noiseMat = sigma2 * np.eye(lN)
        V_P_MMSE = p_mmse_precoder(H_hat, None, D, num_AP, num_UE, p, noiseMat)
        for ap in serving_aps:
            ap_id = ap.id
            W[ap_id] = V_P_MMSE[ap_id, ue_id, :]

    elif precoding_scheme == 'ZF':
        for ap in serving_aps:
            ap_id = ap.id
            served_ues = np.where(D[ap_id, :] == 1)[0]
            if len(served_ues) == 0:
                W[ap_id] = np.zeros(lN, dtype=complex)
            else:
                # 计算伪逆，得到 ZF 预编码向量
                Hhat_served = H_hat[ap_id, :, served_ues]  # shape (lN, num_served)
                pinv_H = np.linalg.pinv(Hhat_served)
                try:
                    idx = np.where(served_ues == ue_id)[0][0]
                    w = pinv_H[idx, :] * np.sqrt(rho_dist[ap_id, ue_id])
                except IndexError:
                    w = np.zeros(lN, dtype=complex)
                W[ap_id] = w
    else:
        for ap in serving_aps:
            ap_id = ap.id
            W[ap_id] = np.zeros(lN, dtype=complex)

    # 每个AP独立归一化，确保发射功率不超过单个AP功率约束
    for ap in serving_aps:
        ap_id = ap.id
        power = np.linalg.norm(W.get(ap_id, np.zeros(lN)))**2
        if power > RHO_TOT_PER_AP:
            scaling_factor = np.sqrt(RHO_TOT_PER_AP / power)
            W[ap_id] = W[ap_id] * scaling_factor

    # ===== 信号与干扰计算 =====
    total_energy = 0.0
    signal = 0j
    interference = 0.0

    # 信号项：仅由服务该 UE 的 AP 贡献
    for ap in serving_aps:
        ap_id = ap.id
        w = W.get(ap_id, np.zeros(lN, dtype=complex))
        h_target = H_true[ap_id, :, ue_id]
        signal += np.vdot(w, h_target)

    # 干扰项：遍历所有 AP（全局）计算对非目标UE的干扰贡献
    for ap in all_aps_global if all_aps_global is not None else serving_aps:
        ap_id = ap.id
        w = W.get(ap_id, np.zeros(lN, dtype=complex))
        for other_ue in all_ues:
            if other_ue.id == ue_id:
                continue
            h_interf = H_true[ap_id, :, other_ue.id]
            interference += np.abs(np.vdot(w, h_interf))**2

    # 累计能耗：所有 AP 的预编码向量能量之和
    for ap in serving_aps:
        ap_id = ap.id      
        total_energy += np.linalg.norm(W.get(ap_id, np.zeros(lN, dtype=complex)))**2

    SINR = np.abs(signal)**2 / (interference + sigma2)
    SE = np.log2(1 + SINR)
    return SE, total_energy

def p_mmse_precoder(hhat, C, D, num_AP, num_UE, p, noiseMat):
    """
    计算 P-MMSE 预编码向量矩阵。
    输入:
      hhat: 信道估计矩阵，形状 (num_AP, lN, num_UE)
      C: （暂不使用，可传入 None）
      D: 服务矩阵 (num_AP, num_UE)，D[l,k]==1 表示 AP l 服务 UE k
      num_AP: AP 数量
      num_UE: UE 数量
      p: 发射功率
      noiseMat: 噪声协方差矩阵，形状 (lN, lN)
    输出:
      V_P_MMSE: 预编码矩阵，形状 (num_AP, num_UE, lN)
                即每个 AP 对每个 UE 有一个预编码向量（长度 lN）
    """
    lN = hhat.shape[1]
    V_P_MMSE = np.zeros((num_AP, num_UE, lN), dtype=complex)
    for l in range(num_AP):
        served_indices = np.where(D[l, :] == 1)[0]
        if served_indices.size == 0:
            continue
        Hhat_l = hhat[l, :, :]  # shape: (lN, num_UE)
        Hhat_l_served = Hhat_l[:, served_indices]  # shape: (lN, num_served)
        sum_in = p * (Hhat_l_served @ Hhat_l_served.conj().T)
        try:
            X = np.linalg.solve(sum_in + noiseMat, Hhat_l_served)
        except np.linalg.LinAlgError as e:
            print(f"AP {l} inversion failed in P-MMSE: {str(e)}")
            X = np.zeros_like(Hhat_l_served)
        X = p * X  # 放大因子
        # X 的转置形状为 (num_served, lN)，将其赋给对应的服务 UE
        V_P_MMSE[l, served_indices, :] = X.T
    return V_P_MMSE

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