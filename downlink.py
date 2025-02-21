# downlink.py

import numpy as np
from scipy.linalg import inv
from config import NOISE_DL, RHO_TOT_PER_AP

def compute_downlink_SE(serving_aps, H_hat, H_true, ue_id, all_ues,
                        lN, p, sigma2, precoding_scheme='MR',
                        rho_dist=None, D=None, pilot_assignments=None, all_aps_global=None):
    """
    计算下行链路频谱效率（SE）。
    预编码方案包括：
      - 'MR'：最大比预编码
      - 'L-MMSE'：本地 MMSE 预编码（排除目标 UE 的贡献）
      - 'P-MMSE'：部分 MMSE 预编码（低复杂度近似全局 MMSE）
    
    输入参数：
      serving_aps: 为目标 UE 提供服务的 AP 列表（对象列表）
      H_hat: 信道估计矩阵，形状 (NUM_AP, lN, NUM_UE)
      H_true: 真实信道矩阵，形状 (NUM_AP, lN, NUM_UE)
      ue_id: 目标 UE 的 id（整数）
      all_ues: 所有 UE 对象列表
      lN: 每个 AP 的天线数
      p: 发射功率（用于预编码计算）
      sigma2: 噪声功率（NOISE_DL）
      precoding_scheme: 字符串，取值 'MR'、'L-MMSE' 或 'P-MMSE'
      rho_dist: 下行功率分配矩阵，形状 (NUM_AP, NUM_UE)
      D: 服务矩阵 (NUM_AP, NUM_UE)，D[l,k]==1 表示 AP l 服务 UE k
      pilot_assignments: 可选，目前未使用
      all_aps_global: 若提供，则用于干扰计算（遍历全局 AP）
      
    输出：
      SE: 下行频谱效率（bit/s/Hz）
    """
    if len(serving_aps) == 0:
        return 0.0

    # 对于干扰计算：如果提供全局AP列表，则使用全局AP，否则仅使用 serving_aps
    if all_aps_global is not None:
        all_aps_for_interference = all_aps_global
    else:
        all_aps_for_interference = serving_aps

    # 预编码向量字典，W[ap_id] 保存该 AP 针对目标 UE 的预编码向量（shape: (lN,)）
    W = {}

    if precoding_scheme == 'MR':
        for ap in serving_aps:
            ap_id = ap.id
            # MR预编码：直接使用目标 UE 的信道估计向量，乘以功率分配平方根
            w = H_hat[ap_id, :, ue_id] * np.sqrt(rho_dist[ap_id, ue_id])
            W[ap_id] = w  # shape (lN,)
    elif precoding_scheme == 'L-MMSE':
        for ap in serving_aps:
            ap_id = ap.id
            # 得到该 AP 服务的 UE 索引（服务矩阵 D 为 (NUM_AP, NUM_UE)）
            served_ues = np.where(D[ap_id, :] == 1)[0]
            # 排除目标 UE 自身，构造干扰部分的协方差矩阵
            interfering_ues = [k for k in served_ues if k != ue_id]
            if interfering_ues:
                C_tmp = sum(
                    np.outer(H_hat[ap_id, :, k], H_hat[ap_id, :, k].conj())
                    for k in interfering_ues
                )
            else:
                C_tmp = np.zeros((lN, lN), dtype=complex)
            # 计算矩阵迹，并设置较低正则化参数（这里 eps 调低）
            trace_C = np.trace(C_tmp) / lN if np.trace(C_tmp) > 0 else 1e-3
            eps = 1e-4 * trace_C
            C_total = C_tmp + (sigma2 + eps) * np.eye(lN)
            try:
                w = inv(C_total) @ H_hat[ap_id, :, ue_id]
                w *= np.sqrt(rho_dist[ap_id, ue_id])
            except np.linalg.LinAlgError as e:
                print(f"AP {ap_id} inversion failed in L-MMSE: {str(e)}")
                w = np.zeros(lN, dtype=complex)
            W[ap_id] = w
    elif precoding_scheme == 'P-MMSE':
        # 调用 P-MMSE 预编码函数，注意此函数返回形状 (num_AP, num_UE, lN)
        num_AP = H_hat.shape[0]
        num_UE = H_hat.shape[2]
        # noiseMat: 使用 sigma2 * I
        noiseMat = sigma2 * np.eye(lN)
        V_P_MMSE = p_mmse_precoder(H_hat, None, D, num_AP, num_UE, p, noiseMat)
        for ap in serving_aps:
            ap_id = ap.id
            W[ap_id] = V_P_MMSE[ap_id, ue_id, :]
    else:
        for ap in serving_aps:
            ap_id = ap.id
            W[ap_id] = np.zeros(lN, dtype=complex)

    # 每个AP独立归一化，确保其发射功率不超过 RHO_TOT_PER_AP
    for ap in serving_aps:
        ap_id = ap.id
        power = np.linalg.norm(W.get(ap_id, np.zeros(lN)))**2
        if power > RHO_TOT_PER_AP:
            scaling_factor = np.sqrt(RHO_TOT_PER_AP / power)
            W[ap_id] = W[ap_id] * scaling_factor

    # ===== 信号与干扰计算 =====
    signal = 0j
    interference = 0.0

    # 信号项：仅由服务该 UE 的 AP 贡献
    for ap in serving_aps:
        ap_id = ap.id
        w = W.get(ap_id, np.zeros(lN, dtype=complex))
        h_target = H_true[ap_id, :, ue_id]
        signal += np.vdot(w, h_target)

    # 干扰项：遍历所有AP（全局），累加其他UE 的干扰贡献
    for ap in all_aps_for_interference:
        ap_id = ap.id
        w = W.get(ap_id, np.zeros(lN, dtype=complex))
        for other_ue in all_ues:
            if other_ue.id == ue_id:
                continue
            h_interf = H_true[ap_id, :, other_ue.id]
            interference += np.abs(np.vdot(w, h_interf))**2

    SINR = np.abs(signal)**2 / (interference + sigma2)
    SE = np.log2(1 + SINR)
    return SE

def p_mmse_precoder(hhat, C, D, num_AP, num_UE, p, noiseMat):
    """
    计算 P-MMSE 预编码向量矩阵。
    输入:
      hhat: 信道估计矩阵，形状 (num_AP, lN, num_UE)
      C: （暂不使用，可传入 None）
      D: 服务矩阵 (num_AP, num_UE)，若 D[l,k]==1 表示 AP l 服务 UE k
      num_AP: AP 数量
      num_UE: UE 数量
      p: 发射功率
      noiseMat: 噪声协方差矩阵，形状 (lN, lN)
    输出:
      V_P_MMSE: 预编码矩阵，形状 (num_AP, num_UE, lN)
                即每个 AP 对每个 UE 有一个预编码向量（长度 lN）
    """
    # 这里我们从 hhat 中获取天线数（lN）
    lN = hhat.shape[1]
    V_P_MMSE = np.zeros((num_AP, num_UE, lN), dtype=complex)
    for l in range(num_AP):
        served_indices = np.where(D[l, :] == 1)[0]
        if served_indices.size == 0:
            continue
        # 对于 AP l，提取其信道估计矩阵 Hhat_l，形状 (lN, num_UE)
        Hhat_l = hhat[l, :, :]
        # 提取服务的 UE 的信道估计，形状 (lN, num_served)
        Hhat_l_served = Hhat_l[:, served_indices]
        # 构造协方差矩阵（仅使用信道估计信息）
        sum_in = p * (Hhat_l_served @ Hhat_l_served.conj().T)
        try:
            # 解线性方程：X = (sum_in + noiseMat)^{-1} * Hhat_l_served
            X = np.linalg.solve(sum_in + noiseMat, Hhat_l_served)  # X shape: (lN, num_served)
        except np.linalg.LinAlgError as e:
            print(f"AP {l} inversion failed in P-MMSE: {str(e)}")
            X = np.zeros_like(Hhat_l_served)
        X = p * X  # 乘以 p 放大因子
        # 注意：X 的形状为 (lN, num_served)，我们希望每个服务 UE 得到一个预编码向量 (lN,)
        # 因此将 X 的转置（形状 (num_served, lN)）赋值给对应位置
        V_P_MMSE[l, served_indices, :] = X.T
    return V_P_MMSE

def power_allocation(gain_matrix, D, rho_tot):
    """
    按式(6.36)实现功率分配：采用平方根加权分配方法
    输入：
      gain_matrix: (L, K) 大尺度衰落系数矩阵（例如使用 trace(R) 作为 beta）
      D: (L, K) 服务矩阵，D[l,k]==1 表示 AP l 服务 UE k
      rho_tot: 每个 AP 的总下行功率约束（RHO_TOT_PER_AP）
    输出：
      rho: (L, K) 功率分配矩阵
    """
    L, K = gain_matrix.shape
    rho = np.zeros((L, K))
    for l in range(L):
        served_ues = np.where(D[l, :] == 1)[0]
        if len(served_ues) == 0:
            continue
        # 防止除0错误
        sqrt_gain = np.sqrt(np.maximum(gain_matrix[l, served_ues], 1e-10))
        total = np.sum(sqrt_gain)
        if total > 0:
            rho[l, served_ues] = rho_tot * sqrt_gain / total
    return rho
