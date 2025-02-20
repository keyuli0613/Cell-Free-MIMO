# downlink.py

import numpy as np
from scipy.linalg import inv
from config import NOISE_DL, RHO_TOT_PER_AP

def compute_downlink_SE(serving_aps, H_hat, H_true, ue_id, all_ues,
                        lN, p, sigma2, precoding_scheme='MR',
                        rho_dist=None, D=None, pilot_assignments=None, all_aps_global=None):
    """
    计算下行链路频谱效率（SE）。
    预编码方案包括 MR 和 L-MMSE。
    
    输入参数：
      serving_aps: 为目标 UE 提供服务的 AP 列表
      H_hat: 信道估计矩阵，形状 (NUM_AP, lN, NUM_UE)
      H_true: 真实信道矩阵，形状 (NUM_AP, lN, NUM_UE)
      ue_id: 目标 UE 的 id（整数）
      all_ues: 所有 UE 对象列表
      lN: 每个 AP 的天线数
      p: 发射功率（用于预编码计算）
      sigma2: 噪声功率（NOISE_DL）
      precoding_scheme: 预编码方案，'MR' 或 'L-MMSE'
      rho_dist: 下行功率分配矩阵，形状 (NUM_AP, NUM_UE)
      D: 服务矩阵 (NUM_AP, NUM_UE)，D[l,k]==1 表示 AP l 服务 UE k
      pilot_assignments: 可选，用于 L-MMSE 预编码（如果存在）
      all_aps_global: 如果提供，则用于干扰计算（遍历全局 AP）
      
    输出：
      SE: 下行频谱效率，单位 bit/s/Hz
    """
    if len(serving_aps) == 0:
        return 0.0

    # 对于干扰计算，如果提供全局AP列表，则使用全局AP，否则仅使用 serving_aps
    if all_aps_global is not None:
        all_aps_for_interference = all_aps_global
    else:
        all_aps_for_interference = serving_aps

    # 预编码向量字典
    W = {}

    for ap in serving_aps:
        ap_id = ap.id
        # 得到该 AP 服务的 UE 列表
        served_ues = np.where(D[ap_id, :] == 1)[0]
        if len(served_ues) == 0:
            W[ap_id] = np.zeros(lN, dtype=complex)
            continue

        if precoding_scheme == 'MR':
            # MR预编码：直接取目标 UE 的信道估计，乘以功率分配的平方根
            w = H_hat[ap_id, :, ue_id] * np.sqrt(rho_dist[ap_id, ue_id])
            W[ap_id] = w

        elif precoding_scheme == 'L-MMSE':
            # L-MMSE预编码：构造局部协方差矩阵，但排除目标UE自身
            interfering_ues = [k for k in served_ues if k != ue_id]
            if interfering_ues:
                C_tmp = sum(
                    np.outer(H_hat[ap_id, :, k], H_hat[ap_id, :, k].conj())
                    for k in interfering_ues
                )
            else:
                # 若无其他UE，则C_tmp为零矩阵
                C_tmp = np.zeros((lN, lN), dtype=complex)
            # 计算矩阵迹并设置正则化参数（调低正则化强度）
            trace_C = np.trace(C_tmp) / lN if np.trace(C_tmp) > 0 else 1e-3
            eps = 1e-4 * trace_C  # 调低正则化参数
            C_total = C_tmp + (sigma2 + eps) * np.eye(lN)

            try:
                # 计算 L-MMSE 预编码向量（公式式(6.25)）
                w = inv(C_total) @ H_hat[ap_id, :, ue_id]
                # 乘以功率分配平方根
                w *= np.sqrt(rho_dist[ap_id, ue_id])
            except np.linalg.LinAlgError as e:
                print(f"AP {ap_id} 求逆失败: {str(e)}")
                w = np.zeros(lN, dtype=complex)
            W[ap_id] = w

        else:
            # 默认：零向量
            W[ap_id] = np.zeros(lN, dtype=complex)

        # 每个AP独立归一化，确保其发射功率不超过 RHO_TOT_PER_AP
        ap_power = np.linalg.norm(W[ap_id])**2
        if ap_power > RHO_TOT_PER_AP:
            scaling_factor = np.sqrt(RHO_TOT_PER_AP / ap_power)
            W[ap_id] = W[ap_id] * scaling_factor

    # ===== 信号与干扰计算 =====
    signal = 0j
    interference = 0.0

    # 信号项：仅服务AP贡献目标UE的信号
    for ap in serving_aps:
        ap_id = ap.id
        w = W.get(ap_id, np.zeros(lN, dtype=complex))
        h_target = H_true[ap_id, :, ue_id]
        signal += np.vdot(w, h_target)

    # 干扰项：遍历所有AP（全局），累加对非目标UE的干扰贡献
    for ap in all_aps_for_interference:
        ap_id = ap.id
        w = W.get(ap_id, np.zeros(lN, dtype=complex))
        for other_ue in all_ues:
            if other_ue.id == ue_id:
                continue
            h_interf = H_true[ap_id, :, other_ue.id]
            interference += np.abs(np.vdot(w, h_interf))**2

    # 计算有效SINR（包含噪声）
    SINR = np.abs(signal)**2 / (interference + sigma2)
    SE = np.log2(1 + SINR)
    return SE

def power_allocation(gain_matrix, D, rho_tot):
    """
    按式(6.36)实现功率分配：采用平方根加权分配方法
    输入：
      gain_matrix: (L, K) 大尺度衰落系数矩阵（例如使用 trace(R) 作为 beta）
      D: (L, K) 服务矩阵（若 D[l, k]==1 表示 AP l 服务 UE k）
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
        # 防止除零错误
        sqrt_gain = np.sqrt(np.maximum(gain_matrix[l, served_ues], 1e-10))
        total = np.sum(sqrt_gain)
        if total > 0:
            rho[l, served_ues] = rho_tot * sqrt_gain / total
    return rho
