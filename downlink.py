# downlink.py

import numpy as np
from scipy.linalg import inv
from config import NOISE_DL, RHO_TOT

def compute_downlink_SE(serving_aps, H_hat, H_true, ue_id, all_ues,
                        lN, p, sigma2, precoding_scheme='MR',
                        rho_dist=None, D=None):
    """
    计算下行链路的频谱效率 (SE)。

    参数说明:
      serving_aps:    为当前UE(ue_id)提供服务的AP对象列表
      H_hat:          (num_AP, lN, num_UE)  下行信道的估计值
      H_true:         (num_AP, lN, num_UE)  下行信道的真实值
      ue_id:          当前UE的索引(ID)
      all_ues:        UE对象的列表
      lN:             每个AP的天线数
      p:              发射功率(可选, 如果本地要参考)
      sigma2:         下行噪声方差, 用于L-MMSE正则等
      precoding_scheme: 字符串, 'MR' 或 'L-MMSE' 等
      rho_dist:       (L, K)的功率分配矩阵(可选), 目前未显式使用
      D:              (L, K) 指示矩阵, D[l, k]==1表示 AP l 服务 UE k

    返回:
      SE:  浮点数, 表示此UE的下行SE
    """

    # 如果没有AP服务,直接SE=0返回
    if len(serving_aps) == 0:
        return 0.0

    # 预编码向量字典: {ap_id: w_vec (shape=(lN,)) }
    W = {}

    for ap in serving_aps:
        ap_id = ap.id

        # 获取该AP真正服务的UE列表
        served_ues = [k for k in range(len(all_ues)) if D[ap_id, k] == 1]

        # 若该AP无任何UE可服务(或D里没有1),则给预编码向量 = 0
        if len(served_ues) == 0:
            W[ap_id] = np.zeros(lN, dtype=complex)
            continue

        # 根据 precoding_scheme 不同,构造相应的预编码向量
        if precoding_scheme == 'MR':
            # MR 直接使用 H_hat[ap, :, ue_id]
            W[ap_id] = H_hat[ap_id, :, ue_id]

        elif precoding_scheme == 'L-MMSE':
            # L-MMSE: 累加外积 (N x N), 再加噪声正则
            C_tmp = np.zeros((lN, lN), dtype=complex)
            for k_ue in served_ues:
                h_vec = H_hat[ap_id, :, k_ue]  # shape = (lN,)
                C_tmp += np.outer(h_vec, h_vec.conj())

            # 加上噪声+小正则保证可逆
            eps = 1e-8
            C_total = C_tmp + sigma2 * np.eye(lN) + eps * np.eye(lN)

            # 尝试求逆; 若仍奇异, fallback用0向量
            try:
                w_vec = inv(C_total) @ H_hat[ap_id, :, ue_id]
            except np.linalg.LinAlgError:
                w_vec = np.zeros(lN, dtype=complex)

            W[ap_id] = w_vec

        else:
            # 其他预编码方式未实现, fallback=0
            W[ap_id] = np.zeros(lN, dtype=complex)

    # ------ 功率归一化(可选) -------
    total_power = sum(np.linalg.norm(w)**2 for w in W.values())
    if RHO_TOT > 0 and total_power > RHO_TOT:
        scale = np.sqrt(RHO_TOT / total_power)
        for ap_id in W:
            w = W[ap_id]
            ap_power = np.linalg.norm(w)**2
            if ap_power > RHO_TOT:
                W[ap_id] *= np.sqrt(RHO_TOT / ap_power)
            # W[ap_id] *= scale

    # ============ 计算有效信号 & 干扰 ============
    signal = 0. + 0j
    interference = 0.

    for ap in serving_aps:
        ap_id = ap.id
        w = W[ap_id]  # 此时不会KeyError
        h_vec = H_true[ap_id, :, ue_id]

        # 累加信号(复数相加)
        signal += np.vdot(w, h_vec)  # np.vdot => conj(w)*h

        # 干扰: other_ue!=ue_id
        for other_ue in all_ues:
            if other_ue.id == ue_id:
                continue
            h_interf = H_true[ap_id, :, other_ue.id]
            interference += np.abs(np.vdot(w, h_interf))**2

    # 计算SINR
    noise = sigma2
    SINR = np.abs(signal)**2 / (interference + noise)
    SE = np.log2(1 + SINR)
    return SE


def centralized_power_allocation(gain_matrix, D, scheme='uniform'):
    """
    集中式功率分配算法 (L,K) => (L,K)
    scheme = 'uniform' or 'proportional'
    """
    L, K = gain_matrix.shape
    rho = np.zeros((L, K))

    if scheme == 'uniform':
        # 均匀分配
        for l in range(L):
            served_ues = np.where(D[l, :] == 1)[0]
            n_served = len(served_ues)
            if n_served > 0:
                rho[l, served_ues] = RHO_TOT / n_served

    elif scheme == 'proportional':
        # 按比例, gain/sum(gain)
        for l in range(L):
            served_ues = np.where(D[l, :] == 1)[0]
            total_gain = np.sum(gain_matrix[l, served_ues])
            if total_gain > 0:
                rho[l, served_ues] = (RHO_TOT * gain_matrix[l, served_ues] / total_gain)

    return rho
