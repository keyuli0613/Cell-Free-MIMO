# precoding.py
import numpy as np
from config import ANTENNAS_PER_AP, UE_MAX_POWER

def centralized_mmse_precoding(ue, ap_list, channel_estimates, power_allocation):
    """
    对于给定UE，根据服务AP的信道估计构造集中式 MMSE 预编码向量。
    本示例将所有服务AP的信道估计拼接后归一化，然后乘以功率平方根，
    作为集中式预编码向量。
    参数：
      ue: UE 对象
      ap_list: 所有AP对象列表
      channel_estimates: dict {(ue.id, ap.id): h_estimated}
      power_allocation: dict {ue.id: transmit_power}
    返回：
      collective_precoding_vector: 拼接后的预编码向量
    """
    # 对服务该UE的AP按 ap.id 排序
    serving_aps = sorted([ap for ap in ap_list if ap.serves(ue)], key=lambda a: a.id)
    w_concat = np.array([], dtype=complex)
    for ap in serving_aps:
        h_est = channel_estimates[(ue.id, ap.id)]
        # 简单起见，这里直接取共轭作为预编码子向量（ZF思路占位）
        w_sub = np.conj(h_est)
        w_concat = np.concatenate((w_concat, w_sub))
    if w_concat.size == 0:
        w_concat = np.zeros(ANTENNAS_PER_AP, dtype=complex)
    p_ue = power_allocation.get(ue.id, UE_MAX_POWER)
    norm_val = np.linalg.norm(w_concat)
    if norm_val < 1e-12:
        return w_concat
    return np.sqrt(p_ue) * w_concat / norm_val
