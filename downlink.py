# downlink.py
import numpy as np
from config import ANTENNAS_PER_AP, NOISE_DL, UE_MAX_POWER

def centralized_downlink_precoding(ue_list, ap_list, channel_estimates, power_allocation):
    """
    集中式下行预编码
    返回:
      precoding_vectors: {ue_id: collective_precoding_vector (complex 1D array)}
      x_signals: {ap_id: x_l (shape: (ANTENNAS_PER_AP,))}
    """
    precoding_vectors = {}
    # 每个UE: 按 AP.id 排序获取服务AP列表, 顺序拼接
    for ue in ue_list:
        serving_aps = sorted([ap for ap in ap_list if ap.serves(ue)], key=lambda a: a.id)
        w_concat = np.array([], dtype=complex)
        for ap in serving_aps:
            h_est = channel_estimates[(ue.id, ap.id)]
            w_concat = np.concatenate((w_concat, h_est))
        if w_concat.size == 0:
            w_concat = np.zeros(ANTENNAS_PER_AP, dtype=complex)
        # 归一化并乘以功率因子
        power = power_allocation.get(ue.id, UE_MAX_POWER)
        norm_val = np.linalg.norm(w_concat)
        if norm_val < 1e-12:
            # 避免除0
            precoding_vectors[ue.id] = w_concat
        else:
            precoding_vectors[ue.id] = np.sqrt(power) * w_concat / norm_val

    # 计算每个AP的发射信号
    x_signals = {ap.id: np.zeros(ANTENNAS_PER_AP, dtype=complex) for ap in ap_list}
    for ue in ue_list:
        serving_aps = sorted([ap for ap in ap_list if ap.serves(ue)], key=lambda a: a.id)
        w_collective = precoding_vectors[ue.id]
        sub_len = ANTENNAS_PER_AP
        for idx, ap in enumerate(serving_aps):
            start = idx * sub_len
            end = (idx + 1) * sub_len
            w_sub = w_collective[start:end]
            # 数据符号
            s = 1 + 0j
            x_signals[ap.id] += w_sub * s
    return precoding_vectors, x_signals

def downlink_signal_model(ue, ap_list, channel_matrix, centralized_precoding):
    """
    计算UE的下行接收信号:
      y = sum_{AP in serving} vdot(h, w_sub)
    """
    y_total = 0+0j
    serving_aps = sorted([ap for ap in ap_list if ap.serves(ue)], key=lambda a: a.id)
    w_collective = centralized_precoding[ue.id]
    sub_len = ANTENNAS_PER_AP
    for idx, ap in enumerate(serving_aps):
        h = channel_matrix[(ue.id, ap.id)]
        start = idx*sub_len
        end = (idx+1)*sub_len
        w_sub = w_collective[start:end]
        s = 1 + 0j
        y_total += np.vdot(h, w_sub)*s
    
    # 加噪声
    noise = (np.random.randn() + 1j*np.random.randn())/np.sqrt(2)*np.sqrt(NOISE_DL)
    y_total += noise
    return y_total
