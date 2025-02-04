# uplink.py
import numpy as np
from config import ANTENNAS_PER_AP, NOISE_UL, TAU_D, TAU_C

def uplink_signal_model(ue_list, ap_list, channel_matrix):
    """
    构造上行接收信号模型：
      每个AP接收到的信号为所有UE信号的叠加，再加上噪声。
      
    参数：
      ue_list: UE对象列表
      ap_list: AP对象列表
      channel_matrix: dict {(ue_id, ap_id): channel_vector}，
                      每个信道向量的形状为 (ANTENNAS_PER_AP,)
      
    返回：
      uplink_signals: dict {ap_id: received_signal_vector}，
                      每个AP接收到的信号向量 (形状为 (ANTENNAS_PER_AP,))
    """
    uplink_signals = {}
    for ap in ap_list:
        y = np.zeros(ANTENNAS_PER_AP, dtype=complex)
        for ue in ue_list:
            h = channel_matrix[(ue.id, ap.id)]
            s = 1 + 0j  # 数据符号
            y += h * s
        noise = (np.random.randn(ANTENNAS_PER_AP) + 1j * np.random.randn(ANTENNAS_PER_AP)) / np.sqrt(2)
        y += noise * np.sqrt(NOISE_UL)
        uplink_signals[ap.id] = y
    return uplink_signals

def compute_uplink_SE(ue, serving_ap_list, v_combining, channel_estimates):
    """
    计算 UE 的上行谱效率 (SE)（简化版本）。
    
    参数：
      ue: 当前 UE 对象
      serving_ap_list: 列表，包含服务该 UE 的 AP 对象
      v_combining: 字典 {ap_id: combining_vector}，每个 AP 为该 UE 计算的合并向量 (形状为 (ANTENNAS_PER_AP,))
      channel_estimates: dict {(ue_id, ap_id): h_estimated}，各 AP 对 UE 的信道估计
      
    返回：
      SE: 上行谱效率 (bit/s/Hz)
      
    简化公式：
      SE = (TAU_D / TAU_C) * log2(1 + effective_SNR)
    其中 effective_SNR 仅考虑目标信号。
    """
    signal = 0
    for ap in serving_ap_list:
        h_est = channel_estimates[(ue.id, ap.id)]
        v = v_combining.get(ap.id, np.ones(ANTENNAS_PER_AP, dtype=complex))
        signal += np.vdot(v, h_est)
    effective_SNR = np.abs(signal)**2 / NOISE_UL
    SE = (TAU_D / TAU_C) * np.log2(1 + effective_SNR)
    return SE
