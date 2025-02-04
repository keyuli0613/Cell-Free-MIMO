# channel_model.py

import numpy as np
from config import ANTENNAS_PER_AP, PATHLOSS_OFFSET, PATHLOSS_EXPONENT, SHADOW_STD

def generate_channel(ap_position, ue_position):
    """
    根据 AP 与 UE 的位置信息生成信道向量，包含：
      - 路径损耗
      - 阴影衰落（以dB表示的正态分布）
      - 小尺度瑞利衰落 (无空间相关性)
      
    参数:
      ap_position: AP 的位置, 格式 [x, y, z]
      ue_position: UE 的位置, 格式 [x, y, z]
      
    返回:
      channel: 复数信道向量，维度为 (ANTENNAS_PER_AP, )
    """
    # 计算欧几里得距离（包括高度差）
    distance = np.linalg.norm(np.array(ap_position) - np.array(ue_position))
    if distance < 1.0:
        distance = 1.0  # 避免距离过小
    
    # 计算路径损耗 (dB)
    pl_dB = PATHLOSS_OFFSET - PATHLOSS_EXPONENT * np.log10(distance)
    pl_linear = 10 ** (pl_dB / 10)
    
    # 阴影衰落（dB），使用正态分布
    shadow = np.random.normal(0, SHADOW_STD)
    shadow_linear = 10 ** (shadow / 10)
    
    # 小尺度瑞利衰落 (无空间相关性)
    fading = (np.random.randn(ANTENNAS_PER_AP) + 1j * np.random.randn(ANTENNAS_PER_AP)) / np.sqrt(2)
    
    # 综合影响：信道幅值 = 路径损耗 * 阴影衰落 的平方根，再乘以瑞利衰落
    channel = np.sqrt(pl_linear * shadow_linear) * fading
    return channel
