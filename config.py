# config.py

import numpy as np

# 网络参数
NUM_AP = 100           # AP 数量
NUM_UE = 40            # UE 数量
ANTENNAS_PER_AP = 4    # 每个AP的天线数

# 相干块参数
TAU_C = 200            # 每个相干块的总符号数
TAU_P = 10             # 导频符号数
TAU_D = TAU_C - TAU_P  # 数据符号数

# 信道模型参数
PATHLOSS_OFFSET = -30.5      # 路径损耗基准 (dB)
PATHLOSS_EXPONENT = 36.7     # 路径损耗对数衰减系数
SHADOW_STD = 4.2             # 阴影衰落标准差 (dB)
# 注：阴影衰落我们简单使用正态分布

# 噪声功率（单位：W），-96 dBm 转换为 W：10^((-96-30)/10) = 10^(-12.6)
NOISE_UL = 10 ** ((-96 - 30) / 10)   # 上行噪声功率
NOISE_DL = 10 ** ((-96 - 30) / 10)   # 下行噪声功率

# 功率设置（单位：W）
UE_MAX_POWER = 0.01    # 每个UE的最大上行功率（例如10 mW）
AP_MAX_POWER = 0.1       # 每个AP的最大下行功率（例如100 mW）

# 地理区域（单位：m）
AREA_SIZE = 400  # 区域边长 400m

# 天线和导频相关参数
# 当使用多天线时，可能需要空间相关参数，例如ASD，这里先简单设置
ASD_AZIMUTH = 15.0      # 方位角标准差，单位度
ASD_ELEVATION = 15.0    # 仰角标准差，单位度

# 带宽 (Hz)
BANDWIDTH = 10e6        # 10 MHz

# 其他参数可根据需要扩展...
MC_TRIALS = 100
RHO_TOT = 1

# 下行参数
RHO_TOT_PER_AP = 1.0  # 每个AP的最大发射功率 (瓦特)
NOISE_DL = 1e-10       # 下行噪声功率 (瓦特)

# 信道估计参数
ESTIMATION_ERROR_VAR = 0.1  # 信道估计误差方差（0.1表示10%误差）