# total_power.py
import numpy as np

def compute_ap_static_power(M_l, P_st):
    """
    计算单个AP的静态硬件功耗
    输入:
      M_l: AP l 激活的天线数
      P_st: 每根天线的静态硬件功耗（W）
    输出:
      静态功耗 = M_l * P_st
    """
    return M_l * P_st

def compute_ap_transmit_power(rho_vec, delta_tr):
    """
    计算单个AP的发射功率部分
    输入:
      rho_vec: AP l 对所有用户的功率分配数组（W），形如 [ρ_{l,1}, ρ_{l,2}, ..., ρ_{l,K}]
      delta_tr: 发射功率负载系数（例如1/η, 若η=0.4，则delta_tr约为2.5）
    输出:
      发射功率部分 = delta_tr * sum(rho_vec)
    """
    return delta_tr * np.sum(rho_vec)

def compute_ap_processing_power(P_proc0, delta_proc_AP, CAP_l, CAP_max):
    """
    计算单个AP的处理功耗部分
    输入:
      P_proc0: AP处理器空闲时的静态功耗（W）
      delta_proc_AP: 与AP算力相关的单位负载能耗系数（W/GOPS）
      CAP_l: AP l 当前需要执行的计算量（GOPS）
      CAP_max: AP的最大处理能力（GOPS）
    输出:
      处理功耗 = P_proc0 + delta_proc_AP * (CAP_l / CAP_max)
    """
    return P_proc0 + delta_proc_AP * (CAP_l / CAP_max)

def compute_total_power(M_array, rho_matrix, CAP_array, P_st, delta_tr, P_proc0, delta_proc_AP, CAP_max):
    """
    计算整个网络所有AP的总功耗，按TDD帧计算。
    
    参数:
      M_array: 长度为 L 的数组，每个元素表示AP l的激活天线数 M_l
      rho_matrix: 形状为 (L, K) 的数组，每行表示AP l对所有UE分配的发射功率（W）
      CAP_array: 长度为 L 的数组，每个元素为AP l当前的处理负载（GOPS）
      P_st: 每根天线静态硬件功耗（W）
      delta_tr: 发射功率负载系数（例如1/η）
      P_proc0: AP处理器空闲功耗（W）
      delta_proc_AP: AP处理功耗的单位负载能耗系数（W/GOPS）
      CAP_max: AP最大处理能力（GOPS）
    
    返回:
      P_total: 所有AP总功耗（W）
    """
    L = len(M_array)
    total_power = 0.0
    for l in range(L):
        P_static = compute_ap_static_power(M_array[l], P_st)
        P_tx = compute_ap_transmit_power(rho_matrix[l, :], delta_tr)
        P_proc = compute_ap_processing_power(P_proc0, delta_proc_AP, CAP_array[l], CAP_max)
        P_l = P_static + P_tx + P_proc
        total_power += P_l
    return total_power

# 测试示例
if __name__ == '__main__':
    # 假设网络中有 L=10 个AP
    L = 10
    # 每个AP固定有4根天线
    M_array = np.full(L, 4)
    
    # 假设每个AP服务K=4个用户，功率分配数组 (W) 随机给出
    K = 4
    rho_matrix = np.random.uniform(0.5, 1.5, (L, K))
    
    # 假设每个AP的处理负载 CAP (GOPS) 随机在100~300之间
    CAP_array = np.random.uniform(100, 300, L)
    # 最大处理能力设为500 GOPS
    CAP_max = 500
    
    # 参数设定（均采用参考文献中的典型值）：
    P_st = 6.8             # 每根天线的静态硬件功耗, 单位W
    delta_tr = 2.5         # 发射功率负载系数（例如功放效率0.4，则1/0.4=2.5）
    P_proc0 = 20.8         # AP处理器空闲功耗, 单位W
    delta_proc_AP = 74     # AP处理功耗斜率, 单位W/GOPS
    
    P_total = compute_total_power(M_array, rho_matrix, CAP_array, P_st, delta_tr, P_proc0, delta_proc_AP, CAP_max)
    print("网络中所有AP的总功耗为: {:.2f} W".format(P_total))
