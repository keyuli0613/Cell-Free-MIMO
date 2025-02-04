# main.py

import numpy as np
from config import (NUM_AP, NUM_UE, ANTENNAS_PER_AP, AREA_SIZE,
                    TAU_P, TAU_D, UE_MAX_POWER, AP_MAX_POWER, NOISE_DL)
from channel_model import generate_channel
from pilot_assignment import assign_pilots
from objects import AP, UE
from uplink import uplink_signal_model
from downlink import centralized_downlink_precoding, downlink_signal_model

def main():
    # 1. 初始化AP和UE对象
    ap_list = []
    ue_list = []
    
    # 在[0, AREA_SIZE]×[0, AREA_SIZE]平面随机分布AP，AP高度为10m
    for ap_id in range(NUM_AP):
        ap_pos = [np.random.uniform(0, AREA_SIZE),
                  np.random.uniform(0, AREA_SIZE),
                  10.0]
        ap_list.append(AP(ap_id, ap_pos, ANTENNAS_PER_AP))
    
    # 同理，为UE随机生成位置，高度1.5m
    for ue_id in range(NUM_UE):
        ue_pos = [np.random.uniform(0, AREA_SIZE),
                  np.random.uniform(0, AREA_SIZE),
                  1.5]
        ue_list.append(UE(ue_id, ue_pos))
    
    # 2. 生成(UE, AP)信道矩阵: channel_matrix[(ue.id, ap.id)] = complex vector
    channel_matrix = {}
    for ue in ue_list:
        for ap in ap_list:
            channel_matrix[(ue.id, ap.id)] = generate_channel(ap.position, ue.position)
    
    # 3. 导频分配 + 合作聚类
    pilot_assignments, dcc = assign_pilots(ue_list, ap_list)
    # 将 dcc 结果写回 ue.assigned_ap_ids
    for ue in ue_list:
        ue.assigned_ap_ids = dcc[ue.id]
    
    # 打印导频分配结果
    print("Pilot assignments:")
    for ue in ue_list:
        p_idx = pilot_assignments[ue.id]
        print(f"UE {ue.id} pilot {p_idx}, served by APs: {ue.assigned_ap_ids}")
    
    # 4. 信道估计(此处直接把真实信道当作估计)
    channel_estimates = channel_matrix.copy()
    
    # 5. 上行测试: 构造上行接收信号
    uplink_signals = uplink_signal_model(ue_list, ap_list, channel_matrix)
    print("\n[Uplink] Sample signal at AP 0:\n", uplink_signals[0])
    
    # 6. 下行测试: 调用集中式下行预编码 + 下行接收信号
    #   假设所有UE的下行功率统一为 UE_MAX_POWER
    power_allocation = {ue.id: UE_MAX_POWER for ue in ue_list}
    
    # 调用 centralized_downlink_precoding 生成预编码向量 & x_signals
    from downlink import centralized_downlink_precoding, downlink_signal_model
    precoding_vectors, x_signals = centralized_downlink_precoding(
        ue_list, ap_list, channel_estimates, power_allocation
    )
    
    # 打印下行发射信号示例
    print("\n[Downlink] Sample x_signals at AP 0:\n", x_signals[0])
    
    # 逐个UE计算下行接收信号
    print("\n[Downlink] Received signals:")
    for ue in ue_list:
        y_dl = downlink_signal_model(ue, ap_list, channel_matrix, precoding_vectors)
        print(f"UE {ue.id} downlink received signal = {y_dl}")
    
    # 7. RL模块占位
    from reinforcement_learning import RLAgent
    state_dim = 10
    action_dim = 5
    rl_agent = RLAgent(state_dim, action_dim)
    
    # 示例
    state = np.random.rand(state_dim)
    action = rl_agent.select_action(state)
    print("\nRL Agent selected action:", action)
    
    print("\nDone.")

if __name__ == '__main__':
    main()
