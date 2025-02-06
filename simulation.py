# simulation.py
import numpy as np
import matplotlib.pyplot as plt
from config import NUM_AP, NUM_UE, AREA_SIZE, ANTENNAS_PER_AP, TAU_C, TAU_P, TAU_D, UE_MAX_POWER, NOISE_DL
from channel_model import generate_channel
from pilot_assignment import assign_pilots
from objects import AP, UE
from uplink import uplink_signal_model, compute_uplink_SE
from downlink import centralized_downlink_precoding, downlink_signal_model
from reinforcement_learning import RLAgent

#############################
# 1. 模拟参数设置
#############################
MC_TRIALS = 1000   # Monte Carlo 仿真次数

# 为简单起见，这里对每个试验都使用相同的功率分配
power_allocation = {}  # {ue_id: UE_MAX_POWER}
# 强制每个UE的下行功率取 UE_MAX_POWER
# 你可以根据实际需求设计不同的功率分配方案
#############################

#############################
# 2. 改进信道估计（占位函数：MMSE信道估计）
#############################
def mmse_channel_estimation(true_channel, estimation_noise_var=1e-2):
    """
    简单模拟 MMSE 信道估计：
      输出的估计值 = 真信道 + 复数高斯估计噪声
      estimation_noise_var 为估计噪声方差
    参数：
      true_channel: 真信道向量（例如形状为 (ANTENNAS_PER_AP,)）
    返回：
      h_est: 信道估计
    """
    noise = (np.random.randn(*true_channel.shape) + 1j * np.random.randn(*true_channel.shape)) / np.sqrt(2)
    h_est = true_channel + np.sqrt(estimation_noise_var) * noise
    return h_est

#############################
# 3. 改进下行预编码（占位示例：ZF预编码）
#############################
def zf_precoding_vector(ue, ap_list, channel_estimates, power_allocation):
    """
    示例：对于给定UE，从服务AP拼接信道估计构造ZF预编码向量。
    这里采用最简单的 ZF 思路：令预编码向量与其他UE信道正交（示例中只考虑本UE的通道）。
    实际上 ZF 预编码需要全局信息，这里仅作占位示例。
    
    返回：
      collective_precoding_vector: 拼接后的预编码向量（1D array）
    """
    serving_aps = sorted([ap for ap in ap_list if ap.serves(ue)], key=lambda ap: ap.id)
    w_concat = np.array([], dtype=complex)
    for ap in serving_aps:
        h_est = channel_estimates[(ue.id, ap.id)]
        # 简单 ZF：令预编码子向量取 h_est 的共轭
        w_sub = np.conj(h_est)
        w_concat = np.concatenate((w_concat, w_sub))
    if w_concat.size == 0:
        w_concat = np.zeros(ANTENNAS_PER_AP, dtype=complex)
    p_ue = power_allocation.get(ue.id, UE_MAX_POWER)
    norm_val = np.linalg.norm(w_concat)
    if norm_val < 1e-12:
        return w_concat
    return np.sqrt(p_ue) * w_concat / norm_val

#############################
# 4. RL 环境占位（简单示例）
#############################
class RLEnvironment:
    def __init__(self, ue_list, ap_list):
        # 状态可以包含网络负载、UE位置、信道统计等信息，这里仅作占位
        self.ue_list = ue_list
        self.ap_list = ap_list

    def get_state(self):
        # 返回一个示例状态向量
        state = np.random.rand(10)
        return state

    def step(self, action):
        # 根据动作更新系统（例如功率分配、调度等），计算奖励
        # 这里仅返回随机奖励和新状态
        reward = np.random.rand()
        new_state = self.get_state()
        done = False
        return new_state, reward, done

#############################
# 5. 主仿真过程
#############################
def run_simulation(mc_trials=MC_TRIALS):
    # 分别收集所有试验下的上行和下行 SE
    uplink_SE_all = []
    downlink_SE_all = []
    
    # 对于每一次仿真（不同的随机部署）
    for trial in range(mc_trials):
        # 1. 初始化AP和UE
        ap_list = []
        ue_list = []
        for ap_id in range(NUM_AP):
            ap_pos = [np.random.uniform(0, AREA_SIZE),
                      np.random.uniform(0, AREA_SIZE),
                      10.0]
            ap_list.append(AP(ap_id, ap_pos, ANTENNAS_PER_AP))
        for ue_id in range(NUM_UE):
            ue_pos = [np.random.uniform(0, AREA_SIZE),
                      np.random.uniform(0, AREA_SIZE),
                      1.5]
            ue_list.append(UE(ue_id, ue_pos))
        
        # 2. 生成信道矩阵和信道估计（采用MMSE估计占位）
        channel_matrix = {}
        channel_estimates = {}
        for ue in ue_list:
            for ap in ap_list:
                true_channel = generate_channel(ap.position, ue.position)
                channel_matrix[(ue.id, ap.id)] = true_channel
                # 调用MMSE估计（占位函数）
                h_est = mmse_channel_estimation(true_channel, estimation_noise_var=1e-2)
                channel_estimates[(ue.id, ap.id)] = h_est
        
        # 3. 导频分配与形成DCC
        pilot_assignments, dcc = assign_pilots(ue_list, ap_list)
        for ue in ue_list:
            ue.assigned_ap_ids = dcc[ue.id]
        
        # 4. 上行SE计算：这里假设使用一个简单的接收合并向量（全1向量），
        #    并仅计算目标信号部分（未考虑多用户干扰）
        v_combining = {}
        uplink_SE_trial = []
        for ue in ue_list:
            serving_aps = [ap for ap in ap_list if ap.serves(ue)]
            for ap in serving_aps:
                # 这里简单设定每个AP的合并向量为全1
                v_combining[ap.id] = np.ones(ANTENNAS_PER_AP, dtype=complex)
            SE_ue = compute_uplink_SE(ue, serving_aps, v_combining, channel_estimates)
            uplink_SE_trial.append(SE_ue)
        avg_uplink_SE = np.mean(uplink_SE_trial)
        uplink_SE_all.append(avg_uplink_SE)
        
        # 5. 下行SE计算：采用集中式预编码（这里选择 MMSE 预编码的简化版本）
        #    可选：也可以调用我们实现的 ZF预编码来对比效果
        precoding_vectors, x_signals = centralized_downlink_precoding(
            ue_list, ap_list, channel_estimates, power_allocation={ue.id: UE_MAX_POWER for ue in ue_list}
        )
        downlink_SE_trial = []
        for ue in ue_list:
            # 这里下行 SE 简化为 log2(1 + effective_SNR)
            # effective_SNR 计算时只使用平均有效信道作为近似（后续可完善）
            y_dl = downlink_signal_model(ue, ap_list, channel_matrix, precoding_vectors)
            # 这里用接收信号功率/噪声计算 SNR (简化)
            effective_SNR = (np.abs(y_dl)**2) / NOISE_DL
            SE_ue = (TAU_D / TAU_C) * np.log2(1 + effective_SNR)
            downlink_SE_trial.append(SE_ue)
        avg_downlink_SE = np.mean(downlink_SE_trial)
        downlink_SE_all.append(avg_downlink_SE)
    
    return uplink_SE_all, downlink_SE_all

#############################
# 6. 绘制CDF曲线
#############################
def plot_cdf(data, title, xlabel):
    data = np.array(data)
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    
    plt.figure()
    plt.plot(sorted_data, cdf, marker='.', linestyle='none')
    plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.title(title)
    plt.grid(True)
    plt.show()

#############################
# 7. 主函数入口
#############################
def main():
    uplink_SE_all, downlink_SE_all = run_simulation(mc_trials=1000)
    print("Average uplink SE over trials: ", np.mean(uplink_SE_all))
    print("Average downlink SE over trials: ", np.mean(downlink_SE_all))
    
    # 绘制CDF曲线
    plot_cdf(uplink_SE_all, "Uplink SE CDF", "Uplink SE (bit/s/Hz)")
    plot_cdf(downlink_SE_all, "Downlink SE CDF", "Downlink SE (bit/s/Hz)")
    
    # RL环境示例
    env = RLEnvironment([], [])  # 这里只是占位
    agent = RLAgent(state_dim=10, action_dim=5)
    state = env.get_state()
    action = agent.select_action(state)
    print("RL Agent selected action:", action)

if __name__ == '__main__':
    main()
