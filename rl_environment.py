import numpy as np
import matplotlib.pyplot as plt
from config import NUM_AP, ANTENNAS_PER_AP, TAU_C, TAU_P, AREA_SIZE, UE_MAX_POWER, RHO_TOT, NOISE_UL, NOISE_DL, POWERTRANS, POWERCIR, POWERPROC, SLEEP, NUM_UE
from objects import AP, UE
from pilot_assignment import assign_pilots
from channel_estimation import mmse_estimate
from downlink import compute_downlink_SE, power_allocation
from scipy.linalg import sqrtm
from ap_power import compute_total_power\

import time

# 固定预编码方案列表：0->MR, 1->L-MMSE
PRECODING_SCHEMES = ['MR', 'L-MMSE']
class RLEnvironment:
    def __init__(self, user_data,
                 init_cluster_size=5,      # 初始聚类大小
                 init_precoding_idx=0,     # 初始预编码方案索引
                 init_rho_tot=RHO_TOT,     # 初始下行总功率
                 lambda_energy=0.01):      # 能耗权重
        """
        初始化环境，加入动态用户数量数据。
        :param user_data: 包含 72 个值的列表，表示一天内每 20 分钟的用户数量
        """
        self.user_data = user_data  # 用户数量时间序列
        self.max_steps = len(user_data)  # 最大时间步数，例如 72
        self.current_step = 0  # 当前时间步
        self.cluster_size = init_cluster_size
        self.precoding_idx = init_precoding_idx
        self.rho_tot = init_rho_tot
        self.lambda_energy = lambda_energy
        self.num_AP = NUM_AP
        self.num_UE = NUM_UE
        self.tau_c = TAU_C
        self.tau_p = TAU_P

        # 固定部署 AP
        self.ap_list = [AP(ap_id=l,
                           position=[np.random.uniform(0, AREA_SIZE),
                                     np.random.uniform(0, AREA_SIZE), 10.0],
                           antennas=ANTENNAS_PER_AP)
                        for l in range(self.num_AP)]
        self.ue_list = [UE(ue_id=k,
                      position=[np.random.uniform(0, AREA_SIZE),
                                np.random.uniform(0, AREA_SIZE), 1.5])
                   for k in range(self.num_UE)]

    def reset(self):
        """
        重置环境，开始新的一天。
        状态包含：[num_UE, cluster_size, precoding_idx, rho_tot]
        """
        self.current_step = 0
        # self.num_UE = self.user_data[0]  # 初始用户数量
        self.cluster_size = 5
        self.precoding_idx = 0
        self.rho_tot = RHO_TOT
        state = np.array([self.num_UE, self.cluster_size, self.precoding_idx, self.rho_tot])
        return state

    def simulate_trial(self):
        """
        运行一次仿真，返回平均下行 SE 和总能耗。
        用户数量由 self.num_UE 确定，动态变化。
        """
        ap_list = self.ap_list
        # ue_list = [UE(ue_id=k,
        #               position=[np.random.uniform(0, AREA_SIZE),
        #                         np.random.uniform(0, AREA_SIZE), 1.5])
        #            for k in range(self.num_UE)]
        ue_list = self.ue_list
        if len(ue_list) == 0:
            return 0.0, 0.0  # 如果没有 UE，返回默认值

        # 计算大尺度衰落系数 beta
        beta_matrix = np.zeros((self.num_AP, self.num_UE))
        for l, ap in enumerate(ap_list):
            for k, ue in enumerate(ue_list):
                distance = np.linalg.norm(np.array(ap.position) - np.array(ue.position))
                path_loss = 10 ** (-(128.1 + 37.6 * np.log10(distance / 1000)) / 10)
                shadowing = 10 ** (np.random.normal(0, 8) / 10)
                beta_matrix[l, k] = path_loss * shadowing

        # 生成空间相关矩阵 R
        R = np.zeros((ANTENNAS_PER_AP, ANTENNAS_PER_AP, self.num_AP, self.num_UE), dtype=complex)
        for l in range(self.num_AP):
            for k in range(self.num_UE):
                R[:, :, l, k] = beta_matrix[l, k] * np.eye(ANTENNAS_PER_AP)

        # 生成真实信道 H_true
        H_true = np.zeros((self.num_AP, ANTENNAS_PER_AP, self.num_UE), dtype=complex)
        for l in range(self.num_AP):
            for k in range(self.num_UE):
                Rsqrt = sqrtm(R[:, :, l, k])
                noise_vec = (np.random.randn(ANTENNAS_PER_AP) + 1j * np.random.randn(ANTENNAS_PER_AP)) / np.sqrt(2)
                H_true[l, :, k] = Rsqrt @ noise_vec

        # 导频分配与 DCC
        pilot_assignments, dcc = assign_pilots(ue_list, ap_list, beta_matrix, L=self.cluster_size)
        for ue in ue_list:
            ue.assigned_ap_ids = dcc[ue.id]
        D = np.zeros((self.num_AP, self.num_UE), dtype=int)
        for ue in ue_list:
            for ap_id in ue.assigned_ap_ids:
                D[ap_id, ue.id] = 1
         # --------------------- 新增 Gamma 计算 ---------------------
        if hasattr(self, 'tau_p') and self.tau_p is not None:
            tau_p_val = self.tau_p
        else:
            tau_p_val = self.num_UE

        epsilon = 1e-9  # 防止除零
        Gamma = np.zeros((self.num_AP, self.num_UE))
        # 对于每个 AP 和每个导频
        for l in range(self.num_AP):
            for t in range(tau_p_val):
                # 找出使用导频 t 的所有 UE，注意 pilot_assignments 中的键为 ue id，值为导频索引
                pilot_UEs = [k for k in range(self.num_UE) if pilot_assignments.get(k) == t]
                if len(pilot_UEs) > 0:
                    sum_beta = np.sum(beta_matrix[l, pilot_UEs])
                    for k in pilot_UEs:
                        # Gamma 的公式：10*log10((tau_p*(beta^2))/(tau_p*sum_beta+epsilon))
                        Gamma[l, k] = (tau_p_val * (beta_matrix[l, k] ** 2)) / (tau_p_val * sum_beta + epsilon)
        # -----------------------------------------------------------

        # Delta：对于每个 AP，将 UE 按 beta 从大到小排序，
        # 直到累计 beta 占总 beta 的比例达到 90%，将这些 UE 标记为 1，
        # 同时记录 tau_sl（每个 AP 的服务 UE 数量）
        Delta = np.zeros((self.num_AP, self.num_UE), dtype=int)
        tau_sl = np.zeros(self.num_AP, dtype=int)
        for l in range(self.num_AP):
            served_ues = np.where(D[l, :] == 1)[0]  # 仅对 AP l 服务的 UE进行划分
            if len(served_ues) == 0:
                continue
            gains = beta_matrix[l, served_ues]  
            sorted_idx_local = np.argsort(gains)[::-1]  # 降序排序在 served_ues 中的索引
            sorted_served_ues = served_ues[sorted_idx_local]
            sorted_gains = gains[sorted_idx_local]
            sum_gain = np.sum(sorted_gains)
            collected_gain = 0
            index_gain = 0
            while (index_gain < len(sorted_gains)) and (collected_gain / sum_gain < 0.8):
                collected_gain += sorted_gains[index_gain]
                index_gain += 1
            tau_sl[l] =  min(index_gain, ANTENNAS_PER_AP)
            Delta[l, sorted_served_ues[:index_gain]] = 1

        # 计算 serving AP 的数量
        serving_ap_ids = set()
        for ue in ue_list:
            serving_ap_ids.update(ue.assigned_ap_ids)
        num_serving_aps = len(serving_ap_ids)
        
        # 信道估计
        
        H_hat = mmse_estimate(ap_list, ue_list, H_true, pilot_assignments, UE_MAX_POWER, NOISE_UL)

        # 功率分配
        gain_matrix = np.zeros((self.num_AP, self.num_UE))
        for l in range(self.num_AP):
            for k in range(self.num_UE):
                gain_matrix[l, k] = np.real(np.trace(R[:, :, l, k]))
        rho_dist = power_allocation(gain_matrix, D, self.rho_tot)

        # 计算 SE 和能耗
        se_list = []
        energy_list = []
    
        tau_c = self.tau_c  # 从配置中读取
        for ue in ue_list:
            se_val, energy = compute_downlink_SE(
                serving_aps=ap_list,  # 这里所有 AP 均参与计算
                ue_id=ue.id,
                all_ues=ue_list,
                lN=ANTENNAS_PER_AP,
                sigma2=NOISE_DL,
                rho_dist=rho_dist,
                pilot_assignments=pilot_assignments,
                beta_matrix=beta_matrix,
                gamma_matrix=Gamma,
                Delta=D,
                tau_sl=tau_sl,
                tau_c=tau_c,
                tau_p=tau_p_val
            )
            se_list.append(se_val)
            energy_list.append(energy)
        avg_SE = np.mean(se_list)
        total_energy = np.sum(energy_list)

         # 获取每个 AP 的实际天线数
        M_array = np.array([ap.antennas for ap in self.ap_list])

        # 粗略估计每个 AP 的计算负载（假设每服务 1 个 UE 需要 50 GOPS）
        CAP_array = np.zeros(self.num_AP)
        for l in range(self.num_AP):
            served_ue_count = np.sum(D[l, :])
            CAP_array[l] = served_ue_count * 50  # 可根据实验再调节这个因子

        # 设置功耗相关参数
        P_st = 6.8             # 每根天线静态硬件功耗 (W)
        delta_tr = 2.5         # 发射功率负载系数 (例如 η=0.4 → 1/0.4 = 2.5)
        P_proc0 = 20.8         # 处理器空闲功耗 (W)
        delta_proc_AP = 74     # 单位计算量功耗 (W/GOPS)
        CAP_max = 500          # 最大处理能力 (GOPS)

        # 计算总功耗
        P_total = compute_total_power(
            M_array, rho_dist, CAP_array,
            P_st, delta_tr, P_proc0, delta_proc_AP, CAP_max
        )
        print("AVGse:", avg_SE)
        print("Power:", P_total)
        return avg_SE, P_total
    
    def step(self, action):
        """
        执行一步动作，更新环境状态。
        动作空间（5 个动作）：
          0: 聚类大小 +1
          1: 聚类大小 -1
          2: 切换预编码方案
          3: 功率 +0.5
          4: 功率 -0.5
        """
        if action == 0:
            self.cluster_size = min(self.cluster_size + 1, self.num_AP)
        elif action == 1:
            self.cluster_size = max(self.cluster_size - 1, 1)
        elif action == 2:
            self.precoding_idx = (self.precoding_idx + 1) % len(PRECODING_SCHEMES)
        elif action == 3:
            self.rho_tot += 0.5
        elif action == 4:
            self.rho_tot = max(self.rho_tot - 0.5, 0.1)

        avg_SE, total_power = self.simulate_trial()
        # max_SE = 10.0  # 需根据实际环境调整
        # max_energy = 1000.0  # 需根据实际环境调整
        # reward = (avg_SE / max_SE) - self.lambda_energy * (total_energy / max_energy)
        # reward = avg_SE + self.lambda_energy * total_energy  # 优化 SE 并惩罚能耗

        
        reward = np.log(1 + avg_SE) -  self.lambda_energy* np.log(1 + total_power)


        if np.isnan(reward):
            reward = 0.0

        self.current_step += 1
        
        # if self.current_step < self.max_steps:
        #     self.num_UE = self.user_data[self.current_step]  # 更新用户数量
        state = np.array([self.num_UE, self.cluster_size, self.precoding_idx, self.rho_tot])
        done = (self.current_step >= self.max_steps)  # 一天结束时 done=True
        
        return state, reward, done
    
    def plot_SE_vs_cluster_size(self, min_cluster=1, max_cluster=None):
        """
        画出聚类大小对系统平均SE的影响
        :param min_cluster: 最小聚类大小
        :param max_cluster: 最大聚类大小（默认为 NUM_AP）
        """
        if max_cluster is None:
            max_cluster = self.num_AP

        cluster_sizes = list(range(min_cluster, max_cluster + 1))
        se_values = []

        

        for size in cluster_sizes:
            self.cluster_size = size
            avg_SE, _ = self.simulate_trial()
            se_values.append(avg_SE)

        

        # 绘图
        plt.figure()
        plt.plot(cluster_sizes, se_values, marker='o')
        plt.xlabel('Cluster Size')
        plt.ylabel('Average Downlink SE')
        plt.title('SE vs. Cluster Size')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 示例用户数量数据（一天 72 个时间步）
    # user_data = [50 + int(50 * np.sin(np.pi * t / 36)) for t in range(72)]
    # user_data = [7 for _ in range(72)]


    # env = RLEnvironment(user_data=user_data)
    # state = env.reset()
    # print("Initial state:", state)
    # for _ in range(10):
    #     action = np.random.randint(0, 5)
    #     next_state, reward, done = env.step(action)
    #     # print(f"Step {_+1}: Action={action}, Reward={reward}, Done={done}, Next State={next_state}")
    #     print()
    user_data = [NUM_UE for _ in range(72)]
    
    env.reset()
    env.plot_SE_vs_cluster_size(min_cluster=1, max_cluster=8)