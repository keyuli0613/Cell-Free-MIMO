import numpy as np
from config import NUM_AP, ANTENNAS_PER_AP, TAU_C, TAU_P, AREA_SIZE, UE_MAX_POWER, RHO_TOT, NOISE_UL, NOISE_DL
from objects import AP, UE
from pilot_assignment import assign_pilots
from channel_estimation import mmse_estimate
from downlink import compute_downlink_SE, power_allocation
from scipy.linalg import sqrtm

# 固定预编码方案列表：0->MR, 1->L-MMSE
PRECODING_SCHEMES = ['MR', 'L-MMSE']

class RLEnvironment:
    def __init__(self, user_data,
                 init_cluster_size=5,      # 初始聚类大小
                 init_precoding_idx=0,     # 初始预编码方案索引
                 init_rho_tot=RHO_TOT,     # 初始下行总功率
                 lambda_energy=0.001):      # 能耗权重
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
        self.tau_c = TAU_C
        self.tau_p = TAU_P

        # 固定部署 AP
        self.ap_list = [AP(ap_id=l,
                           position=[np.random.uniform(0, AREA_SIZE),
                                     np.random.uniform(0, AREA_SIZE), 10.0],
                           antennas=ANTENNAS_PER_AP)
                        for l in range(self.num_AP)]

    def reset(self):
        """
        重置环境，开始新的一天。
        状态包含：[num_UE, cluster_size, precoding_idx, rho_tot]
        """
        self.current_step = 0
        self.num_UE = self.user_data[0]  # 初始用户数量
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
        ue_list = [UE(ue_id=k,
                      position=[np.random.uniform(0, AREA_SIZE),
                                np.random.uniform(0, AREA_SIZE), 1.5])
                   for k in range(self.num_UE)]
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
        for ue in ue_list:
            serving_aps = [ap for ap in ap_list if ap.id in ue.assigned_ap_ids]
            if not serving_aps:
                se_list.append(0.0)
                energy_list.append(0.0)
                continue
            se_val, energy = compute_downlink_SE(
                serving_aps=serving_aps,
                H_hat=H_hat,
                H_true=H_true,
                ue_id=ue.id,
                all_ues=ue_list,
                lN=ANTENNAS_PER_AP,
                p=UE_MAX_POWER,
                sigma2=NOISE_DL,
                precoding_scheme=PRECODING_SCHEMES[self.precoding_idx],
                rho_dist=rho_dist,
                D=D,
                pilot_assignments=pilot_assignments,
                all_aps_global=ap_list
            )
            se_list.append(se_val)
            energy_list.append(energy)
        avg_SE = np.mean(se_list)
        total_energy = np.sum(energy_list)
        return avg_SE, total_energy

    def step(self, action):
        """
        执行一步动作，更新环境状态。
        动作空间（5 个动作）：
          0: 聚类大小 +1
          1: 聚类大小 -1
          2: 切换预编码方案（MR 和 L-MMSE 之间切换）
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

        avg_SE, total_energy = self.simulate_trial()
        # max_SE = 10.0  # 需根据实际环境调整
        # max_energy = 1000.0  # 需根据实际环境调整
        # reward = (avg_SE / max_SE) - self.lambda_energy * (total_energy / max_energy)
        reward = avg_SE + self.lambda_energy * total_energy  # 优化 SE 并惩罚能耗

        if np.isnan(reward):
            reward = 0.0

        self.current_step += 1
        
        if self.current_step < self.max_steps:
            self.num_UE = self.user_data[self.current_step]  # 更新用户数量
        state = np.array([self.num_UE, self.cluster_size, self.precoding_idx, self.rho_tot])
        done = (self.current_step >= self.max_steps)  # 一天结束时 done=True
        
        return state, reward, done

if __name__ == "__main__":
    # 示例用户数量数据（一天 72 个时间步）
    # user_data = [50 + int(50 * np.sin(np.pi * t / 36)) for t in range(72)]
    user_data = [16 for _ in range(72)]

    env = RLEnvironment(user_data=user_data)
    state = env.reset()
    print("Initial state:", state)
    for _ in range(10):
        action = np.random.randint(0, 5)
        next_state, reward, done = env.step(action)
        print(f"Step {_+1}: Action={action}, Reward={reward}, Done={done}, Next State={next_state}")
