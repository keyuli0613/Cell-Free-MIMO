import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from config import NUM_AP, NUM_UE, ANTENNAS_PER_AP, AREA_SIZE, UE_MAX_POWER, NOISE_UL, TAU_C, TAU_P

# 简单生成空间相关矩阵（不考虑实际局部散射，只使用单位矩阵）
def generate_unit_correlation(N):
    return np.eye(N)

def run_cellular_simulation(mc_trials=100):
    se_list = []
    
    for trial in range(mc_trials):
        # 1. 随机部署 AP 和 UE
        ap_positions = np.array([[np.random.uniform(0, AREA_SIZE), np.random.uniform(0, AREA_SIZE), 10.0] for _ in range(NUM_AP)])
        ue_positions = np.array([[np.random.uniform(0, AREA_SIZE), np.random.uniform(0, AREA_SIZE), 1.5] for _ in range(NUM_UE)])
        
        # 2. 计算大尺度衰落（简单模型：只使用距离，忽略阴影）
        beta_matrix = np.zeros((NUM_AP, NUM_UE))
        for l in range(NUM_AP):
            for k in range(NUM_UE):
                distance = np.linalg.norm(ap_positions[l, :2] - ue_positions[k, :2])
                path_loss = 10 ** ( - (128.1 + 37.6 * np.log10(distance/1000)) / 10 )
                beta_matrix[l, k] = path_loss
        
        # 3. 生成空间相关矩阵 R (此处简单设为 beta * I)
        R = np.zeros((ANTENNAS_PER_AP, ANTENNAS_PER_AP, NUM_AP, NUM_UE), dtype=complex)
        for l in range(NUM_AP):
            for k in range(NUM_UE):
                R[:, :, l, k] = beta_matrix[l, k] * generate_unit_correlation(ANTENNAS_PER_AP)
        
        # 4. 生成真实信道 H_true (形状: (NUM_AP, ANTENNAS_PER_AP, NUM_UE))
        H_true = np.zeros((NUM_AP, ANTENNAS_PER_AP, NUM_UE), dtype=complex)
        for l in range(NUM_AP):
            for k in range(NUM_UE):
                Rsqrt = sqrtm(R[:, :, l, k])
                noise_vec = (np.random.randn(ANTENNAS_PER_AP) + 1j * np.random.randn(ANTENNAS_PER_AP)) / np.sqrt(2)
                H_true[l, :, k] = Rsqrt @ noise_vec
        
        # 5. 选择每个UE最近的AP（传统单小区模型）
        serving_ap = np.zeros(NUM_UE, dtype=int)
        for k in range(NUM_UE):
            distances = np.linalg.norm(ap_positions[:, :2] - ue_positions[k, :2], axis=1)
            serving_ap[k] = np.argmin(distances)
        
        # 6. 计算下行 SE：假设每个AP对所服务UE使用 MR 预编码
        SE_cellular = []
        for k in range(NUM_UE):
            l = serving_ap[k]
            h = H_true[l, :, k]  # 目标UE的信道向量
            # MR 预编码：使用信道估计的共轭（这里假设完美估计）
            w = h.conj()  
            # AP对该UE发射信号
            signal = np.abs(np.vdot(w, h))**2
            # 仅计算噪声干扰（不考虑多用户干扰，因为每个AP只服务一个UE）
            noise = NOISE_UL  # 下行用 NOISE_DL 可能更合适，但这里为了示例简单用NOISE_UL
            SINR = signal / noise
            SE = np.log2(1 + SINR)
            SE_cellular.append(SE)
        
        se_list.append(np.mean(SE_cellular))
    
    return se_list

def plot_cdf(data, title, xlabel):
    data = np.array(data)
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.figure()
    plt.plot(sorted_data, cdf, marker='.', linestyle='none')
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True)
    plt.show()

def main():
    se_cellular = run_cellular_simulation(mc_trials=100)
    print("Average SE (Cellular model):", np.mean(se_cellular))
    plot_cdf(se_cellular, "Cellular SE CDF", "SE (bit/s/Hz)")

if __name__ == "__main__":
    main()
