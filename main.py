import numpy as np
import matplotlib.pyplot as plt
from config import NUM_AP, NUM_UE, ANTENNAS_PER_AP, TAU_C, TAU_P, AREA_SIZE, UE_MAX_POWER, MC_TRIALS, NOISE_UL, NOISE_DL, RHO_TOT
from objects import AP, UE
from pilot_assignment import assign_pilots
from channel_estimation import mmse_estimate
from uplink import compute_uplink_SE_advanced
from downlink import compute_downlink_SE, centralized_power_allocation
from scipy.linalg import sqrtm

def generate_spatial_correlation(N, angle_spread_deg=10):
    """
    生成ULA天线的空间相关矩阵，基于简单的局部散射模型。
    输入:
      N: 天线数
      angle_spread_deg: 方位角扩展角度（度）
    输出:
      R: N x N 相关矩阵
    """
    angles = np.linspace(-angle_spread_deg/2, angle_spread_deg/2, 100)
    a = np.array([np.exp(-1j * np.pi * np.sin(np.deg2rad(theta)) * np.arange(N)) for theta in angles])
    R = a.T @ a.conj() / len(angles)
    return R

def run_simulation(mc_trials=MC_TRIALS):
    uplink_SE_all = []
    # 我们为下行不同预编码方案分别保存 SE（例如 MR、L-MMSE、P-MMSE）
    downlink_SE_all = {'MR': [], 'L-MMSE': [], 'P-MMSE': []}
    
    for trial in range(mc_trials):
        # -------------------------------
        # 1) 初始化 AP 与 UE 对象（位置随机）
        # -------------------------------
        ap_list = [AP(ap_id=l, 
                     position=[np.random.uniform(0, AREA_SIZE), 
                               np.random.uniform(0, AREA_SIZE), 10.0],
                     antennas=ANTENNAS_PER_AP)
                   for l in range(NUM_AP)]
        ue_list = [UE(ue_id=k, 
                     position=[np.random.uniform(0, AREA_SIZE), 
                               np.random.uniform(0, AREA_SIZE), 1.5])
                   for k in range(NUM_UE)]
        
        # -------------------------------
        # 2) 计算大尺度衰落系数 beta (例如使用3GPP UMa模型)
        # -------------------------------
        beta_matrix = np.zeros((NUM_AP, NUM_UE))
        for l, ap in enumerate(ap_list):
            for k, ue in enumerate(ue_list):
                distance = np.linalg.norm(np.array(ap.position) - np.array(ue.position))
                # 注意：距离单位转换（例如米转千米）根据具体模型调整
                path_loss = 10 ** ( - (128.1 + 37.6 * np.log10(distance/1000)) / 10 )
                shadowing = 10 ** (np.random.normal(0, 8) / 10)  # 8dB 阴影
                beta_matrix[l, k] = path_loss * shadowing
        
        # -------------------------------
        # 3) 生成空间相关矩阵 R (形状: (N, N, NUM_AP, NUM_UE))
        # -------------------------------
        R = np.zeros((ANTENNAS_PER_AP, ANTENNAS_PER_AP, NUM_AP, NUM_UE), dtype=complex)
        for l in range(NUM_AP):
            for k in range(NUM_UE):
                R[:,:,l,k] = beta_matrix[l, k] * generate_spatial_correlation(ANTENNAS_PER_AP)
        
        # -------------------------------
        # 4) 生成真实信道 H_true (形状: (NUM_AP, ANTENNAS_PER_AP, NUM_UE))
        # -------------------------------
        H_true = np.zeros((NUM_AP, ANTENNAS_PER_AP, NUM_UE), dtype=complex)
        for l in range(NUM_AP):
            for k in range(NUM_UE):
                Rsqrt = sqrtm(R[:,:,l,k])
                noise_vec = (np.random.randn(ANTENNAS_PER_AP) + 1j*np.random.randn(ANTENNAS_PER_AP)) / np.sqrt(2)
                H_true[l, :, k] = Rsqrt @ noise_vec
        
        # -------------------------------
        # 5) 导频分配与形成 DCC（基于 pilot_assignment 算法）
        # -------------------------------
        pilot_assignments, dcc = assign_pilots(ue_list, ap_list, beta_matrix)
        for ue in ue_list:
            ue.assigned_ap_ids = dcc[ue.id]
        
        # 构造 D 矩阵，D[l, k]=1 表示 AP l 为 UE k 服务
        D = np.zeros((NUM_AP, NUM_UE), dtype=int)
        for ue in ue_list:
            for ap_id in ue.assigned_ap_ids:
                D[ap_id, ue.id] = 1
        
        # -------------------------------
        # 6) 信道估计：利用上行导频生成 MMSE 估计 H_hat（形状同 H_true）
        # -------------------------------
        H_hat = mmse_estimate(ap_list, ue_list, H_true, pilot_assignments, UE_MAX_POWER, NOISE_UL)
        # 此处保证上行估计 H_hat 将用于后续的上行与下行处理（TDD 互易性）
        
        # -------------------------------
        # 7) 上行 SE 计算：利用 H_hat 和 H_true 计算上行 SE
        # -------------------------------
        se_list_uplink = []
        for ue in ue_list:
            serving_aps = [ap for ap in ap_list if ap.id in ue.assigned_ap_ids]
            if not serving_aps:
                se_list_uplink.append(0.0)
            else:
                se_val = compute_uplink_SE_advanced(
                    serving_aps=serving_aps,
                    H_hat=H_hat,
                    H_true=H_true,
                    ue_id=ue.id,
                    all_ues=ue_list,
                    lN=ANTENNAS_PER_AP,
                    p=UE_MAX_POWER,
                    sigma2=NOISE_UL,
                    tau_c=TAU_C,
                    tau_p=TAU_P
                )
                se_list_uplink.append(se_val)
        avg_uplink_SE = np.mean(se_list_uplink)
        uplink_SE_all.append(avg_uplink_SE)
        
        # -------------------------------
        # 8) 下行 SE 计算：利用同一 H_hat 进行下行预编码，并计算 SE
        # 这里考虑不同预编码方案：MR、L-MMSE、P-MMSE（如有实现）
        # -------------------------------
        # 首先构造功率分配矩阵（集中式和分布式各自计算）
        gain_matrix = np.zeros((NUM_AP, NUM_UE))
        for l in range(NUM_AP):
            for k in range(NUM_UE):
                gain_matrix[l, k] = np.real(np.trace(R[:, :, l, k]))
                
        rho_central = centralized_power_allocation(gain_matrix, D, scheme='proportional')
        
        # 分布式功率分配：对每个AP，将总功率 RHO_TOT 均分给服务该AP的 UE
        rho_dist = np.zeros((NUM_AP, NUM_UE))
        for l in range(NUM_AP):
            served_ues = np.where(D[l, :] == 1)[0]
            num_served = len(served_ues)
            if num_served > 0:
                rho_dist[l, served_ues] = RHO_TOT / num_served
        
        # 对于每个预编码方案计算下行 SE（所有UE）
        for scheme in ['MR', 'L-MMSE', 'P-MMSE']:
            se_list_downlink = []
            for ue in ue_list:
                serving_aps = [ap for ap in ap_list if ap.id in ue.assigned_ap_ids]
                if not serving_aps:
                    se_list_downlink.append(0.0)
                    continue
                se_val = compute_downlink_SE(
                    serving_aps=serving_aps,
                    H_hat=H_hat,
                    H_true=H_true,
                    ue_id=ue.id,
                    all_ues=ue_list,
                    lN=ANTENNAS_PER_AP,
                    p=UE_MAX_POWER,
                    sigma2=NOISE_DL,
                    precoding_scheme=scheme,
                    rho_dist=rho_dist,
                    D=D
                )
                se_list_downlink.append(se_val)
            avg_downlink_SE = np.mean(se_list_downlink)
            downlink_SE_all[scheme].append(avg_downlink_SE)
        
        # 在每个 trial 内，上下行 SE 都基于同一 H_hat（TDD互易性）
    
    return uplink_SE_all, downlink_SE_all

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

def plot_cdf_dict(schemes_dict, title, xlabel):
    plt.figure()
    for scheme_name, se_list in schemes_dict.items():
        arr = np.sort(np.array(se_list))
        cdf = np.arange(1, len(arr) + 1) / len(arr)
        plt.plot(arr, cdf, marker='.', linestyle='none', label=scheme_name)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    uplink_SE_all, downlink_SE_all = run_simulation(mc_trials=50)
    
    print("Average Uplink SE:", np.mean(uplink_SE_all))
    print("Average Downlink SE (MR):", np.mean(downlink_SE_all['MR']))
    print("Average Downlink SE (L-MMSE):", np.mean(downlink_SE_all['L-MMSE']))
    print("Average Downlink SE (P-MMSE):", np.mean(downlink_SE_all['P-MMSE']))
    
    plot_cdf(uplink_SE_all, "Uplink SE CDF", "SE (bit/s/Hz)")
    plot_cdf_dict(downlink_SE_all, "Downlink SE CDF", "SE (bit/s/Hz)")
    
if __name__ == "__main__":
    main()
