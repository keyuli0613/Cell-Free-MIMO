import numpy as np
import matplotlib.pyplot as plt
from config import NUM_AP, NUM_UE, ANTENNAS_PER_AP, TAU_C, TAU_P, AREA_SIZE, UE_MAX_POWER, MC_TRIALS, NOISE_UL, NOISE_DL, RHO_TOT
from objects import AP, UE
from pilot_assignment import assign_pilots
from channel_estimation import mmse_estimate
from uplink import compute_uplink_SE_advanced
from downlink import compute_downlink_SE, power_allocation
from scipy.linalg import sqrtm
import time

def generate_spatial_correlation(N, angle_spread_deg=10):
    """生成ULA天线的空间相关矩阵"""
    angles = np.linspace(-angle_spread_deg/2, angle_spread_deg/2, 100)
    a = np.array([np.exp(-1j * np.pi * np.sin(np.deg2rad(theta)) * np.arange(N)) for theta in angles])
    R = a.T @ a.conj() / len(angles)
    return R

def run_simulation(mc_trials=MC_TRIALS):
    uplink_SE_all = []
    downlink_SE_all = {'MR': [], 'L-MMSE': []}
    processing_times = {'MR': [], 'L-MMSE': []}  # 分别记录MR和L-MMSE的处理时间

    for trial in range(mc_trials):
        # 初始化AP和UE
        ap_list = [AP(ap_id=l, position=[np.random.uniform(0, AREA_SIZE), np.random.uniform(0, AREA_SIZE), 10.0], antennas=ANTENNAS_PER_AP) for l in range(NUM_AP)]
        ue_list = [UE(ue_id=k, position=[np.random.uniform(0, AREA_SIZE), np.random.uniform(0, AREA_SIZE), 1.5]) for k in range(NUM_UE)]

        # 计算大尺度衰落系数 beta
        beta_matrix = np.zeros((NUM_AP, NUM_UE))
        for l, ap in enumerate(ap_list):
            for k, ue in enumerate(ue_list):
                distance = np.linalg.norm(np.array(ap.position) - np.array(ue.position))
                path_loss = 10 ** (-(128.1 + 37.6 * np.log10(distance/1000)) / 10)
                shadowing = 10 ** (np.random.normal(0, 8) / 10)
                beta_matrix[l, k] = path_loss * shadowing

        # 生成空间相关矩阵 R
        R = np.zeros((ANTENNAS_PER_AP, ANTENNAS_PER_AP, NUM_AP, NUM_UE), dtype=complex)
        for l in range(NUM_AP):
            for k in range(NUM_UE):
                R[:,:,l,k] = beta_matrix[l, k] * generate_spatial_correlation(ANTENNAS_PER_AP)

        # 生成真实信道 H_true
        H_true = np.zeros((NUM_AP, ANTENNAS_PER_AP, NUM_UE), dtype=complex)
        for l in range(NUM_AP):
            for k in range(NUM_UE):
                Rsqrt = sqrtm(R[:,:,l,k])
                noise_vec = (np.random.randn(ANTENNAS_PER_AP) + 1j*np.random.randn(ANTENNAS_PER_AP)) / np.sqrt(2)
                H_true[l, :, k] = Rsqrt @ noise_vec

        # 导频分配与DCC
        pilot_assignments, dcc = assign_pilots(ue_list, ap_list, beta_matrix)
        for ue in ue_list:
            ue.assigned_ap_ids = dcc[ue.id]
        D = np.zeros((NUM_AP, NUM_UE), dtype=int)
        for ue in ue_list:
            for ap_id in ue.assigned_ap_ids:
                D[ap_id, ue.id] = 1

        # 信道估计
        start_time = time.time()  # 记录开始时间
        H_hat = mmse_estimate(ap_list, ue_list, H_true, pilot_assignments, UE_MAX_POWER, NOISE_UL)

        # 上行SE计算
        se_list_uplink = [compute_uplink_SE_advanced([ap for ap in ap_list if ap.id in ue.assigned_ap_ids], H_hat, H_true, ue.id, ue_list, ANTENNAS_PER_AP, UE_MAX_POWER, NOISE_UL, TAU_C, TAU_P) if ue.assigned_ap_ids else 0.0 for ue in ue_list]
        uplink_SE_all.append(np.mean(se_list_uplink))

        # 下行SE计算与处理时间记录
        gain_matrix = np.zeros((NUM_AP, NUM_UE))
        for l in range(NUM_AP):
            for k in range(NUM_UE):
                gain_matrix[l, k] = np.real(np.trace(R[:,:,l,k]))
        rho_dist = power_allocation(gain_matrix, D, RHO_TOT)

        for scheme in ['MR', 'L-MMSE']:
            
            se_list_downlink = []
            for ue in ue_list:
                serving_aps = [ap for ap in ap_list if ap.id in ue.assigned_ap_ids]
                if not serving_aps:
                    se_list_downlink.append(0.0)
                    continue
                se_val, _ = compute_downlink_SE(serving_aps, H_hat, H_true, ue.id, ue_list, ANTENNAS_PER_AP, UE_MAX_POWER, NOISE_DL, scheme, rho_dist, D, pilot_assignments)
                se_list_downlink.append(se_val)
            downlink_SE_all[scheme].append(np.mean(se_list_downlink))
            end_time = time.time()  # 记录结束时间
            processing_times[scheme].append(end_time - start_time)  # 存储处理时间
            print(f"[Trial {trial+1}] {scheme} Processing Time: {end_time - start_time:.4f} seconds")

    return uplink_SE_all, downlink_SE_all, processing_times

def plot_cdf_dict(data_dict, title, xlabel, legend_label):
    """绘制多个数据集的CDF图"""
    plt.figure()
    for scheme, data in data_dict.items():
        sorted_data = np.sort(np.array(data))
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, cdf, marker='.', linestyle='none', label=f"{legend_label} {scheme}")
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    uplink_SE_all, downlink_SE_all, processing_times = run_simulation(mc_trials=50)

    # 打印平均值
    print("平均上行SE:", np.mean(uplink_SE_all))
    for scheme in ['MR', 'L-MMSE']:
        print(f"平均下行SE ({scheme}):", np.mean(downlink_SE_all[scheme]))
        print(f"平均处理时间 ({scheme}):", np.mean(processing_times[scheme]), "秒")

    # 绘制CDF图
    plot_cdf_dict(downlink_SE_all, "下行SE CDF", "SE (bit/s/Hz)", "方案")
    plot_cdf_dict(processing_times, "处理时间 CDF", "时间 (秒)", "方案")

if __name__ == "__main__":
    main()