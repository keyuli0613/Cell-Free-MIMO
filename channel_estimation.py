# channel_estimation.py
import numpy as np

def mmse_estimate(ap_list, ue_list, H_true, pilot_assignments, p, sigma2):
    """基于接收导频信号的MMSE估计"""
    num_APs = len(ap_list)
    num_UEs = len(ue_list)
    N = ap_list[0].antennas
    tau_p = len(ue_list)  # 导频长度等于UE数量
    
    # 生成正交导频序列
    pilots = np.fft.fft(np.eye(tau_p)) / np.sqrt(tau_p)
    
    # 模拟AP接收导频信号
    Y_pilot = np.zeros((num_APs, N, tau_p), dtype=complex)
    for l, ap in enumerate(ap_list):
        for t in range(tau_p):
            # 收集所有使用导频t的UE
            ue_indices = [ue.id for ue in ue_list if pilot_assignments[ue.id] == t]
            H_sum = np.sum(H_true[l, :, ue_indices], axis=0)
            
            noise = (np.random.randn(N) + 1j*np.random.randn(N)) * np.sqrt(sigma2)
            Y_pilot[l, :, t] = np.sqrt(p) * H_sum + noise
    
    # MMSE估计
    H_hat = np.zeros_like(H_true)
    for l, ap in enumerate(ap_list):
        for k, ue in enumerate(ue_list):
            t = pilot_assignments[ue.id]
            Psi = p * tau_p * np.sum([H_true[l,:,i] @ H_true[l,:,i].conj().T for i in range(num_UEs) if pilot_assignments[i]==t], axis=0) + sigma2 * np.eye(N)
            H_hat[l,:,k] = np.sqrt(p) * H_true[l,:,k] @ np.linalg.inv(Psi) @ Y_pilot[l,:,t]
            


    return H_hat