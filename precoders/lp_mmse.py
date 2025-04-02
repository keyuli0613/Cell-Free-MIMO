# precoders/lp_mmse.py
import numpy as np

def precoding_vector(serving_aps, H_hat, ue_id, rho_dist, D, lN, sigma2):
    W = {}
    num_UEs = H_hat.shape[1]
    for ap in serving_aps:
        # 初始化干扰+噪声协方差矩阵为 sigma2 * I
        R = sigma2 * np.eye(lN)
        # 累加其他用户造成的干扰协方差 (基于大尺度衰落系数)
        for other_ue in range(num_UEs):
            if other_ue == ue_id:
                continue
            # 取AP ap对用户other_ue的大尺度系数D和功率系数rho
            beta_li = D[ap, other_ue] if D is not None else 1.0
            power_coeff = rho_dist[ap, other_ue] if rho_dist is not None else 1.0
            # 将干扰视为等增益噪声：rho * beta * I
            R += power_coeff * beta_li * np.eye(lN)
        # 计算预编码向量
        h_t = H_hat[ap, ue_id, :].reshape(lN, 1)
        w = np.linalg.solve(R, h_t)
        # 归一化和功率分配
        if rho_dist is not None:
            coeff = rho_dist[ap, ue_id]
            w = w / (np.linalg.norm(w) + 1e-12)
            w = np.sqrt(coeff) * w
        else:
            w = w / (np.linalg.norm(w) + 1e-12)
        W[ap] = w.ravel()
    return W
