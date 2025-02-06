# Cell-Free Massive MIMO 仿真平台


基于《Foundations of User-Centric Cell-Free Massive MIMO》实现的分布式大规模 MIMO 系统仿真平台，支持 **上下行链路频谱效率分析、动态协作聚类（DCC）、多种预编码方案**。

---

## 📂 代码框架

```bash
Cell-Free-MIMO/
├── main.py              # 主仿真循环
├── config.py            # 系统参数配置
├── objects.py           # AP/UE 类定义
├── channel_estimation.py# MMSE
├── pilot_assignment.py  # 导频分配与动态协作聚类（DCC）
├── uplink.py            # 上行链路 SE 计算（含 MMSE/MR 合并）
├── downlink.py          # 下行链路预编码与 SE 计算
├── utils/               # 辅助工具
│   ├── visualization.py  # CDF 绘图模块
│   └── power_allocation.py  # 功率分配算法
└── README.md
```

---

## 🛠️ 实现算法

### **1. 信道建模（式 2.25）**

$$
R = \frac{1}{N_a} \sum_{i=1}^{N_a} a(\phi_i) a(\phi_i)^H
$$

```python
import numpy as np

def generate_spatial_correlation(N, angle_spread=10):
    angles = np.linspace(-angle_spread/2, angle_spread/2, 100)
    a = [np.exp(-1j*np.pi*np.sin(np.deg2rad(theta))*np.arange(N)) for theta in angles]
    R = (np.array(a).T @ np.conj(a)) / len(angles)  # 式(2.25)
    return R
```

---

### **2. MMSE 信道估计（式 4.5）**

<p>
$$
\hat{h}_{mk} = \tau_p \rho_p R_{mk} \left(\tau_p \rho_p \sum_{i \in P_k} R_{mi} + \sigma^2 I\right)^{-1} y_{mp}
$$
</p>




```python
Hhat = np.sqrt(p) * R @ np.linalg.inv(p * tau_p * sum_R + sigma2 * np.eye(N)) @ Y_pilot
```

---

### **3. 上行频谱效率计算（式 5.5）**

$$
SE_k^{ul} = \frac{\tau_c - \tau_p}{\tau_c} \log_2 \left(1 + \frac{\rho_u \left| \sum\limits_{m \in M_k} v_{mk}^H h_{mk} \right|^2}{\rho_u \sum\limits_{i \neq k} \left| \sum\limits_{m \in M_k} v_{mk}^H h_{mi} \right|^2 + \sigma^2 \sum\limits_{m \in M_k} \| v_{mk} \|^2} \right)
$$

```python
SINR = np.abs(np.sum(v_mk.conj().T @ h_mk))**2 / (np.sum(np.abs(np.sum(v_mk.conj().T @ h_mi))**2) + noise)
SE = (tau_c - tau_p) / tau_c * np.log2(1 + SINR)
```

---

### **4. 下行预编码（式 6.25 / 6.33）**

<p>
$$
w_{mk}^{\text{L-MMSE}} = \rho_d \left(\sum\limits_{i \in K_m} \hat{h}_{mi} \hat{h}_{mi}^H + \sigma^2 I \right)^{-1} \hat{h}_{mk}
$$
<p>
  
```python
# L-MMSE 预编码计算
C_total = sum(h_hat @ h_hat.conj().T for UE in served_UEs) + sigma2 * np.eye(N)
w = np.linalg.inv(C_total) @ h_hat_k  # 式(6.25)
```

---

## 📈 仿真结果示例

### **1. 上行链路 SE CDF**
![Uplink SE CDF](./Uplink%20SE%20CDF.png)

### **2. 下行链路 SE CDF**
![Downlink SE CDF](./Downlink%20SE%20CDF.png)

---

## 🚀 未来优化方向

强化学习模块 (Reinforcement Learning for QoS & Energy)，用于后续结合系统运行数据（例如不同聚类、weekday、时间、延时敏感/宽松等）来设计基于 RL 的资源（功率）和 QoS 优化算法。

---

## 📢 参考文献

1. **[Foundations of User-Centric Cell-Free Massive MIMO](https://www.cell-free.net/book/)**
2. **Marzetta, T. L., et al., "Fundamentals of Massive MIMO"**
3. **Björnson, E., et al., "Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency"**

