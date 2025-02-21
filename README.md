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
├── pilot_assignment.py  # 导频分配与动态协作聚类（DCC）：先根据主AP（或大尺度衰落）选取服务 AP 集合，再利用邻域信息（DCC）选择导频，从而减少导频污染
├── uplink.py            # 上行链路 SE 计算（含 MMSE/MR 合并）
├── downlink.py          # 下行链路预编码与 SE 计算
├── utils/               # 辅助工具
│   ├── visualization.py  # CDF 绘图模块
│   └── power_allocation.py  # 功率分配算法
└── README.md

```
AP 与 UE 部署：
在覆盖区域内随机生成 AP（固定高度10 m）和 UE（固定高度1.5 m）。

大尺度衰落与空间相关性：
根据 3GPP UMa 模型计算 AP-UE 间的大尺度衰落系数（包括路径损耗和阴影衰落），并利用局部散射模型生成空间相关矩阵 R。

真实信道生成：
利用空间相关矩阵 R 生成真实信道 H_true（假设为 Rayleigh 衰落）。

导频分配与合作聚类（DCC）：
根据大尺度衰落信息为 UE 分配导频，并确定每个 UE 的服务 AP 集合（DCC）。

信道估计：
利用 MMSE 估计方法（基于上行导频）得到信道估计 H_hat，用于后续上行检测和下行预编码（TDD 系统中的互易性）。

上行 SE 计算：
利用 H_hat 和 H_true 计算上行链路频谱效率（SE）。

下行预编码与 SE 计算：
根据不同预编码方案（MR、L-MMSE、P-MMSE），利用 H_hat 构造下行预编码向量，。

下行功率采用平方根加权大尺度衰落系数的分配策略，确保每个 AP 的功率不超过限制。

结果统计与绘图：
进行 Monte Carlo 仿真，统计上行与下行 SE，并绘制 CDF 曲线。

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

$$
\hat{h}_{mk} = \tau_p \rho_p R_{mk} \left(\tau_p \rho_p \sum_{i \in P_k} R_{mi} + \sigma^2 I\right)^{-1} y_{mp}
$$

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

### **4. 下行频谱效率计算**

#### **MR 预编码（6.26）**
直接使用目标 UE 的信道估计向量：

$$
\mathbf{w}_{mk}^{MR} = D_{kl} \hat{h}_{kl}
$$

#### **L-MMSE 预编码（6.25）**
计算局部协方差矩阵时排除目标 UE 的贡献（即只考虑其他服务 UE 的信道），并加上噪声项和一个正则化项（eps 与协方差矩阵迹有关），再求逆与目标 UE 的信道估计相乘。

$$
\mathbf{w}_{mk}^{L-MMSE} = \rho_k \left( \sum_{i \in K_m} \rho_i \hat{h}_{mi} \hat{h}_{mi}^{H} + C_{il}^2 + \sigma_{ul}^2 \mathbf{I}_N \right)^{-1} D_{kl} \hat{h}_{kl}
$$

#### **P-MMSE 预编码（6.33）**
使用 AP 本地服务 UE 的所有信道（包括目标 UE）构造协方差矩阵，这是对全局 MMSE 的近似，计算方式类似，但不排除目标 UE 的信道。

### **5. 功率分配函数（式 6.36）**
每个 AP 与 UE 大尺度衰落系数的平方根进行分配，从而更加注重信道质量好的 UE。

$$
\rho_{lk} = \rho_{tot} \frac{\beta_{lk}}{\sum_{i \in K_l} \beta_{li}}
$$

---

## 📈 仿真结果示例

### **1. 上行链路 SE CDF**
![Uplink SE CDF](./Uplink%20SE%20CDF.png)

### **2. 下行链路 SE CDF**
![Downlink SE CDF](./Downlink%20SE%20CDF.png)

---

## 🚀 未来优化方向


- **强化学习模型构建：**
#### 状态 (State)



- **信道状态信息**：可用的 H_hat（或其统计信息，如平均 SINR、信道硬化程度等）。
- **大尺度参数**：如路径损耗、阴影衰落、AP 与 UE 之间的距离。
- **当前功率分配**：各 AP 的发射功率分配矩阵（或统计信息，如平均功率）。
- **当前聚类情况**：每个 UE 的服务 AP 集合（例如 DCC 信息）。
- **网络负载**：例如同时活跃的 UE 数量、队列长度等（如果考虑时延）。
- **能耗指标**：例如各 AP 的功耗水平。

可以将这些信息整合成一个向量或字典，作为 RL 环境的状态。

#### 动作 (Action)



- **调整功率分配**：例如针对每个 AP 的功率分配，动作可以是对每个 AP 或整体功率做微调（增加、减少、保持）。
- **切换预编码方案**：动作可以是选择不同的预编码方案（例如 MR、L-MMSE、P-MMSE），对某个 UE 或整个系统生效。
- **调整服务用户集合（聚类）**：动作可以是调整某个 AP 的服务列表，例如允许/禁止某个 AP 服务某个 UE。


#### 奖励 (Reward)

奖励函数应同时考虑 QoS（例如 SE、用户公平性、时延要求）和能耗（AP 发射功率、能耗成本）。例如：

- 奖励可以定义为总 SE 减去能耗惩罚项，或者构造一个多目标奖励（例如加权求和）。
- 如果网络满足时延要求，则奖励正向；若超出时延阈值，则给予负奖励。

---

## 📢 参考文献

1. **[Foundations of User-Centric Cell-Free Massive MIMO](https://www.cell-free.net/book/)**
2. **Marzetta, T. L., et al., "Fundamentals of Massive MIMO"**
3. **Björnson, E., et al., "Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency"**

