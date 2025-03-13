# 信道建模总结

## 1. 计算大尺度衰落系数

大尺度衰落主要由 **路径损耗（Path Loss）** 和 **阴影衰落（Shadow Fading）** 组成：

$$
\beta = \text{路径损耗} \times \text{阴影衰落}
$$

### 路径损耗

使用 3GPP UMa 模型计算：

$$
PL(d) = 128.1 + 37.6 \log_{10} (d)
$$

### 阴影衰落

阴影衰落服从均值 \( 0 \) dB，标准差 \( 8 \) dB 的对数正态分布。

---

## 2. 局部散射模型（Local Scattering Model）

### 2.1 均匀线性阵列（ULA, Uniform Linear Array）

在 AP 侧使用 ULA，相邻天线间的信号存在相位差：

$$
\Delta \phi = \frac{\lambda}{2\pi} \cdot d = \frac{\lambda}{2\pi} \cdot \frac{2}{\lambda} \sin(\theta) = \pi \sin(\theta)
$$

其中：
- \( \lambda \) 是信号波长
- \( d = \lambda/2 \) 是相邻天线间距
- \( \theta \) 是入射角

### 阵列响应向量（Array Response Vector）

$$
a_n (\theta) = e^{-j\pi n \sin(\theta)}
$$

表示信号从 \( \theta \) 角度入射时，各天线接收的相对相位偏移。

---

### 2.2 计算空间相关矩阵 \( R \)

空间相关矩阵 \( R \) 用于建模 AP 上天线之间的信号相关性：

$$
R = \sum_{m=1}^{M} a_m a_m^H
$$

#### 代码实现：

```python
R = a.T @ a.conj() / len(angles)
```

✅ `np.pi * sin(θ)` 计算相位偏移，因为 \( \lambda/2 \) 间距的天线相位差是 \( \pi \sin(\theta) \)。  
✅ `np.exp(-1j * π * sin(θ) * np.arange(N))` 计算阵列响应向量，表示不同天线的相对相位偏移。  
✅ `R = a.T @ a.conj() / len(angles)` 计算空间相关矩阵，用于模拟天线之间的信号相关性，并决定 MIMO 信道的特性。

---

## 3. 小尺度衰落（Small-Scale Fading）

在 **相关瑞利衰落（Correlated Rayleigh Fading）** 下，信道的协方差矩阵 \( R \) 影响了信道的结构：

$$
H \sim CN(0, R)
$$

即信道矩阵 \( H \) 服从均值为 0，协方差为 \( R \) 的复高斯分布。

### 3.1 为什么要计算 \( R^{1/2} \)

如果信道是**无相关瑞利衰落**，即 \( R = I \)，那么：

$$
H = z, \quad z \sim CN(0, I)
$$

其中 \( z \) 是 i.i.d. 复高斯随机变量。

但在**相关瑞利衰落**下，需要对 \( z \) 施加空间相关性 \( R \)，使信道满足：

$$
H \sim CN(0, R)
$$

### 实现方法：

$$
H = R^{1/2} z
$$

#### 代码实现：

```python
from scipy.linalg import sqrtm

Rsqrt = sqrtm(R[:,:,l,k])  # 计算 R^(1/2)
noise_vec = (np.random.randn(ANTENNAS_PER_AP) + 1j*np.random.randn(ANTENNAS_PER_AP)) / np.sqrt(2)
H_true[l, :, k] = Rsqrt @ noise_vec
```

这样，`H_true` 就符合 **相关瑞利衰落模型**：

$$
H \sim CN(0, R)
$$

---

## 4. 相关瑞利衰落 vs. 莱斯衰落（Rician Fading）

当 LoS（视距）路径存在时，需要用 Rician Fading。

$$
H = \frac{K}{K+1} H_{\text{LoS}} + \frac{1}{K+1} H_{\text{Rayleigh}}
$$

### 代码修改（支持 Rician Fading）：

```python
K_factor = 3  # 设定 Rician K 因子

for l in range(NUM_AP):
    for k in range(NUM_UE):
        Rsqrt = sqrtm(R[:,:,l,k])
        noise_vec = (np.random.randn(ANTENNAS_PER_AP) + 1j*np.random.randn(ANTENNAS_PER_AP)) / np.sqrt(2)

        # 计算 LoS 成分
        theta_LoS = np.random.uniform(-30, 30)  # 直射路径角度
        H_LoS = np.exp(1j * np.pi * np.sin(np.deg2rad(theta_LoS)) * np.arange(ANTENNAS_PER_AP))

        # 计算 Rician 信道
        H_true[l, :, k] = np.sqrt(K_factor / (K_factor + 1)) * H_LoS + np.sqrt(1 / (K_factor + 1)) * Rsqrt @ noise_vec
```

---

## **总结**
✅ **大尺度衰落**： \( \beta = \text{路径损耗} \times \text{阴影衰落} \)  
✅ **小尺度衰落**： \( H = R^{1/2} z \)  
✅ **相关瑞利衰落**： \( H \sim CN(0, R) \)  
✅ **如果有 LoS，应该用 Rician Fading**  
✅ **Cell-Free Massive MIMO 依赖信道建模优化 AP-UE 协同**


DCC（Dynamic Cooperation Cluster）：：每个 UE 只由信道条件最好的 L 个 AP 进行服务，而不是让所有 AP 为所有 UE 提供信号。基于大尺度衰落 β_matrix，先组成dcc[ue.id]。
Step 2: 形成 DCC（Dynamic Cooperation Cluster）	每个 UE 选择前 L 个最强 AP，形成 动态 AP 组	dcc[ue.id] = top-L AP
Step 3: 遍历每个 UE，检测是否有相同 AP 的干扰	如果 两个 UE 的 DCC 有重叠 AP，则标记为干扰	if set(dcc[ue.id]) & set(dcc[neighbor_ue.id]):
Step 4: 分配导频（Pilot Assignment）	给 干扰最小的 UE 先分配导频，确保相同 AP 服务的 UE 尽量使用不同导频

# 导频分配（Pilot Assignment）

## 1.1 理论基础

在 Cell-Free Massive MIMO 系统中，导频分配是上行信道估计的关键步骤。根据书中的《Section 4.1》，系统使用一组 $\tau_p$ 个相互正交的导频序列 $\{\phi_1, \dots, \phi_{\tau_p}\} \in \mathbb{C}^{\tau_p}$，这些导频满足正交性条件：

$$
\phi_{t_1}^H \phi_{t_2} = \begin{cases}
\tau_p, & \text{if } t_1 = t_2 \\
0, & \text{if } t_1 \neq t_2
\end{cases}
$$

每个用户设备（UE） $k$ 被分配一个导频索引 $t_k \in \{1, \dots, \tau_p\}$。定义集合：

$$
P_k = \{ i : t_i = t_k, i = 1, \dots, K \}
$$

表示所有使用与 UE $k$ 相同导频的 UE。

由于 UE 数量 $K$ 通常大于可用导频数 $\tau_p$，多个 UE 共享同一个导频，这种现象称为**导频污染（pilot contamination）**，会导致信道估计时共享导频的 UE 之间产生干扰。

## 1.2 代码实现

在你的代码中，`assign_pilots` 函数负责为每个 UE 分配导频，并结合**动态合作聚类（Dynamic Cooperation Clustering, DCC）**优化分配过程。

### **实现步骤**

#### 选择服务 AP（DCC）：
对于每个 UE $k$，根据大尺度衰落系数 $\beta_{l,k}$（通常存储在 `beta_matrix` 中），选择信号强度最强的 $L$ 个接入点（AP）作为其服务 AP 集合 $M_k$。

**代码示例：**
```python
ranked_aps = np.argsort(beta_matrix[:, ue.id])[::-1][:L]
dcc[ue.id] = [ap.id for ap in ap_list if ap.id in ranked_aps]
```

#### **导频分配策略：**
- 使用**贪心算法**，尽量避免 DCC 有重叠的 UE 分配到相同的导频，以减少导频污染。
- 对于每个 UE $k$：
  1. 检查其服务 AP 集合 $M_k$ 与已分配导频的其他 UE 的 $M_i$ 是否有交集。
  2. 如果有交集，则将这些 UE 使用的导频加入“不可用”集合。
  3. 从剩余的可用导频 $\{1, \dots, \tau_p\}$ 中选择一个未被标记的导频分配给 UE $k$。
  4. 如果所有导频都不可用（即 $\tau_p < K$ 时无法完全避免重叠），则选择一个默认导频（例如使用最少的导频）。

**代码示例：**
```python
for ue in ue_list:
    used_pilots = set()
    for neighbor_ue in ue_list:
        if neighbor_ue.id in pilot_assignments:
            if len(set(dcc[ue.id]) & set(dcc[neighbor_ue.id])) > 0:
                used_pilots.add(pilot_assignments[neighbor_ue.id])
    for pilot in available_pilots:
        if pilot not in used_pilots:
            pilot_assignments[ue.id] = pilot
            break
    if ue.id not in pilot_assignments:
        pilot_assignments[ue.id] = available_pilots[0]  # 默认分配
```

#### **与理论的联系：**
- 通过避免 DCC 重叠的 UE 使用相同导频，减少了集合 $P_k$ 中干扰 UE 的数量，从而降低了导频污染的影响。
- 但由于 $\tau_p < K$，完全消除导频共享是不可能的，导频污染仍然是系统性能的限制因素。

---

# **导频信号（Pilot Signal）**

## **2.1 理论基础**

在导频传输阶段，每个 UE $k$ 发送其导频信号：

$$
s_k = \eta_k \phi_{t_k}
$$

其中：
- $\eta_k$ 是上行导频功率，
- $\phi_{t_k}$ 是分配的导频序列。

AP $l$ 接收到的导频信号为：

$$
Y_{\text{pilot},l} = \sum_{i=1}^{K} \eta_i h_{il} \phi_{t_i}^T + N_l
$$

其中：
- $h_{il}$ 是 UE $i$ 到 AP $l$ 的信道向量。
- $N_l \sim \mathcal{N}_C(0, \sigma_{ul}^2 I_N)$ 是加性高斯白噪声。

为了估计信道 $h_{kl}$，AP $l$ 将接收信号与归一化的导频 $\phi_{t_k}^* / \tau_p$ 相乘，得到：

$$
y_{\text{pilot}, t_k l} = \frac{Y_{\text{pilot},l} \phi_{t_k}^*}{\tau_p} = \frac{\eta_k}{\tau_p} h_{kl} + \sum_{i \in P_k \setminus \{k\}} \frac{\eta_i}{\tau_p} h_{il} + n_{t_k l}
$$

其中：
- 期望项 $\frac{\eta_k}{\tau_p} h_{kl}$ 是 UE $k$ 的信道贡献。
- 干扰项 $\sum_{i \in P_k \setminus \{k\}} \frac{\eta_i}{\tau_p} h_{il}$ 是导频污染引起的干扰。
- 噪声项 $n_{t_k l} \sim \mathcal{N}_C(0, \sigma_{ul}^2 I_N)$。

## **2.2 代码实现**

在 `mmse_estimate` 函数中，模拟了 AP 接收到的导频信号。

### **实现步骤**

#### 生成正交导频序列：

使用 $\tau_p \times \tau_p$ 的 FFT 矩阵或单位矩阵的列作为导频序列，确保正交性。

**代码示例：**
```python
pilots = np.fft.fft(np.eye(tau_p)) / np.sqrt(tau_p)
```

#### **计算接收信号：**
对于每个 AP $l$ 和导频索引 $t$，计算使用导频 $t$ 的所有 UE 的信道和，并添加噪声。

**代码示例：**
```python
for l in range(num_APs):
    for t in range(tau_p):
        ue_indices = [ue.id for ue in ue_list if pilot_assignments[ue.id] == t]
        if ue_indices:
            H_sum = np.sum(H_true[l, :, ue_indices], axis=0)
        else:
            H_sum = np.zeros(N, dtype=complex)
        noise = (np.random.randn(N) + 1j * np.random.randn(N)) * np.sqrt(sigma2) / np.sqrt(2)
        Y_pilot[l, :, t] = np.sqrt(p) * H_sum + noise
```

---

**完整的 MMSE 信道估计、导频污染等内容，可继续添加。**

