第一页：
项目目标： 建立cell free 模型，然后模拟一天之类接入UE的变化，根据强化学习模型，训练出一个动态调整AP配置的模型

第二页
cell free 背景：
Cellular 传统基站：信号覆盖不均匀，边缘区域用户信号速率弱，边缘处多个基站覆盖
重叠导致干扰，没有有效协调机制，用户在小区间移动时无法及时切换基站，单一基站
能耗高
cell free：无小区间干扰；用户通常可以找到距离较近的天线进行通信，降低路径损耗；
所有能够接收到用户信号的天线都会协同提供服务，提高 SNR 和数据速率；由中央处
理单元（CPU）协调所有 AP，实现全局优化，实现更好的负载均衡# 信道建模总结

第三页
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

第四页
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

第五页
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

## 2.2 代码实现

在 `mmse_estimate` 函数中，模拟了 AP 接收到的导频信号。

### 实现步骤

#### 生成正交导频序列

使用大小为 $\tau_p \times \tau_p$ 的FFT矩阵或单位矩阵的列作为导频序列，确保正交性。

**代码示例：**

```python
pilots = np.fft.fft(np.eye(tau_p)) / np.sqrt(tau_p)
```

#### 计算接收信号

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

这里 $p$ 对应 $\eta_k$（假设所有 UE 功率相同），$\sigma^2$ 对应 $\sigma_{ul}^2$。

### 与理论的联系

代码中的 $Y_{pilot}[l, :, t]$ 对应书中的 $y_{pilot, t}^{l}$，表示解扩后的接收信号。
干扰项隐含在 $H_{sum}$ 中，包含了所有 $i \in P_k$ 的信道贡献，体现了导频污染。

第六页

## 3. MMSE信道估计（MMSE Channel Estimation）

### 3.1 理论基础

根据书中的推导，MMSE信道估计器为：

$$
\hat{h}_{kl} = \eta_k \tau_p R_{kl} \Psi_{t_k l}^{-1} y_{pilot, t_k}^{l}
$$

其中：

$$
\Psi_{t_k l} = \sum_{i \in P_k} \eta_i \tau_p R_{il} + \sigma_{ul}^2 I_N
$$

$R_{kl}$ 是信道 $h_{kl}$ 的相关矩阵（通常为对角矩阵，元素为大尺度衰落系数）。$\Psi_{t_k l}$ 是接收信号 $y_{pilot, t_k}^{l}$ 的协方差矩阵，包含所有共享导频 UE 的统计信息和噪声。

估计误差的协方差矩阵为：

$$
C_{kl} = R_{kl} - \eta_k \tau_p R_{kl} \Psi_{t_k l}^{-1} R_{kl}
$$

### 3.2 代码实现

在 `mmse_estimate` 函数中，实现了 MMSE 信道估计。

**代码示例：**

```python
for l in range(num_APs):
    for t in range(tau_p):
        Psi = sum(eta[i] * tau_p * R[l,:,:,i] for i in pilot_groups[t]) + sigma2 * np.eye(N)
        Psi_inv = np.linalg.inv(Psi)
```

注：
- $R[l,:,: ,i]$ 是 UE $i$ 到 AP $l$ 的相关矩阵。

#### 计算MMSE信道估计

```python
H_hat = np.zeros((num_APs, N, num_UEs), dtype=complex)
for l in range(num_APs):
    for k, t_k in enumerate(pilot_assignments):
        Psi = sum(eta[i] * tau_p * R[l, :, :, i] for i in pilot_groups[t_k]) + sigma_ul ** 2 * np.eye(N)
        Psi_inv = np.linalg.inv(Psi)
        H_hat[l, :, k] = eta_k * tau_p * R_kl @ Psi_inv @ Y_pilot[l, :, t_k]
```

#### 添加估计误差

为了模拟实际的不完美估计，可添加随机误差：

```python
estimation_error = np.sqrt(ESTIMATION_ERROR_VAR / 2) * (np.random.randn(*H_hat.shape) + 1j * np.random.randn(*H_hat.shape))
H_hat += estimation_error
```

### 与理论的联系

- 误差项 $C_{kl}$ 在理论上由公式给出，代码中通过随机噪声近似模拟。

## 4. 导频污染（Pilot Contamination）

### 4.1 理论基础

导频污染的理论模型为：

$$
\Psi_{t_k l} = \eta_k \tau_p R_{kl} + \sum_{i \in P_k \setminus \{k\}} \eta_i \tau_p R_{il} + \sigma_{ul}^2 I_N
$$

干扰项使得逆矩阵变大，降低了信道估计精度。

### 4.2 在代码中的体现

- 当 $\tau_p < K$ 时，污染不可避免。
- MMSE估计中的 $\Psi_{t_k l}$ 明确包含了共享导频的UE统计信息。
- 污染越严重（即共享导频的UE越多），估计越不准确。



第七页
## 5. 功率分配（rho_dist）

```python
rho_dist = power_allocation(gain_matrix, D, RHO_TOT)
```

**作用**：为每个 AP-UE 对分配下行发射功率 $\rho_{l,k}$，结果存储在形状为 $(NUM\_AP, NUM\_UE)$ 的矩阵中。

### 实现细节
输入包括：
- 增益矩阵 $gain\_matrix$
- 服务矩阵 $D$，其中 $D[l,k] = 1$ 表示 AP $l$ 服务 UE $k$
- 每个 AP 的总功率约束 $RHO\_TOT\_PER\_AP$

对于每个 AP $l$：
1. 找到它服务的 UE。
2. 计算 $\beta_{l,k}$（平方根增益）。
3. 按比例分配功率，确保总和不超过 $RHO\_TOT\_PER\_AP$。

**理论联系**：
这种平方根加权分配受到书中 Theorem 6.2（上行-下行对偶性）的启发。定理指出，通过适当的功率分配，下行 SINR 可以等于上行 SINR。然而，代码中的方法更注重实际功率约束和公平性优化，而非严格遵循对偶性的理论推导。

---

## 6. 下行 SE 计算

```python
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
            D=D,
            pilot_assignments=pilot_assignments
        )
        se_list_downlink.append(se_val)

    downlink_SE_all[scheme].append(np.mean(se_list_downlink))
```

### 结构

#### 外层循环：
遍历三种预编码方案：
- **MR (Maximum Ratio)**：简单、低复杂度的预编码。
- **L-MMSE (Local MMSE)**：本地最小均方误差预编码，抑制干扰。
- **P-MMSE (Partial MMSE)**：部分 MMSE，适用于大规模网络。

#### 内层循环：
为每个 UE 计算 SE：
1. 根据 $D$ 和 UE 的服务 AP 列表确定服务 AP。
2. 调用 `compute_downlink_SE` 计算 SE。

**输出**：
计算每种方案下所有 UE 的平均 SE，存储在 `downlink_SE_all[scheme]` 中。

---

## `compute_downlink_SE` 函数

### 预编码向量计算

#### MR 预编码：
\[
    w_{l,k} = \hat{h}_{l,k} \rho_{l,k}
\]
基于信道估计和功率分配。

#### L-MMSE 预编码：
考虑本地干扰，计算为：
\[
    w_{l,k} = \rho_{l,k} \cdot \left( C_{int} + \sigma^2 I \right)^{-1} \hat{h}_{l,k}
\]

#### P-MMSE 预编码：
近似全局干扰，基于部分 MMSE 方法。

---

## SINR 和 SE 计算

### 信号功率：
\[
    \left| \sum_{l} w_{l,k}^H h_{l,k} \right|^2
\]

### 干扰功率：
\[
    \sum_{i \neq k} \sum_{l} \left| w_{l,i}^H h_{l,k} \right|^2
\]

### SINR：
\[
    \frac{\text{信号功率}}{\text{干扰功率} + \sigma^2}
\]

### SE 计算：
\[
    \log_2(1 + \text{SINR})
\]


第八页
强化学习模型场景：就是大概是我策略动作是 1.调整聚类的大小，几个AP服务一个UE（用衰落系数选），2，下行传输时用不同的预编码方案（MR或LMMSE）3. AP的最大功率。奖励是下行速率SE+总能耗。然后我想用之前数据里那个一周内每隔20分钟的流量，转化成UE数量，加到状态里，根据不同的时延要求，可以调整SE在奖励中的系数。然后状态是UE数量加动作参数的那些。每次UE数量更新的时候，dqn训练一次，更新一下状态。然后用经验池去联系上各个episode。看奖励曲线。