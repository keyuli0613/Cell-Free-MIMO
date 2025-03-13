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

函数部分解释
1. assign_pilots 函数
python

Collapse

Wrap

Copy
def assign_pilots(ue_list, ap_list, beta_matrix, L=3, tau_p=None):
功能
基于大尺度衰落系数 beta_matrix，为每个 UE 选择前 L 个最强 AP，形成 DCC。
为每个 UE 分配导频，尽量避免 DCC 有重叠的 UE 使用相同导频，减少导频污染。
主要步骤
初始化导频数：
python

Collapse

Wrap

Copy
if tau_p is None:
    tau_p = len(ue_list)
如果未指定 tau_p，则默认导频数等于 UE 数量，表示每个 UE 可以获得唯一的导频。若 tau_p < len(ue_list)，则会发生导频复用。
形成 DCC：
python

Collapse

Wrap

Copy
dcc = {}
for ue in ue_list:
    ranked_aps = np.argsort(beta_matrix[:, ue.id])[::-1][:L]
    dcc[ue.id] = [ap.id for ap in ap_list if ap.id in ranked_aps]
对每个 UE，从 beta_matrix[:, ue.id] 中提取所有 AP 的大尺度衰落系数。
使用 np.argsort(...)[::-1] 按降序排列，[:L] 取前 L 个最大值的索引（即最强 AP）。
将这些 AP 的 id 存储在 dcc[ue.id] 中。
导频分配（贪心算法）：
python

Collapse

Wrap

Copy
pilot_assignments = {}
available_pilots = list(range(tau_p))
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
        pilot_assignments[ue.id] = available_pilots[np.argmin(list(used_pilots))] if used_pilots else available_pilots[0]
初始化可用导频集合 available_pilots 为 [0, 1, ..., tau_p-1]。
对每个 UE：
检查其他已分配导频的 UE，若它们的 DCC 与当前 UE 的 DCC 有交集（set(dcc[ue.id]) & set(dcc[neighbor_ue.id])），则将它们的导频加入 used_pilots。
从 available_pilots 中选择一个不在 used_pilots 中的导频分配给当前 UE。
如果所有导频都被占用（used_pilots 包含所有导频），则选择一个备选导频（这里简单取第一个可用导频或最小干扰的导频）。
输出
pilot_assignments：每个 UE 的导频分配。
dcc：每个 UE 的服务 AP 集合。
2. mmse_estimate 函数
python

Collapse

Wrap

Copy
def mmse_estimate(ap_list, ue_list, H_true, pilot_assignments, p, sigma2):
功能
基于接收到的导频信号，使用 MMSE 方法估计信道矩阵 H_hat。

主要步骤
初始化和生成正交导频序列：
python

Collapse

Wrap

Copy
num_APs = len(ap_list)
num_UEs = len(ue_list)
N = ap_list[0].antennas
tau_p = len(ue_list)
pilots = np.fft.fft(np.eye(tau_p)) / np.sqrt(tau_p)
tau_p 默认等于 UE 数量。
使用 FFT 生成正交导频矩阵 pilots，形状为 (tau_p, tau_p)，每列是一个正交导频序列。
模拟 AP 接收导频信号：
python

Collapse

Wrap

Copy
Y_pilot = np.zeros((num_APs, N, tau_p), dtype=complex)
for l, ap in enumerate(ap_list):
    for t in range(tau_p):
        ue_indices = [ue.id for ue in ue_list if pilot_assignments[ue.id] == t]
        if ue_indices:
            H_sum = np.sum(H_true[l, :, ue_indices], axis=0)
        else:
            H_sum = np.zeros(N, dtype=complex)
        noise = (np.random.randn(N) + 1j * np.random.randn(N)) * np.sqrt(sigma2)
        Y_pilot[l, :, t] = np.sqrt(p) * H_sum + noise
Y_pilot 形状为 (num_APs, N, tau_p)，表示每个 AP 在每个导频时隙接收到的信号。
对于每个 AP l 和导频时隙 t：
找到使用导频 t 的所有 UE（ue_indices）。
计算这些 UE 的真实信道和 H_sum。
添加高斯噪声，得到接收信号 Y_pilot[l, :, t]。
MMSE 估计：
python

Collapse

Wrap

Copy
H_hat = np.zeros_like(H_true)
for l, ap in enumerate(ap_list):
    for k, ue in enumerate(ue_list):
        t = pilot_assignments[ue.id]
        Psi = p * tau_p * np.sum(
            [H_true[l, :, i].conj() @ H_true[l, :, i].T for i in range(num_UEs) if pilot_assignments[i] == t], 
            axis=0
        ) + sigma2 * np.eye(N)
        H_hat[l, :, k] = np.sqrt(p) * np.linalg.inv(Psi) @ Y_pilot[l, :, t]
对每个 AP l 和 UE k：
获取该 UE 使用的导频 t。
计算 Psi 矩阵，表示所有使用导频 t 的 UE 的干扰加上噪声。
使用 MMSE 公式 H_hat[l, :, k] = sqrt(p) * inv(Psi) @ Y_pilot[l, :, t] 估计信道。
添加估计误差：
python

Collapse

Wrap

Copy
error_shape = H_hat.shape
estimation_error = np.sqrt(ESTIMATION_ERROR_VAR / 2) * (
    np.random.randn(*error_shape) + 1j * np.random.randn(*error_shape)
)
return H_hat + estimation_error
生成一个与 H_hat 形状相同的复高斯噪声，标准差由 ESTIMATION_ERROR_VAR 确定。
将噪声加到 H_hat 上，模拟实际系统中的估计不完美。
输出
H_hat：估计的信道矩阵，包含估计误差。