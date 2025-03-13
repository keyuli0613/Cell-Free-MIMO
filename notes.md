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

