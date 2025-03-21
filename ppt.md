无小区大规模 MIMO（Cell-Free Massive MIMO）建模精炼笔记
引言与基本原理
无小区大规模 MIMO 是一种新型网络架构，其中大量分布式接入点（APs）在没有传统蜂窝小区边界的情况下，通过协同传输来服务区域内的所有用户​
MA-MIMO.ELLINTECH.SE
。与传统蜂窝网络每个用户仅由单个基站服务不同，无小区架构下所有AP（或选定的AP集合）可以同时为任意用户提供服务，从而消除小区边界造成的干扰和“边缘”效应​
MA-MIMO.ELLINTECH.SE
。这种用户中心（user-centric）的网络理念来源于网络MIMO/协作多点(CoMP)技术，但通过Massive MIMO技术的结合，实现了在整个覆盖区域内的统一高性能服务​
MA-MIMO.ELLINTECH.SE
​
MA-MIMO.ELLINTECH.SE
。这种架构的核心思想是：AP分布式部署、集中或协同处理，以达到更均匀的服务质量和更高的频谱效率。 
MDPI.COM
​图1：传统蜂窝网络 (a) vs. 无小区大规模 MIMO (b) vs. 用户中心的无小区大规模 MIMO 架构 (c)。在无小区架构中，大量AP通过前传链路连接到中央处理单元（CPU），协同为所有用户提供服务；而用户中心的无小区架构中，每个用户由其附近的一组AP动态组网服务。 无小区大规模 MIMO 系统通常假设工作在时分双工（TDD）模式下，以利用信道的互易性简化下行链路的预编码设计。大量AP通过高速前传链路与中央处理单元（CPU）相连（或分布式处理），以协调传输和接收。由于所有AP协同工作，理想情况下每个用户终端（UE）都能从多个AP的相干联合传输中获益，提升接收信号强度并降低干扰​
ARXIV.ORG
​
ARXIV.ORG
。然而，全网协作也带来了可扩展性挑战：当AP和用户数量增加时，信道状态信息的开销、前传通信和处理复杂度都会急剧上升​
ARXIV.ORG
。为此，实用的无小区系统通常采用用户中心的动态合作簇（DCC）等策略，使每个用户由有限的若干AP服务，从而在保证性能的同时控制系统开销。下面，我们将围绕关键理论背景对该系统的建模要点进行精炼阐述。
信道建模
在无小区 Massive MIMO 系统中，精确的信道建模对性能评估至关重要。信道通常包含大尺度衰落（路径损耗和阴影衰落）和小尺度衰落（快衰落）两部分。如果AP和用户设备都配备多根天线，还需考虑空间相关性（用相关矩阵描述）。下面分别介绍这些要素：
大尺度衰落
大尺度衰落描述了因距离衰减和阴影阻挡导致的平均信号强度变化。常用模型是假设信号的平均功率随距离呈幂律衰减，并叠加对数正态阴影衰落。例如，可以定义AP $m$ 与用户 $k$ 之间的大尺度衰落系数 $\beta_{mk}$ 表示平均信道增益，其典型模型为:
路径损耗: 根据传播模型，$\beta_{mk}$ 随距离 $d_{mk}$ 衰减，如 $\mathrm{PL}(d_{mk}) \propto d_{mk}^{-\gamma}$（$\gamma$为路径损耗指数）。在对数域可以表示为 $10\log_{10}\beta_{mk} = -\gamma \cdot 10\log_{10}(d_{mk}) + C$（$C$为频段相关常数），并可根据经验模型（如COST-231-Hata模型等）确定具体公式。
阴影衰落: 用一个零均值的对数正态随机变量表示。实际实现中可令 $\beta_{mk} = \mathrm{PL}(d_{mk}) \times 10^{\frac{X_{mk}}{10}}$，其中 $X_{mk}\sim \mathcal{N}(0,\sigma_{\text{shadow}}^2)$ 表示以分贝计的阴影衰落​
RESEARCHGATE.NET
。$\sigma_{\text{shadow}}$常取值如 8 dB，表示阴影衰落的标准差。
通过上述模型，我们得到 $\beta_{mk}$，它反映了在长时间尺度上AP-$m$到用户-$k$信道的平均强度和衰减。仿真中，常假设大尺度系数在多播时延内保持不变（慢衰落），只随用户位置变化。
小尺度衰落
小尺度衰落刻画信号在较小时间/空间尺度上的快速波动，通常由多径干涉引起，可看作随机分布。大规模MIMO通常假设瑞利衰落（各径无直视成分）或莱斯衰落（存在一定直视径成分）。在经典瑞利模型下，AP $m$ 到用户 $k$ 的小尺度信道可以建模为复高斯随机变量 $g_{mk}\sim \mathcal{CN}(0,1)$，即幅度服从瑞利分布、相位均匀分布。若考虑AP和用户都可能有多天线，则对应的信道向量 $\mathbf{h}{mk}$ 可建模为 $\mathbf{h}{mk}\sim \mathcal{CN}(\mathbf{0}, \mathbf{I})$ 在无相关假设下。通常假设块衰落模型：在一个相干块（如几毫秒内若干OFDM符号）内，$g_{mk}$ 保持恒定，块间变化独立。这样，实际信道系数可表示为： 
ℎ
𝑚
𝑘
=
𝛽
𝑚
𝑘
  
𝑔
𝑚
𝑘
,
h 
mk
​
 = 
β 
mk
​
 
​
 g 
mk
​
 , 其中 $\sqrt{\beta_{mk}}$ 表示大尺度衰落的振幅衰减，而 $g_{mk}$ 表示小尺度快衰落。这一模型意味着平均功率为$\beta_{mk}$，瞬时归一化为$\mathcal{CN}(0,\beta_{mk})$分布。 在仿真实现中，可以如下生成Rayleigh信道矩阵的元素（Python示例）：
python
复制
# 假设已有距离矩阵 dist[m,k] 以及计算路径损耗的函数 path_loss(d)
beta = path_loss(dist[m,k]) * 10**(sigma_shadow * np.random.randn() / 10)  # 大尺度衰落
h_mk = np.sqrt(beta/2) * (np.random.randn() + 1j*np.random.randn())       # 小尺度瑞利衰落
上述代码中，我们先根据距离计算路径损耗，再乘以阴影衰落因子得到 $\beta_{mk}$，随后生成复高斯随机数并乘以 $\sqrt{\beta_{mk}}$ 得到信道系数 $h_{mk}$。
空间相关矩阵
当AP或用户终端配备有多根天线时，信道的各天线分量之间往往存在空间相关性。例如，AP阵列天线间距较小时，不同天线看到的信号角度分布相似，会导致其信道增益不独立。为此，可用空间相关矩阵来刻画信道相关性。如果AP $m$ 有 $N$ 根天线，用户 $k$ 单天线（下行情况），则 $\mathbf{h}_{mk}$ 可视作 $N\times1$的向量，其统计分布可定义为： 
ℎ
𝑚
𝑘
∼
𝐶
𝑁
(
0
,
𝑅
𝑚
𝑘
)
,
h 
mk
​
 ∼CN(0,R 
mk
​
 ), 其中 $\mathbf{R}{mk}$ 是 $N\times N$ 的正定协方差矩阵，即相关矩阵。$\mathbf{R}{mk}$ 的对角元素通常取1（各天线增益归一化），而非对角元素反映天线之间的相关系数（可依据经验模型，如指数相关模型：$[\mathbf{R}{mk}]{ij} = \rho^{|i-j|}$，$0\le \rho<1$）。在仿真中，可先根据阵列几何和散射环境生成 $\mathbf{R}{mk}$，再通过矩阵平方根法获得相关信道：$\mathbf{h}{mk} = \mathbf{R}{mk}^{1/2} \mathbf{g}{mk}$，其中 $\mathbf{g}_{mk}\sim \mathcal{CN}(0,\mathbf{I}_N)$ 是独立瑞利信道向量。这种方法确保仿真信道符合指定相关性质。 需要注意，在大规模分布式AP场景中，不同AP间的阴影衰落也可能相关（例如相邻AP受到相似环境遮挡）。不过在基础研究中，经常假设不同AP对同一用户的阴影衰落独立，以简化分析。
导频分配与动态合作聚类 (DCC)
无小区Massive MIMO系统的一大挑战是导频分配和AP合作集的形成。由于所有AP共享整个频带，如果用户数 $K$ 超过正交导频的数量 $\tau_p$，则必须重用导频，从而导致导频污染：多个用户共享同一导频时，AP无法将其信道估计完全区分开​
ARXIV.ORG
。合理的导频分配策略和动态合作簇 (Dynamic Cooperation Clustering) 机制可以缓解这一问题，并提高系统可扩展性。
导频分配
在 TDD 模式下，系统通常设有 $\tau_p$ 个正交导频序列（长度为$\tau_p$符号），用户在上行训练时发送导频，AP据此估计信道​
ARXIV.ORG
。当 $K > \tau_p$ 时，不可避免会有多个用户共享同一序列（记共享同一导频$t$的用户集合为 $S_t$）。为了减小由此带来的干扰，导频分配需要考虑用户的地理分布和服务AP的重叠情况。一种有效的原则是：避免将同一导频分配给由相同AP服务的多个用户​
ARXIV.ORG
。换言之，每个AP每个导频时隙内只服务至多一个使用该导频的用户​
ARXIV.ORG
。这样可保证在任何单个AP上，同频导频的用户不会相互干扰其信道估计，从物理上降低了导频污染的影响。实现这种原则的具体方法可以包括：根据用户的位置或大尺度衰落划分导频组，或采用图着色算法确保相邻（共享AP）的用户导频不同。 导频污染会降低信道估计精度并引入额外干扰。特别地，它有两个主要后果：​
ARXIV.ORG
(1) 降低信道估计质量，使得协同传输的效果变差；(2) 使共享导频用户的估计信道彼此相关，在数据传输时造成相干干扰。因此，除了巧妙的导频指派外，还可以在导频阶段采用功控（例如离AP近的用户降低导频功率，以减少对远距用户的干扰）等措施，进一步减轻污染。
动态合作聚簇 (DCC)
动态合作聚簇是无小区网络提高可扩展性和干扰管理的关键机制。由于让每个AP服务所有用户在实践中开销巨大，DCC提倡用户中心的AP选择：为每个用户动态挑选一组“服务AP簇”，AP只需与簇内其他AP协同服务该特定用户​
ARXIV.ORG
​
ARXIV.ORG
。不同用户的簇可以重叠，且可随时间动态调整，这与传统固定小区或静态分簇形成鲜明对比​
ARXIV.ORG
。动态聚簇根据大尺度信道条件（如$\beta_{mk}$大小）来决定哪个AP应属于哪个用户的簇。例如，一个简单的用户中心策略是：每个用户选择对自己信道增益最大的 $M_0$ 个AP组成服务簇​
ARXIV.ORG
。这些AP将负责为该用户进行数据传输和接收，而远距增益小的AP可以不参与，降低前传开销和处理复杂度。 DCC 的“动态”体现在簇的组成会随环境和网络状况改变。例如，当用户移动时，其附近高增益的AP集合也会相应变化；又或者在不同时刻、不同频段，不同用户的干扰情况变化，系统可以按需调整AP簇以优化性能​
ARXIV.ORG
​
ARXIV.ORG
。值得注意的是，DCC 策略通常与导频分配联动：由于每个用户主要由自己簇内的AP服务，那么尽量保证同簇用户使用不同导频，可避免簇内导频污染。因此，初始接入时的联合算法会同时考虑导频分配和簇形成，以达到全局最优​
ARXIV.ORG
。研究表明，采用DCC的用户中心无小区系统，在保证每个用户由足够AP服务的前提下，其性能可以逼近全协作（所有AP服务所有用户）的上界，但开销大幅降低​
ARXIV.ORG
。 在实际仿真或实现中，DCC算法可能如下伪代码描述：
python
复制
# 输入: 每个AP对每个用户的平均信道增益 beta[m,k]
for each user k:
    # 选取增益最大的 M0 个AP作为服务簇
    cluster_k = np.argsort(beta[:, k])[-M0:]  
    assign user k to all APs in cluster_k
    # 确保这些AP在此用户的导频上不服务其他用户（满足每AP每导频单用户原则）
上述伪代码体现的策略是一种简单实现。更高级的实现中，还可能加入对AP负载的控制（避免某些AP同时服务过多用户）以及簇间干扰的评估。不过核心思想都是：根据大尺度增益划分用户-AP关联关系，并动态更新。
MMSE 信道估计及误差分析
最小均方误差 (MMSE) 信道估计是在导频训练后获取信道状态信息的最优线性估计方法。在无小区Massive MIMO中，由于每个AP可能服务多个用户且存在导频复用，各AP通常各自基于接收到的导频信号进行本地的MMSE估计，然后将结果发送到CPU（集中方案）或直接用于本地数据检测/预编码（分布式方案）。MMSE估计充分利用了信道的统计先验（例如大尺度衰落$\beta_{mk}$和空间相关矩阵$\mathbf{R}_{mk}$），相比简单的最小二乘(LS)估计能显著提高精度。 信道估计过程：假设导频长度为 $\tau_p$，所有用户以功率 $p_p$ 同时发送导频。AP $m$ 接收到导频序列 $t$ 上的信号为： 
𝑦
𝑚
,
𝑡
=
∑
𝑖
∈
𝑆
𝑡
𝑝
𝑝
 
ℎ
𝑚
𝑖
+
𝑛
𝑚
,
y 
m,t
​
 =∑ 
i∈S 
t
​
 
​
  
p 
p
​
 
​
 h 
mi
​
 +n 
m
​
 , 其中 $S_t$ 是与用户 $k$ 共用导频$t$的用户集合（如果用户$k$的导频是独享的，则 $S_t$ 仅有用户$k$自己），$\mathbf{n}m$ 是观测噪声向量（假设 $\mathcal{CN}(0,\sigma^2 \mathbf{I})$）。对于单天线AP的简单情况，上式可以看作标量相加。同理，$\mathbf{y}{m,t}$ 是不同用户导频信道的叠加。MMSE估计器利用已知的统计量${\beta_{mi}, \mathbf{R}_{mi}}$计算期望和协方差，从而给出对特定用户$k$信道的最佳线性估计： 
ℎ
^
𝑚
𝑘
=
𝐸
[
ℎ
𝑚
𝑘
𝑦
𝑚
,
𝑡
𝐻
]
(
𝐸
[
𝑦
𝑚
,
𝑡
𝑦
𝑚
,
𝑡
𝐻
]
)
−
1
𝑦
𝑚
,
𝑡
.
h
^
  
mk
​
 =E[h 
mk
​
 y 
m,t
H
​
 ](E[y 
m,t
​
 y 
m,t
H
​
 ]) 
−1
 y 
m,t
​
 . 直观来说，MMSE估计得到的是接收信号$\mathbf{y}{m,t}$的一个加权版本，权重选择使得均方误差最小。对于上式，$\mathbb{E}[\mathbf{y}{m,t}\mathbf{y}_{m,t}^H]$ 是导频接收的自相关矩阵，包括了期望信道功率和噪声（在协作簇和导频分配策略合理时，这个矩阵尽可能接近对角矩阵，以减少不同用户分量的耦合）。在实际计算中，AP可预先计算好这些统计矩阵，然后对每次收到的导频信号直接应用矩阵求逆乘法得到信道估计​
ARXIV.ORG
。例如，对于单天线AP的标量情况，MMSE估计简化为： 
ℎ
^
𝑚
𝑘
=
𝑝
𝑝
𝛽
𝑚
𝑘
∑
𝑖
∈
𝑆
𝑡
𝑝
𝑝
𝛽
𝑚
𝑖
+
𝜎
2
/
𝜏
𝑝
 
𝑦
𝑚
,
𝑡
,
h
^
  
mk
​
 = 
∑ 
i∈S 
t
​
 
​
 p 
p
​
 β 
mi
​
 +σ 
2
 /τ 
p
​
 
p 
p
​
 
​
 β 
mk
​
 
​
 y 
m,t
​
 , 其中分母部分反映了用户$k$在AP $m$处的导频信号干扰加噪声功率。如果导频$t$仅由用户$k$使用，那么 $S_t$ 无其他用户，估计退化为 $\hat{h}{mk} = \frac{\sqrt{p_p}\beta{mk}}{p_p \beta_{mk} + \sigma^2/\tau_p} y_{m,t}$。 误差分析：MMSE信道估计的误差 $\tilde{\mathbf{h}}{mk} = \mathbf{h}{mk} - \hat{\mathbf{h}}{mk}$ 依然是随机的，可计算其均方误差 (MSE) 或方差来衡量估计精度。以上述标量情况为例，$\hat{h}{mk}$ 是线性高斯估计，则估计和真实信道呈Joint Gaussian，可知误差也是高斯且与估计值独立。误差方差为： 
M
S
E
𝑚
𝑘
=
𝐸
{
∣
ℎ
~
𝑚
𝑘
∣
2
}
=
𝛽
𝑚
𝑘
−
𝑝
𝑝
𝛽
𝑚
𝑘
2
∑
𝑖
∈
𝑆
𝑡
𝑝
𝑝
𝛽
𝑚
𝑖
+
𝜎
2
/
𝜏
𝑝
.
MSE 
mk
​
 =E{∣ 
h
~
  
mk
​
 ∣ 
2
 }=β 
mk
​
 − 
∑ 
i∈S 
t
​
 
​
 p 
p
​
 β 
mi
​
 +σ 
2
 /τ 
p
​
 
p 
p
​
 β 
mk
2
​
 
​
 . 从中我们可以看出几个要点：
当没有导频污染（$S_t$仅含$k$自己）且信噪比足够高（$p_p\beta_{mk}\gg \sigma^2/\tau_p$），则 $\mathrm{MSE}{mk} \approx \beta{mk} - \beta_{mk} = 0$，表示可以非常精确地估计信道；实际中有限功率下也会剩下一点误差。
当存在导频污染时，即使$ p_p \to \infty$，误差方差也趋近于 $\beta_{mk} - \frac{\beta_{mk}^2}{\sum_{i\in S_t}\beta_{mi}}$，这是一个误差下限，说明导频共享用户之间的干扰使估计不可能完美。这正是导频污染导致的估计精度损失之一​
ARXIV.ORG
。特别地，不同用户共导频导致的估计向量相关还会在数据传输时引入相干干扰，使某些干扰项在结合波束赋形时按天线数放大​
ARXIV.ORG
。
总体而言，MMSE 信道估计通过结合统计先验大大改善了信道CSI获取的准确性，但其性能仍受限于导频资源受限和干扰。实际系统设计中，会力图增加导频序列长度（提高$\tau_p$）、合理分配导频以及采用DCC策略，来尽可能降低误差对性能的影响。
5. 功率分配（rho_dist）
在无小区大规模 MIMO 系统的下行操作中，功率分配是优化频谱效率（SE）和用户公平性的关键环节。合理的功率分配能够平衡信号强度与干扰抑制，确保每个用户设备（UE）获得适当的服务质量。以下是功率分配的实现方式及其理论背景：

代码实现
python

Collapse

Wrap

Copy
rho_dist = power_allocation(gain_matrix, D, RHO_TOT)
作用：为每个 AP-UE 对分配下行发射功率 $\rho_{l,k}$，结果存储在形状为 $(NUM_AP, NUM_UE)$ 的矩阵中，表示每个接入点（AP） $l$ 为用户 $k$ 分配的功率。
输入参数：
增益矩阵 $gain_matrix$：反映大尺度衰落系数 $\beta_{l,k}$，表示 AP $l$ 到 UE $k$ 的平均信道增益。
服务矩阵 $D$：其中 $D[l,k] = 1$ 表示 AP $l$ 服务 UE $k$，$D[l,k] = 0$ 表示不服务。
总功率约束 $RHO_TOT_PER_AP$：每个 AP 可用的最大下行发射功率。
实现细节
对于每个 AP $l$：

确定服务 UE 集合：根据服务矩阵 $D$，找到 AP $l$ 需要服务的用户集合 ${k | D[l,k] = 1}$。
计算分配权重：采用平方根加权分配方法，计算 $\sqrt{\beta_{l,k}}$ 作为功率分配的权重。这是因为 $\sqrt{\beta_{l,k}}$ 能够反映路径损耗的补偿需求，同时避免对少数强信道用户过度分配功率。
功率分配：将 $RHO_TOT_PER_AP$ 按 $\sqrt{\beta_{l,k}}$ 的比例分配给服务集合中的每个 UE，确保总功率满足约束：
𝜌
𝑙
,
𝑘
=
𝑅
𝐻
𝑂
_
𝑇
𝑂
𝑇
_
𝑃
𝐸
𝑅
_
𝐴
𝑃
⋅
𝛽
𝑙
,
𝑘
∑
𝑘
′
∈
{
𝑘
∣
𝐷
[
𝑙
,
𝑘
′
]
=
1
}
𝛽
𝑙
,
𝑘
′
ρ 
l,k
​
 =RHO_TOT_PER_AP⋅ 
∑ 
k 
′
 ∈{k∣D[l,k 
′
 ]=1}
​
  
β 
l,k 
′
 
​
 
​
 
β 
l,k
​
 
​
 
​
 
若 AP $l$ 不服务任何 UE，则 $\rho_{l,k} = 0$。
理论联系
上行-下行对偶性：平方根加权分配受到书中 Theorem 6.2 的启发。该定理表明，通过适当的功率分配，下行信号与干扰加噪声比（SINR）可以等于上行 SINR，从而实现对称性能。然而，实际系统中，功率分配需同时考虑每个 AP 的功率约束和用户公平性，因此代码实现并非严格遵循对偶性推导，而是更倾向于实用优化。
公平性与效率平衡：通过 $\sqrt{\beta_{l,k}}$ 分配功率，系统能够在补偿远距离用户路径损耗的同时，避免将过多功率集中于近距离用户，从而提升整体网络的公平性和频谱效率。
6. 下行 SE 计算
下行频谱效率（SE）是衡量无小区大规模 MIMO 系统性能的核心指标，反映了系统在给定带宽下能够传输的有效数据速率。下行 SE 的计算依赖于信道估计质量、预编码方案以及功率分配策略。以下是其实现方式和理论分析：

代码结构
python

Collapse

Wrap

Copy
for scheme in ['MR', 'L-MMSE', 'P-MMSE']:
    se_list_downlink = []
    for ue in ue_list:
        serving_aps = [ap for ap in ap_list if ap.id in ue.assigned_ap_ids]
        if not serving_aps:
            se_list_downlink.append(0.0)
            continue
    
        se_val = compute_downlink_SE(
            serving_aps=serving_aps,
            H_hat=H_hat,           # 信道估计
            H_true=H_true,         # 真实信道
            ue_id=ue.id,
            all_ues=ue_list,
            lN=ANTENNAS_PER_AP,    # 每 AP 天线数
            p=UE_MAX_POWER,        # 用户最大功率（用于上行参考）
            sigma2=NOISE_DL,       # 下行噪声功率
            precoding_scheme=scheme,
            rho_dist=rho_dist,     # 功率分配结果
            D=D,                   # 服务矩阵
            pilot_assignments=pilot_assignments  # 导频分配方案
        )
        se_list_downlink.append(se_val)

    downlink_SE_all[scheme].append(np.mean(se_list_downlink))
结构说明
外层循环：遍历三种预编码方案：
MR (Maximum Ratio)：最大比率预编码，简单高效，适用于信道条件较好的场景。
L-MMSE (Local MMSE)：本地最小均方误差预编码，通过抑制 AP 内的干扰提升性能。
P-MMSE (Partial MMSE)：部分 MMSE 预编码，适用于大规模网络，通过近似全局优化平衡性能与计算复杂度。
内层循环：为每个 UE 计算 SE：
根据服务矩阵 $D$ 和 UE 的服务 AP 列表（ue.assigned_ap_ids）确定服务 AP 集合。
若某 UE 无服务 AP，则 SE 为 0。
调用 compute_downlink_SE 函数计算该 UE 的下行 SE。
输出：每种预编码方案下，所有 UE 的平均 SE 存储在 downlink_SE_all[scheme] 中，用于后续性能对比。
compute_downlink_SE 函数
该函数负责为特定 UE 计算下行 SE，核心步骤包括预编码向量计算和 SINR/SE 计算。

预编码向量计算
根据选择的预编码方案，计算服务 AP 对目标 UE 的预编码向量 $w_{l,k}$：

MR 预编码：
𝑤
𝑙
,
𝑘
=
ℎ
^
𝑙
,
𝑘
𝜌
𝑙
,
𝑘
w 
l,k
​
 = 
h
^
  
l,k
​
  
ρ 
l,k
​
 
​
 
其中，$\hat{h}{l,k}$ 是信道估计，$\rho{l,k}$ 是功率分配值。MR 预编码直接利用信道共轭，计算复杂度低，但不抑制干扰。
L-MMSE 预编码：
𝑤
𝑙
,
𝑘
=
𝜌
𝑙
,
𝑘
⋅
(
𝐶
int
+
𝜎
2
𝐼
)
−
1
ℎ
^
𝑙
,
𝑘
w 
l,k
​
 = 
ρ 
l,k
​
 
​
 ⋅(C 
int
​
 +σ 
2
 I) 
−1
  
h
^
  
l,k
​
 
其中，$C_{\text{int}}$ 是本地干扰协方差矩阵，考虑服务 AP 内其他 UE 的信号干扰。L-MMSE 通过权衡信号增益与干扰抑制提升性能。
P-MMSE 预编码： 采用部分 MMSE 方法，近似全局干扰抑制，具体实现参考书中公式 (6.17)。P-MMSE 在大规模网络中通过折中计算复杂度和性能，提供优于 MR 和 L-MMSE 的结果。
SINR 和 SE 计算
信号功率：
∣
∑
𝑙
∈
serving_aps
𝑤
𝑙
,
𝑘
𝐻
ℎ
𝑙
,
𝑘
∣
2
​
  
l∈serving_aps
∑
​
 w 
l,k
H
​
 h 
l,k
​
  
​
  
2
 
表示服务 AP 对目标 UE $k$ 的相干信号贡献，其中 $h_{l,k}$ 是真实信道。
干扰功率：
∑
𝑖
≠
𝑘
∑
𝑙
∣
𝑤
𝑙
,
𝑖
𝐻
ℎ
𝑙
,
𝑘
∣
2
i

=k
∑
​
  
l
∑
​
  
​
 w 
l,i
H
​
 h 
l,k
​
  
​
  
2
 
表示所有其他 UE $i$ 的预编码向量对目标 UE $k$ 造成的干扰。
SINR：
SINR
𝑘
=
信号功率
干扰功率
+
𝜎
2
SINR 
k
​
 = 
干扰功率+σ 
2
 
信号功率
​
 
其中，$\sigma^2$ 是下行噪声功率（NOISE_DL）。
SE：
SE
𝑘
=
log
⁡
2
(
1
+
SINR
𝑘
)
SE 
k
​
 =log 
2
​
 (1+SINR 
k
​
 )
基于香农容量公式，计算 UE $k$ 的下行频谱效率（单位：bit/s/Hz）。
理论联系
信道估计依赖：下行 SE 高度依赖 MMSE 信道估计的准确性。若导频污染严重，$\hat{h}{l,k}$ 与 $h{l,k}$ 的误差将增加干扰功率，降低 SINR。
预编码方案比较：
MR 预编码简单，适用于低干扰场景，但在高负载下性能受限。
L-MMSE 通过本地干扰抑制改善 SINR，适用于中等规模网络。
P-MMSE 通过部分全局优化，在高干扰或大规模场景中表现更优。
功率分配影响：平方根加权分配通过调整 $\rho_{l,k}$，直接影响信号功率和干扰功率的分布，进而优化 SE。
强化学习在 AP 选择中的作用
在无小区大规模 MIMO网络中，AP选择（即每个用户由哪些AP服务）是决定系统性能和资源利用的关键问题之一。传统上，AP选择可以基于瞬时或统计信道信息采取贪婪算法（例如选最大增益的若干AP）或者固定规则。然而，在复杂动态环境中，例如用户移动、流量负载变化、无线信道时变等因素，静态或启发式的AP选择策略可能难以适应最优。强化学习 (Reinforcement Learning, RL) 提供了一种数据驱动的方法，能够在与环境的反复交互中自动逼近最优策略，对于解决AP选择这样复杂的组合优化问题展现出潜力。 AP选择的优化目标可以多种多样，例如：最大化系统总吞吐量、保证用户公平性的最小化某百分位用户的干扰、降低AP能耗等等。我们可以将AP选择建模为一个马尔可夫决策过程 (MDP)：
状态（State）: 系统当前的无线环境状态，例如所有用户的大尺度信道增益矩阵${\beta_{mk}}$，或经过降维处理的特征（如每个用户最佳几条链路的增益值、队列长度等）。状态反映了做决策时的环境条件。
动作（Action）: 为每个用户分配服务AP集合的方案。例如可以表示为一个二进制矩阵决策：$a_{mk}=1$表示选择AP $m$服务用户$k$，$0$表示不选。由于每个用户通常限制选择若干AP，动作空间虽然大但可以通过分解为每个用户各自的选择来处理。
奖励（Reward）: 执行动作后所得到的性能反馈。例如可定义整个网络的总速率或某种效用函数作为即时奖励。为了体现长期利益，奖励也可综合考虑随时间的性能（折扣累计值）。
RL 智能体通过反复尝试不同的AP选择方案并观察奖励，不断更新策略。常用的方法有Q-learning（对每个状态-动作对估计价值）和策略梯度或深度神经网络的方法（例如Deep Q Network, 深度强化学习）来应对连续大型状态空间。对于无小区网络，这通常意味着采用深度神经网络来近似状态->动作的策略映射。 强化学习的优势在于：它不需要显式的数学信道模型优化推导，而是通过与真实或仿真的环境交互自动寻找好的策略。这对于一些难以解析的问题（如带有限前传容量、多用户干扰耦合的AP子集选择）特别有用。举例来说，系统可以运行一个智能体，其在每个时隙根据当前观测到的大尺度信道情况选择各用户的服务AP集合，然后根据实际测得的吞吐量或能效作为奖励反馈。随着足够多的训练迭代，智能体会逐渐学到某种近似最优的AP选择策略。例如它可能学会在高负载时缩小每用户簇以减少干扰，在低负载时扩大簇提高信号增益，或者为某些关键用户保留更多AP资源等等。这种自适应调配能够超越简单阈值或固定规则，在复杂场景下取得更佳性能。 需要注意的是，RL方法也有挑战：训练过程可能较长，特别是在大型网络下状态动作空间巨大时，需要借助分布式多智能体学习、状态抽象等技术。此外，保证学习过程的稳定和所得策略的可靠（避免极端情况下性能骤降）也很重要。因此，在学术研究中，强化学习常用于AP选择的概念验证，展示相对传统方法的增益，并探索智能通信系统的前景。
小结
通过以上要点整理，我们精炼了无小区大规模 MIMO 系统建模中的核心概念和理论背景，包括：无小区架构的基本原理、信道模型构建（大尺度/小尺度衰落及相关性）、导频复用下的干扰及应对策略（导频分配与动态合作簇）、MMSE信道估计的方法与误差，以及智能算法（强化学习）在AP选择优化中的应用前景。这些概念相互关联：精确的信道建模是基础，合理的导频和簇管理保证了信道估计和协同传输的效果，而新兴的智能算法则有望在复杂场景下进一步提升系统性能。希望经过优化的笔记结构和内容能帮助读者更清晰地理解无小区大规模 MIMO的模型与关键技术，为后续学术研究或汇报提供有力支撑。




