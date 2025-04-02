import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 收敛后的状态（归一化后的4维状态）
states = np.array([
    [0.13333333, 0.44444444, 0.0, 0.47368421],
    [0.2,        0.44444444, 0.0, 0.21052632],
    [0.6,        0.33333333, 0.0, 0.21052632],
    [0.73333333, 0.33333333, 0.0, 0.0],
    [0.8,        0.33333333, 0.0, 0.0],
    [1.0,        0.33333333, 0.0, 0.0],
    [0.53333333, 0.33333333, 0.0, 0.0],
    [0.33333333, 0.33333333, 0.0, 0.0],
    [0.4,        0.33333333, 0.0, 0.0],
    [0.2,        0.33333333, 0.0, 0.0],
    [0.0,        0.33333333, 0.0, 0.0],
])

# 转换为 DataFrame 便于可视化
df = pd.DataFrame(states, columns=['num_UE', 'cluster_size', 'precoding_idx', 'rho_tot'])
df['Step'] = np.arange(len(df))

# 绘图
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Step', y='num_UE', label='num_UE')
sns.lineplot(data=df, x='Step', y='cluster_size', label='cluster_size')
sns.lineplot(data=df, x='Step', y='precoding_idx', label='precoding_idx')
sns.lineplot(data=df, x='Step', y='rho_tot', label='rho_tot')
plt.title("State Evolution in Episode 60 (Normalized)")
plt.ylabel("Normalized Value")
plt.xlabel("Time Step")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
