import numpy as np
from rl_environment import RLEnvironment  # 确保路径和模块一致

# 固定参数
FIXED_CLUSTER_SIZE = 1
FIXED_PRECODING_IDX = 0
FIXED_RHO_TOT = 0.47368421
LAMBDA_ENERGY = 0.001

def run_fixed_strategy(user_data, verbose=True):
    env = RLEnvironment(
        user_data=user_data,
        init_cluster_size=FIXED_CLUSTER_SIZE,
        init_precoding_idx=FIXED_PRECODING_IDX,
        init_rho_tot=FIXED_RHO_TOT,
        lambda_energy=LAMBDA_ENERGY
    )
    
    state = env.reset()
    total_reward = 0.0
    rewards = []

    for t in range(len(user_data)):
        # 这里不采取动作，固定参数状态
        next_state, reward, done = env.step(action=None)  # 不执行任何动作
        rewards.append(reward)
        total_reward += reward

        if verbose:
            print(f"Step={t}, UE={state[0]}, Reward={reward:.4f}, NextState={next_state}")
        state = next_state

        if done:
            break
    
    avg_reward = total_reward / len(user_data)
    print(f"\n[Fixed Strategy Result] Total Reward: {total_reward:.2f}, Avg Reward per Step: {avg_reward:.2f}")
    return rewards

if __name__ == "__main__":
    # 示例用户数量数据
    user_data = [5, 6, 12, 14, 15, 18, 11, 8, 9, 6, 3] 
    run_fixed_strategy(user_data)
