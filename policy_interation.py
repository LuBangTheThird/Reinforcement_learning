import numpy as np

# 环境状态转移概率和奖励 (P[s][a]) 是一个字典，表示状态s下执行动作a后的状态转移和奖励
# 假设一个简单的2个状态、3个动作的例子
# 环境:{动作:[(概率 状态 奖励)] ---------}
# 状态 0 1 表示 s1 s2 动作 0 1 2 表示 左 不动 右
# 规定状态 2 为 target_area 到target 奖励为1 碰边界奖励为 -1 其他为 0
P = {
    0: {0: [(1.0, 0, -1)], 1: [(1.0, 0, 0)], 2: [(1.0, 1, 1)]},
    1: {0: [(1.0, 0, 0)], 1: [(1.0, 1, 1)], 2: [(1.0, 1, -1)]},
}

# 初始化环境的状态和动作数量
n_states = 2
n_actions = 3
# 折扣因子
gamma = 0.9

# 最初的策略
# 创建的是 n_states 行 n_actions 列的矩阵 策略表！
policy = np.zeros([n_states, n_actions])
policy[0][0] = 1.0
policy[1][0] = 1.0
print("最初的策略是:", policy)


def policy_evaluation(policy, V, threshold=1e-6, gamma=0.9):
    # 利用迭代算法求解 state value PDF 41
    while True:  # 是否收敛，收敛则退出循环
        delta = 0
        # 遍历 s
        for s in range(n_states):
            v = 0
            # 求和得出v(s)
            # policy = np.ones([n_states, n_actions]) / n_actions
            # 初始化"随机"策略 这个策略其实是给定的 [[0.5 0.5] [0.5 0.5] [0.5 0.5] [0.5 0.5]]

            # print("当前的策略是:", policy)

            # 它用于将一个可迭代的数据对象（如列表、元组或字符串）组合为一个索引序列，
            # 同时列出数据和数据下标。这个函数返回一个枚举对象，该对象可以被迭代，每次迭代都会返回一个包含索引和对应值的元组。
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward in P[s][a]:
                    # 这里的 prob 是我们初始化设定的P中的值 根据PDF 35面大标题2.5上面的那个公式 为 p(s'|s,a) = p(r|s,a)
                    # action_prob 是 pi(a|s) 策略实施的概率
                    v += action_prob * prob * (reward + gamma * V[next_state])  # PDF 35面大标题2.5上面的那个公式
                    print("v= ", v)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
            # PDF P 41  保证了迭代的收敛性
        if delta < threshold:
            break
    return V


def policy_improvement(V, policy, gamma=0.9):
    # 这只是一轮的策略改进，我们要进行很多轮的策略改进，所以外面还得有一个循环，终止条件就是我们的策略不再更新
    policy_stable = True

    for s in range(n_states):
        print("当前", s, "的策略是:", policy[s])

        #  n_states 行 n_actions 列的矩阵 策略表！
        # policy = np.ones([n_states, n_actions]) / n_actions
        # 初始化"随机"策略 这个策略其实是给定的 [[0.5 0.5] [0.5 0.5] [0.5 0.5] [0.5 0.5]]

        # np.argmax 函数用于返回数组中最大值的索引
        old_action = np.argmax(policy[s])  # 针对 状态 s 的所有策略 定义这个是为了后面判断动作是否更新
        print("the old_action is ", old_action)

        # 初始化 action_value为0
        # 这个初始化为空是不是好一点？numpy中好像没有这种函数？---可以用 inf 负无穷来初始化   、
        # 这个action_values的定义其实不必担心，因为我们最后还有一个求和去更新他，而不是直接通过比较大小去更新！！！！！
        # print(f"初始化action_value如果为空{np.empty(n_actions)}")

        action_values = np.zeros(n_actions)  # 对动作函数的初始化 这里是关于每个动作的一张表
        for a in range(n_actions):
            for prob, next_state, reward in P[s][a]:
                # p(s'|s,a) = p(r|s,a)
                action_values[a] = prob * (reward + gamma * V[next_state])  # PDF 35 的公式
                print("动作", a, "的action_value 是", action_values[a])
        # 找最大的action
        new_action = np.argmax(action_values)
        if old_action != new_action:
            policy_stable = False
            # np.eye(n_actions) 创建一个 n_actions x n_actions 的单位矩阵，其中对角线上的元素为1，其余元素为0。
            # np.eye(n_actions)[new_action] 从单位矩阵中选择第 new_action 行，得到一个长度为 n_actions 的向量
            # 策略数组 policy 在状态 s 下的策略向量就表示在该状态下选择动作 new_action 的概率为1，而选择其他动作的概率为0。
        policy[s] = np.eye(n_actions)[new_action]
        print("状态", s, "的最优策略是", policy[s])
    return policy, policy_stable


#print(policy_evaluation(policy, np.zeros(n_states)))
def policy_iteration(policy):
    #创建的是 n_states 行 n_actions 列的矩阵 策略表！
    #policy = np.ones([n_states, n_actions]) / n_actions  # 初始化"随机"策略 这个策略其实是给定的 [[0.5 0.5] [0.5 0.5][0.5 0.5][0.5 0.5]]
    #print(f"初始化策略是 ：{policy}")
    V = np.zeros(n_states)  # 初始化价值函数 这里所有的V都是0
    while True:
        # 策略评估
        V = policy_evaluation(policy, V)
        # 策略改进
        policy, policy_stable = policy_improvement(V, policy)
        # 如果策略已经不再改变，则结束
        if policy_stable:
            break
    return policy, V


# 执行策略迭代
optimal_policy, optimal_value = policy_iteration(policy)

print("最优策略:")
print(optimal_policy)
print("最优价值函数:")
print(optimal_value)
