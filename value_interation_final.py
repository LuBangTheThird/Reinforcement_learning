import numpy as np

# 环境状态转移概率和奖励 (P[s][a]) 是一个字典，表示状态s下执行动作a后的状态转移和奖励
# 假设一个简单的4个状态、5个动作的环境
# 动作  概率 状态 奖励

# 规定 2为禁入区,target 为 4
P = {
    # 状态1 动作从1-5代表上 左 下 右 不动
    1: {1: [(1.0, 1, -1)], 2: [(1.0, 2, -1)], 3: [(1.0, 3, 0)], 4: [(1.0, 1, -1)],
        5: [(1.0, 1, 0)]},

    2: {1: [(1.0, 2, -1)], 2: [(1.0, 2, -1)], 3: [(1.0, 4, 1)], 4: [(1.0, 1, 0)],
        5: [(1.0, 2, -1)]},

    3: {1: [(1.0, 1, 0)], 2: [(1.0, 4, 1)], 3: [(1.0, 3, -1)], 4: [(1.0, 3, -1)],
        5: [(1.0, 3, 0)]},

    4: {1: [(1.0, 2, -1)], 2: [(1.0, 4, -1)], 3: [(1.0, 4, -1)], 4: [(1.0, 3, 0)],
        5: [(1.0, 4, 1)]},

}

#4个状态5个动作
n_states = 4
n_actions = 5
# 折扣因子
gamma = 0.9

threshold = 1e-6  # 终止条件

V = np.zeros(n_states)  # 初始化状态价值V(s) np.zeros 创建一个由零组成的（n_states）维数组
policy = np.zeros(n_states+1, dtype=int)  # 初始化策略 每个状态应采取的策略 我们这里 只取 索引为1开始的
# print(policy[1:])
# print(V)


# ----------------------------------------------------------------------------------------------------

# 环境的状态和动作数量


def bellman_update(V, P, gamma):
    delta = 0
    for s in range(1, n_states + 1):

        max_q = -np.inf  # 调用np 里面的 inf 意思是初始化一个 负无穷的数，这样就更好比较了
        #初始化max_q哪里有一点问题。我们应该要找最小的q而不是简单的初始化为0

        best_action = None  # 定义best_action 为空值，避免 初始化的歧义！ 需要注意的是 None 不能与其他值进行比较

        # prob 概率  下一个状态  奖励
        for a in range(1, n_actions + 1):
            for prob, next_state, reward in P[s][a]:
                # PDF35 面的公式 需要注意的是这个求和是针对这个问题的，不具有一般性，如果要一般性的话，最好分两部求解！
                q = prob * (reward + gamma * V[next_state-1])  # 这个代码里面初始的V是0 V[next_state]是一个数
                # print()
                # print(q)
               # max_q = q
                if q > max_q:
                    max_q = q
                    best_action = a
        v = max_q
        # print()
        # print(v)
        delta = max(delta, np.abs(v - V[s-1]))
        # 这里就更新了v
        V[s-1] = v
        policy[s] = best_action
        # print(V[s-1])
        # print(V)
    return delta, policy

#迭代更新价值函数
def value_iteration(V, P, gamma, threshold):
    while True:
        delta ,policy= bellman_update(V, P, gamma)
        if delta < threshold:
            break
    return V,policy

# 计算状态价值
V_final,best_policy = value_iteration(V, P, gamma, threshold)
print("最终的状态价值: ", V_final)
print("最终的最优策略: ", best_policy[1:])
