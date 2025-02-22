import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evalution_deterministic import print_values, print_policy
import pandas as pd
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def play_game(grid, policy, max_steps=20):

    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    a = np.random.choice(ALL_POSSIBLE_ACTIONS) # first action is uniformly random

    states = [s]
    actions = [a]
    rewards = [0]

    for _ in range(max_steps):
        r = grid.move(a)
        s = grid.current_state()

        rewards.append(r)
        states.append(s)
        
        if grid.game_over():
            break
        else:
            a = epcilon_greedy_policy(s, policy)
            actions.append(a)

    # states  = [s(0), s(1), ..., s(T-1), s(T)]
    # actions = [a(0), a(1), ..., a(T-1),     ]
    # rewards = [   0, R(1), ..., R(T-1), R(T)]

    return states, actions, rewards

def epcilon_greedy_policy(s, policy, epsilon=0.2):
    if np.random.random() < epsilon:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)
    else:
        return policy[s]
    
def max_dict(d):
    max_val = max(d.values())
    max_keys = [key for key, val in d.items() if val == max_val]

    ### slow version
    # max_keys = []
    # for key, val in d.items():
    #   if val == max_val:
    #     max_keys.append(key)

    return np.random.choice(max_keys), max_val


if __name__ == '__main__':

    grid = negative_grid(-0.1)


    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)

    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # initialize Q(s,a) and returns
    Q = {}
    sample_counts = {}
    state_sample_counts = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions: # not a terminal state
            Q[s] = {}
            sample_counts[s] = {}
            state_sample_counts[s] = 0
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0
                sample_counts[s][a] = 0
        else:
            # terminal state or state we can't otherwise get to
            pass

    deltas = []
    for it in range(10000):
        
        biggest_change = 0
        states, actions, rewards = play_game(grid, policy)

        states_actions = list(zip(states, actions))

        T = len(states)
        G = 0
        for t in range(T - 2, -1, -1):
            # retrieve current s, a, r tuple
            s = states[t]
            a = actions[t]

            G = rewards[t+1] + GAMMA * G

            # check if we have already seen (s, a) ("first-visit")
            if (s, a) not in states_actions[:t]:
                old_q = Q[s][a]
                sample_counts[s][a] += 1
                lr = 1 / sample_counts[s][a]
                Q[s][a] = old_q + lr * (G - old_q)

                # update policy
                policy[s] = max_dict(Q[s])[0]
                state_sample_counts[s] = state_sample_counts[s] + 1
                # update delta
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.xscale('log')
    plt.show()

    print("final policy:")
    print_policy(policy, grid)

    # find V
    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]

    print("final values:")
    print_values(V, grid)
    
    print("state_sample_count:")
    state_sample_count_arr = np.zeros((grid.rows, grid.cols))
    for i in range(grid.rows):
        for j in range(grid.cols):
            if (i, j) in state_sample_counts:
                state_sample_count_arr[i,j] = state_sample_counts[(i, j)]
    df = pd.DataFrame(state_sample_count_arr)
    print(df)