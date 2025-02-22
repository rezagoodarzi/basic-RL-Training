import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evalution_deterministic import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def epsilon_greedy(policy,s,eps=0.1):
    p = np.random.random()
    if p < (1 - eps):
        return policy[s]
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)
    
if __name__ == '__main__':
    
    grid = negative_grid()

    print("rewards:")
    print_values(grid.rewards, grid)

    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }

    # initialize V(s) and returns
    V = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0
        
    deltas = []
    
    number_of_steps = 10000
    for i in range(number_of_steps):
        grid.reset()
        deltaa = 0
        while not grid.game_over():
            s = grid.current_state()
            a = epsilon_greedy(policy,s)
            r = grid.move(a)
            new_state = grid.current_state()
            v_old  = V[s]
            V[s] = V[s] + ALPHA * (GAMMA * V[new_state] + r - V[s])
            deltaa = max(deltaa , np.abs(V[s]- v_old))
            s = new_state

        deltas.append(deltaa)
        
    plt.plot(deltas)
    plt.show()

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)