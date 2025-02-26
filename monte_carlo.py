import numpy as np
from grid_world import windy_grid,standard_grid,negative_grid,ACTION_SPACE
from iterative_policy_evalution_deterministic import print_values, print_policy

GAMMA = 0.9

def play_game(grid,policy,max_step = 20):
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])
    s = grid.current_state()
    rewards = [0]
    states = [s]
    
    for _ in range(max_step):
        a = policy[s]
        r = grid.move(a)
        s = grid.current_state()
        rewards.append(r)
        states.append(s)
        if grid.game_over():
            break

        
    return states, rewards

if __name__ == '__main__':
    grid = standard_grid()
    
    print("rewards: ")
    print_values(grid.rewards, grid)
    
    policy = {
        (2, 0): 'R',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'D',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }
    
    V = {}
    returns = {}
    states = grid.all_states() 
    for s in states:
        if s in grid.actions:
            returns[s] = []
        else:
            V[s] = 0
            
    for t in range(100):
        states , rewards = play_game(grid, policy)
        G = 0

        for i in range (len(states)-2,-1,-1):
            s = states[i]
            r = rewards[i+1]
            G = r + GAMMA * G
            if s not in states[:i]:
                returns[s].append(G)
                V[s] = np.mean(returns[s])

#print states and rewards


print("values: ")   
print_values(V, grid)
print("policy: ")
print_policy(policy, grid)

