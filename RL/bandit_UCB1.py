import numpy as np
import matplotlib.pyplot as plt

class MyCasinoMachine:
    def __init__(self,true_mean):
        self.true_mean = true_mean
        self.estimate = 0
        self.count = 1

    def pull(self):
        return np.random.random() < self.true_mean

    def update(self,x):
        self.count += 1
        self.estimate = (((self.count - 1))* self.estimate + x) / self.count
         
np.random.seed(32)
true_means = [0.2,0.5,0.75]
machines = [MyCasinoMachine(true_mean) for true_mean in true_means]
epsilon = 0.1
num_steps = 10000
num_machines = len(machines)

best_machine_indx = np.argmax(true_means)

counts = np.zeros(num_machines)
values = np.zeros(num_machines)

rewards = []
avg_rewards = []
best_choise = []
avg_estimate = [[] for _ in range(num_machines)]
total_step = 0

for j in range(len(machines)):
    x = machines[j].pull()
    total_step += 1
    machines[j].update(x)
#count_eachmachine = [[] for _ in range(num_machines)]
for i in range(num_steps):

    machine_indx = np.argmax(([casino.estimate + np.sqrt((2 * np.log2(total_step)/casino.count)) for casino in machines] ))
    
    reward = machines[machine_indx].pull()
    machines[machine_indx].update(reward)
    counts[machine_indx] += 1
    values[machine_indx] += (reward - values[machine_indx]) / counts[machine_indx]
    total_step += 1
    rewards.append(reward)
    avg_rewards.append(np.mean(rewards))
    best_choise.append(machine_indx == best_machine_indx)
    avg_estimate[machine_indx].append(machines[machine_indx].estimate)

cumulative_best = np.cumsum(best_choise)
percentage_best = cumulative_best / (np.arange(num_steps) + 1)
#avg estimanate
cumulative_reward = np.cumsum(rewards)
winrate = cumulative_reward / (np.arange(num_steps) + 1)

#show the estimate of the best machine plot for all itearation
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(avg_estimate[best_machine_indx], label='Best Machine Estimate')
plt.axhline(y=true_means[best_machine_indx], color='r', linestyle='--', label='True Best Machine')
plt.xlabel('Plays')
plt.ylabel('Average Reward')   
plt.title('Best Machine Estimate Over Time')
plt.xscale('log')
plt.legend()
#plt.show()


plt.subplot(1, 2, 2)
plt.plot(winrate)
plt.plot(np.ones(num_steps) * np.max(true_means))  
#plt.show()
#plt estimate
for i in range(num_machines):
    plt.plot(avg_estimate[i], label='Machine {}'.format(i))
plt.axhline(y=true_means[best_machine_indx], color='r', linestyle='--', label='Max Possible Reward')
plt.xlabel('Plays')
plt.ylabel('Average Reward')
plt.title('Average Reward Estimates Over Time')
plt.xscale('log')
plt.legend()
plt.show()
plt.tight_layout()

# Average reward plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(avg_rewards, label='Observed Average Reward')
plt.axhline(y=true_means[best_machine_indx], color='r', linestyle='--', 
            label='Max Possible Reward')
plt.xlabel('Plays')
plt.ylabel('Average Reward')
plt.title('Average Reward Over Time')
plt.xscale('log')
#plt.legend()

# Best machine selection plot
plt.subplot(1, 2, 2)
plt.plot(percentage_best * 100, label='Best Machine Selection Rate')
plt.axhline(y=100, color='r', linestyle='--', label='Ideal Performance')
plt.xlabel('Plays')
plt.ylabel('Percentage (%)')
plt.title('Percentage of Optimal Machine Selection')
plt.xscale('log')
plt.legend()


plt.tight_layout()
plt.show()