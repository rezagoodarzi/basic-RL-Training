from __future__ import print_function, division
from builtins import range

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

np.random.seed(32)
NUM_TRIALS = 2000
BANDIT_Means = [1, 2, 3]


class Bandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.m = 0
        self.lambda_ = 1
        self.tau = 1
        self.N = 0

    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean

    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.m

    def update(self, x):
        self.m = (self.tau * x + self.lambda_ * self.m) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1


def plot(bandits, trial):
    x = np.linspace(-3, 6, 200)
    for b in bandits:
        y = norm.pdf(x, b.m, np.sqrt(1. / b.lambda_))
        plt.plot(x, y, label=f"real mean: {b.true_mean:.4f}, num plays: {b.N}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
#    plt.show()


def experiment():
    bandits = [Bandit(p) for p in BANDIT_Means]

    sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
    rewards = np.zeros(NUM_TRIALS)
    for i in range(NUM_TRIALS):
    
        j = np.argmax([b.sample() for b in bandits])

        if i in sample_points:
            plt.subplot(5, 2, sample_points.index(i) + 1)
            plot(bandits, i)

        x = bandits[j].pull()

        rewards[i] = x

        bandits[j].update(x)
    plt.show()
    plt.legend()
    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)
  # plot moving average ctr
    plt.plot(cumulative_average)
    for m in BANDIT_Means:
        plt.plot(np.ones(NUM_TRIALS)*m)
    plt.legend(["Bandit 1 true mean", "Bandit 2 true mean", "Bandit 3 true mean"])
    plt.show()

    # print total reward
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [b.N for b in bandits])


if __name__ == "__main__":
    plt.figure(figsize=(18, 10))
    experiment()