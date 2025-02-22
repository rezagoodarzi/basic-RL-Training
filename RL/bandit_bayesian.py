from __future__ import print_function, division
from builtins import range
#import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

np.random.seed(32)
NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    def __init__(self, p):
        self.p = p
        self.a = 1
        self.b = 1
        self.N = 0 

    def pull(self):
        return np.random.random() < self.p

    def sample(self):
        return np.random.beta(self.a, self.b)

    def update(self, x):
        self.a += x
        self.b += 1 - x
        self.N += 1


def plot(bandits, trial):
    x = np.linspace(0, 1, 200)

    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label=f"real p: {b.p:.4f}, win rate = {b.a - 1}/{b.N}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
#    plt.show()


def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

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
    # print total reward
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [b.N for b in bandits])


if __name__ == "__main__":
    plt.figure(figsize=(18, 10))
    experiment()