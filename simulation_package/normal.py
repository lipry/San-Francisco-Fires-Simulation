import numpy as np
from scipy import random
from .distribution import Distribution
import math


class Normal(Distribution):

    def __init__(self, mu=0.0, variance=1.0, t="bm", log=False):
        super(Normal, self).__init__(varred=False)
        self.mu = mu
        self.sigma = np.sqrt(variance)
        self.technique = t
        self.log = log
        self.buffer = []

    def random(self, u=0):
        z = self.general_normal()
        if self.log:
            return np.exp(z)
        return z

    # generate a random value from a normal distribution
    def general_normal(self):
        if len(self.buffer) > 0:
            z = self.buffer.pop()
        else:
            z = self.std_normal()
        if self.mu == 0.0 and self.sigma == 1.0:
            return z
        return self.mu + self.sigma * z

    def std_normal(self):
        if self.technique is "bm":
            return self.std_box_muller_random()
        return self.std_approx_random()

    # generate a random value from a normal distribution,
    # the formula was derived using the box-muller technique
    def std_box_muller_random(self):
        u_theta = random.random()
        u_rho = random.random()
        theta = 2 * math.pi * u_theta
        rho = math.sqrt(-2 * math.log(u_rho))
        self.buffer.append((rho * np.sin(theta)))
        return rho * np.cos(theta)

    # generate a random value from a normal distribution,
    # the formula was derived using a approximated technique
    @staticmethod
    def std_approx_random(n=12):
        return np.sqrt(12/n)*(np.sum(np.random.uniform(size=n))-n/2)