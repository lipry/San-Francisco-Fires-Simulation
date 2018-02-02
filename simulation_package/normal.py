import numpy as np
from scipy import random
from .distribution import Distribution
import math


class Normal(Distribution):

    def __init__(self, mu=0.0, sigma=1.0, t="bm", log=False):
        self.mu = mu
        self.sigma = sigma
        self.technique = t
        self.log = log

    def random(self):
        z = self.general_normal()
        if self.log:
            return tuple([np.exp(i) for i in z]) if isinstance(z, tuple) else np.exp(z)
        return z

    # generate a random value from a normal distribution
    def general_normal(self):
        z = self.std_normal()
        if self.mu == 0.0 and self.sigma == 1.0:
            return z
        return tuple([self.mu + self.sigma * i for i in z])

    def std_normal(self):
        if self.technique is "bm":
            bm = self.std_box_muller_random()
            return bm
        return self.std_approx_random()

    # generate a random value from a normal distribution,
    # the formula was derived using the box-muller technique
    @staticmethod
    def std_box_muller_random():
        u_theta = random.random()
        u_rho = random.random()
        theta = 2 * math.pi * u_theta
        rho = math.sqrt(-2 * math.log(u_rho))
        return rho * np.cos(theta), rho * np.sin(theta)

    # generate a random value from a normal distribution,
    # the formula was derived using a approximated technique
    @staticmethod
    def std_approx_random(n=12):
        return np.sqrt(12/n)*(np.sum(np.random.uniform(size=n))-n/2)
