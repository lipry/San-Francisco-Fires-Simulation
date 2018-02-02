import numpy as np
from scipy import random
from .distribution import Distribution


class Gamma(Distribution):

    def __init__(self, n, l):
        self.n = n
        self.lam = l

    # generate a random value from a gamma distribution,
    # the formula was derived using the integral transformation technique
    def random(self):
        return -(1/self.lam)*np.log(np.prod(np.random.uniform(size=self.n)))
