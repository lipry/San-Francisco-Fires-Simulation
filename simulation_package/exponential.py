import numpy as np
from scipy import random
from .distribution import Distribution


class Exponential(Distribution):

    def __init__(self, l):
        self.lam = l

    # generate a random value from an exponential distribution,
    # the formula was derived using the integral transformation technique
    def random(self):
        return -(1 / self.lam) * np.log(random.random())
