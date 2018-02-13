import numpy as np
from .distribution import Distribution


class Exponential(Distribution):

    def __init__(self, l, varred=True):
        super(Exponential, self).__init__(varred=varred)
        self.lam = l

    # generate a random value from an exponential distribution,
    # the formula was derived using the integral transformation technique
    def random(self, u):
        return -(1 / self.lam) * np.log(u)
