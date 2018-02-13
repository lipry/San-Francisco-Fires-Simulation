from abc import ABC, abstractmethod
import numpy as np


class Distribution(ABC):

    def __init__(self, varred=True):
        self.varred = varred

    @abstractmethod
    def random(self, u):
        pass

    def rvs(self, n):
        u = np.random.uniform(0, 1, int(np.floor(n)))
        first = [self.random(x) for x in list(u)]
        if self.varred:
            second = [self.random(1-x) for x in list(u)]
            return first+second

        return first

    def cdf(self, n):
        data = self.rvs(n)
        s = np.sort(data)
        yvals = np.arange(len(s))/float(len(s))
        return sorted, yvals

    def pdf(self, n, n_bins):
        data = self.rvs(n)
        counts, bins = np.histogram(data, bins=n_bins, normed=True)
        return bins[1:], counts
