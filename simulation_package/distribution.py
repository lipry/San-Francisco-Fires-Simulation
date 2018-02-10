from abc import ABC, abstractmethod
import numpy as np


class Distribution(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def random(self):
        pass

    def rvs(self, n):
        return [self.random() for _ in range(0, n)]

    def cdf(self, n):
        data = self.rvs(n)
        s = np.sort(data)
        yvals = np.arange(len(s))/float(len(s))
        return sorted, yvals

    def pdf(self, n, n_bins):
        data = self.rvs(n)
        counts, bins = np.histogram(data, bins=n_bins, normed=True)
        return bins[1:], counts
