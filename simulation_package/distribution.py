from abc import ABC, abstractmethod
import numpy as np


class Distribution(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def random(self):
        pass

    def rvs(self, n):
        values = [self.random() for _ in range(0, n)]
        values = values if not isinstance(self.random(), tuple) else np.reshape(values, n*2)
        return np.array(values)

    def cdf(self, n):
        data = self.rvs(n)
        s = np.sort(data)
        yvals = np.arange(len(s))/float(len(s))
        return sorted, yvals

    def pdf(self, n, n_bins):
        data = self.rvs(n)
        print(data)
        counts, bins = np.histogram(data, bins=n_bins, normed=True)
        return bins[1:], counts
