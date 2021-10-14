import numpy as np
from scipy import stats


class Mean(object):

    def fit(self, X, y):
        self.mean = np.mean(y)

    def predict(self, X):
        return self.mean

class Linear(object):

    def fit(self, X, y):
        slope, intercept, r, p, std_err = stats.linregress(X, y)
        self.f = lambda x: slope * x + intercept

    def predict(self, X):
        return self.f(X)

class MeanLinear(object):

    def fit(self, X, y):
        slope, intercept, r, p, std_err = stats.linregress(X, y)
        self.mean = np.mean(y)
        self.f = lambda x: slope * x + intercept
        return stats.linregress(X, y)

    def predict(self, X):
        return (self.f(X) + self.mean) / 2