import numpy as np


class GeneralUtil:
    """
    Class with utility functions
    """

    @staticmethod
    def running_average(x, N):
        """
        Function used to compute the running average of the last N elements of a vector x
        """
        if len(x) < N:
            N = len(x)
        if len(x) >= N:
            y = np.copy(x)
            y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
        else:
            y = np.zeros_like(x)
        return round(y[-1], 2)
