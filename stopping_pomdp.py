import numpy as np
from scipy.stats import betabinom


class StoppingPOMDP:
    """
    The stopping POMDP from (Hammar, Stadler 2021 - Intrusion Prevention through Optimal Stopping)
    """

    @staticmethod
    def b0():
        """
        :return: the initial belief
        """
        return [1.0, 0.0, 0.0]

    @staticmethod
    def x0():
        """
        :return: the initial state
        """
        return 0

    @staticmethod
    def X():
        """
        The state space
        """
        return [0, 1, 2]

    @staticmethod
    def U():
        """
        The control space, 0 (continue), 1 (stop)
        """
        return [0, 1]

    @staticmethod
    def O(n):
        """
        The observation space
        """
        return list(range(n + 1))

    @staticmethod
    def C(intrusion_stop_gain: float, stopping_cost: float, intrusion_cost: float):
        """
        A |X|x|U| cost matrix
        """
        return [
            [0.0, stopping_cost],
            [intrusion_cost, -intrusion_stop_gain],
            [0.0, 0.0]
        ]

    @staticmethod
    def P(intrusion_start_probability: float):
        """
        A |U|x|X|x|X| transition tensor
        """
        return [
            # Continue
            [
                [1.0 - intrusion_start_probability, intrusion_start_probability, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ],
            # Stop
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0]
            ]
        ]

    @staticmethod
    def Z(n):
        """
        A |X|x|O| tensor, where |O|=n+1
        """
        intrusion_dist = []
        no_intrusion_dist = []
        terminal_dist = np.zeros(n + 1)
        terminal_dist[-1] = 1
        intrusion_rv = betabinom(n=n, a=1, b=0.7)
        no_intrusion_rv = betabinom(n=n, a=0.7, b=3)
        for i in range(n + 1):
            intrusion_dist.append(intrusion_rv.pmf(i))
            no_intrusion_dist.append(no_intrusion_rv.pmf(i))
        return [no_intrusion_dist, intrusion_dist, terminal_dist]
