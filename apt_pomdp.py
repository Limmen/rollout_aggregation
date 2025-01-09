import numpy as np
from scipy.stats import betabinom
import math


class APTPOMDP:
    """
    The APT POMDP from (Hammar, Li, Stadler, Zhu 2024 -
    Automated Security Response through Online Learning with Adaptive Conjectures)
    """

    @staticmethod
    def b0(N):
        """
        :return: the initial belief
        """
        b0 = [0]*(N+1)
        b0[0]=1
        return b0

    @staticmethod
    def x0():
        """
        :return: the initial state
        """
        return 0

    @staticmethod
    def X(N):
        """
        The state space
        """
        return list(range(N+1))

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
    def cost_function(x, u):
        """
        Computes c(x,u)
        """
        if x > 0:
            return math.pow(x, 5/4)*(1-u) -u
        else:
            return math.pow(x, 5/4)*(1-u) + u

    @staticmethod
    def C(X, U):
        """
        A |X|x|U| cost matrix
        """
        C = []
        for x in X:
            C.append([APTPOMDP.cost_function(x=x, u=u) for u in U])
        return C

    @staticmethod
    def f(x_prime, x, u, N, p_a):
        """
        Computes P(x_prime | x,u)
        """
        if u == 1 and x_prime == 0:
            return 1
        if u == 0 and x == N and x_prime == N:
            return 1
        if u == 0 and x < N and x_prime == x:
            return 1-p_a
        if u == 0 and x < N and x_prime == (x + 1):
            return p_a
        return 0

    @staticmethod
    def P(p_a: float, X, U):
        """
        A |U|x|X|x|X| transition tensor
        """
        P = []
        N = len(X)-1
        for u in U:
            u_p = []
            for x in X:
                u_p.append([APTPOMDP.f(x_prime, x, u, N, p_a) for x_prime in X])
            P.append(u_p)
        return P

    @staticmethod
    def Z(n, X):
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
        Z = []
        for x in X:
            if x == 0:
                Z.append(no_intrusion_dist)
            else:
                Z.append(intrusion_dist)
        return Z
