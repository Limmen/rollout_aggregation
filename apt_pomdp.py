import numpy as np
import random
from scipy.stats import betabinom
import math
import itertools


class POMDP:
    """
    Example POMDP for Cybersecurity
    """

    @staticmethod
    def erdos_renyi_graph(K, p_c):
        """
        Generates the adjacency matrix of a random Erdös-Renyi graph
        """
        adjacency_matrix = np.zeros((K, K), dtype=int)
        for i in range(K):
            for j in range(i + 1, K):  # Only consider the upper triangle to avoid duplicate edges
                if random.random() < p_c:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1  # Ensure the graph is undirected
        return adjacency_matrix

    @staticmethod
    def b0(K, X, vec_to_x):
        """
        The initial belief
        """
        b0 = [0] * (len(X))
        x0 = POMDP.x0(X=X, K=K, vec_to_x=vec_to_x)
        b0[x0] = 1
        return b0

    @staticmethod
    def x0(X, K, vec_to_x):
        """
        The initial state
        """
        return X.index(vec_to_x[tuple([0] * K)])

    @staticmethod
    def X(K):
        """
        The state space, each server can be in two states: 0 (healthy) and 1 (compromised)
        """
        vector_space = list(itertools.product(*([[0, 1]] * K)))
        x = 0
        X = []
        x_to_vec = {}
        vec_to_x = {}
        for x_vec in vector_space:
            X.append(x)
            x_to_vec[x] = x_vec
            vec_to_x[tuple(x_vec)] = x
            x += 1
        return X, x_to_vec, vec_to_x

    @staticmethod
    def U(K):
        """
        The control space, 0 (continue), 1 (stop) per server (N)
        """
        vector_space = list(itertools.product(*([[0, 1]] * K)))
        u = 0
        U = []
        u_to_vec = {}
        vec_to_u = {}
        for u_vec in vector_space:
            U.append(u)
            u_to_vec[u] = u_vec
            vec_to_u[tuple(u_vec)] = u
            u += 1
        return U, u_to_vec, vec_to_u, [[0, 1]] * K

    @staticmethod
    def O(n, K):
        """
        The observation space (0,...n) for each server i in N.
        """
        vector_space = list(itertools.product(*([range(n + 1)] * K)))
        o = 0
        O = []
        o_to_vec = {}
        vec_to_o = {}
        for o_vec in vector_space:
            O.append(o)
            o_to_vec[o] = o_vec
            vec_to_o[tuple(o_vec)] = o
            o += 1
        return O, o_to_vec, vec_to_o

    @staticmethod
    def cost_function(x, u, x_to_vec, u_to_vec, eta):
        """
        Computes c(x,u)
        """
        x_vec = list(x_to_vec[x])
        u_vec = list(u_to_vec[u])
        compromised_costs = 0
        response_costs = 0
        for i in range(len(u_vec)):
            compromised_costs += x_vec[i] * (1 - u_vec[i])
            if x_vec[i] == 1:
                response_costs -= u_vec[i]
            else:
                response_costs += u_vec[i]
        return eta * compromised_costs + response_costs

    @staticmethod
    def C(X, U, x_to_vec, u_to_vec, eta):
        """
        A |X|x|U| cost matrix
        """
        C = []
        for x in X:
            C.append([POMDP.cost_function(x=x, u=u, x_to_vec=x_to_vec, u_to_vec=u_to_vec, eta=eta) for u in U])
        return C

    @staticmethod
    def f(x_prime, x, u, K, p_a, x_to_vec, u_to_vec, A):
        """
        Computes P(x_prime | x,u)
        """
        x_vec = list(x_to_vec[x])
        x_prime_vec = list(x_to_vec[x_prime])
        u_vec = list(u_to_vec[u])
        probabilities = []
        for i in range(K):
            num_compromised_neighbors = POMDP.get_num_compromised_neighbors(i=i, A=A, x_vec=x_vec)
            probabilities.append(POMDP.f_local(x_prime=x_prime_vec[i], x=x_vec[i], u=u_vec[i], p_a=p_a,
                                               num_compromised_neighbors=num_compromised_neighbors))
        return math.prod(probabilities)

    @staticmethod
    def get_num_compromised_neighbors(i, A, x_vec):
        """
        Gets the number of compromised neighbors
        """
        num_compromised = 0
        for j in range(len(A[i])):
            if A[i][j] == 1:
                num_compromised += x_vec[j]
        return num_compromised

    @staticmethod
    def f_local(x_prime, x, u, p_a, num_compromised_neighbors):
        """
        Transition function of a single node, P(x_prime_i | x_i, u_i)
        """
        compromise_probability = min(1.0, p_a * (num_compromised_neighbors + 1))
        if u == 1 and x_prime == 0:
            return 1
        if u == 0 and x == 1 and x_prime == 1:
            return 1
        if u == 0 and x == 0 and x_prime == 0:
            return 1 - compromise_probability
        if u == 0 and x == 0 and x_prime == 1:
            return compromise_probability
        return 0

    @staticmethod
    def P(p_a, X, U, x_to_vec, u_to_vec, N, A):
        """
        A |U|x|X|x|X| transition tensor
        """
        P = []
        for u in U:
            u_p = []
            for x in X:
                distribution = [POMDP.f(x_prime, x, u, N, p_a, x_to_vec, u_to_vec, A) for x_prime in X]
                assert round(sum(distribution), 2) == 1
                u_p.append(distribution)
            P.append(u_p)
        return P

    @staticmethod
    def Z(n, X, K, x_to_vec, o_to_vec, O):
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
            x_vec = x_to_vec[x]
            x_distribution = []
            for o in O:
                o_vec = o_to_vec[o]
                probs = []
                for i in range(K):
                    if x_vec[i] == 0:
                        probs.append(no_intrusion_dist[o_vec[i]])
                    else:
                        probs.append(intrusion_dist[o_vec[i]])
                x_distribution.append(math.prod(probs))  # Conditional independence -> product
            assert round(sum(x_distribution), 2) == 1
            Z.append(x_distribution)
        return Z
