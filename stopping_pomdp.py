import numpy as np
from scipy.stats import betabinom
import itertools


class StoppingPOMDP:
    """
    The stopping POMDP from (Hammar, Stadler 2021 - Intrusion Prevention through Optimal Stopping))
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
        A |S|x|U| cost matrix
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

    @staticmethod
    def belief_operator(z, u, b, X, Z, P):
        """
        Computes b' after observing (b,o)
        """
        b_prime = [0.0] * len(X)
        for x_prime in X:
            b_prime[x_prime] = StoppingPOMDP.bayes_filter(
                x_prime=x_prime, z=z, u=u, b=b, X=X, P=P, Z=Z)
        assert round(sum(b_prime), 2) == 1
        return b_prime

    @staticmethod
    def bayes_filter(x_prime, z, u, b, X, Z, P):
        """
        A Bayesian filter to compute b[x_prime] after observing (z,u)
        """
        norm = 0.0
        for x in X:
            for x_prime_1 in X:
                prob_1 = Z[x_prime_1][z]
                norm += b[x] * prob_1 * P[u][x][x_prime_1]
        temp = 0.0
        for x in X:
            temp += Z[x_prime][z] * P[u][x][x_prime] * b[x]
        b_prime_s_prime = temp / norm
        assert round(b_prime_s_prime, 2) <= 1
        return float(b_prime_s_prime)

    @staticmethod
    def nearest_neighbor(B_n, b):
        """
        Returns the nearest neighbor of b in B_n
        """
        distances = np.linalg.norm(np.array(B_n) - np.array(b), axis=1)
        nearest_index = int(np.argmin(distances))
        return B_n[nearest_index]

    @staticmethod
    def B_n(n, X):
        """
        Creates the aggregate belief space B_n, where n is the resolution
        """
        combinations = [k for k in itertools.product(range(n + 1), repeat=len(X)) if sum(k) == n]
        belief_points = [list(float(k_i / n) for k_i in k) for k in combinations]
        belief_points = list(filter(lambda x: x[-1] == 1.0 or x[-1] == 0.0, belief_points))
        return belief_points

    @staticmethod
    def C_b(B_n, X, U, C):
        """
        Generates a cost tensor for the aggregate belief MDP
        """
        belief_C = list(np.zeros((len(B_n), len(U))).tolist())
        for u in U:
            for b in B_n:
                belief_C[B_n.index(b)][u] = StoppingPOMDP.expected_cost(b=b, u=u, C=C, X=X)
        return belief_C

    @staticmethod
    def expected_cost(b, u, C, X):
        """
        Computes E[C[x][u] | b]
        """
        return sum([C[x][u] * b[x] for x in X])

    @staticmethod
    def P_z_b_u(b, z, Z, X, U, P, u):
        """
        Computes P(z | b, u)
        """
        return sum([Z[x_prime][z] * b[x] * P[u][x][x_prime] for x in X for x_prime in X])

    @staticmethod
    def P_b(B_n, X, U, O, P, Z):
        """
        Generates an aggregate belief space transition operator
        """
        belief_T = list(np.zeros((len(U), len(B_n), len(B_n))).tolist())
        for u in U:
            for b1 in B_n:
                for b2 in B_n:
                    belief_T[u][B_n.index(b1)][B_n.index(b2)] \
                        = StoppingPOMDP.P_b2_b1_u(
                        b1=b1, b2=b2, u=u, X=X, O=O, P=P, Z=Z, B_n=B_n)
        return belief_T

    @staticmethod
    def P_b2_b1_u(b1, b2, u, X, O, P, Z, B_n):
        """
        Calculates P(b2 | b1, u)
        """
        prob = 0
        for z in O:
            if sum([Z[s_prime][z] * b1[s] * P[u][s][s_prime] for s in X for s_prime in X]) == 0:
                continue
            b_prime = StoppingPOMDP.belief_operator(z=z, u=u, b=b1, X=X, Z=Z, P=P)
            nearest_neighbor = StoppingPOMDP.nearest_neighbor(B_n=B_n, b=b_prime)
            if nearest_neighbor == b2:
                for x in X:
                    for x_prime in X:
                        prob += Z[x_prime][z] * b1[x] * P[u][x][x_prime]
        return prob

    @staticmethod
    def evaluate(mu, P, Z, C, O, X, U, x0, b0, B_n, J_mu, gamma, base: bool = True) -> float:
        """
        Estimates J for a base or rollout policy
        """
        returns = []
        for i in range(100):
            x = x0
            b = b0
            Cost = 0
            t = 0
            while t <= 100:
                if base:
                    u = StoppingPOMDP.base_policy(mu=mu, U=U, b=b, B_n=B_n)
                else:
                    u = StoppingPOMDP.rollout_policy(U=U, O=O, Z=Z, X=X, P=P, b=b, C=C, J_mu=J_mu, gamma=gamma, B_n=B_n)
                Cost += C[x][u]
                x = int(np.random.choice(X, p=P[u][x]))
                z = np.random.choice(O, p=Z[x])
                b = StoppingPOMDP.belief_operator(z=z, u=u, b=b, X=X, Z=Z, P=P)
                t += 1
            returns.append(Cost)
        avg_return = np.mean(returns)
        return float(avg_return)

    @staticmethod
    def base_policy(mu, U, b, B_n):
        """
        Returns mu[b]
        """
        return np.random.choice(U, p=mu[B_n.index(StoppingPOMDP.nearest_neighbor(B_n=B_n, b=b))])

    @staticmethod
    def rollout_policy(U, O, Z, X, P, b, C, J_mu, gamma, B_n):
        """
        Returns \tilde{\mu}[b]
        """
        Q_b = np.zeros(len(U))
        for u in U:
            for z in O:
                P_b_z_u = StoppingPOMDP.P_z_b_u(b=b, z=z, Z=Z, X=X, U=U, P=P, u=u)
                if P_b_z_u > 0:
                    b_prime = StoppingPOMDP.belief_operator(z=z, u=u, b=b, X=X, Z=Z, P=P)
                    Q_b[u] += P_b_z_u * (StoppingPOMDP.expected_cost(b=b, u=u, C=C, X=X) +
                                         gamma * J_mu[B_n.index(StoppingPOMDP.nearest_neighbor(B_n=B_n, b=b_prime))])
        return int(np.argmin(Q_b))

    @staticmethod
    def pomdp_solver_file(gamma, X, U, O, P, b0, Z, C):
        """
        Generates the POMDP environment specification based on the format at http://www.pomdp.org/code/index.html,
        """
        file_str = ""
        file_str = file_str + f"discount: {gamma}\n\n"
        file_str = file_str + "values: cost\n\n"
        file_str = file_str + f"states: {len(X)}\n\n"
        file_str = file_str + f"actions: {len(U)}\n\n"
        file_str = file_str + f"observations: {len(O)}\n\n"
        initial_belief_str = " ".join(list(map(lambda x: str(x), b0)))
        file_str = file_str + f"start: {initial_belief_str}\n\n\n"
        num_transitions = 0
        for x in X:
            for u in U:
                for x_prime in X:
                    num_transitions += 1
                    file_str = file_str + f"T: {u} : {x} : {x_prime} {P[u][x][x_prime]:.80f}\n"
        file_str = file_str + "\n\n"
        for u in U:
            for x_prime in X:
                for o in O:
                    file_str = file_str + f"O : {u} : {x_prime} : {o} {Z[x_prime][o]:.80f}\n"
        file_str = file_str + "\n\n"
        for x in X:
            for u in U:
                for x_prime in X:
                    for o in O:
                        file_str = file_str + f"R: {u} : {x} : {x_prime} : {o} {C[x][u]:.80f}\n"
        return file_str
