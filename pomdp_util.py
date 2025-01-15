import time
import random
import math
from multiprocessing import Pool
import numpy as np
import itertools
from collections import Counter


class POMDPUtil:
    """
    Utility functions for belief aggregation in POMDPs
    """

    @staticmethod
    def belief_operator(z, u, b, X, Z, P):
        """
        Computes b' after observing (b,o)
        """
        b_prime = [0.0] * len(X)
        for x_prime in X:
            b_prime[x_prime] = POMDPUtil.bayes_filter(
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
        return belief_points

    @staticmethod
    def C_b(B_n, X, U, C):
        """
        Generates a cost tensor for the aggregate belief MDP
        """
        belief_C = list(np.zeros((len(B_n), len(U))).tolist())
        for u in U:
            for b in B_n:
                belief_C[B_n.index(b)][u] = POMDPUtil.expected_cost(b=b, u=u, C=C, X=X)
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
                    belief_T[u][B_n.index(b1)][B_n.index(b2)] = \
                        POMDPUtil.P_b2_b1_u(b1=b1, b2=b2, u=u, X=X, O=O, P=P, Z=Z, B_n=B_n)
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
            b_prime = POMDPUtil.belief_operator(z=z, u=u, b=b1, X=X, Z=Z, P=P)
            nearest_neighbor = POMDPUtil.nearest_neighbor(B_n=B_n, b=b_prime)
            if nearest_neighbor == b2:
                for x in X:
                    for x_prime in X:
                        prob += Z[x_prime][z] * b1[x] * P[u][x][x_prime]
        return prob

    @staticmethod
    def parallel_monte_carlo_evaluate(mu, P, Z, C, O, X, U, b0, B_n, gamma, J_mu, initial_controls, N=100, M=100):
        """
        Runs N parallel evaluation episodes following mu. Then it returns
        the average cost as well as the list of episodes
        (each episode is a list of tuples (b_0,u_0,c_0),(b_1,u_1,c_1),..)
        """
        inputs = [(mu, P, Z, C, O, X, U, b0, B_n, gamma, J_mu, N, initial_controls,
                   int(time.time()) + i) for i in range(M)]
        with Pool() as pool:
            results = pool.starmap(POMDPUtil.monte_carlo_evaluate, inputs)
            costs = list(map(lambda res: res[0], results))
            episodes = list(map(lambda res: res[1], results))
            return np.mean(costs), episodes

    @staticmethod
    def monte_carlo_evaluate(mu, P, Z, C, O, X, U, b0, B_n, gamma, J_mu, N, initial_controls, seed):
        """
        Monte-Carlo evaluation to estimate J for a base or rollout policy
        """
        np.random.seed(seed)
        x = np.random.choice(X, p=b0)
        b = b0
        Cost = 0
        t = 0
        episode = []
        while t <= N-1:
            if t < len(initial_controls):
                u = initial_controls[t]
            else:
                u = POMDPUtil.base_policy(mu=mu, U=U, b=b, B_n=B_n)
            Cost += math.pow(gamma, t) * C[x][u]
            episode.append((B_n.index(POMDPUtil.nearest_neighbor(B_n=B_n, b=b)), u, C[x][u]))
            x = int(np.random.choice(X, p=P[u][x]))
            z = np.random.choice(O, p=Z[x])
            b = POMDPUtil.belief_operator(z=z, u=u, b=b, X=X, Z=Z, P=P)
            t += 1
        if J_mu is not None:
            Cost += math.pow(gamma, N)*J_mu[B_n.index(POMDPUtil.nearest_neighbor(B_n=B_n, b=b))]
        return (Cost, episode)

    @staticmethod
    def base_policy(mu, U, b, B_n):
        """
        Returns mu[b]
        """
        return np.random.choice(U, p=mu[B_n.index(POMDPUtil.nearest_neighbor(B_n=B_n, b=b))])

    @staticmethod
    def rollout_policy(U, O, Z, X, P, b, C, J_mu, gamma, B_n, l, mu):
        """
        Returns \tilde{\mu}[b]
        """
        Q_b = np.zeros(len(U))
        for u in U:
            print(f"{u}/{len(U)}, l: {l}")
            for z in O:
                print(f"{z}/{len(O)}, l: {l}")
                P_b_z_u = POMDPUtil.P_z_b_u(b=b, z=z, Z=Z, X=X, U=U, P=P, u=u)
                if P_b_z_u > 0:
                    b_prime = POMDPUtil.belief_operator(z=z, u=u, b=b, X=X, Z=Z, P=P)
                    if l == 1:
                        J_mu_val, _ = POMDPUtil.parallel_monte_carlo_evaluate(
                            mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b0=b, B_n=B_n, J_mu=None, gamma=gamma,
                            N=500, M=5000, initial_controls=[])
                    else:
                        J_mu_val = POMDPUtil.rollout_policy(U=U, O=O, Z=Z, X=X, P=P, b=b_prime,
                                                            C=C, J_mu=J_mu, gamma=gamma, B_n=B_n, l=l - 1, mu=mu)[1]
                    Q_b[u] += P_b_z_u * (POMDPUtil.expected_cost(b=b, u=u, C=C, X=X) + gamma * J_mu_val)
        u_star = int(np.argmin(Q_b))
        return u_star, Q_b[u_star]


    @staticmethod
    def rollout_certainty_equivalence_policy(U, O, Z, X, P, b, C, J_mu, gamma, B_n, l, mu):
        """
        Returns \tilde{\mu}[b]
        """
        Q_b = np.zeros(len(U))
        for u in U:
            print(f"{u}/{len(U)}, l: {l}")
            probs = [POMDPUtil.P_z_b_u(b=b, z=z, Z=Z, X=X, U=U, P=P, u=u) for z in O]
            max_prob = int(np.argmax(probs))
            z = O[max_prob]
            b_prime = POMDPUtil.belief_operator(z=z, u=u, b=b, X=X, Z=Z, P=P)
            if l == 1:
                J_mu_val, _ = POMDPUtil.parallel_monte_carlo_evaluate(
                    mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b0=b, B_n=B_n, J_mu=J_mu, gamma=gamma,
                    N=500, M=100, initial_controls=[])
            else:
                J_mu_val = POMDPUtil.rollout_policy(U=U, O=O, Z=Z, X=X, P=P, b=b_prime,
                                                    C=C, J_mu=J_mu, gamma=gamma, B_n=B_n, l=l - 1, mu=mu)[1]
            Q_b[u] += POMDPUtil.expected_cost(b=b, u=u, C=C, X=X) + gamma * J_mu_val
        u_star = int(np.argmin(Q_b))
        return u_star, Q_b[u_star]

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

    @staticmethod
    def evaluate_particle_filter_parallel(max_num_particles, Z, O, P, b0, U, X, N):
        """
        Parallel evaluation of the accuracy of a particle filter by comparing it with the true belief
        """
        inputs = [(max_num_particles, Z, O, P, b0, U, X, int(time.time()) + i) for i in range(N)]
        with Pool() as pool:
            errors = pool.starmap(POMDPUtil.evaluate_particle_filter, inputs)
            std = np.std(errors)
            return np.mean(errors), std

    @staticmethod
    def evaluate_particle_filter(max_num_particles, Z, O, P, b0, U, X, seed):
        """
        Evaluates the accuracy of a particle filter by comparing it with the true belief
        """
        np.random.seed(seed)
        x = np.random.choice(X, p=b0)
        b = b0
        t = 0
        particles = [x]
        errors = []
        while t <= 25:
            u = random.choice(U)
            x = int(np.random.choice(X, p=P[u][x]))
            z = np.random.choice(O, p=Z[x])
            b = POMDPUtil.belief_operator(z=z, u=u, b=b, X=X, Z=Z, P=P)
            particles = POMDPUtil.particle_filter(particles=particles, max_num_particles=max_num_particles, P=P, o=z,
                                                  u=u, X=X, O=O, Z=Z)
            errors.append(POMDPUtil.compare_belief_and_particles(b=b, particles=particles))
            t += 1
        return float(np.mean(errors))

    @staticmethod
    def compare_belief_and_particles(b, particles):
        """
        Computes the difference between a given belief b and a given particle filter
        """
        counter = Counter(particles)
        particle_b = []
        for i in range(len(b)):
            if i in counter:
                particle_b.append(counter[i] / len(particles))
            else:
                particle_b.append(0)
        return np.linalg.norm(np.array(particle_b) - np.array(b), axis=0)

    @staticmethod
    def particle_filter(particles, max_num_particles, P, o, u, X, O, Z):
        """
        Implements a particle filter
        """
        new_particles = []
        while len(new_particles) < max_num_particles:
            x = random.choice(particles)
            x_prime = np.random.choice(X, p=P[u][x])
            o_hat = np.random.choice(O, p=Z[x_prime])
            if o == o_hat:
                new_particles.append(x_prime)
        return new_particles

    @staticmethod
    def monte_carlo_policy_evaluation(episodes, gamma, B_n, B_n_indices):
        """
        Implements the first visit Monte-Carlo policy evaluation method (Sutton and Barton, p. 92)
        """
        returns = {b: [] for b in range(len(B_n))}
        V_pi = np.zeros(len(B_n))
        for episode in episodes:
            G = 0
            for t in reversed(range(len(episode))):
                b, u, c = episode[t]
                G = gamma * G + c
                if all(b != episode[k][0] for k in range(0, t)):  # First-visit Monte Carlo
                    returns[b].append(G)
        for b in B_n_indices:
            V_pi[b] = np.mean(returns[b])
        return V_pi
