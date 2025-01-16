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
    def monte_carlo_evaluate_sequential(mu, P, Z, C, O, X, U, b0, B_n, gamma, J_mu, base_policy, l, rollout_length,
                                        rollout_mc_samples, component_spaces, multiagent, certainty_equivalence,
                                        u_to_vec, vec_to_u, N=100, M=100):
        """
        Runs N parallel evaluation episodes following mu. Then it returns
        the average cost as well as the list of episodes
        (each episode is a list of tuples (b_0,u_0,c_0),(b_1,u_1,c_1),..)
        """
        costs = []
        episodes = []
        for i in range(M):
            seed = int(time.time()) + i
            result = POMDPUtil.monte_carlo_evaluate(
                mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b0=b0, B_n=B_n, gamma=gamma, N=N, base_policy=base_policy,
                l=l, seed=seed, J_mu=J_mu, rollout_length=rollout_length, rollout_mc_samples=rollout_mc_samples,
                component_spaces=component_spaces, multiagent=multiagent, certainty_equivalence=certainty_equivalence,
                u_to_vec=u_to_vec, vec_to_u=vec_to_u)
            costs.append(result[0])
            episodes.append(result[1])
        return np.mean(costs), episodes

    @staticmethod
    def parallel_monte_carlo_evaluate(mu, P, Z, C, O, X, U, b0, B_n, gamma, J_mu, base_policy, l,
                                      rollout_length, rollout_mc_samples, component_spaces, multiagent,
                                      certainty_equivalence, u_to_vec, vec_to_u, N=100, M=100):
        """
        Runs N parallel evaluation episodes following mu. Then it returns
        the average cost as well as the list of episodes
        (each episode is a list of tuples (b_0,u_0,c_0),(b_1,u_1,c_1),..)
        """
        inputs = [(mu, P, Z, C, O, X, U, b0, B_n, gamma, J_mu, N, base_policy, l,
                   int(time.time()) + i, rollout_length, rollout_mc_samples, component_spaces, multiagent,
                   certainty_equivalence, u_to_vec, vec_to_u) for i in range(M)]
        with Pool() as pool:
            results = pool.starmap(POMDPUtil.monte_carlo_evaluate, inputs)
            costs = list(map(lambda res: res[0], results))
            episodes = list(map(lambda res: res[1], results))
            return np.mean(costs), episodes

    @staticmethod
    def monte_carlo_evaluate(mu, P, Z, C, O, X, U, b0, B_n, gamma, J_mu, N, base_policy, l, seed, rollout_length,
                             rollout_mc_samples, component_spaces, multiagent, certainty_equivalence, u_to_vec,
                             vec_to_u):
        """
        Monte-Carlo evaluation to estimate J for a base or rollout policy
        """
        np.random.seed(seed)
        b = b0
        Cost = 0
        t = 0
        episode = []
        while t <= N - 1:
            if base_policy:
                u = POMDPUtil.base_policy(mu=mu, U=U, b=b, B_n=B_n)
            else:
                u, _ = POMDPUtil.rollout_policy(mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b=b, B_n=B_n, J_mu=J_mu,
                                                gamma=gamma, l=l, N=N, t=t, rollout_length=rollout_length,
                                                rollout_mc_samples=rollout_mc_samples, monte_carlo=True,
                                                component_spaces=component_spaces, multiagent=multiagent,
                                                certainty_equivalence=certainty_equivalence, u_to_vec=u_to_vec,
                                                vec_to_u=vec_to_u)
                print(f"{t}/{N - 1}, {u}")
                # print(f"{t}/{N-1}, u_tilde: {u}, u_base: {POMDPUtil.base_policy(mu=mu, U=U, b=b, B_n=B_n)}, b: {b}")
            Cost += math.pow(gamma, t) * POMDPUtil.expected_cost(b=b, u=u, C=C, X=X)
            episode.append((B_n.index(POMDPUtil.nearest_neighbor(B_n=B_n, b=b)), u,
                            POMDPUtil.expected_cost(b=b, u=u, C=C, X=X)))
            z = np.random.choice(O, p=[POMDPUtil.P_z_b_u(b=b, z=z, Z=Z, X=X, U=U, P=P, u=u) for z in O])
            b = POMDPUtil.belief_operator(z=z, u=u, b=b, X=X, Z=Z, P=P)
            t += 1
        return (Cost, episode)

    @staticmethod
    def exact_eval(t, b, base_policy, mu, U, B_n, P, Z, C, O, X, J_mu, gamma, l, N, certainty_equivalence,
                   rollout_horizon, rollout_length, J, monte_carlo, rollout_mc_samples, multiagent, u_to_vec,
                   component_spaces, vec_to_u):
        """
        Computes the exact value function for either the base policy or the rollout policy
        """
        if t >= min(N, rollout_horizon):
            if rollout_horizon < N and J_mu is not None:
                candidate_beliefs = []
                for k, v in J_mu.items():
                    if k[1] == t:
                        candidate_beliefs.append(list(k[0]))
                J[(tuple(b), t)] = J_mu[(tuple(POMDPUtil.nearest_neighbor(candidate_beliefs, b)), t)]
            else:
                J[(tuple(b), t)] = 0
            return J
        if base_policy:
            u = POMDPUtil.base_policy(mu=mu, U=U, b=b, B_n=B_n)
        else:
            start = time.time()
            u, _ = POMDPUtil.rollout_policy(mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b=b, B_n=B_n, J_mu=J_mu,
                                            gamma=gamma, l=l, t=t, N=N, certainty_equivalence=certainty_equivalence,
                                            rollout_length=rollout_length, monte_carlo=monte_carlo,
                                            rollout_mc_samples=rollout_mc_samples, multiagent=multiagent,
                                            u_to_vec=u_to_vec, component_spaces=component_spaces, vec_to_u=vec_to_u)
            print(time.time()-start)
            import sys
            sys.exit()
        Cost = POMDPUtil.expected_cost(b=b, u=u, C=C, X=X)
        if t == 0:
            inputs = [(z, u, b, X, Z, P, base_policy, mu, U, t, B_n, C, O, J_mu, gamma,
                       l, N, certainty_equivalence, rollout_horizon, rollout_length, J.copy(),
                       monte_carlo, rollout_mc_samples, multiagent, u_to_vec, component_spaces, vec_to_u) for z in O]
            with Pool() as pool:
                results = pool.starmap(POMDPUtil.parallel_lookahead, inputs)
                for i in range(len(results)):
                    Cost += results[i][0]
                    J = J | results[i][1]
        else:
            for z in O:
                b_prime = POMDPUtil.belief_operator(z=z, u=u, b=b, X=X, Z=Z, P=P)
                J_prime = POMDPUtil.exact_eval(
                    t=t + 1, b=b_prime, base_policy=base_policy, mu=mu, U=U,
                    B_n=B_n, P=P, Z=Z, C=C, O=O, X=X, J_mu=J_mu, gamma=gamma, l=l, N=N,
                    certainty_equivalence=certainty_equivalence, rollout_horizon=rollout_horizon,
                    rollout_length=rollout_length, J=J.copy(), monte_carlo=monte_carlo,
                    rollout_mc_samples=rollout_mc_samples, multiagent=multiagent, u_to_vec=u_to_vec,
                    component_spaces=component_spaces, vec_to_u=vec_to_u)
                cost_to_go = J_prime[(tuple(b_prime), t + 1)]
                Cost += gamma * POMDPUtil.P_z_b_u(b=b, z=z, Z=Z, X=X, U=U, P=P, u=u) * cost_to_go
                J = J | J_prime
        J[(tuple(b), t)] = Cost
        return J

    @staticmethod
    def parallel_lookahead(z, u, b, X, Z, P, base_policy, mu, U, t, B_n, C, O, J_mu, gamma, l, N,
                           certainty_equivalence, rollout_horizon, rollout_length, J, monte_carlo,
                           rollout_mc_samples, multiagent, u_to_vec, component_spaces, vec_to_u):
        """
        Auxillary function for parallelizing the lookahead per observation in exact_eval()
        """
        b_prime = POMDPUtil.belief_operator(z=z, u=u, b=b, X=X, Z=Z, P=P)
        J_prime = POMDPUtil.exact_eval(
            t=t + 1, b=b_prime, base_policy=base_policy, mu=mu, U=U,
            B_n=B_n, P=P, Z=Z, C=C, O=O, X=X, J_mu=J_mu, gamma=gamma, l=l, N=N,
            certainty_equivalence=certainty_equivalence, rollout_horizon=rollout_horizon,
            rollout_length=rollout_length, J=J.copy(), monte_carlo=monte_carlo,
            rollout_mc_samples=rollout_mc_samples, multiagent=multiagent, u_to_vec=u_to_vec,
            component_spaces=component_spaces, vec_to_u=vec_to_u)
        cost_to_go = J_prime[(tuple(b_prime), t + 1)]
        J = J | J_prime
        return gamma * POMDPUtil.P_z_b_u(b=b, z=z, Z=Z, X=X, U=U, P=P, u=u) * cost_to_go, J

    @staticmethod
    def base_policy(mu, U, b, B_n):
        """
        Returns mu[b]
        """
        return np.random.choice(U, p=mu[B_n.index(POMDPUtil.nearest_neighbor(B_n=B_n, b=b))])

    @staticmethod
    def rollout_policy(U, O, Z, X, P, b, C, J_mu, gamma, B_n, l, mu, t, N, rollout_length, u_to_vec,
                       component_spaces, vec_to_u, certainty_equivalence=False, monte_carlo=False,
                       rollout_mc_samples=1, multiagent=False):
        """
        Returns \tilde{\mu}[b]
        """
        if t >= N:
            return random.choice(U), 0
        rollout_horizon = min(t + 1 + rollout_length, N)
        if multiagent:
            aggregate_u = []
            rollout_value = 0
            base_u = u_to_vec[POMDPUtil.base_policy(mu=mu, U=U, b=b, B_n=B_n)]
            for agent in range(len(component_spaces)):
                u_agent = aggregate_u.copy()
                u_agent.append(-1)
                for agent_i in range(agent + 1, len(component_spaces)):
                    u_agent.append(base_u[agent_i])
                Q_b_agent = np.zeros(len(component_spaces[agent]))
                for local_u in component_spaces[agent]:
                    u_agent[agent] = local_u
                    u = vec_to_u[tuple(u_agent)]
                    cost = POMDPUtil.rollout_optimization(certainty_equivalence=certainty_equivalence, b=b,
                                                          u=u, C=C, P=P, X=X, Z=Z, O=O, monte_carlo=monte_carlo, l=l,
                                                          B_n=B_n, U=U, J_mu=J_mu, gamma=gamma, N=N,
                                                          rollout_horizon=rollout_horizon, t=t,
                                                          rollout_length=rollout_length,
                                                          rollout_mc_samples=rollout_mc_samples, u_to_vec=u_to_vec,
                                                          multiagent=multiagent, mu=mu,
                                                          component_spaces=component_spaces, vec_to_u=vec_to_u)
                    Q_b_agent[local_u] = cost
                u_star_local = int(np.argmin(Q_b_agent))
                rollout_value = Q_b_agent[u_star_local]
                aggregate_u.append(u_star_local)
            return vec_to_u[tuple(aggregate_u)], rollout_value
        else:
            Q_b = np.zeros(len(U))
            for u in U:
                cost = POMDPUtil.rollout_optimization(certainty_equivalence=certainty_equivalence, b=b,
                                                      u=u, C=C, P=P, X=X, Z=Z, O=O, monte_carlo=monte_carlo, l=l,
                                                      B_n=B_n, U=U, J_mu=J_mu, gamma=gamma, N=N,
                                                      rollout_horizon=rollout_horizon, t=t,
                                                      rollout_length=rollout_length,
                                                      rollout_mc_samples=rollout_mc_samples, u_to_vec=u_to_vec,
                                                      multiagent=multiagent, mu=mu, component_spaces=component_spaces,
                                                      vec_to_u=vec_to_u)
                Q_b[u] = cost
            u_star = int(np.argmin(Q_b))
            return u_star, Q_b[u_star]

    @staticmethod
    def rollout_optimization(certainty_equivalence, b, u, C, P, X, Z, O, monte_carlo, l, B_n, U, J_mu, gamma,
                             N, rollout_horizon, t, rollout_length, rollout_mc_samples, u_to_vec, multiagent, mu,
                             component_spaces, vec_to_u):
        """
        Performs the rollout lookahead optimization
        """
        cost = POMDPUtil.expected_cost(b=b, u=u, C=C, X=X)
        if certainty_equivalence:
            b_prime = np.sum([POMDPUtil.P_z_b_u(b=b, z=z, Z=Z, X=X, U=U, P=P, u=u) *
                              np.array(POMDPUtil.belief_operator(z=z, u=u, b=b, X=X, Z=Z, P=P)) for z in O],
                             axis=0)
            if l == 1:
                if not monte_carlo:
                    J = POMDPUtil.exact_eval(
                        mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b=b_prime, B_n=B_n, J_mu=J_mu, gamma=gamma,
                        N=N, base_policy=True, l=-1, t=t + 1, certainty_equivalence=certainty_equivalence,
                        rollout_horizon=rollout_horizon, rollout_length=rollout_length, J={},
                        monte_carlo=monte_carlo, rollout_mc_samples=rollout_mc_samples, u_to_vec=u_to_vec,
                        multiagent=multiagent, component_spaces=component_spaces, vec_to_u=vec_to_u)
                    J_mu_val = J[(tuple(b_prime), t + 1)]
                else:
                    J_mu_val, _ = POMDPUtil.monte_carlo_evaluate_sequential(
                        mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b0=b_prime, B_n=B_n, gamma=gamma, J_mu=J_mu,
                        base_policy=True, l=l, N=N - t, M=rollout_mc_samples, rollout_length=rollout_length,
                        rollout_mc_samples=rollout_mc_samples, component_spaces=component_spaces,
                        multiagent=multiagent, vec_to_u=vec_to_u, u_to_vec=u_to_vec,
                        certainty_equivalence=certainty_equivalence)
            else:
                J_mu_val = POMDPUtil.rollout_policy(U=U, O=O, Z=Z, X=X, P=P, b=b_prime,
                                                    C=C, J_mu=J_mu, gamma=gamma, B_n=B_n, l=l - 1, mu=mu,
                                                    t=t + 1, N=N, rollout_length=rollout_length,
                                                    monte_carlo=monte_carlo,
                                                    rollout_mc_samples=rollout_mc_samples,
                                                    certainty_equivalence=certainty_equivalence,
                                                    u_to_vec=u_to_vec, multiagent=multiagent,
                                                    component_spaces=component_spaces, vec_to_u=vec_to_u)[1]
            cost += gamma * J_mu_val
            return cost

        for z in O:
            P_b_z_u = POMDPUtil.P_z_b_u(b=b, z=z, Z=Z, X=X, U=U, P=P, u=u)
            if P_b_z_u > 0:
                b_prime = POMDPUtil.belief_operator(z=z, u=u, b=b, X=X, Z=Z, P=P)
                if l == 1:
                    if not monte_carlo:
                        J = POMDPUtil.exact_eval(
                            mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b=b_prime, B_n=B_n, J_mu=J_mu, gamma=gamma,
                            N=N, base_policy=True, l=-1, t=t + 1, certainty_equivalence=certainty_equivalence,
                            rollout_horizon=rollout_horizon, rollout_length=rollout_length, J={},
                            monte_carlo=monte_carlo, rollout_mc_samples=rollout_mc_samples, u_to_vec=u_to_vec,
                            multiagent=multiagent, component_spaces=component_spaces, vec_to_u=vec_to_u)
                        J_mu_val = J[(tuple(b_prime), t + 1)]
                    else:
                        J_mu_val, _ = POMDPUtil.monte_carlo_evaluate_sequential(
                            mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b0=b_prime, B_n=B_n, gamma=gamma, J_mu=J_mu,
                            base_policy=True, l=l, N=N - t, M=rollout_mc_samples, rollout_length=rollout_length,
                            rollout_mc_samples=rollout_mc_samples, component_spaces=component_spaces,
                            multiagent=multiagent, vec_to_u=vec_to_u, u_to_vec=u_to_vec,
                            certainty_equivalence=certainty_equivalence)
                else:
                    J_mu_val = POMDPUtil.rollout_policy(U=U, O=O, Z=Z, X=X, P=P, b=b_prime,
                                                        C=C, J_mu=J_mu, gamma=gamma, B_n=B_n, l=l - 1, mu=mu,
                                                        t=t + 1, N=N, rollout_length=rollout_length,
                                                        monte_carlo=monte_carlo,
                                                        rollout_mc_samples=rollout_mc_samples,
                                                        certainty_equivalence=certainty_equivalence,
                                                        u_to_vec=u_to_vec, multiagent=multiagent,
                                                        component_spaces=component_spaces, vec_to_u=vec_to_u)[1]
                cost += P_b_z_u * gamma * J_mu_val
        return cost

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
