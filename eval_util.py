import numpy as np
from policy_iteration import PI
from value_iteration import VI
from pomdp_util import POMDPUtil


class EvalUtil:
    """
    Util functions for evaluation
    """

    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Sets the random seed
        """
        np.random.seed(seed)

    @staticmethod
    def print_mu(mu, J, B_n, b_n_0, u_to_vec):
        """
        Prints the base policy mu and its cost-to-go
        """

        print(" ")
        print("---- mu ----")
        for i in range(len(B_n)):
            print(f"b: {list(map(lambda x: round(x, 2), B_n[i]))}, u: {u_to_vec[np.argmax(mu[i])]}")
        print(f"J(b_n_0): {J[b_n_0]}")

    @staticmethod
    def compute_base_policy(B_n, P_b, C_b, U, b_n_0, gamma, u_to_vec, pi=True, verbose=False, epsilon=0.01,
                            pi_iterations=10):
        """
        Computes the base policy in the aggregate belief MDP
        """
        B_n_indices = []
        mu_1 = []
        for i in range(len(B_n)):
            mu_1.append([1.0, 0.0])
            B_n_indices.append(i)

        if pi:
            mu, J = PI.pi(
                P=P_b, mu=mu_1, N=pi_iterations, gamma=gamma, C=C_b, X=B_n_indices, U=U, x0=b_n_0, verbose=verbose)
        else:
            mu, J = VI.vi(
                P=P_b, epsilon=epsilon, gamma=gamma, C=C_b, X=B_n_indices, U=U, verbose=verbose)
        if verbose:
            EvalUtil.print_mu(mu=mu, J=J, B_n=B_n, b_n_0=b_n_0, u_to_vec=u_to_vec)
        return mu, J

    @staticmethod
    def exact_eval(X, b0, U, O, P, Z, C, gamma, l, u_to_vec, N, rollout_length, rollout_mc_samples, monte_carlo,
                   certainty_equivalence, multiagent, component_spaces, vec_to_u):
        """
        Runs the exact evaluation
        """
        ns = list(range(1, 200))
        for n in ns:
            B_n = POMDPUtil.B_n(n=n, X=X)
            B_n_indices = []
            for i in range(len(B_n)):
                B_n_indices.append(i)
            b_n_0 = B_n.index(b0)
            P_b = POMDPUtil.P_b(B_n=B_n, X=X, U=U, O=O, P=P, Z=Z)
            C_b = POMDPUtil.C_b(B_n=B_n, X=X, U=U, C=C)
            mu, J_mu = EvalUtil.compute_base_policy(B_n=B_n, P_b=P_b, C_b=C_b, U=U, b_n_0=b_n_0, gamma=gamma,
                                                    pi=False, verbose=False, u_to_vec=u_to_vec)
            J_b0_mu = POMDPUtil.exact_eval(
                mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b=b0, B_n=B_n, J_mu=None, gamma=gamma, N=N, l=1,
                base_policy=True, t=0, certainty_equivalence=False, rollout_horizon=N,
                rollout_length=rollout_length, J={}, monte_carlo=monte_carlo, rollout_mc_samples=rollout_mc_samples,
                multiagent=multiagent, u_to_vec=u_to_vec, component_spaces=component_spaces, vec_to_u=vec_to_u)
            # J_b0_mu = {}
            J_b0_mu_tilde = POMDPUtil.exact_eval(
                mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b=b0, B_n=B_n, J_mu=J_b0_mu, gamma=gamma, N=N, base_policy=False,
                l=l, t=0, certainty_equivalence=certainty_equivalence, rollout_horizon=N,
                rollout_length=rollout_length, J={},
                monte_carlo=monte_carlo, rollout_mc_samples=rollout_mc_samples, multiagent=multiagent,
                u_to_vec=u_to_vec, component_spaces=component_spaces, vec_to_u=vec_to_u)
            print(f"{n} {round(J_b0_mu[(tuple(b0), 0)], 3)} {round(J_b0_mu_tilde[(tuple(b0), 0)], 3)}")

    @staticmethod
    def monte_carlo_eval(X, b0, U, O, P, Z, C, gamma, l, u_to_vec, N, M, rollout_length, rollout_mc_samples,
                         multiagent, component_spaces, vec_to_u, certainty_equivalence):
        """
        Runs the Monte-Carlo evaluation
        """
        ns = list(range(1, 200))
        for n in ns:
            B_n = POMDPUtil.B_n(n=n, X=X)
            B_n_indices = []
            for i in range(len(B_n)):
                B_n_indices.append(i)
            b_n_0 = B_n.index(b0)
            P_b = POMDPUtil.P_b(B_n=B_n, X=X, U=U, O=O, P=P, Z=Z)
            C_b = POMDPUtil.C_b(B_n=B_n, X=X, U=U, C=C)
            mu, J_mu = EvalUtil.compute_base_policy(B_n=B_n, P_b=P_b, C_b=C_b, U=U, b_n_0=b_n_0, gamma=gamma,
                                                    pi=False, verbose=False, u_to_vec=u_to_vec)
            J_b0_mu, episodes = POMDPUtil.parallel_monte_carlo_evaluate(
                mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b0=b0, B_n=B_n, J_mu=None, gamma=gamma, N=N, M=M, l=l,
                base_policy=True, rollout_length=rollout_length, rollout_mc_samples=rollout_mc_samples,
                multiagent=multiagent, component_spaces=component_spaces, u_to_vec=u_to_vec, vec_to_u=vec_to_u,
                certainty_equivalence=certainty_equivalence)
            V_pi = POMDPUtil.monte_carlo_policy_evaluation(episodes=episodes, gamma=gamma, B_n=B_n,
                                                           B_n_indices=B_n_indices)
            J_b0_mu_tilde, episodes = POMDPUtil.parallel_monte_carlo_evaluate(
                mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b0=b0, B_n=B_n, J_mu=V_pi, gamma=gamma, N=N, M=M, l=l,
                base_policy=False, rollout_length=rollout_length, rollout_mc_samples=rollout_mc_samples,
                multiagent=multiagent, component_spaces=component_spaces, u_to_vec=u_to_vec, vec_to_u=vec_to_u,
                certainty_equivalence=certainty_equivalence)
            print(f"{n} {round(J_b0_mu, 3)} {round(J_b0_mu_tilde, 3)}")
