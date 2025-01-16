import time
import csv
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
    def eval(results_file, X, b0, U, O, P, Z, C, gamma, l, u_to_vec):
        """
        Runs the evaluation and saves the results to a csv file
        """
        # with open(results_file, mode='w', newline='', encoding='utf-8') as file:
        #     csv.writer(file).writerow(["n", "B_n", "T_mdp", "T_mu", "J_mu", "J_mu_tilde", "l", "|U|", "|O|", "X"])
        ns = list(range(1, 200))
        # ns = [7]
        # #12
        for n in ns:
            start = time.time()
            B_n = POMDPUtil.B_n(n=n, X=X)
            B_n_indices = []
            for i in range(len(B_n)):
                B_n_indices.append(i)
            b_n_0 = B_n.index(b0)
            P_b = POMDPUtil.P_b(B_n=B_n, X=X, U=U, O=O, P=P, Z=Z)
            C_b = POMDPUtil.C_b(B_n=B_n, X=X, U=U, C=C)
            # T_mdp = time.time() - start
            mu, J_mu = EvalUtil.compute_base_policy(B_n=B_n, P_b=P_b, C_b=C_b, U=U, b_n_0=b_n_0, gamma=gamma,
                                                    pi=False, verbose=False, u_to_vec=u_to_vec)

            N=12
            M=1
            # start = time.time()
            J_b0_mu = POMDPUtil.exact_eval(
                mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b=b0, B_n=B_n, J_mu=None, gamma=gamma, N=N, l=1,
                base_policy = True, t=0)
            # J_b0_mu2 = POMDPUtil.monte_carlo_evaluate_sequential(
            #     mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b0=b0, B_n=B_n, J_mu=None, gamma=gamma, N=5, M=1, l=1,
            #     base_policy = True)
            # print(f"{J_b0_mu}, {J_b0_mu2}, {b0}")
            # import sys
            # sys.exit()
            # V_pi = POMDPUtil.monte_carlo_policy_evaluation(episodes=episodes, gamma=gamma, B_n=B_n,
            #                                                B_n_indices=B_n_indices)
            # u, _ = POMDPUtil.rollout_policy(mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b=b0, B_n=B_n, J_mu=V_pi,
            #                                 gamma=gamma, l=l)
            # u, _ = POMDPUtil.rollout_certainty_equivalence_policy(mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b=b0, B_n=B_n, J_mu=V_pi,
            #                                 gamma=gamma, l=l)
            J_b0_mu_tilde = POMDPUtil.exact_eval(
                mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b=b0, B_n=B_n, J_mu=None, gamma=gamma, N=N, base_policy=False,
                l=l, t=0)
            print(f"{n} {round(J_b0_mu, 3)} {round(J_b0_mu_tilde, 3)}")
            # if J_b0_mu < J_b0_mu_tilde:
            #     raise ValueError("ERROR")
            # import sys
            # sys.exit()
            # print(V_pi)
            # print(f"{n} {abs(J_mu[b_n_0]-J_b0_mu)}")
            # T_mu = time.time() - start
            # print(f"{n} {time.time() - start}")
            # J_b0_mu = POMDPUtil.parallel_evaluate(mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b0=b0, B_n=B_n, J_mu=J_mu,
            #                              gamma=gamma, base=True)
            # print(J_b0_mu)
            # J_b0_mu_tilde = POMDPUtil.parallel_evaluate(mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b0=b0, B_n=B_n, J_mu=J_mu,
            #                                    gamma=gamma, base=False, l=l)
            # with open(results_file, mode='a', newline='', encoding='utf-8') as file:
            #     csv.writer(file).writerow([n, len(B_n), f"{T_mdp:.2f}", f"{T_mu:.2f}", f"{J_b0_mu:.2f}",
            #                                f"{J_b0_mu_tilde:.2f}", l, len(U), len(O), len(X)])
            #     print(f"n: {n}, B_n_size: {len(B_n)}, T_mdp: {T_mdp:.3f}s, T_mu: {T_mu:.3f}s, J_b0_mu: {J_b0_mu:.2f},"
            #           f"J_b0_mu_tilde: {J_b0_mu_tilde:.2f}, l: {l}")
