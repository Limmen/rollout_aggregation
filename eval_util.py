import time
import csv
import multiprocessing
from policy_iteration import PI
from value_iteration import VI
from pomdp_util import POMDPUtil


class EvalUtil:
    """
    Util functions for evaluation
    """

    @staticmethod
    def print_mu(mu, J, B_n, b_n_0):
        """
        Prints the base policy mu and its cost-to-go
        """

        print(" ")
        print("---- mu ----")
        for i in range(len(B_n)):
            print(f"b: {list(map(lambda x: round(x, 2), B_n[i]))}, u: {np.argmax(mu[i])}")
        print(f"J(b_n_0): {J[b_n_0]}")

    @staticmethod
    def compute_base_policy(B_n, P_b, C_b, U, b_n_0, gamma, pi=True, verbose=False):
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
                P=P_b, mu=mu_1, N=100, gamma=gamma, C=C_b, X=B_n_indices, U=U, x0=b_n_0, verbose=verbose)
        else:
            mu, J = VI.vi(
                P=P_b, epsilon=0.01, gamma=gamma, C=C_b, X=B_n_indices, U=U, verbose=verbose)
        if verbose:
            EvalUtil.print_mu(mu=mu, J=J, B_n=B_n, b_n_0=b_n_0)
        return mu, J

    @staticmethod
    def eval(results_file, X, b0, U, O, P, Z, C, gamma, l):
        """
        Runs the evaluation and saves the results to a csv file
        """
        with open(results_file, mode='w', newline='', encoding='utf-8') as file:
            csv.writer(file).writerow(["n", "B_n", "T_mdp", "T_mu", "J_mu", "J_mu_tilde", "l"])
        ns = list(range(1, 50))
        for n in ns:
            start = time.time()
            B_n = POMDPUtil.B_n(n=n, X=X)
            b_n_0 = B_n.index(b0)
            P_b = POMDPUtil.P_b(B_n=B_n, X=X, U=U, O=O, P=P, Z=Z)
            C_b = POMDPUtil.C_b(B_n=B_n, X=X, U=U, C=C)
            T_mdp = time.time() - start
            start = time.time()
            mu, J_mu = EvalUtil.compute_base_policy(B_n=B_n, P_b=P_b, C_b=C_b, U=U, b_n_0=b_n_0, gamma=gamma, pi=False)
            T_mu = time.time() - start
            J_b0_mu = POMDPUtil.parallel_evaluate(mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b0=b0, B_n=B_n, J_mu=J_mu,
                                         gamma=gamma, base=True)
            J_b0_mu_tilde = POMDPUtil.parallel_evaluate(mu=mu, P=P, Z=Z, C=C, O=O, X=X, U=U, b0=b0, B_n=B_n, J_mu=J_mu,
                                               gamma=gamma, base=False, l=l)
            with open(results_file, mode='a', newline='', encoding='utf-8') as file:
                csv.writer(file).writerow([n, len(B_n), f"{T_mdp:.2f}", f"{T_mu:.2f}", f"{J_b0_mu:.2f}",
                                           f"{J_b0_mu_tilde:.2f}", l])
                print(f"n: {n}, B_n_size: {len(B_n)}, T_mdp: {T_mdp:.3f}s, T_mu: {T_mu:.3f}s, J_b0_mu: {J_b0_mu:.2f},"
                      f"J_b0_mu_tilde: {J_b0_mu_tilde:.2f}, l: {l}")
