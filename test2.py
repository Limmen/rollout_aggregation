from apt_pomdp import APTPOMDP
from eval_util import EvalUtil
from pomdp_util import POMDPUtil
import time


if __name__ == '__main__':
    N = 6
    p_a = 0.25
    k = 12
    seed = 29123
    gamma=0.75
    l = 1

    for N in [1, 2, 3, 4, 5, 6, 7, 8]:
        X = APTPOMDP.X(N=N)
        U, control_id_to_vec, control_vec_to_id  = APTPOMDP.U(N=N)
        C = APTPOMDP.C(X=X, U=U)
        O = APTPOMDP.O(k)
        Z = APTPOMDP.Z(k, X=X)
        P = APTPOMDP.P(p_a=p_a, X=X, U=U, control_id_to_vec=control_id_to_vec)
        b0 = APTPOMDP.b0(N=N)
        print(f"N={N}")
        for n in range(1, 9):
            start = time.time()
            B_n = POMDPUtil.B_n(n=n, X=X)
            P_b = POMDPUtil.P_b(B_n=B_n, X=X, U=U, O=O, P=P, Z=Z)
            C_b = POMDPUtil.C_b(B_n=B_n, X=X, U=U, C=C)
            b_n_0 = B_n.index(b0)
            mu, J_mu = EvalUtil.compute_base_policy(B_n=B_n, P_b=P_b, C_b=C_b, U=U, b_n_0=b_n_0, gamma=gamma,
                                                    pi=False, verbose=False)
            elapsed = (time.time() - start)
            print(f"{n} {elapsed:.10f}")
