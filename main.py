import numpy as np
from policy_iteration import PolicyIteration
from stopping_pomdp import  StoppingPOMDP

if __name__ == '__main__':
    X = StoppingPOMDP.X()
    U = StoppingPOMDP.U()
    C = StoppingPOMDP.C(intrusion_stop_gain=5, intrusion_cost=1, stopping_cost=0.5)
    O = StoppingPOMDP.O(100)
    Z = StoppingPOMDP.Z(100)
    P = StoppingPOMDP.P(intrusion_start_probability=0.2)
    B_n = StoppingPOMDP.B_n(n=70, X=X)
    P_b = StoppingPOMDP.P_b(B_n=B_n, X=X, U=U, O=O, P=P, Z=Z)
    C_b = StoppingPOMDP.C_b(B_n=B_n, X=X, U=U, C=C)
    print(f"Belief space size: {len(B_n)}")

    B_n_indices = []
    mu_1 = []
    for i in range(len(B_n)):
        mu_1.append([1.0, 0.0])
        B_n_indices.append(i)
    x0 = B_n.index([1.0,0.0,0.0])

    mu, J, avg_returns, running_avg_returns = PolicyIteration.pi(P=P_b, mu=mu_1, N=10, gamma=0.99, C=C_b, X=B_n_indices,
                                                                 U=U, x0=x0)

    print(" ")
    print("---- mu ----")
    for i in range(len(B_n)):
        print(f"b: {list(map(lambda x: round(x, 2), B_n[i]))}, u: {np.argmax(mu[i])}")

    print(f"J(x_0): {J[x0]}")