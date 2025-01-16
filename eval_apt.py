from apt_pomdp import POMDP
from eval_util import EvalUtil

if __name__ == '__main__':
    K = 5
    p_a = 0.2
    p_c = 0.2
    k = 1
    seed = 29123
    eta = 2
    N = 5
    M = 300
    A = POMDP.erdos_renyi_graph(K=K, p_c=p_c)
    X, x_to_vec, vec_to_x = POMDP.X(K=K)
    U, u_to_vec, vec_to_u, component_spaces = POMDP.U(K=K)
    C = POMDP.C(X=X, U=U, x_to_vec=x_to_vec, u_to_vec=u_to_vec, eta=eta)
    O, o_to_vec, vec_to_o = POMDP.O(k, K=K)
    Z = POMDP.Z(k, X=X, K=K, x_to_vec=x_to_vec, o_to_vec=o_to_vec, O=O)
    P = POMDP.P(p_a=p_a, X=X, U=U, x_to_vec=x_to_vec, u_to_vec=u_to_vec, N=K, A=A)
    b0 = POMDP.b0(K=K, X=X, vec_to_x=vec_to_x)
    gamma=0.75
    l = 1
    rollout_length = N
    rollout_mc_samples = 100
    monte_carlo = True
    certainty_equivalence = True
    multiagent = True
    EvalUtil.exact_eval(X=X, b0=b0, U=U, O=O, P=P, Z=Z, C=C, gamma=gamma, l=l, u_to_vec=u_to_vec, N=N,
                        rollout_length=rollout_length, monte_carlo=monte_carlo,
                        rollout_mc_samples=rollout_mc_samples, certainty_equivalence=certainty_equivalence,
                        multiagent=multiagent, component_spaces=component_spaces, vec_to_u=vec_to_u)
    # EvalUtil.monte_carlo_eval(X=X, b0=b0, U=U, O=O, P=P, Z=Z, C=C, gamma=gamma, l=l, u_to_vec=u_to_vec, N=N, M=M,
    #                           rollout_length=rollout_length, rollout_mc_samples=rollout_mc_samples,
    #                           multiagent=multiagent, certainty_equivalence=certainty_equivalence, vec_to_u=vec_to_u,
    #                           component_spaces=component_spaces)
