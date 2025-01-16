from apt_pomdp import POMDP
from eval_util import EvalUtil

if __name__ == '__main__':
    N = 1
    p_a = 0.2
    p_c = 0.2
    k = 1
    seed = 29123
    eta = 2
    A = POMDP.erdos_renyi_graph(N=N, p_c=p_c)
    X, x_to_vec, vec_to_x = POMDP.X(N=N)
    U, u_to_vec, vec_to_u = POMDP.U(N=N)
    C = POMDP.C(X=X, U=U, x_to_vec=x_to_vec, u_to_vec=u_to_vec, eta=eta)
    O, o_to_vec, vec_to_o = POMDP.O(k, N=N)
    Z = POMDP.Z(k, X=X, N=N, x_to_vec=x_to_vec, o_to_vec=o_to_vec, O=O)
    P = POMDP.P(p_a=p_a, X=X, U=U, x_to_vec=x_to_vec, u_to_vec=u_to_vec, N=N, A=A)
    b0 = POMDP.b0(N=N, X=X, vec_to_x=vec_to_x)
    gamma=0.75
    l = 2
    results_file = "apt_results.csv"
    EvalUtil.eval(results_file=results_file, X=X, b0=b0, U=U, O=O, P=P, Z=Z, C=C, gamma=gamma, l=l, u_to_vec=u_to_vec)
