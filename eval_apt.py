from apt_pomdp import APTPOMDP
from eval_util import EvalUtil

if __name__ == '__main__':
    N = 6
    p_a = 0.25
    k = 12
    seed = 29123
    X = APTPOMDP.X(N=N)
    U = APTPOMDP.U()
    C = APTPOMDP.C(X=X, U=U)
    O = APTPOMDP.O(k)
    Z = APTPOMDP.Z(k, X=X)
    P = APTPOMDP.P(p_a=p_a, X=X, U=U)
    b0 = APTPOMDP.b0(N=N)
    gamma=0.75
    l = 1
    results_file = "apt_results.csv"
    EvalUtil.eval(results_file=results_file, X=X, b0=b0, U=U, O=O, P=P, Z=Z, C=C, gamma=gamma, l=l)
