from stopping_pomdp import StoppingPOMDP
from eval_util import EvalUtil

if __name__ == '__main__':
    X = StoppingPOMDP.X()
    U = StoppingPOMDP.U()
    C = StoppingPOMDP.C(intrusion_stop_gain=10, intrusion_cost=1, stopping_cost=10)
    O = StoppingPOMDP.O(50)
    Z = StoppingPOMDP.Z(50)
    P = StoppingPOMDP.P(intrusion_start_probability=0.2)
    b0 = StoppingPOMDP.b0()
    gamma = 0.99
    l = 1
    results_file = "stopping_results.csv"
    EvalUtil.eval(results_file=results_file, X=X, b0=b0, U=U, O=O, P=P, Z=Z, C=C, gamma=gamma, l=l, stopping=True)