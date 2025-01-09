import numpy as np
from pomdp_util import POMDPUtil
from apt_pomdp import APTPOMDP
from csle_tolerance.util.pomdp_solve_parser import PomdpSolveParser

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
    #pomdp-solve -pomdp apt.pomdp -method incprune
    with open("apt.pomdp", 'w') as file:
        file.write(POMDPUtil.pomdp_solver_file(gamma, X, U, O, P, b0, Z, C))
    alpha_vectors = PomdpSolveParser.parse_alpha_vectors(
        file_path="./apt-77320.alpha")
    val = np.min([np.dot(b0, list(-np.array(alpha[1]))) for alpha in alpha_vectors])
    print(val)

    # for i in range(len(alpha_vectors)):
    #     dot_vals.append(np.dot(b_vec, list(np.array(alpha_vectors[i][1][0:2]))))
    # val = np.min([np.dot([1,0], list(np.array(alpha[1][0:2]))) for alpha in alpha_vectors])
    # print(val)
    # print(alpha_vectors[0][1])

    # belief_space = np.linspace(0.0, 1, int(1.0 / 0.01))
    # print(belief_space)
    # for i in range(len(alpha_vectors)):
    #     print(f"a*:{alpha_vectors[i][0]}, vector: {list(-np.array(alpha_vectors[i][1][0:2]))}")
    # values_01 = []
    # for j, b in enumerate(belief_space):
    #     b_vec = [1 - b, b]
    #     dot_vals = []
    #     for i in range(len(alpha_vectors)):
    #         dot_vals.append(np.dot(b_vec, list(np.array(alpha_vectors[i][1][0:2]))))
    #     min_index = np.argmin(dot_vals)
    #     values_01.append(dot_vals[min_index])
    #     vec_dots = []
    #     print(f"{b} {values_01[-1]}")
    #     for b in belief_space:
    #         b_vec = [1 - b, b]
    #         vec_dots.append(-np.dot(b_vec, list(-np.array(alpha_vectors[min_index][1][0:2]))))