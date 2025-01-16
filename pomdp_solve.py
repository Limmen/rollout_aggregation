import numpy as np
from pomdp_util import POMDPUtil
from apt_pomdp import POMDP
from csle_tolerance.util.pomdp_solve_parser import PomdpSolveParser

if __name__ == '__main__':
    N = 1
    p_a = 0.2
    p_c = 0.2
    k = 8
    seed = 29123
    eta = 2
    A = POMDP.erdos_renyi_graph(K=N, p_c=p_c)
    X, x_to_vec, vec_to_x = POMDP.X(K=N)
    U, u_to_vec, vec_to_u = POMDP.U(K=N)
    C = POMDP.C(X=X, U=U, x_to_vec=x_to_vec, u_to_vec=u_to_vec, eta=eta)
    O, o_to_vec, vec_to_o = POMDP.O(k, K=N)
    Z = POMDP.Z(k, X=X, K=N, x_to_vec=x_to_vec, o_to_vec=o_to_vec, O=O)
    P = POMDP.P(p_a=p_a, X=X, U=U, x_to_vec=x_to_vec, u_to_vec=u_to_vec, N=N, A=A)
    b0 = POMDP.b0(K=N, X=X, vec_to_x=vec_to_x)
    gamma=0.99
    #pomdp-solve -pomdp apt.pomdp -method incprune
    with open("apt.pomdp", 'w') as file:
        file.write(POMDPUtil.pomdp_solver_file(gamma, X, U, O, P, b0, Z, C))
    alpha_vectors = PomdpSolveParser.parse_alpha_vectors(
        file_path="./apt-33578.alpha")
    val = np.min([np.dot(b0, list(-np.array(alpha[1]))) for alpha in alpha_vectors])
    # print(val)

    belief_space = np.linspace(0.0, 1, int(1.0 / 0.01))
    for b in belief_space:
        val = np.min([np.dot([1-b, b], list(-np.array(alpha[1]))) for alpha in alpha_vectors])
        print(f"{b} {val}")


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