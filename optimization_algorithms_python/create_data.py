import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.linalg import matrix_rank
from nptyping import NDArray
from optim_algorithms import QuadraticUncon


def create_data_quad(dim):
    """Creates matrix P, vector q for (1/2)x^TPx + qx quadratic problem
    :arg
        dim (int): dimension of the parameter
    :returns
        P (NDArray[float]): a positive define matrix in (1/2)x^TPx + qx
        q (NDArray[float]): the vector in (1/2)x^TPx + qx
        x_init (NDArray[float]): the initial value of the parameter x
        L (float): maximum eigenvalue of the Hessian matrix (smoothness parameter)
        mu (float): minimum eigenvalue of the Hessian matrix (strong convexity parameter)
        f_val_init (float): initial value of the cost function
    """
    # Create a positive Definite matrix P
    dims = tuple([dim, dim])
    P_tmp = np.random.normal(0, 1, size=dims)
    U, S, Vh = np.linalg.svd(P_tmp, full_matrices=False)

    # Tune the eigenvalues of the problem
    eig_min = 10
    eig_max = 100000
    eigs_tmp = eig_min + (eig_max - eig_min) * np.random.rand(dims[1]-2)
    eigs = np.concatenate((eigs_tmp, [eig_max, eig_min]), axis=0)

    # Create a positive definite matrix
    P = U@np.diag(eigs)@U.T

    # Create random vector q
    q = np.random.normal(0, 1, size=(dims[1], 1))

    # Compute the optimal values
    x_star = -np.dot(LA.pinv(P), q)
    f_star = (1/2)*(x_star.T@P@x_star) + q.T@x_star
    print(f"Optimal Value: {f_star[0][0]}")

    # Initialize parameters
    x_init = np.zeros(shape=(dims[1], 1))
    f_val_init = (1/2)*(x_init.T@P@x_init) + q.T@x_init
    print(f"Initial Value: {f_val_init[0][0]}")
    print("Condition Number of the problem")
    print(np.amax(eigs)/np.amin(eigs))

    return P, q, x_init, eig_max, eig_min, f_val_init[0][0]


if __name__ == '__main__':
    pass


