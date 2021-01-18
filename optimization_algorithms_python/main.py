import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.linalg import matrix_rank
from optim_algorithms import QuadraticUncon


def create_data_quad(dim):

    # Create a positive Definite matrix P
    P_tmp = np.random.rand(dim, dim)
    P = P_tmp.T@P_tmp
    U, S, Vh = np.linalg.svd(P, full_matrices=True)
    eigs_tmp = np.random.rand(dim)
    eigs = (eigs_tmp+1)*1100
    P = (U@np.diag(eigs))@Vh

    # Create q in the column space of P
    x_tmp = np.random.rand(dim, 1)
    q = np.dot(P, x_tmp)

    # Compute the optimal values
    x_star = -np.dot(LA.pinv(P), q)
    f_star = (1/2)*(x_star.T@P@x_star) + q.T@x_star
    print(f"Optimal Value: {f_star[0][0]}")

    # Initialize parameters
    x_init = np.random.rand(dim, 1)
    f_val_init = (1/2)*(x_init.T@P@x_init) + q.T@x_init
    print(f"Initial Value: {f_val_init[0][0]}")
    print("Condition Number of the problem")
    print(np.amax(eigs)/np.amin(eigs))

    L = np.amax(eigs)
    mu = np.amin(eigs)
    return P, q, x_init, L, mu, f_val_init[0][0]


def main():

    print("Gradient descent for quadratic problem")
    dim = 100                                                            # shape of the square matrix
    P, q, x, stepsize, mu, f_val_init = create_data_quad(dim=dim)
    print("Solution via gradient")
    (fval_grad, iter_gd) = QuadraticUncon.gradient_descent(P, q, x, 1/stepsize, f_val_init, criterion='gradient_norm', max_iter=100)
    print("Solution via Nesterov")
    (fval_nes, iter_nes) = QuadraticUncon.accel_gradient_descent(P, q, x, 1/stepsize, f_val_init, max_iter=100, mu=mu, criterion='gradient_norm', momentum='mu_s_convex')

    plt.subplot(2, 1, 1)

    # Make the first plot
    plt.plot(fval_grad[0:iter_gd])
    plt.title('Gradient Descent')

    # Set the second subplot as active, and make the second plot.
    plt.subplot(2, 1, 2)
    plt.plot(fval_nes[0:iter_nes])
    plt.title('Nesterov')

    plt.show()

if __name__ == '__main__':
    main()