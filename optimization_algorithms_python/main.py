import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.linalg import matrix_rank
from optim_algorithms import QuadraticUncon
from create_data import create_data_quad


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