import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.linalg import matrix_rank
from optim_algorithms import QuadraticUncon
from create_data import create_data_quad


def main():

    print("Gradient descent for quadratic problem")
    m = 100
    n = 50
    P, q, x, stepsize, mu, f_val_init = create_data_quad(dims=tuple([m, n]))
    print("Solution via gradient")
    (fval_grad, iter_gd) = QuadraticUncon.gradient_descent(P, q, x, 1/stepsize, f_val_init, criterion='gradient_norm', max_iter=100)
    print("Solution via Nesterov")
    (fval_nes, iter_nes) = QuadraticUncon.accel_gradient_descent(P, q, x, 1/stepsize, f_val_init, max_iter=100, mu=mu, criterion='gradient_norm', momentum='mu_s_convex')

    # Make the first plot
    plt.plot(fval_grad[0:iter_gd])
    plt.plot(fval_nes[0:iter_nes])
    plt.xlabel('iterations')
    plt.ylabel('f_value')
    plt.title('Compare first order algorithms on quadratic problem')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()