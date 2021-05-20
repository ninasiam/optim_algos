import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.linalg import matrix_rank
from optim_algorithms import QuadraticUncon
from create_data import create_data_LS

def main():

    print("Gradient descent for Least squares")
    m, n = 500, 100
    A, b, x_init, f_val_init = create_data_LS(dims=(m,n))

    print("Solution via SGD")
    fval_sgd, iter_sgd = QuadraticUncon.stochastic_gradient_descent(A, b, x_init, 0.1, f_val_init, criterion='gradient_norm', max_iter=100)

   # Make the first plot
    plt.plot(fval_sgd[0:iter_sgd])
    plt.xlabel('iterations')
    plt.ylabel('f_value')
    plt.title('Least Squares minimization')
    plt.legend(['SGD'])
    plt.grid()
    plt.show()
if __name__ == '__main__':
    main()