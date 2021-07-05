import numpy as np
from nptyping import NDArray
from numpy import linalg as LA
from numpy.linalg import matrix_rank
import matplotlib.pylab as plt


class QuadraticUncon:
    """Class that contains algorithms for the unconstrained quadratic problem (1/2)x^TPx + qx"""
    @staticmethod
    def update_alpha(alpha: float, Q: float):
        """ Update the parameters to compute the NAG parameters (for non strongly convex case)"""
        a = 1
        b = alpha**2 - Q
        c = -alpha**2
        D = b**2 - 4 * a * c
        new_alpha = (-b + np.sqrt(D))/2
        new_beta = (alpha * (1 - alpha))/(alpha**2 + new_alpha)

        return new_alpha, new_beta

    @staticmethod
    def gradient_descent(P: NDArray[float], q: NDArray[float], x: NDArray[float], stepsize: float,
                         f_val_init: float, max_iter=1000, epsilon=10e-4, criterion=None) -> tuple:
        """ Gradient Descent for the unconstrained Quadratic problem (1/2)x^TPx + qx
        :arg
        P (NDArray[float]): a positive define matrix in (1/2)x^TPx + qx
        q (NDArray[float]): the vector in (1/2)x^TPx + qx
        x NDArray[float]: the parameter
        stepsize (float): the stepsize for the gradient step
        f_val_init (float): the initial cost function value
        max_iter (int): maximum number of iterations
        epsilon (float): tolerance
        criteriom (string): available 'gradient_norm', 'max_iterations', 'rel_change'

        :returns
        fval, iter (tuple): the cost function value at each iteration
        """

        iter = 1
        f_val = []
        f_val.append(f_val_init)
        while iter < max_iter:

            gradient = np.dot(P, x) + q
            x_next = x - stepsize*gradient
            f_val_iter = (1/2)*(x_next.T@P@x_next) + q.T@x_next
            f_val.append(f_val_iter[0][0])
            x_prev = x
            x = x_next
            print(f"f_value: {f_val_iter[0][0]} at iteration: {iter}")

            if (criterion == 'gradient_norm' and LA.norm(gradient) < epsilon) \
               or (criterion == 'max_iterations' and iter > max_iter) \
               or (criterion == 'rel_change' and LA.norm(x - x_prev) < epsilon):
                print("Terminating Condition attained")
                break
            else:
                iter += 1

        return f_val, iter

    @staticmethod
    def accel_gradient_descent(P: NDArray[float], q: NDArray[float], x: NDArray[float], stepsize: float, f_val_init: float,
                               alpha=1, beta=0.3, mu=10e-6, max_iter=1000, epsilon=10e-4, criterion=None,
                               momentum='mu_s_convex') -> tuple:
        """ Accelerated Gradient Descent (Nesterov Accelerated Gradient) for the unconstrained Quadratic problem (1/2)x^TPx + qx
        :arg
        P (NDArray[float]): a positive define matrix in (1/2)x^TPx + qx
        q (NDArray[float]): the vector in (1/2)x^TPx + qx
        x NDArray[float]: the parameter
        stepsize (float): the stepsize for the gradient step
        f_val_init (float): the initial cost function value
        alpha (float): value to compute the momentum parameter
        max_iter (int): maximum number of iterations
        epsilon (float): tolerance
        criterion (string): available 'gradient_norm', 'max_iterations', 'rel_change'
        momentum (string): 'mu_s_convex' -> (1-sqrt(Q))/(1+sqrt(Q))

        :returns
        fval, iter (tuple): the cost function value at each iteration
        """

        iter = 1
        Q = mu * stepsize                                                 # in case of (1/L)
        if momentum == 'mu_s_convex':
            beta = (1 - np.sqrt(Q))/(1 + np.sqrt(Q))

        f_val = []
        f_val.append(f_val_init)
        y = x                                                           # initialize the y sequence to x
        while iter < max_iter:

            gradient = np.dot(P, y) + q
            x_next = y - stepsize*gradient
            if momentum != 'mu_s_convex':
                alpha, beta = QuadraticUncon.update_parameters(alpha, Q)
            y_next = x_next + beta*(x_next - x)
            f_val_iter = (1/2)*(x_next.T@P@x_next) + q.T@x_next
            f_val.append(f_val_iter[0][0])
            x_prev = x
            x = x_next
            y = y_next
            print(f"f_value: {f_val_iter[0][0]} at iteration: {iter}")
            if (criterion == 'gradient_norm' and LA.norm(gradient) < epsilon) \
                    or (criterion == 'max_iterations' and iter > max_iter) \
                    or (criterion == 'rel_change' and LA.norm(x - x_prev) < epsilon):
                print("Terminating Condition attained")
                break
            else:
                iter += 1

        return f_val, iter
class LeastSquares:
    """Class that contains algorithms for the Linear Least Squares problem """
    
    @staticmethod
    def stochastic_gradient_descent(A: NDArray[float], b: NDArray[float], x: NDArray[float], stepsize: float, f_val_init: float,
                               max_iter=1000, epsilon=10e-4, criterion=None) -> tuple:
        """Stochastic Gradient Descent for (1/2)|| Ax - b||_2^2 quadratic problem.
        :arg
        A (NDArray[float]): a positive define matrix in (1/2)x^TPx + qx
        b (NDArray[float]): the vector in (1/2)x^TPx + qx
        x NDArray[float]: the parameter
        stepsize (float): the stepsize for the gradient step
        f_val_init (float): the initial cost function value
        max_iter (int): maximum number of iterations
        epsilon (float): tolerance
        criterion (string): available 'gradient_norm', 'max_iterations', 'rel_change'

        :returns
        fval, iter (tuple): the cost function value at each iteration
        """
        # Decomposable cost function (preferably L2 norm)
        iter = 1
        f_val = []
        f_val.append(f_val_init)
        while iter < max_iter:
            i = int(np.random.randint(low=0, high=A.shape[0], size=1))
            gradient = (A[i, :]@x - b[i])*A[i, :].T # element wise
            x_next = x - stepsize*gradient.reshape((x.shape))
            f_val_iter = (1/2)*LA.norm(A@x_next - b) ** 2
            f_val.append(f_val_iter)
            x_prev = x
            x = x_next
            print(f"f_value: {f_val_iter} at iteration: {iter}")

            if (criterion == 'gradient_norm' and LA.norm(gradient) < epsilon) \
               or (criterion == 'max_iterations' and iter > max_iter) \
               or (criterion == 'rel_change' and LA.norm(x - x_prev) < epsilon):
                print("Terminating Condition attained")
                break
            else:
                iter += 1

        return f_val, iter

if __name__ == '__main__':
    pass


