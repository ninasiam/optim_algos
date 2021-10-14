# optim_algos

## Implementations of algorithms for solving basic but often encountered optimization problems for educational purposes.

  1.  Python directory: numpy, nptyping, matplotlib are required. 
        - Available Objectives:
          - L-smooth, mu-strongly convex objective. 
          - Least square objective, || Ax - b ||_2^2.
        - Available Algorithms:
          - Gradient Descent.
          - Accelerated Gradient Descent (Nesterov Accelerated Gradient).
          - Stochastic gradient Descent.
          
###### For both algorithmic settings the default values for the stepsize parameters are the optimal ones.
      
  2.  MatrixLS_Sparsity_Constraints: Eigen library is required (C++ code).
      Solving a Matrix Least Squares Problem under sparsity constraints. \
      Available Algorithms:
      - Alternating Direction Method of Multipliers (ADMM).
      - Fast Iterative Shrinkage/Thresholding Algorithm.

###### More problem settings and algorithms will be available soon.
      
      
      
