#ifndef MASTER_LIB_HPP // Guard: if master_lib.hpp hasn't been included yet...
#define MASTER_LIB_HPP // #define this so the compiler knows it has been included

#include <iomanip>
#include <fstream>
#include <iostream>
#include <time.h>
#include <math.h>
#include <string>
#include <limits>

#include <array>

#define EIGEN_DONT_PARALLELIZE
#define PRINT_INFO
#define FACTORS_ARE_TRANSPOSED


#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

#endif
