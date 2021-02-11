CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(MLS_SC)

# SET DIRECTORIES
SET(CMAKE_MINIMUM_REQUIRED_VERSION                           "3.5")
SET(CPP_LIBS                                                 "/SDKS/CPP_LIBS")
SET(EIGEN_ROOT_DIR                                           ${CPP_LIBS}/eigen)
SET(INCLUDE_DIR                                              ${CMAKE_CURRENT_LIST_DIR}/include)
SET(BIN_DIR                                                  ${CMAKE_CURRENT_LIST_DIR}/bin)

# SET GCC
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_CXX_FLAGS              "${CMAKE_CXX_FLAGS} -fopenmp -Wall -Wextra -std=c++17")
SET(CMAKE_CXX_FLAGS_RELEASE      "-mtune=native -march=native -O3 -w")
SET(CMAKE_CXX_FLAGS_DEBUG        "-g")


SET(compiler.vendor              ${CMAKE_CXX_COMPILER_ID})
SET(compiler.version             ${CMAKE_CXX_COMPILER_VERSION})

# SET THE INCLUDE DIR
INCLUDE_DIRECTORIES(${EIGEN_ROOT_DIR})
INCLUDE_DIRECTORIES(${INCLUDE_DIR})

# TESTS
ADD_SUBDIRECTORY(tests/matrix_ADMM)
ADD_SUBDIRECTORY(tests/FISTA)

MESSAGE(STATUS "C++ Compiler:     {vendor=\"${compiler.vendor}\", version=\"${compiler.version}\"}")


