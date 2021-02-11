CMAKE_MINIMUM_REQUIRED(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

PROJECT(test_ADMM)

SET(TEST_SRC_FILE_NAMES    test_ADMM.cpp})

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
########################################################################################################################

########################################################################################################################
### CREATE EXECUTABLE FROM SOURCE AND HEADER FILES
########################################################################################################################
SET(LIBRARIES_TO_LINK       OpenMP::OpenMP_CXX)

MESSAGE(STATUS " [ ${PROJECT_NAME_TOUPPER} ] Creating executable test_FISTA.cpp")

ADD_EXECUTABLE(test_ADMM test_ADMM.cpp)

TARGET_LINK_LIBRARIES(${LIBRARIES_TO_LINK})
SET_TARGET_PROPERTIES(
   test_ADMM 
   PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}"
)
