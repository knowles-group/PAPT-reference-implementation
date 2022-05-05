cmake_minimum_required(VERSION 3.15)
project(PAPT-reference-implementation)
include(FetchContent)


set(CMAKE_CXX_STANDARD 17)

add_executable(PAPT-reference-implementation main.cpp Hamiltonian.cpp Amplitudes.cpp Amplitudes.h ManyBody.cpp ManyBody.h LA.cpp LA.h)
FetchContent_Declare(fcidump GIT_REPOSITORY https://gitlab.com/molpro/fcidump.git)

set(FORTRAN OFF)
FetchContent_MakeAvailable(fcidump)
target_link_libraries(PAPT-reference-implementation molpro::FCIdump)

option(USE_MKL "Whether to use MKL library to speed up Eigen" OFF)
if (USE_MKL)
    set(BLA_VENDOR "Intel10_64lp_seq")
    find_package(BLAS)
    if (BLAS_FOUND)
        message(STATUS "Eigen will use MKL")
        target_compile_definitions(PAPT-reference-implementation PRIVATE EIGEN_USE_MKL_ALL=ON)
        target_link_libraries(PAPT-reference-implementation BLAS::BLAS)
    endif ()
endif ()
find_package(Eigen3 3.4.0 QUIET)
if (EIGEN3_FOUND)
    message(STATUS "Eigen3 found on system")
else ()
    message(STATUS "Eigen3::Eigen not found on system, and will be downloaded")
    include(FetchContent)
    FetchContent_Declare(eigen3
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
            )
    FetchContent_MakeAvailable(eigen3)
    add_library(Eigen3::Eigen ALIAS eigen)
endif ()
target_link_libraries(PAPT-reference-implementation Eigen3::Eigen)

enable_testing()
add_subdirectory(examples)
