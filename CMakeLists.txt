cmake_minimum_required(VERSION 3.15)
project(PAPT)
#include(FetchContent)

#set(CMAKE_CXX_STANDARD 17)
#set(FORTRAN OFF)
#set(DEPENDENCYMANAGER_FETCHCONTENT OFF)
#
#add_executable(PAPT.exe src/main.cpp src/Hamiltonian.cpp src/Amplitudes.cpp src/Amplitudes.h src/ManyBody.cpp src/ManyBody.h src/LA.cpp src/LA.h)
#
#FetchContent_Declare(molpro-plugin GIT_REPOSITORY https://gitlab.com/molpro/molpro-plugin.git)
#FetchContent_MakeAvailable(molpro-plugin)
#target_link_libraries(PAPT.exe molpro::molpro-plugin)
#
#option(USE_MKL "Whether to use MKL library to speed up Eigen" OFF)
#if (USE_MKL)
#    set(BLA_VENDOR "Intel10_64lp_seq")
#    find_package(BLAS)
#    if (BLAS_FOUND)
#        message(STATUS "Eigen will use MKL")
#        target_compile_definitions(PAPT.exe PRIVATE EIGEN_USE_MKL_ALL=ON)
#        target_link_libraries(PAPT.exe BLAS::BLAS)
#    endif ()
#endif ()
#find_package(Eigen3 3.4.0 QUIET)
#if (EIGEN3_FOUND)
#    message(STATUS "Eigen3 found on system")
#else ()
#    message(STATUS "Eigen3::Eigen not found on system, and will be downloaded")
#    include(FetchContent)
#    FetchContent_Declare(eigen3
#            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
#            GIT_TAG 3.4.0
#            )
#    FetchContent_MakeAvailable(eigen3)
#    add_library(Eigen3::Eigen ALIAS eigen)
#endif ()
#target_link_libraries(PAPT.exe Eigen3::Eigen)

add_subdirectory(src)
enable_testing()
add_subdirectory(examples)
