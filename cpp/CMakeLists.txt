cmake_minimum_required(VERSION 3.14)
project(su2_3d_lgt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(su2_3d_lgt
    src/main.cpp
)
target_include_directories(su2_3d_lgt PRIVATE src)

add_executable(larger_loops_run
    src/main_larger_loops.cpp
)
target_include_directories(larger_loops_run PRIVATE src)

add_executable(scaling_smear_run
    src/main_scaling_smear.cpp
)
target_include_directories(scaling_smear_run PRIVATE src)
