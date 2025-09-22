cmake_minimum_required(VERSION 3.17)
project (fff)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find_package(FTK REQUIRED HINTS "/project/xli281_uksr/mxia/git/ftk/build")
find_package(FTK REQUIRED HINTS "/project/xli281_uksr/mxia/git/ftk/install/lib64/cmake/FTK")
# find_package (FTK REQUIRED)
# find_package (VTK REQUIRED)
find_package(VTK REQUIRED HINTS "/project/xli281_uksr/mxia/git/VTK/build/lib64/cmake/vtk-9.4")
find_package(zstd REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable (ftk-cylinder2D cylinder2D.cpp)
target_include_directories(ftk-cylinder2D PRIVATE "/project/xli281_uksr/mxia/git/ftk/install/include/ftk")
target_link_libraries (ftk-cylinder2D FTK::libftk)


file(GLOB SRC_FILES "src/*.cpp")
add_executable(main2d test/main2d.cpp ${SRC_FILES})
target_include_directories(main2d PRIVATE "./include" PRIVATE "/project/xli281_uksr/mxia/git/ftk/install/include/ftk" PRIVATE "./src")
target_link_libraries(main2d FTK::libftk zstd::libzstd_shared Eigen3::Eigen)

add_executable(cp_extraction2d test/cp_extraction_2d_sos.cpp ${SRC_FILES})
target_include_directories(cp_extraction2d PRIVATE "./include" PRIVATE "/project/xli281_uksr/mxia/git/ftk/install/include/ftk" PRIVATE "./src")
target_link_libraries(cp_extraction2d FTK::libftk zstd::libzstd_shared)

add_executable(test test/test.cpp ${SRC_FILES})
target_include_directories(test PRIVATE "./include" PRIVATE "/project/xli281_uksr/mxia/git/ftk/install/include/ftk" PRIVATE "./src")

if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found and enabled for CXX.")
    target_link_libraries(test PRIVATE FTK::libftk zstd::libzstd_shared  OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found. Continuing without OpenMP support.")
    target_link_libraries(test PRIVATE FTK::libftk zstd::libzstd_shared)
endif()
# target_link_libraries(test FTK::libftk zstd::libzstd_shared Eigen3::Eigen)

#编译时用cmake .. -DCMAKE_PREFIX_PATH=/project/xli281_uksr/mxia/git/VTK/build/lib64/cmake/vtk-9.4