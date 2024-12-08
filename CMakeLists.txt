cmake_minimum_required(VERSION 3.5)

project(cal_intrinsic)

set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")  # 禁用优化并启用调试信息
#set(CMAKE_C_FLAGS_DEBUG "-O0 -g")    # 对 C 语言文件也禁用优化
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3 -Wall -std=c++11")

find_package(OpenCV 3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
 ${PROJECT_SOURCE_DIR}/include
 ${CERES_INCLUDE_DIRS}
)

add_executable(cam_calib_func src/main.cpp include/cam_calib.cpp)
target_link_libraries(cam_calib_func ${OpenCV_LIBS} ${CERES_LIBRARIES})
