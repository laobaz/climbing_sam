cmake_minimum_required(VERSION 3.22)
project (rtdetr_test)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # 必须使用指定标准
set(CMAKE_CXX_EXTENSIONS OFF)  # 禁用编译器扩展（如 GNU 扩展）
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_BUILD_TYPE Debug)  # 生成调试信息
find_package(Threads)
find_package(CUDA REQUIRED)
find_package(spdlog REQUIRED) 

find_package(OpenCV REQUIRED)
find_library(TBB_LIBRARY
    NAMES tbb
    PATHS "/usr/lib/x86_64-linux-gnu"  
          "/usr/local/lib"            
    REQUIRED
)




set(CMAKE_BUILD_TYPE Debug)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++1y -pedantic -Wextra -ftemplate-depth=100000")


FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


set(ONNXRuntime_ROOT "/usr/local/onnxruntime") 


set(ONNXRuntime_INCLUDE_DIRS "${ONNXRuntime_ROOT}/include")
set(ONNXRuntime_LIBRARIES "${ONNXRuntime_ROOT}/lib/libonnxruntime.so") 



find_package( Boost 1.52 COMPONENTS system thread REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )

include_directories("/usr/local/cuda/include")

include_directories("/usr/local/include/")
include_directories("/usr/local/onnxruntime/include")
link_directories("/usr/local/lib")

include_directories(${OPENCV_INCLUDE_DIRS})


add_executable(rtdetr_test main.cc  ./Sam/SAM.cpp ./rtdetr/rtdetr.cpp ./code/object_detector.cc  ./code/preprocess_op.cc  ./code/utils.cc )

target_link_libraries(rtdetr_test /usr/local/cuda/lib64/libnvinfer.so yaml-cpp)
target_link_libraries(rtdetr_test  ${CUDA_CUDART_LIBRARY})
target_link_libraries(rtdetr_test ${OpenCV_LIBS})
target_link_libraries(rtdetr_test  ${ONNXRuntime_LIBRARIES})
target_link_libraries(rtdetr_test  ${TBB_LIBRARY})
target_link_libraries(rtdetr_test  spdlog::spdlog)



target_link_libraries(rtdetr_test ${Boost_LIBRARIES})

# add_definitions(-O2 -pthread)




message("CUDA_CUDART_LIBRARY: ${CUDA_CUDART_LIBRARY}")



