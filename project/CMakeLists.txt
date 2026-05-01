cmake_minimum_required(VERSION 3.18)
project(img-compressor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer the system g++ as CUDA host compiler when possible
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  find_program(GPP g++)
  if(GPP)
    set(CMAKE_CUDA_HOST_COMPILER ${GPP} CACHE PATH "CUDA host compiler")
  endif()
endif()

include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/third_party)

find_package(JPEG REQUIRED)

set(SRCS
  src/main.cu
  src/compressor_gpu.cu
  src/compressor_cpu.cpp
  src/jpeg_scanline_writer.cpp
  src/saliency.cpp
)

add_executable(img-compressor ${SRCS})

target_include_directories(img-compressor PRIVATE ${JPEG_INCLUDE_DIR})
target_link_libraries(img-compressor PRIVATE ${JPEG_LIBRARY})

set_target_properties(img-compressor PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(img-compressor PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O2>
)
