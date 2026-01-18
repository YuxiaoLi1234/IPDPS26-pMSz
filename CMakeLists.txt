cmake_minimum_required(VERSION 3.18)
project(pMSz LANGUAGES CXX CUDA)

# --- Basic settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# Use mpicxx as host compiler for nvcc (equivalent to: nvcc -ccbin mpicxx)
# You can also pass this via command line:
#   cmake .. -DCMAKE_CUDA_HOST_COMPILER=mpicxx
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  set(CMAKE_CUDA_HOST_COMPILER mpicxx CACHE FILEPATH "CUDA host compiler" FORCE)
endif()

# --- Find dependencies ---
find_package(OpenMP REQUIRED)

# Prefer searching common system paths for libs
find_library(ZSTD_LIB zstd REQUIRED)
find_library(ZFP_LIB  zfp  REQUIRED)

# If you also need SZ3, uncomment:
# find_library(SZ3_LIB sz3 REQUIRED)

# --- Helper function to define each CUDA executable ---
function(add_pmsz_exe exe_name cu_file)
  add_executable(${exe_name} ${cu_file})

  # Enable separable compilation if you ever split device code across TUs (safe to keep ON)
  set_target_properties(${exe_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  # Match your flags:
  # nvcc: -std=c++17 already handled via CUDA_STANDARD
  # host compiler: -fopenmp
  target_compile_options(${exe_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=177>
    $<$<COMPILE_LANGUAGE:CUDA>:--disable-warnings>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
    $<$<COMPILE_LANGUAGE:CXX>:-w>
    )



  # Link libs: -lzstd -lzfp and OpenMP
  target_link_libraries(${exe_name} PRIVATE
    ${ZSTD_LIB}
    ${ZFP_LIB}
    OpenMP::OpenMP_CXX
  )

  # If you need SZ3:
  # target_link_libraries(${exe_name} PRIVATE ${SZ3_LIB})

endfunction()

# --- Build three executables ---
add_pmsz_exe(pMSz      ${CMAKE_SOURCE_DIR}/src/pMSz.cu)
add_pmsz_exe(sync_pMSz ${CMAKE_SOURCE_DIR}/src/sync_pMSz.cu)
add_pmsz_exe(naive_MSz ${CMAKE_SOURCE_DIR}/src/naive_MSz.cu)
add_pmsz_exe(find_best_ocr ${CMAKE_SOURCE_DIR}/src/find_best_ocr.cu)
