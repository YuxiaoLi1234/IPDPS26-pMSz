cmake_minimum_required(VERSION 3.18)
project(pMSz LANGUAGES CXX CUDA)

# ========================
# Standards & build type
# ========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ========================
# Compilers (NERSC / MPI)
# ========================
# Recommended:
# cmake -S . -B build -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CUDA_HOST_COMPILER=mpicxx
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  set(CMAKE_CUDA_HOST_COMPILER mpicxx CACHE FILEPATH "CUDA host compiler" FORCE)
endif()

# ========================
# Core dependencies
# ========================
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# ZSTD / ZFP (library only)
find_library(ZSTD_LIB zstd REQUIRED)
find_library(ZFP_LIB  zfp  REQUIRED)

# --- Provide PkgConfig::ZSTD for SZ3 exported target (some SZ3 installs require this) ---
find_package(PkgConfig QUIET)

if(PkgConfig_FOUND)
  # Try common pkg-config names for zstd
  pkg_check_modules(ZSTD QUIET IMPORTED_TARGET libzstd)
  if(NOT TARGET PkgConfig::ZSTD)
    pkg_check_modules(ZSTD QUIET IMPORTED_TARGET zstd)
  endif()
endif()

# If still not found, create a minimal fallback target using the library we already found
if(NOT TARGET PkgConfig::ZSTD)
  message(STATUS "Creating fallback target PkgConfig::ZSTD using ZSTD_LIB=${ZSTD_LIB}")
  add_library(PkgConfig::ZSTD INTERFACE IMPORTED)
  set_target_properties(PkgConfig::ZSTD PROPERTIES
    INTERFACE_LINK_LIBRARIES "${ZSTD_LIB}"
  )
endif()

# ========================
# SZ3 autodetection
# ========================
option(USE_SZ3 "Enable SZ3 support" ON)

set(SZ3_FOUND FALSE)
set(SZ3_TARGET "")

if(USE_SZ3)
  # 1) Try config package first (preferred)
  find_package(SZ3 CONFIG QUIET)

  # Accept common exported target names
  if(TARGET SZ3::sz3)
    set(SZ3_TARGET SZ3::sz3)
  elseif(TARGET SZ3::SZ3)
    set(SZ3_TARGET SZ3::SZ3)
  elseif(TARGET SZ3::sz3c)
    set(SZ3_TARGET SZ3::sz3c)
  elseif(TARGET SZ3::SZ3c)
    set(SZ3_TARGET SZ3::SZ3c)
  endif()

  if(SZ3_TARGET)
    message(STATUS "Found SZ3 via CMake package target: ${SZ3_TARGET}")
    set(SZ3_FOUND TRUE)
  else()
    # 2) Fallback: manual search (headers + lib)
    set(_SZ3_HINTS "")
    if(DEFINED SZ3_ROOT)
      list(APPEND _SZ3_HINTS "${SZ3_ROOT}")
    endif()
    if(DEFINED ENV{SZ3_ROOT})
      list(APPEND _SZ3_HINTS "$ENV{SZ3_ROOT}")
    endif()

    find_path(SZ3_INCLUDE_DIR
      NAMES SZ3/api/sz.hpp
      HINTS ${_SZ3_HINTS}
      PATH_SUFFIXES include
    )

    find_library(SZ3_LIBRARY
      NAMES sz3 sz3c SZ3 SZ3c
      HINTS ${_SZ3_HINTS}
      PATH_SUFFIXES lib lib64
    )

    if(SZ3_INCLUDE_DIR AND SZ3_LIBRARY)
      message(STATUS "Found SZ3 manually:")
      message(STATUS "  include = ${SZ3_INCLUDE_DIR}")
      message(STATUS "  library = ${SZ3_LIBRARY}")

      # Create an imported target so downstream usage is uniform
      add_library(SZ3::sz3 UNKNOWN IMPORTED)
      set_target_properties(SZ3::sz3 PROPERTIES
        IMPORTED_LOCATION "${SZ3_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SZ3_INCLUDE_DIR}"
      )
      set(SZ3_TARGET SZ3::sz3)
      set(SZ3_FOUND TRUE)
    else()
      set(SZ3_FOUND FALSE)
    endif()
  endif()

  if(NOT SZ3_FOUND)
    message(FATAL_ERROR
      "USE_SZ3=ON but SZ3 not found.\n"
      "Solutions:\n"
      "  1) cmake ... -DCMAKE_PREFIX_PATH=/path/to/sz3_prefix (or -DSZ3_DIR=...)\n"
      "  2) cmake ... -DSZ3_ROOT=/path/to/sz3_prefix\n"
      "  3) export SZ3_ROOT=/path/to/sz3_prefix\n"
      "  4) or disable: -DUSE_SZ3=OFF\n"
      "\n"
      "Tip (your case): you have SZ3Config.cmake under ~/lib64/cmake/SZ3, so try:\n"
      "  cmake -S . -B build -DSZ3_DIR=$ENV{HOME}/lib64/cmake/SZ3\n"
    )
  endif()
endif()

# ========================
# Executable helper
# ========================
function(add_pmsz_exe exe_name cu_file)
  add_executable(${exe_name} ${cu_file})

  set_target_properties(${exe_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  target_compile_options(${exe_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=177>
    $<$<COMPILE_LANGUAGE:CUDA>:--disable-warnings>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-w>
  )

  target_link_libraries(${exe_name} PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    ${ZSTD_LIB}
    ${ZFP_LIB}
  )

  if(USE_SZ3)
    target_link_libraries(${exe_name} PRIVATE ${SZ3_TARGET})
    target_compile_definitions(${exe_name} PRIVATE USE_SZ3=1)
  endif()
endfunction()

# ========================
# Targets
# ========================
add_pmsz_exe(pMSz          ${CMAKE_SOURCE_DIR}/src/pMSz.cu)
add_pmsz_exe(sync_pMSz     ${CMAKE_SOURCE_DIR}/src/sync_pMSz.cu)
add_pmsz_exe(naive_MSz     ${CMAKE_SOURCE_DIR}/src/naive_MSz.cu)
add_pmsz_exe(find_best_ocr ${CMAKE_SOURCE_DIR}/src/find_best_ocr.cu)
