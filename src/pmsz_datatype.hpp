#pragma once

#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <zfp.h>

namespace pmsz {

#ifdef PMSZ_USE_UINT16
using DataType = uint16_t;
inline constexpr const char* kDataTypeName = "uint16";
#else
using DataType = double;
inline constexpr const char* kDataTypeName = "double";
#endif

template <typename T>
__host__ __device__ constexpr T invalid_val();

template <>
__host__ __device__ constexpr double invalid_val<double>() {
    return 1e30;
}

template <>
__host__ __device__ constexpr uint16_t invalid_val<uint16_t>() {
    return std::numeric_limits<uint16_t>::max();
}

template <typename T>
inline MPI_Datatype mpi_datatype() {
    if constexpr (std::is_same_v<T, uint8_t>) return MPI_UINT8_T;
    else if constexpr (std::is_same_v<T, uint16_t>) return MPI_UINT16_T;
    else if constexpr (std::is_same_v<T, uint32_t>) return MPI_UINT32_T;
    else if constexpr (std::is_same_v<T, uint64_t>) return MPI_UINT64_T;
    else if constexpr (std::is_same_v<T, int8_t>) return MPI_INT8_T;
    else if constexpr (std::is_same_v<T, int16_t>) return MPI_INT16_T;
    else if constexpr (std::is_same_v<T, int32_t>) return MPI_INT32_T;
    else if constexpr (std::is_same_v<T, int64_t>) return MPI_INT64_T;
    else if constexpr (std::is_same_v<T, float>) return MPI_FLOAT;
    else if constexpr (std::is_same_v<T, double>) return MPI_DOUBLE;
    else {
        throw std::runtime_error("Unsupported MPI datatype");
    }
}

template <typename T>
inline zfp_type zfp_type_for() {
    if constexpr (std::is_same_v<T, double>) return zfp_type_double;
    return zfp_type_float;
}

template <typename T>
inline std::vector<float> to_zfp_float_buffer(const T* input, size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(input[i]);
    }
    return out;
}

template <typename T>
inline void from_zfp_float_buffer(const float* input, size_t n, T* output) {
    if constexpr (std::is_same_v<T, uint16_t>) {
        for (size_t i = 0; i < n; ++i) {
            const float v = input[i];
            if (!std::isfinite(v)) {
                throw std::runtime_error("ZFP produced non-finite value for uint16_t output");
            }
            const float rounded = std::round(v);
            if (rounded < 0.0f || rounded > static_cast<float>(std::numeric_limits<uint16_t>::max())) {
                throw std::runtime_error("ZFP produced out-of-range value for uint16_t output");
            }
            output[i] = static_cast<uint16_t>(rounded);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            output[i] = static_cast<T>(input[i]);
        }
    }
}

}  // namespace pmsz
