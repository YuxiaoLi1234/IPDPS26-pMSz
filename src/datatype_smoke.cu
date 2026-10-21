#include <mpi.h>
#include <zfp.h>

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include "pmsz_datatype.hpp"

using DataType = pmsz::DataType;

__global__ void smoke_kernel(DataType* data, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = static_cast<DataType>(data[idx] + static_cast<DataType>(1));
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    const std::string tmp_path = "/tmp/pmsz_uint16_smoke_input.bin";
    std::vector<DataType> host{1, 2, 3, 4, 5, 6, 7, 8};

    {
        std::ofstream out(tmp_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(host.size() * sizeof(DataType)));
    }
    {
        std::vector<DataType> loaded(host.size());
        std::ifstream in(tmp_path, std::ios::binary);
        in.read(reinterpret_cast<char*>(loaded.data()), static_cast<std::streamsize>(loaded.size() * sizeof(DataType)));
        assert(loaded == host);
    }

#ifdef PMSZ_USE_UINT16
    assert(pmsz::mpi_datatype<DataType>() == MPI_UINT16_T);
#else
    assert(pmsz::mpi_datatype<DataType>() == MPI_DOUBLE);
#endif

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count > 0) {
        DataType* d_data = nullptr;
        cudaMalloc(&d_data, host.size() * sizeof(DataType));
        cudaMemcpy(d_data, host.data(), host.size() * sizeof(DataType), cudaMemcpyHostToDevice);
        smoke_kernel<<<1, 64>>>(d_data, host.size());
        cudaDeviceSynchronize();
        cudaMemcpy(host.data(), d_data, host.size() * sizeof(DataType), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
        assert(host[0] == static_cast<DataType>(2));
    } else {
        std::cout << "[datatype_smoke] CUDA device unavailable, kernel check skipped" << std::endl;
    }

    const size_t nx = 2, ny = 2, nz = 2;
    std::vector<DataType> zfp_in{10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<DataType> zfp_out(zfp_in.size());

    std::vector<float> zfp_in_float;
    std::vector<float> zfp_out_float;
    void* field_in_ptr = zfp_in.data();
    void* field_out_ptr = zfp_out.data();
    zfp_type type = pmsz::zfp_type_for<DataType>();
    if constexpr (std::is_same_v<DataType, uint16_t>) {
        zfp_in_float = pmsz::to_zfp_float_buffer(zfp_in.data(), zfp_in.size());
        zfp_out_float.resize(zfp_in.size());
        field_in_ptr = zfp_in_float.data();
        field_out_ptr = zfp_out_float.data();
    }

    zfp_field* field = zfp_field_3d(field_in_ptr, type, nx, ny, nz);
    zfp_stream* zfp = zfp_stream_open(nullptr);
    zfp_stream_set_reversible(zfp);
    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    void* buffer = malloc(bufsize);
    bitstream* stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);
    size_t csize = zfp_compress(zfp, field);
    assert(csize > 0);

    zfp_field* out_field = zfp_field_3d(field_out_ptr, type, nx, ny, nz);
    zfp_stream_rewind(zfp);
    bool ok = zfp_decompress(zfp, out_field);
    assert(ok);

    if constexpr (std::is_same_v<DataType, uint16_t>) {
        pmsz::from_zfp_float_buffer(zfp_out_float.data(), zfp_out_float.size(), zfp_out.data());
    }

    assert(zfp_in == zfp_out);

    zfp_field_free(field);
    zfp_field_free(out_field);
    zfp_stream_close(zfp);
    stream_close(stream);
    free(buffer);

    MPI_Finalize();
    std::cout << "[datatype_smoke] passed for " << pmsz::kDataTypeName << std::endl;
    return 0;
}
