#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <bitset>
#include <zfp.h>
#include <numeric>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <atomic>
#include <cstring>
#include <iomanip>
#include <string>
#include "SZ3/api/sz.hpp"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <filesystem>
#include "zstd.h"
#include <zdict.h>
#include <unistd.h>
#include <omp.h>
#include <map> 

double additional_time = 0.0;
double compression_time = 0.0;
size_t cmpSize = 0;
double maxValue, minValue;
std::uintmax_t storageOverhead = 0;
size_t edit_cnt = 0;
int ite = 0;
int check = 0;
double editsTime = 0;
__device__ bool filtered = false;
std::vector<std::vector<float>> time_counter;
// ===== Simple MPI phase timer (drop-in) =====
#include <vector>
#include <string>
#include <numeric>

#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <cstdio>

int PX, PY, PZ;
float datatransfer = 0.0;
float finddirection = 0.0;
float getfcp = 0.0;
float fixtime_cp = 0.0;
float packing = 0.0;
float unpacking = 0.0;
float sending = 0.0;
double start_time, end_time, start_time_total;
std::vector<std::map<std::string, double>> times;
std::vector<double> comm_time, comp_time, pack_time, send_time;
int maxNeighbors_host = 14;
__device__ int maxNeighbors = 14;
double *sendbuff_right, *recvbuff_right, 
*sendbuff_left, *recvbuff_left, 
*sendbuff_top, *recvbuff_top, 
*sendbuff_bottom, *recvbuff_bottom,
*sendbuff_ld, *recvbuff_ld,
*sendbuff_ru, *recvbuff_ru;
size_t total_edit_cnt = 0;
size_t *all_max, *all_min;
int *DS_M, *AS_M, *dec_DS_M, *dec_AS_M, *de_direction_as, *de_direction_ds;
int *updated_vertex;
int *or_types;
int *edits;
uint8_t *delta_counter;
__device__ unsigned int count_f_max = 0, count_f_min = 0, edit_count = 0, count_f_dir = 0;
double delta;
int threshold, q;

using namespace std;
__device__ int directions_host[42] = 
{1,0,0,-1,0,0,
0,1,0,0,-1,0,
0,0,1, 0,0,-1,
-1,1,0,1,-1,0, 
-1,0,1,1,0,-1,
0,1,1,0,-1,-1,  
-1,1,1, 1,-1,-1};

__device__ int dx[] = {+1,  0, -1,  0, +1, -1, +1, -1};
__device__ int dy[] = { 0, +1,  0, -1, +1, +1, -1, -1};

int directions_host1[42] = 
{1,0,0,-1,0,0,
0,1,0,0,-1,0,
0,0,1, 0,0,-1,
-1,1,0,1,-1,0, 
-1,0,1,1,0,-1,
0,1,1,0,-1,-1,  
-1,1,1, 1,-1,-1};


void zfp_compress_decompress_3d(double* input_data, double* decp_data,
                                size_t nx, size_t ny, size_t nz, double tolerance) {
   
    zfp_field* field = zfp_field_3d(input_data, zfp_type_double, nx, ny, nz);

    zfp_stream* zfp = zfp_stream_open(NULL);

    zfp_stream_set_accuracy(zfp, tolerance);

    size_t bufsize = zfp_stream_maximum_size(zfp, field);

    void* buffer = malloc(bufsize);
    bitstream* stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    cmpSize = zfp_compress(zfp, field);
    if (!cmpSize) {
        fprintf(stderr, "compression failed\n");
        exit(EXIT_FAILURE);
    }

    zfp_field* field_dec = zfp_field_3d(decp_data, zfp_type_double, nx, ny, nz);
    zfp_stream_rewind(zfp);
    if (!zfp_decompress(zfp, field_dec)) {
        fprintf(stderr, "decompression failed\n");
        exit(EXIT_FAILURE);
    }

    zfp_field_free(field);
    zfp_field_free(field_dec);
    zfp_stream_close(zfp);
    stream_close(stream);
    free(buffer);
}

void zfp_compress_decompress_2d(double* input_data, double* decp_data,
                                size_t nx, size_t ny, double tolerance) {
    
    zfp_field* field = zfp_field_2d(input_data, zfp_type_double, nx, ny);

    zfp_stream* zfp = zfp_stream_open(NULL);

    zfp_stream_set_accuracy(zfp, tolerance);

    size_t bufsize = zfp_stream_maximum_size(zfp, field);

    void* buffer = malloc(bufsize);
    bitstream* stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    cmpSize = zfp_compress(zfp, field);
    if (!cmpSize) {
        fprintf(stderr, "compression failed\n");
        exit(EXIT_FAILURE);
    }

    zfp_field* field_dec = zfp_field_2d(decp_data, zfp_type_double, nx, ny);
    zfp_stream_rewind(zfp);
    if (!zfp_decompress(zfp, field_dec)) {
        fprintf(stderr, "decompression failed\n");
        exit(EXIT_FAILURE);
    }

    zfp_field_free(field);
    zfp_field_free(field_dec);
    zfp_stream_close(zfp);
    stream_close(stream);
    free(buffer);
}




template <typename T>
__device__ bool islarger_shared(const size_t v, const size_t u, 
                                T value_v1, T value_v2){
    return value_v1 > value_v2 || (value_v1 == value_v2 && v > u);
}

template <typename T>
__device__ bool isless_shared(const size_t v, const size_t u, 
                            T value_v1, T value_v2){
    return value_v1 < value_v2 || (value_v1 == value_v2 && v < u);
}



template <typename T>
__global__ void iscriticle(T *input_data, T *decp_data, 
                            size_t width_host, size_t height_host, size_t depth_host,
                            size_t data_size, int *or_types, int data_type = 1){

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i>=data_size) return;
    T data_value = decp_data[i];
    T input_data_value = input_data[i];
    
    
    bool is_maxima = true;
    bool is_minima = true;
    
    
    int x = i % width_host;
    int y = (i / width_host) % height_host;
    int z = (i / (width_host * height_host)) % depth_host;
   

    size_t largest_index = i;
    T largest_value = data_value;
    size_t global_largest_index = i;

    size_t smallest_index = i;
    T smallest_value = data_value;
    size_t global_smallest_index = i;
    
    for (int d = 0; d < maxNeighbors; d++) {
        // for(int k = 0; k < 8; k++){
    
            int dx = directions_host[3 * d];
            int dy = directions_host[3 * d + 1];
            int dz = directions_host[3 * d + 2];
            
            int nx = x + dx;
            int ny = y + dy;
            int nz = z + dz;
            
            size_t neighbor = nx + (ny + nz * height_host) * width_host;

            if (static_cast<int64_t>(nx) < 0 || nx >= width_host || static_cast<int64_t>(ny) < 0 || ny >= height_host || nz < 0 || nz >= depth_host || neighbor >= data_size) continue;
            T neighbor_value = decp_data[neighbor];
           
            if (neighbor_value > data_value) {
                is_maxima = false;
            }
            else if (neighbor_value == data_value and neighbor > i) {
                is_maxima = false;
            }
            
            if (neighbor_value < data_value) {
                is_minima = false;
            }
            else if (neighbor_value == data_value and neighbor < i) {
                is_minima = false;
            }

            if(islarger_shared<T>(neighbor, largest_index, neighbor_value, largest_value)){
                largest_index = neighbor;
                largest_value = neighbor_value;

            }

            
            if(isless_shared<T>(neighbor, smallest_index, neighbor_value, smallest_value)){
                smallest_index = neighbor;
                smallest_value = neighbor_value;
                
            }
        
    }
    
    
    if (data_type == 1){
        if(is_maxima) {
            
            or_types[2 * i] = -1;
            or_types[2 * i + 1] = smallest_index;

        }
        else if(is_minima) {
            
            or_types[2 * i] = -2;
            or_types[2 * i + 1] = largest_index;

            
            
        }
        else {
            
            or_types[2 * i] = smallest_index;
            or_types[2 * i + 1] = largest_index;

            
            
        }
        return;
    }
    
    int original_type = or_types[2 * i];
    
    if (data_type == 0) {
    
        if ((is_maxima &&  original_type!= -1) || (!is_maxima && original_type == -1)) {
            count_f_max+=1;

        }

        if ((is_minima &&  original_type!= -2) || (!is_minima && original_type == -2)) {
            count_f_min+=1;

        }

        
        
        
        // check vertex's largest neighbor if not a max;
        if ((is_minima && original_type!= -2) || (!is_minima && original_type == -2) || (is_maxima &&  original_type!= -1) || (!is_maxima && original_type == -1)){
                count_f_dir+=1;
                return;
        }

        int original_largest_index  = or_types[2 * i + 1];
        if(or_types[2 * i] == -1) original_largest_index =i;
        else if(or_types[2 * i] == -2) original_largest_index = or_types[2 * i + 1];

        if(largest_index != original_largest_index){
            count_f_dir+=1;
            
            
            
        }

        int original_smallest_index  = or_types[2 * i];
        if(or_types[2 * i] == -1) original_smallest_index = or_types[2 * i + 1];
        else if(or_types[2 * i] == -2) original_smallest_index = i;

        if(smallest_index != original_smallest_index){
            count_f_dir+=1;
            
        }
        return;
        
    }
}



template <typename T>
void compressLocalData(const std::string file_path, std::string cpfilename, const std::string filename,  
    const T *input_data_host, T *&decp_data_host, size_t width_host, size_t height_host, size_t depth_host, 
    T &bound, std::string decpfilename,int rank, int processData = 0) {

    auto start = std::chrono::high_resolution_clock::now();
    SZ3::Config conf(static_cast<int>(width_host), static_cast<int>(height_host),
    static_cast<int>(depth_host));
    
    
    conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
    conf.errorBoundMode = SZ3::EB_ABS;
    conf.absErrorBound = bound; 

    size_t data_size = width_host * height_host * depth_host;

    char *compressedData = SZ_compress(conf, input_data_host, cmpSize);

    decp_data_host = new T[data_size];
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    compression_time = double(duration.count())/1000;
    
    start = std::chrono::high_resolution_clock::now();
    SZ_decompress(conf, compressedData, cmpSize, decp_data_host);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    delete[] compressedData;

    
    
    std::uintmax_t original_dataSize = std::filesystem::file_size(file_path);
    double cr = double(original_dataSize) / cmpSize;
    
}

template <typename T>
void getdata(const std::string &filename, T *&input_data_host, size_t data_size) {

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size != static_cast<std::streamsize>(data_size * sizeof(T))) {
        std::cout<< "datasize is: "<< data_size<< ", datapoint size: "<< sizeof(T)<<std::endl;
        std::cout<<"file size: "<<size<<" calculated size: "<<static_cast<std::streamsize>(data_size * sizeof(T))<<std::endl;
        std::cerr << "File size does not match expected data size." << std::endl;
        return;
    }


    input_data_host = new T[data_size];
    file.read(reinterpret_cast<char *>(input_data_host), size);
    if (!file) {
        std::cerr << "Error reading file." << std::endl;
        return;
    }
    minValue = *std::min_element(input_data_host, input_data_host + data_size);
    maxValue = *std::max_element(input_data_host, input_data_host + data_size);

}



std::string extractFilename(const std::string& path) {
    
    int lastSlash = path.find_last_of("/\\");
    std::string filename = (lastSlash == std::string::npos) ? path : path.substr(lastSlash + 1);

    int dotPos = filename.find_last_of('.');
    std::string name = (dotPos == std::string::npos) ? filename : filename.substr(0, dotPos);

    return name;
}






int main(int argc, char** argv) {


    
    std::cout << std::fixed << std::setprecision(16);
    std::string dimension = argv[1];
    
    
    double bound = std::stod(argv[2]);
    std::string compressor_id = argv[3];

    size_t width_host, height_host, depth_host;
    std::string file_path;
    std::istringstream iss(dimension);
    char delimiter;

    
    
    if (std::getline(iss, file_path, ',')) {
        if (iss >> width_host >> delimiter && delimiter == ',' &&
            iss >> height_host >> delimiter && delimiter == ',' &&
            iss >> depth_host) {
                
        } else {
            std::cerr << "Parsing error for dimensions" << std::endl;
            
        }
    } else {
        std::cerr << "Parsing error for file" << std::endl;

    }
    
    std::string filename = extractFilename(file_path);
    size_t num_Elements = static_cast<int>(width_host) * height_host * depth_host;
    size_t global_x = width_host, global_y = height_host, global_z = depth_host;



    size_t padded_x = global_x;
    size_t padded_y = global_y;
    size_t padded_z = global_z;

    
    size_t data_size = static_cast<int>(padded_x) * padded_y * padded_z;
    

    
    double* decompressed_host_data = nullptr;
    double *input_data = nullptr;
    double *decp_data = nullptr;
    double* input_data_host = new double[data_size];
    double* decp_data_host = new double[data_size];
    
    getdata<double>(file_path, input_data_host, num_Elements);   
    

    bound = bound * (maxValue - minValue);
    std::string decpfilename = "datasets/decp_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".bin";
    std::string cpfilename;
    if(compressor_id == "sz3") cpfilename = "datasets/compressed_"+filename+"_"+std::to_string(bound)+".sz";
    else if(compressor_id == "zfp") cpfilename = cpfilename = "datasets/compressed_"+filename+"_"+std::to_string(bound)+".zfp";
    else cpfilename = "datasets/compressed_"+filename+"_"+std::to_string(bound)+".raw";

    int result = 0;
    std::string command;
    
    double range  = bound;
    cpfilename = "datasets/compressed_"+filename+"_"+std::to_string(bound)+".sz";
    
         
    std::ostringstream oss1;
    oss1 << std::scientific << std::setprecision(17) << bound;
    
    if(compressor_id == "sz3") compressLocalData<double>(file_path, cpfilename, filename, input_data_host, decp_data_host, width_host, height_host, depth_host, bound, decpfilename, 0, 1);
    else if(compressor_id=="zfp"){
        if(depth_host == 1) zfp_compress_decompress_2d(input_data_host, decp_data_host,
                                width_host, height_host, bound);
        else zfp_compress_decompress_3d(input_data_host, decp_data_host,
                                width_host, height_host, depth_host, bound);
    }

    

    cudaError_t err = cudaMalloc(&or_types, 2*data_size * sizeof(int));
    if (err != cudaSuccess) {
        printf("Rank %d: cudaMalloc failed Malloc for or_types: %s\n", 0, cudaGetErrorString(err));
        fflush(stdout);

    }

    
    double *device_input, *device_decp, *device_decp_copy;
    size_t local_bytes = data_size * sizeof(double);
    err = cudaMalloc((void**)&device_input, local_bytes);
    if (err != cudaSuccess) {
        printf("Rank %d: cudaMalloc failed Malloc for device_input: %s\n", 0, cudaGetErrorString(err));
        fflush(stdout);
       
    }

    cudaMalloc((void**)&device_decp, local_bytes);
    
    
    err = cudaMemcpy(device_input, input_data_host, local_bytes, cudaMemcpyHostToDevice);

    err = cudaMemcpy(device_decp, decp_data_host, local_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Rank %d: cudaMalloc failed Malloc for device_decp: %s\n", 0, cudaGetErrorString(err));
        fflush(stdout);

    }
    
    dim3 blockSize(256);
    dim3 gridSize((data_size + blockSize.x - 1) / blockSize.x);

    unsigned int h_count = 0;
    std::vector<double> edits_time = {};
    
    
    cudaDeviceSynchronize();
    int initialValue = 0;
    unsigned int host_count_f_min = 1, host_count_f_max = 1, host_count_f_dir = 1;
    err = cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(unsigned int));
    err = cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(unsigned int));
    err = cudaMemcpyToSymbol(count_f_dir, &initialValue, sizeof(unsigned int));
    iscriticle<double><<<gridSize, blockSize>>>(device_input, device_input, 
        padded_x, padded_y, padded_z, data_size, or_types, 1);
    err = cudaDeviceSynchronize();
    
    iscriticle<double><<<gridSize, blockSize>>>(device_decp, device_decp, 
        padded_x, padded_y, padded_z, data_size, or_types, 0);
    err = cudaDeviceSynchronize();
    
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_count_f_dir, count_f_dir, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);

    int ite = 0;
    while((host_count_f_max!=0 || host_count_f_min !=0 || host_count_f_dir != 0) && ite < 20 && bound >= 1e-14){
        err = cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(unsigned int));
        err = cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(unsigned int));
        err = cudaMemcpyToSymbol(count_f_dir, &initialValue, sizeof(unsigned int));
        bound /= 2;
        if(compressor_id == "sz3") compressLocalData<double>(file_path, cpfilename, filename, input_data_host, decp_data_host, width_host, height_host, depth_host, bound, decpfilename, 0, 1);
        else if(compressor_id=="zfp"){
            if(depth_host == 1) zfp_compress_decompress_2d(input_data_host, decp_data_host,
                                    width_host, height_host, bound);
            else zfp_compress_decompress_3d(input_data_host, decp_data_host,
                                    width_host, height_host, depth_host, bound);
            
        }
        err = cudaMemcpy(device_decp, decp_data_host, local_bytes, cudaMemcpyHostToDevice);
        iscriticle<double><<<gridSize, blockSize>>>(device_decp, device_decp, 
            padded_x, padded_y, padded_z, data_size, or_types, 0);
        err = cudaDeviceSynchronize();

        cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_count_f_dir, count_f_dir, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
    }
    size_t raw_size = width_host * height_host * depth_host * sizeof(double);
    if(host_count_f_max == 0 && host_count_f_min == 0 && host_count_f_dir == 0){
        std::cout<< "bound founded at: " << bound << " CR is: " << (double) raw_size / (double) cmpSize << std::endl;
    }
    else{
        std::cout<<"bound not founded: " << bound << ", set to lossless" <<std::endl;
    }

    std::ofstream outFilep("../stat_result/loss_cr_"+compressor_id+".txt", std::ios::app);
    if (!outFilep) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return 1; // 返回错误码
    }

    outFilep << "Filename: "<< filename <<std::endl;
    if(host_count_f_max == 0 && host_count_f_min == 0 && host_count_f_dir == 0) {
        outFilep << "abs bound founded: "<< bound <<std::endl;
        outFilep << "rel bound founded: "<< bound / (maxValue - minValue) <<std::endl;
        outFilep << "CR: "<< (double) raw_size / (double) cmpSize <<std::endl;
    }
    else outFilep << "bound not founded: "<< bound <<std::endl;
    outFilep << "\n"<< std::endl;

    return 0;
}


