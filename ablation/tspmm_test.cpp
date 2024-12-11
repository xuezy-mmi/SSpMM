#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include <float.h>
#include <random>
#include <assert.h>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
// #include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "./include/bm_test_utils.h"
#include "./include/tspmm.cuh"

using namespace std;
#define repeat 100
#define warmup 10
template <typename InType, typename OutType, typename DTypeVec, typename ITypeVec>
void BmFN(std::string benchmark, int N, int vec_length, int kernel, bool func, bool record, std::string arch){
    // Open the benchmark file
    std::ifstream infile(benchmark, std::ifstream::in);
    std::string line;

    // get the Size of the benchmark
    std::getline(infile, line, ',');
    const int m_vec = std::stoi(line);/////////M
    const int M = m_vec * vec_length;
    std::getline(infile, line, ',');
    const int K = std::stoi(line);////////////K
    std::getline(infile, line, '\n');
    /////////////////////////////////////////////////////////////
    int nonzeros_vec = std::stoi(line);//////A_nnz
    const int nnz = nonzeros_vec;
    /////////////////////////////////////////////////////////////
    // const int nnz = std::stoi(line);//////A_nnz
    // Create the A column indices

    std::default_random_engine generator;

    // SpMM
    // if (sparse == 1){
    {
        int *row_offsets = new int[m_vec + 1];
        for (int i = 0; i < m_vec + 1; i ++){
            if (i == m_vec) std::getline(infile, line, '\n');
            else std::getline(infile, line, ' ');
            row_offsets[i] = std::stoi(line);
        }
        nonzeros_vec = row_offsets[m_vec];// 
        const int nonzeros = nonzeros_vec * vec_length;
        int *col_indices = new int[nonzeros_vec];
        for (int i = 0; i < nonzeros_vec; i ++){
            std::getline(infile, line, ' ');
            col_indices[i] = std::stoi(line);
        }
        // int vec_num = (vec_length > 8) ? vec_length / 8 : 1;
        // Initialize the input operands
        InType *values = new InType[nonzeros];
        InType *rhs_matrix = new InType[K * N];
        MakeDenseMatrix<InType>(K, N, rhs_matrix, generator); 
        MakeDenseMatrix<InType>(1, nonzeros, values, generator);//

        float *output_value_host = new float[M * N];


        if (func){// verify the result
            // Initialize the output matrix with 0
            for (int i=0; i < M * N; i++){
                output_value_host[i] = 0.0f;
            }

            // traverse all the vector rows
            for (int i=0; i < m_vec; i++){
                // traverse all the nonzero columns in this row
                for (int j=row_offsets[i]; j < row_offsets[i+1]; j++){//K
                    int col_idx = col_indices[j];
                    // traverse all the elements in the vector
                    for (int v=0; v < vec_length; v++){
                        int row_idx = i * vec_length + v;
                        for (int l=0; l < N; l++){
                            output_value_host[row_idx * N + l] += (float)values[j * vec_length + v] * (float)rhs_matrix[col_idx * N + l];
                        }
                    }
                }
            }
        }// end if func

        int *row_indices = new int[m_vec];
        IdentityRowSwizzle(m_vec, row_indices);

        // Device
        int *d_row_offsets, *d_col_indices, *d_row_indices;
        InType * d_value, *d_rhs_matrix;
        OutType * d_output_value;
        checkCuda(cudaMalloc(&d_row_offsets, (m_vec + 1) * sizeof(int)));
        checkCuda(cudaMalloc(&d_col_indices, nonzeros_vec * sizeof(int)));
        checkCuda(cudaMalloc(&d_row_indices, m_vec * sizeof(int)));

        checkCuda(cudaMalloc(&d_value, nonzeros * sizeof(InType)));
        checkCuda(cudaMalloc(&d_rhs_matrix, (K * N) * sizeof(InType)));
        checkCuda(cudaMalloc(&d_output_value, (M * N) * sizeof(OutType)));

        checkCuda(cudaMemcpy(d_row_offsets, row_offsets, (m_vec + 1) * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_col_indices, col_indices, nonzeros_vec * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_row_indices, row_indices, m_vec * sizeof(int), cudaMemcpyHostToDevice));

        checkCuda(cudaMemcpy(d_value, values, nonzeros * sizeof(InType), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix, K * N * sizeof(InType), cudaMemcpyHostToDevice));
        // cudaProfilerStart();
        if (kernel == 2){
            printf("Using WMMA-SpMM, V = %d\n", vec_length);
            float total_msec = 0;
            double Gperf = 0;
            for(int i = 0; i < repeat+warmup; i++){
                float msec = 0;
                cudaEvent_t start;
                cudaEvent_t end;
                checkCuda(cudaEventCreate(&start));
                checkCuda(cudaEventCreate(&end));
                checkCuda(cudaEventRecord(start));
                
                spmm::WMMA_SpMM(m_vec, vec_length, N, K, d_row_offsets, d_col_indices, d_value, d_rhs_matrix, d_output_value);

                checkCuda(cudaEventRecord(end));
                checkCuda(cudaEventSynchronize(end));
                checkCuda(cudaEventElapsedTime(&msec, start, end));
                if(i >= warmup) {total_msec = total_msec + msec;}
            }

            Gperf = ((double)nnz) * N * 2 / 1000/ 1000 / total_msec * repeat;//nonzeros
            double sp = 100.0 - ((double)nnz * 100 / M / K);
            printf("M = %d, K = %d, N = %d, NNZ = %d, Sparsity = %lf%\n", M, K, N, nnz, sp);
            printf("\033[33mWMMA on TC Avager Performance = \033[35m%lf\033[0m \033[33mGFlops\033[0m\n", Gperf);
            if(record){
                std::ofstream outFile;
            //-------------edit here to change the output file-----------------------------
                std::string output_dir = "./data/";
                std::string line;

                if(vec_length == 8){
                    line = "_WMMASpMM_v8.csv";
                }
                
                else if(vec_length == 16){
                    line = "_WMMASpMM_v16.csv";
                }

                output_dir = output_dir + arch + line;

                outFile.open(output_dir, std::ios::app);
            //-----------------------------------------------------------------------------
                outFile << nnz << ',' << N << ',' << Gperf << std::endl;
                outFile.close();
            }
        }
        else{
            printf("Unsupported Kernel \n");
        }

        if (func){
            OutType *output_value_cuda = new OutType[M * N];
            checkCuda(cudaMemcpy(output_value_cuda, d_output_value, M * N * sizeof(OutType), cudaMemcpyDeviceToHost));

            // Verify the result
            int errors = 0;
            float max_error = 0.0f;
            for (int j=0; j < M * N; j++){
                // if (j < 256) printf("item %d, expect %.4f, got %.4f\n", j, (float)output_value_host[j], (float)output_value_cuda[j]);
                if (abs((float)output_value_cuda[j] - (float)output_value_host[j]) > 0.5f){
                    // if (j < 2560) printf("item %d, expect %.4f, got %.4f\n", j, (float)output_value_host[j], (float)output_value_cuda[j]);
                    if(abs((float)output_value_cuda[j] - (float)output_value_host[j]) > max_error){
                        max_error = abs((float)output_value_cuda[j] - (float)output_value_host[j]);
                    }
                    errors ++;
                }
            }
            if (errors > 0) {
                printf( "SPMM does not agree with SEQUENTIAL! %d errors! max error: %f\n",errors, max_error);
            }else {
                printf("Results verified: they agree.\n");
            }
            delete output_value_cuda;
        }

        // Free the memory
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_row_indices);
        cudaFree(d_value);
        cudaFree(d_rhs_matrix);
        cudaFree(d_output_value);

        delete row_offsets;
        delete col_indices;
        delete row_indices;
        delete values;
        delete rhs_matrix;
        delete output_value_host;
    }
}

void usage(void){
    printf("Input Help!\n");
    printf("Input Format is  ./tspmm_test [paths.txt] [dimN] [vec_length] [kernel] [sort] [verify] [mixed] [reocrd]\n");
    printf("dimN: N in M-K-N\n");
    printf("veclength: {1, 2, 4, 8, 16}\n");
    printf("kernel: {2 WMMA SpMM}\n");
    printf("sort: {0 unsorted by row_swizzle, 1 sorted by row_swizzle}\n");
    printf("func: {0 not verify result with cpu, 1: verify result}\n");
    printf("mixed: {0:fp16*fp16=fp32, 1:fp16*fp16=fp16}\n");

    printf("\n");
    printf("\n");
    printf("\n");
    exit(1);
}
int main(int argc, char **argv){

    if(argc != 11 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0){
        usage();
    }
    // // Run the benchmark
    // else{
    std::string paths(argv[1]);
    const int dimN = std::atoi(argv[2]);//B_cols_num
    const int vec_length = std::atoi(argv[3]);//8 16
    const int kernel = std::atoi(argv[4]);//2:wmma spmm
    const int sorted = std::atoi(argv[5]);//0:unsorted 1:sorted
    const int func = std::atoi(argv[6]);//0:not verify  1:verify
    const int mixed = std::atoi(argv[7]);//0:16-16-16  1:16-16-32
    const int record = std::atoi(argv[8]);//
    const int reorder = std::atoi(argv[9]);//
    std::string arch(argv[10]);
    {
        // Open the benchmark file
        std::ifstream infile(paths, std::ifstream::in);
        std::string line;

        std::string benchmark;
        if(vec_length == 8){
            benchmark = "../benchmark/dlmc-v8/";
        }
        else if(vec_length == 16){
            benchmark = "../benchmark/dlmc-v16/";
        }
        else{
            benchmark = "../benchmark/dlmc/";
        }

        int file_count = 0;
        while (std::getline(infile, line)){
            std::string benchmark1;
            benchmark1 = benchmark + line;
            //printf("path: %s\n", line);
            std::cout << "NO." << file_count << "  matrix: " << line << std::endl;
            if (mixed) BmFN<half, half, float, int>(benchmark1, dimN, vec_length, kernel, func, record, arch);
            else BmFN<half, float, float, int>(benchmark1, dimN, vec_length, kernel, func, record, arch);
            file_count++;
        }
        infile.close();
        printf("matrix num:%d\n", file_count);
    }
    return 0;
}
