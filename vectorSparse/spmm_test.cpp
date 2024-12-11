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
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "./include/bm_test_utils.h"
#include "./include/cuda_spmm.cuh"
#include "./include/wmma_spmm.cuh"
#include "./include/cublas_gemm.cuh"
#include "sputnik/sputnik.h"

// #include "absl/random/random.h"
// #include "benchmark/benchmark.h"

// void ReportThroughput(benchmark::State& state) {
//   state.SetBytesProcessed(
//       static_cast<int64_t>(state.iterations()) *
//       state.range(3) * state.range(2) * 2);
// }

using namespace std;
#define repeat 200
#define warmup 10
template <typename InType, typename OutType, typename IndexType, typename DTypeVec, typename ITypeVec, cudaDataType_t DCuSPARSE>
void BmFN(std::string benchmark, int dimK, int vec_length, int kernel, bool sorted, bool func, int sparse, int record, int SVC, std::string arch){

    // Open the benchmark file
    std::ifstream infile(benchmark, std::ifstream::in);
    std::string line;

    // get the Size of the benchmark
    std::getline(infile, line, ',');
    const int m_vec = std::stoi(line);/////////M
    const int m = m_vec * vec_length;
    std::getline(infile, line, ',');
    const int n = std::stoi(line);////////////K
    std::getline(infile, line, '\n');
    /////////////////////////////////////////////////////////////
    int nonzeros_vec = std::stoi(line);//////A_nnz
    const int nnz = nonzeros_vec;
    const int nonzeros = nonzeros_vec * vec_length;
    /////////////////////////////////////////////////////////////
    const int k = dimK;

    // Create the A column indices

    std::default_random_engine generator;

    // SpMM
    if (sparse == 1){
        int *row_offsets = new int[m_vec + 1];

        for (int i = 0; i < m_vec + 1; i ++){
            if (i == m_vec) std::getline(infile, line, '\n');
            else std::getline(infile, line, ' ');
            row_offsets[i] = std::stoi(line);
        }
        nonzeros_vec = row_offsets[m_vec];// 
        int *col_indices = new int[nonzeros_vec];
        IndexType *col_indices_sputnik = new IndexType[nonzeros_vec];
        for (int i = 0; i < nonzeros_vec; i ++){
            std::getline(infile, line, ' ');
            col_indices[i] = std::stoi(line);
            col_indices_sputnik[i] = (IndexType)std::stoi(line);
            // printf("%d\n",i);
        }

        // Initialize the input operands
        InType *values = new InType[nonzeros];
        InType *rhs_matrix = new InType[n * k];
        MakeDenseMatrix<InType>(n, k, rhs_matrix, generator);
        MakeDenseMatrix<InType>(1, nonzeros, values, generator);


        // Allocate the host output
        float *output_value_host = new float[m * k];

        if (func){// verify the result
            // Initialize the output matrix with 0
            for (int i=0; i < m * k; i++){
                output_value_host[i] = 0.0f;
            }
            
            // traverse all the vector rows
            for (int i=0; i < m_vec; i++){
                // traverse all the nonzero columns in this row
                for (int j=row_offsets[i]; j < row_offsets[i+1]; j++){
                    int col_idx = col_indices[j];
                    // traverse all the elements in the vector
                    for (int v=0; v < vec_length; v++){
                        int row_idx = i * vec_length + v;
                        for (int l=0; l < k; l++){
                            output_value_host[row_idx * k + l] += (float)values[j * vec_length + v] * (float)rhs_matrix[col_idx * k + l];
                        }
                    }
                }
            }
        }// end if func


        int *row_indices = new int[m_vec];
        if (sorted) {
            //printf("Sort CSR based on row length\n");
            SortedRowSwizzle(m_vec, row_offsets, row_indices);
        }
        else{
            //printf("Process the rows in order\n");
            IdentityRowSwizzle(m_vec, row_indices);
        }



        // Device
        int *d_row_offsets, *d_col_indices, *d_row_indices;
        IndexType *d_col_indices_sputnik;
        InType *d_value, *d_rhs_matrix;
        OutType *d_output_value;
        checkCuda(cudaMalloc(&d_row_offsets, (m_vec + 1) * sizeof(int)));
        checkCuda(cudaMalloc(&d_col_indices, nonzeros_vec * sizeof(int)));
        checkCuda(cudaMalloc(&d_col_indices_sputnik, nonzeros_vec * sizeof(IndexType)));
        checkCuda(cudaMalloc(&d_row_indices, m_vec * sizeof(int)));

        checkCuda(cudaMalloc(&d_value, nonzeros * sizeof(InType)));
        checkCuda(cudaMalloc(&d_rhs_matrix, (n * k) * sizeof(InType)));
        checkCuda(cudaMalloc(&d_output_value, (m * k) * sizeof(OutType)));

        checkCuda(cudaMemcpy(d_row_offsets, row_offsets, (m_vec + 1) * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_col_indices, col_indices, nonzeros_vec * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_col_indices_sputnik, col_indices_sputnik, nonzeros_vec * sizeof(IndexType), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_row_indices, row_indices, m_vec * sizeof(int), cudaMemcpyHostToDevice));

        checkCuda(cudaMemcpy(d_value, values, nonzeros * sizeof(InType), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix, n * k * sizeof(InType), cudaMemcpyHostToDevice));
        // cudaProfilerStart();
        if (kernel == 0){
            printf("Using vectorsparse \n");

            float total_msec = 0;
            double Gperf = 0;
            for(int i = 0; i < repeat+warmup; i++){
                float msec = 0;
                cudaEvent_t start;
                cudaEvent_t end;
                checkCuda(cudaEventCreate(&start));
                checkCuda(cudaEventCreate(&end));
                checkCuda(cudaEventRecord(start));

                spmm::wmmaSpmm(m_vec, vec_length, k, n, d_row_indices, d_row_offsets, d_col_indices, d_value, d_rhs_matrix, d_output_value);

                checkCuda(cudaEventRecord(end));
                checkCuda(cudaEventSynchronize(end));
                checkCuda(cudaEventElapsedTime(&msec, start, end));
                if(i >= warmup) total_msec = total_msec + msec;
            }
            if(SVC == 1){
                Gperf = ((double)nnz) * k * 2 / 1000/ 1000 / total_msec * repeat;
            }
            else{
                Gperf = ((double)nonzeros) * k * 2 / 1000/ 1000 / total_msec * repeat;
            }
            if(record){
                std::ofstream outFile;
                std::string output_dir = "./data/";
                std::string line;
                if(SVC == 0){
                    line = "_VS_structured.csv";
                }
                else if(SVC == 1){
                    line = "_VS_v1.csv";
                }
                else if(SVC == 8){
                    line = "_VS_v8.csv";
                }
                else if(SVC == 16){
                    line = "_VS_v16.csv";
                }
                output_dir = output_dir + arch + line;

                outFile.open(output_dir, std::ios::app);
            //-----------------------------------------------------------------------------
                outFile << nnz << ',' << k << ',' << Gperf << std::endl;
                outFile.close();

            }
            printf("\033[33mWMMA on TC Avager Performance = \033[35m%lf\033[0m \033[33mGFlops\033[0m\n", Gperf);
        }
        else if (kernel == 1){
            printf("Using CUDA \n");

            float total_msec = 0;
            double Gperf = 0;
            for(int i = 0; i < repeat+warmup; i++){
                float msec = 0;
                cudaEvent_t start;
                cudaEvent_t end;
                checkCuda(cudaEventCreate(&start));
                checkCuda(cudaEventCreate(&end));
                checkCuda(cudaEventRecord(start));

                spmm::cudaSpmm(m_vec, vec_length, k, n, d_row_indices, d_row_offsets, d_col_indices, d_value, d_rhs_matrix, d_output_value);

                checkCuda(cudaEventRecord(end));
                checkCuda(cudaEventSynchronize(end));
                checkCuda(cudaEventElapsedTime(&msec, start, end));
                if(i >= warmup) total_msec = total_msec + msec;
            }

            Gperf = ((double)nnz) * k * 2 / 1000/ 1000 / total_msec * repeat;
            
            if(record){
                std::ofstream outFile;
                std::string output_dir = "./data/";
                std::string line;
                line = "_cudaspmm.csv";
                output_dir = output_dir + arch + line;

                outFile.open(output_dir, std::ios::app);
            //-----------------------------------------------------------------------------
                outFile << nnz << ',' << k << ',' << Gperf << std::endl;
                outFile.close();
            }
            printf("\033[33mMMA_SpMM on CUDA Core Avager Performance = \033[35m%lf\033[0m \033[33mGFlops\033[0m\n", Gperf);
        }
        else if (kernel == 2){
            printf("Using Sputnik\n");

            float total_msec = 0;
            double Gperf = 0;
            for(int i = 0; i < repeat+warmup; i++){

                DTypeVec* d_value_vec = reinterpret_cast<DTypeVec *>(d_value);
                DTypeVec* d_rhs_matrix_vec = reinterpret_cast<DTypeVec *>(d_rhs_matrix);
                DTypeVec* d_output_value_vec = reinterpret_cast<DTypeVec *>(d_output_value);
                ITypeVec* d_col_indices_sputnik_vec = reinterpret_cast<ITypeVec *>(d_col_indices_sputnik);

                float msec = 0;
                cudaEvent_t start;
                cudaEvent_t end;
                checkCuda(cudaEventCreate(&start));
                checkCuda(cudaEventCreate(&end));
                checkCuda(cudaEventRecord(start));

                sputnik::CudaSpmm(m, n, k, nonzeros, d_row_indices, d_value_vec, d_row_offsets, d_col_indices_sputnik_vec, d_rhs_matrix_vec, d_output_value_vec, 0);

                checkCuda(cudaEventRecord(end));
                checkCuda(cudaEventSynchronize(end));
                checkCuda(cudaEventElapsedTime(&msec, start, end));
                // total_msec = total_msec + msec;
                if(i >= warmup) {
                    total_msec = total_msec + msec;
                }
            }

            Gperf = ((double)nnz) * k * 2 / 1000/ 1000 / total_msec * repeat;
            if(record){
                std::ofstream outFile;
                std::string output_dir = "./data/";
                std::string line;
                line = "_sputnik.csv";
                output_dir = output_dir + arch + line;

                outFile.open(output_dir, std::ios::app);
            //-----------------------------------------------------------------------------
                outFile << nnz << ',' << k << ',' << Gperf << std::endl;
                outFile.close();
            }
            printf("\033[33mSputnik Avager Performance = \033[35m%lf\033[0m \033[33mGFlops\033[0m\n", Gperf);
        }
        else if (kernel == 3){
            printf("Using CuSPARSE \n");

            float total_msec = 0;
            double Gperf = 0;
            

                cusparseHandle_t handle = NULL;
                cusparseDnMatDescr_t rhs_dense, output_dense;
                cusparseSpMatDescr_t lhs_sparse;

                cusparseCreate(&handle);

                // // create lhs sparse matrix
                // cusparseCreateCsr(
                //     &lhs_sparse, m, n, nonzeros_vec, d_row_offsets, d_col_indices, d_value,
                //     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, DCuSPARSE
                // );
                // // create rhs dense matrix
                // cusparseCreateDnMat(
                //     &rhs_dense, n, k, n, d_rhs_matrix, DCuSPARSE, CUSPARSE_ORDER_COL
                // );
                // // create output dense matrix
                // cusparseCreateDnMat(
                //     &output_dense, m, k, m, d_output_value, DCuSPARSE, CUSPARSE_ORDER_COL
                // );
                // create lhs sparse matrix
                cusparseCreateCsr(
                    &lhs_sparse, m, n, nonzeros_vec, d_row_offsets, d_col_indices, d_value,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, DCuSPARSE
                );
                // create rhs dense matrix
                cusparseCreateDnMat(
                    &rhs_dense, n, k, k, d_rhs_matrix, DCuSPARSE, CUSPARSE_ORDER_ROW
                );
                // create output dense matrix
                cusparseCreateDnMat(
                    &output_dense, m, k, k, d_output_value, DCuSPARSE, CUSPARSE_ORDER_ROW
                );


                InType alpha = 1.0;
                InType beta  = 0.0;
                size_t buffer_size;
                void* dBuffer = NULL;

                // get buffer
                cusparseSpMM_bufferSize(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, lhs_sparse, rhs_dense, &beta, output_dense, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &buffer_size
                );

                checkCuda(cudaMalloc(&dBuffer, buffer_size));
                
                
                // preprocess to get additional speedup
                cusparseSpMM_preprocess(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, lhs_sparse, rhs_dense, &beta, output_dense, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2,
                    dBuffer
                );
                
            for(int i = 0; i < repeat+warmup; i++){
                float msec = 0;
                cudaEvent_t start;
                cudaEvent_t end;
                checkCuda(cudaEventCreate(&start));
                checkCuda(cudaEventCreate(&end));
                checkCuda(cudaEventRecord(start));
                // execute SpMM
                cusparseSpMM(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, lhs_sparse, rhs_dense, &beta, output_dense, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                    dBuffer
                );
                checkCuda(cudaEventRecord(end));
                checkCuda(cudaEventSynchronize(end));
                checkCuda(cudaEventElapsedTime(&msec, start, end));
                // total_msec = total_msec + msec;
                if(i >= warmup) {
                    total_msec = total_msec + msec;
                }
            }

            checkCuda(cudaFree(dBuffer));
            cusparseDestroyDnMat(rhs_dense);
            cusparseDestroyDnMat(output_dense);
            cusparseDestroySpMat(lhs_sparse);
            cusparseDestroy(handle);
            
            Gperf = ((double)nnz) * k * 2 / 1000/ 1000 / total_msec * repeat;
            if(record){
                std::ofstream outFile;
                std::string output_dir = "./data/";
                std::string line;
                line = "_cusparse.csv";
                output_dir = output_dir + arch + line;

                outFile.open(output_dir, std::ios::app);
            //-----------------------------------------------------------------------------
                outFile << nnz << ',' << k << ',' << Gperf << std::endl;
                outFile.close();

            }
            printf("\033[33mcuSPARSE Avager Performance = \033[35m%lf\033[0m \033[33mGFlops\033[0m\n", Gperf);

        }
        else{
            printf("Unsupported Kernel \n");
        }
        // cudaProfilerStop();


        if (func){
            OutType *output_value_cuda = new OutType[m * k];
            checkCuda(cudaMemcpy(output_value_cuda, d_output_value, m * k * sizeof(OutType), cudaMemcpyDeviceToHost));

            // Verify the result
            int errors = 0;
            for (int j=0; j < m * k; j++){
                // if (j < 256) printf("item %d, expect %.4f, got %.4f\n", j, (float)output_value_host[j], (float)output_value_cuda[j]);
                if (abs((float)output_value_cuda[j] - (float)output_value_host[j]) > 0.5){
                    // if (j < 2560) printf("item %d, expect %.4f, got %.4f\n", j, (float)output_value_host[j], (float)output_value_cuda[j]);
                    errors ++;
                }
            }
            if (errors > 0) {
                printf( "SPMM does not agree with SEQUENTIAL! %d errors!\n",errors);
            }else {
                printf("Results verified: they agree.\n");
            }
            delete output_value_cuda;
        }


        // Free the memory
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_row_indices);
        cudaFree(d_col_indices_sputnik);
        cudaFree(d_value);
        cudaFree(d_rhs_matrix);
        cudaFree(d_output_value);

        delete row_offsets;
        delete col_indices;
        delete col_indices_sputnik;
        delete row_indices;
        delete values;
        delete rhs_matrix;
        delete output_value_host;
    }
    // CuBLAS Dense GeMM
    else{
        // Create cublas handles
        printf("Dense Baseline\n");

        float total_msec = 0;
        double Gperf = 0;

        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle));
        half alpha = 1.0;
        half beta = 0.0;

        // Initialize the input operands
        InType *lhs_matrix = new InType[m_vec * n];
        InType *rhs_matrix = new InType[n * k];
        MakeDenseMatrix<InType>(m_vec, n, lhs_matrix, generator);
        MakeDenseMatrix<InType>(n, k, rhs_matrix, generator);

        
        // Allocate and initialize device memory
        InType *d_lhs_matrix, *d_rhs_matrix;
        InType *d_output_values;

        checkCuda(cudaMalloc(&d_lhs_matrix, (m_vec * n) * sizeof(InType)));
        checkCuda(cudaMalloc(&d_rhs_matrix, (n * k) * sizeof(InType)));
        checkCuda(cudaMalloc(&d_output_values, (m_vec * k) * sizeof(InType)));
            
        checkCuda(cudaMemcpy(d_lhs_matrix, lhs_matrix, (m_vec * n) * sizeof(InType), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix, (n * k) * sizeof(InType), cudaMemcpyHostToDevice));

        for(int i = 0; i < repeat+warmup; i++){
            // cudaProfilerStart();
        
            float msec = 0;
            cudaEvent_t start;
            cudaEvent_t end;
            checkCuda(cudaEventCreate(&start));
            checkCuda(cudaEventCreate(&end));
            checkCuda(cudaEventRecord(start));
            // cublasGeMM(handle, m_vec, n, k, d_rhs_matrix, d_lhs_matrix, d_output_values);
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m_vec, n,
                        &alpha, d_rhs_matrix, CUDA_R_16F, k, d_lhs_matrix, CUDA_R_16F, n, 
                        &beta, d_output_values, CUDA_R_16F, k,
                        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            // cudaProfilerStop();
            checkCuda(cudaEventRecord(end));
            checkCuda(cudaEventSynchronize(end));
            checkCuda(cudaEventElapsedTime(&msec, start, end));
            if(i >= warmup) total_msec = total_msec + msec;
        }
        Gperf = ((double)nnz) * k * 2 / 1000/ 1000 / total_msec * repeat;
        if(record){
            std::ofstream outFile;
            std::string output_dir = "./data/";
            std::string line;
            line = "_cublas.csv";
            output_dir = output_dir + arch + line;

            outFile.open(output_dir, std::ios::app);
            //-----------------------------------------------------------------------------
            outFile << nnz << ',' << k << ',' << Gperf << std::endl;
            outFile.close();
            
        }
        printf("\033[33mCuBLAS Avager Performance = \033[35m%lf\033[0m \033[33mGFlops\033[0m\n", Gperf);

        // Copy the output back to the host
        InType * output_value_host = new InType[m * k];

        if (func){
            // All the rows in the output matrix
            for (int i=0; i < m; i++){
                // All the columns in the output matrix
                for (int j=0; j < k; j++){
                    // the inner product dimension
                    float out_temp = 0;
                    for (int v=0; v < n; v++){
                        out_temp += (float)lhs_matrix[i * n + v] * (float)rhs_matrix[v * k + j];
                    }
                    output_value_host[i * k + j] = (InType)out_temp;
                }
            }

            InType *output_value_cuda = new InType[m * k];
            checkCuda(cudaMemcpy(output_value_cuda, d_output_values, m * k * sizeof(InType), cudaMemcpyDeviceToHost));

            // Verify the result
            int errors = 0;
            for (int j=0; j < m * k; j++){
                // if (j < 256) printf("item %d, expect %.4f, got %.4f\n", j, (float)output_value_host[j], (float)output_value_cuda[j]);
                if (abs((float)output_value_cuda[j] - (float)output_value_host[j]) > 0.5){
                    // printf("item %d, expect %.4f, got %.4f\n", j, (float)output_value_host[j], (float)output_value_cuda[j]);
                    errors ++;
                }
            }
            if (errors > 0) {
                printf( "CuBLAS does not agree with SEQUENTIAL! %d errors!\n",errors);
            }
            else {
                printf("Results verified: they agree.\n");
            }
            delete output_value_cuda;
            delete output_value_host;
        }

        checkCublas(cublasDestroy(handle));

        cudaFree(d_lhs_matrix);
        cudaFree(d_rhs_matrix);
        cudaFree(d_output_values);

        delete lhs_matrix;
        delete rhs_matrix;

    }
}

void usage(void){
    printf("Input Help!\n");
    printf("Input Format is  ./spmm_test [paths.txt] [dimN] [vec_length] [kernel] [SVC] [func] [sparse] [mixed] [reocrd]\n");
    printf("dimN: N in M-K-N\n");
    printf("veclength: {1, 2, 4, 8, 16}\n");
    printf("kernel: {0 wmmaSpMM, 1 cudaSpMM, 2 Sputnik, 3 cuSPARSE}\n");
    printf("SVC: {0 structured(V=8), 1 V=1, 8 V=8, 16 V=16}\n");
    printf("func: {0 not verify result with cpu, 1: verify result}\n");
    printf("sparse: {0 dense(cuBLAS), 1 SpMM(kernels)}\n");
    printf("mixed: {0:fp32*fp32=fp32, 1:fp16*fp16=fp16, 2:fp16*fp16=fp32}\n");
    printf("wmmaSpMM(0)\t: V=2,4,8,16, mixed=1,2\n");
    printf("cudaSpMM(1)\t: V=1,2,4,8\n");
    printf("Sputnik(2)\t: V=1\n");
    printf("cuSPARSE(3)\t: V=1\n");
    printf("\n");
    printf("\n");
    printf("\n");
    exit(1);
}
int main(int argc, char **argv){

    if(argc != 11 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0){
        usage();
    }
    std::string paths(argv[1]);
    int dimK = std::atoi(argv[2]);
    int vec_length = std::atoi(argv[3]);
    int kernel = std::atoi(argv[4]);
    int SVC = std::atoi(argv[5]);
    int func = std::atoi(argv[6]);
    int sparse = std::atoi(argv[7]);
    int mixed = std::atoi(argv[8]);
    int record = std::atoi(argv[9]);
    std::string arch(argv[10]);
    // int dimK = 512;//B_cols_num
    // int vec_length = 8;//1 2 4 8
    // int kernel = 0;//0:wmma 1:cuda 2:sputnik 3:cusparse
    int sorted = 0;//0:unsorted 1:sorted
    // int func = 0;///0:not verify  1:verify
    // int sparse = 1;///0:cublas 1:spmm
    // int mixed = 1;//0:fp32*fp32=fp32  1:fp16*fp16=fp16 2:fp16*fp16=fp32
    // int record = 0;//0:not record 1:record
    // Open the benchmark file
    std::ifstream infile(paths, std::ifstream::in);
    std::string line;

    std::string benchmark_dir;
    if(SVC == 0){//structured v= 8
        benchmark_dir = "../benchmark/dlmc/";
    }
    else if(SVC == 1){//v=1
        benchmark_dir = "../benchmark/dlmc/";
    }
    else if(SVC == 8){//v=8
        benchmark_dir = "../benchmark/dlmc-v8/";
    }
    else if(SVC == 16){//v=16
        benchmark_dir = "../benchmark/dlmc-v16/";
    }
    else{
        printf("unsupported SVC!\n");
        exit(1);
    }
    int file_count = 0;
    while (std::getline(infile, line)){
        std::string benchmark;
        benchmark = benchmark_dir + line;
        //printf("path: %s\n", line);
        std::cout << benchmark << std::endl;
        if (mixed == 1) BmFN<half, half, short, half2, short2, CUDA_R_16F>(benchmark, dimK, vec_length, kernel, sorted, func, sparse, record, SVC, arch);
        else if (mixed == 2) BmFN<half, float, int, half2, short2, CUDA_R_16F>(benchmark, dimK, vec_length, kernel, sorted, func, sparse, record, SVC, arch);
        else BmFN<float, float, int, float, int, CUDA_R_32F>(benchmark, dimK, vec_length, kernel, sorted, func, sparse, record, SVC, arch); 
        file_count++;
    }
    infile.close();
    printf("matrix num:%d\n", file_count);

    return 0;
}
