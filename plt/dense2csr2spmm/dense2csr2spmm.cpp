#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <string>
#include <random>
using namespace std;

typedef struct
{
    int M;
    int K;
    int NNZ;
    int * rowPtr;
    int * colInd;
    float * values;
} csr_format;

template <typename ValueType>
void MakeDenseMatrix(int rows, int columns, ValueType *matrix,
                     std::default_random_engine generator, float sparsity)
{
    std::uniform_real_distribution<float> distribution(0, 10);
    int new_sparsity = int(10*sparsity);
    for (int64_t i = 0; i < static_cast<int64_t>(rows) * columns; ++i){
        float temp = distribution(generator);
		if(temp < new_sparsity){
			matrix[i] = ValueType(0);
		}
		else{
			matrix[i] = ValueType(temp);
		}
        
    }
}

csr_format dense2csr(int M, int K, float sparsity){
	csr_format csr;
	int NNZ = 0;
	std::default_random_engine generator;
	int * rowPtr = new int[M + 1];
	rowPtr[0] = 0;
    float *lhs_matrix = new float[M * K];
    // float *rhs_matrix = new float[K * N];
	int id;

	// init sparse matrix
    // MakeDenseMatrix<float>(K, N, rhs_matrix, generator, sparsity); 
    MakeDenseMatrix<float>(M, K, lhs_matrix, generator, sparsity);
	for(int i = 0; i < M; i++){
		for(int j = 0; j < K; j++){
			id = i*K+j;
			printf("%f\t", lhs_matrix[id]);
			if(lhs_matrix[id] > 0){
				NNZ++;
			}			
		}
		printf("\n");
	}
	int * colInd = new int[NNZ];
	float * values = new float[NNZ];
	int _id = 0;
	int this_row_nnz = 0;
	for(int i = 0; i < M; i++){
		// this_row_nnz = 0;
		for(int j = 0; j < K; j++){
			id = i*K+j;
			if(lhs_matrix[id] > 0){
				this_row_nnz++;
				colInd[_id] = j;
				values[_id] = lhs_matrix[id];
				_id++;
			}
		}
		rowPtr[i+1] = this_row_nnz;
	}
	printf("number of non-zero: %d\n", NNZ);

	csr.M = M;
	csr.K = K;
	csr.NNZ = NNZ;
	csr.rowPtr = rowPtr;
	csr.colInd = colInd;
	csr.values = values;

	printf("\ntransfer dense matrix data to csr format\n");
	printf("rowPtr:\n");
	for(int i = 0; i < csr.M+1; i++){
		printf("%d ", csr.rowPtr[i]);
	}
	printf("\ncolInd:\n");
	for(int i = 0; i < csr.NNZ; i++){
		printf("%d ", csr.colInd[i]);
	}
	printf("\nsparse value:\n");
	for(int i = 0; i < csr.NNZ; i++){
		printf("%f ", csr.values[i]);
	}
	return csr;
}
float * gen_dense_matrix(float * matrix, int K, int N){
	std::default_random_engine generator;
	MakeDenseMatrix<float>(K, N, matrix, generator, 0);
	printf("\n\ndense rhs_matrix values:\n");
	for(int i = 0; i <K; i++){
		for (int j = 0; j < N; j++){
			printf("%f\t", matrix[i * N + j]);
		}
		printf("\n");
	}
	return matrix;
}
void csr2spmm(int M, int K, int N, csr_format lhs_matrix, float * rhs_matrix, float * output){
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			output[i*N+j] = 0.0f;
			for(int k = lhs_matrix.rowPtr[i]; k < lhs_matrix.rowPtr[i+1]; k++){
				int col_index = lhs_matrix.colInd[k];
				output[i*N+j] += lhs_matrix.values[k] * rhs_matrix[col_index*N+j];
			}
		}
	}
	// return output;
}
void usage(){
    
	printf("Input Format is: ./dense2csr2spmm [int M] [int K] [int N] [float Sparsity]\n");
	printf("Sparsity only include 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\n");
    exit(1);
}

int main(int argc, char **argv){
    if(argc != 5 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0){
        usage();
    }
    const int M = std::atoi(argv[1]);
	const int K = std::atoi(argv[2]);
	const int N = std::atoi(argv[3]);
	const float sparsity = std::stof(argv[4]);
	printf("M:%d K:%d N:%d Sparsity:%f\n\n", M, K, N, sparsity);

    csr_format lhs_matrix = dense2csr(M,K,sparsity);
	float * rhs_matrix = new float[K*N];
	gen_dense_matrix(rhs_matrix, K, N);
    float * output = new float[M * N];
	csr2spmm(M, K, N, lhs_matrix, rhs_matrix, output);

	printf("\n\noutput values:\n");
	for(int i = 0; i <M; i++){
		for (int j = 0; j < N; j++){
			printf("%f\t", output[i * N + j]);
		}
		printf("\n");
	}

    return 0;
}