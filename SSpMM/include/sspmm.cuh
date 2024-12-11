#include "cuda_fp16.h"
#ifndef SSPMM_H
#define SSPMM_H

namespace spmm{
// original version
cudaError_t SpMM(int m_vec, int vec_length, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix) ;

cudaError_t SpMM(int m_vec, int vec_length, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix) ;

cudaError_t SpMM(int m_vec, int vec_length, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const float* __restrict__ values,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix) ;

cudaError_t SSpMM(int m_vec, int vec_length, int N, int K, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix) ;

cudaError_t SSpMM(int m_vec, int vec_length, int N, int K, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix) ;

cudaError_t SSpMM(int m_vec, int vec_length, int N, int K, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const float* __restrict__ values,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix) ;
    
} // namespace spmm

#endif