#include "cuda_fp16.h"
#ifndef TSPMM_H
#define TSPMM_H

namespace spmm{
// original version
cudaError_t WMMA_SpMM(int m_vec, int vec_length, int N, int K, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix) ;
cudaError_t WMMA_SpMM(int m_vec, int vec_length, int N, int K, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix) ;
cudaError_t WMMA_SpMM(int m_vec, int vec_length, int N, int K, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const float* __restrict__ values,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix) ;
} // namespace spmm

#endif