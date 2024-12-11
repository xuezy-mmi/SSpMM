#include "cuda_fp16.h"
#ifndef WMMA_SPMM_H
#define WMMA_SPMM_H

namespace spmm{

cudaError_t wmmaSpmm_8b(int m_vec, int vec_length, int n, int k,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix);

cudaError_t wmmaSpmm_16b8b(int m_vec, int vec_length, int n, int k,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix);

cudaError_t wmmaSpmm_16b(int m_vec, int vec_length, int n, int k,
    const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix);

} // namespace spmm

#endif
