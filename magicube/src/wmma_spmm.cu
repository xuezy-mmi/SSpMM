#include "../include/wmma_spmm.cuh"
#include "spmm_utils/dense_tile.h"
#include "spmm_utils/sparse_tile.h"
#include "spmm_utils/compute_utils.h"
#include "spmm_utils/output_tile.h"
#include <stdio.h>
#include <mma.h>

using namespace nvcuda;

namespace spmm{

//8-bit Tile_N = 128 with 4 warps
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_8b(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int dimN_index = blockIdx.y * Tile_N;
    const int lane_id = threadIdx.x;
    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices
    __shared__ int values_tile_array[Tile_K*VecLength/2];
    __shared__ int column_indices_tile_array[Tile_K*2];

    // One int32 has four 8-bit integers
    // Padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/4 + 8*7];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    // One int32 has four 8-bit integers
    wmmaSparseTile_8b<LoadType, VecType, Tile_K * VecLength / 4, Tile_K> sparse_tile_loader(
        row_offset_vec, threadIdx.x % 32, threadIdx.x / 32, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[4] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_8b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/4, dimN_index/4, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    // Tile_N / warps / four threads in x-dim of output matrix
    __align__(16) int output_fragment[Tile_N / Warps / 4] = {};
    wmmaComputeUtils_8b<Tile_K * VecLength / 4> computer(values_tile, dense_tile, output_fragment, lane_id);

    int steps = nonzeros / Tile_K;
    int residue = nonzeros % Tile_K;

    if(steps > 0){
        sparse_tile_loader.Load(0);
        __syncthreads();
        dense_tile_loader.Prefetch(0);

        int i = 1;
        #pragma unroll
        for(; i < steps; i++){
            dense_tile_loader.LoadRowfromRegister(i-1);
            sparse_tile_loader.Load(i);
            __syncthreads();
            dense_tile_loader.Prefetch(i);
            computer.TileMAC(i-1);
            __syncthreads();
        }

        dense_tile_loader.LoadRowfromRegister(i-1);
        __syncthreads();
        computer.TileMAC(i-1);
    }

    if(residue > 0){
        sparse_tile_loader.Residue();
        __syncthreads();
        dense_tile_loader.ResidueLoad(residue);
        __syncthreads();
        computer.TileMACResidue();
    } 

    wmmaOutputTile_8b<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix);
    output_tile_storer.Store();
}

//16-bit 8-bit Tile_N = 128 with 4 warps
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_16b8b(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int dimN_index = blockIdx.y * Tile_N;
    const int lane_id = threadIdx.x;
    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices
    __shared__ int values_tile_array[Tile_K*VecLength];
    __shared__ int column_indices_tile_array[Tile_K*2];

    // One int32 has four 8-bit integers
    // Padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/4 + 8*7];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    // One int32 has two 16-bit integers
    wmmaSparseTile_16b8b<LoadType, VecType, Tile_K * VecLength / 2, Tile_K> sparse_tile_loader(
        row_offset_vec, threadIdx.x % 32, threadIdx.x / 32, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[4] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_8b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/4, dimN_index/4, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    // Tile_N / warps / four threads in x-dim of output matrix
    __align__(16) int output_fragment[Tile_N / Warps / 4] = {};
    wmmaComputeUtils_16b8b<Tile_K * VecLength / 2> computer(values_tile, dense_tile, output_fragment, lane_id);

    int steps = nonzeros / Tile_K;
    int residue = nonzeros % Tile_K;

    if(steps > 0){
        sparse_tile_loader.Load(0);
        __syncthreads();
        dense_tile_loader.Prefetch(0);

        int i = 1;
        #pragma unroll
        for(; i < steps; i++){
            dense_tile_loader.LoadRowfromRegister(i-1);
            sparse_tile_loader.Load(i);
            __syncthreads();
            dense_tile_loader.Prefetch(i);
            computer.TileMAC(i-1);
            __syncthreads();
        }

        dense_tile_loader.LoadRowfromRegister(i-1);
        __syncthreads();
        computer.TileMAC(i-1);
    }
   
    if(residue > 0){
        sparse_tile_loader.Residue();
        __syncthreads();
        dense_tile_loader.ResidueLoad(residue);
        __syncthreads();
        computer.TileMACResidue();
    } 

    wmmaOutputTile_16b8b<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix);
    output_tile_storer.Store();
}

//16-bit 8-bit Tile_N = 128 with 4 warps 8v
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_16b8b8v(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int dimN_index = blockIdx.y * Tile_N;
    const int lane_id = threadIdx.x;
    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices
    __shared__ int values_tile_array[Tile_K*VecLength];
    __shared__ int column_indices_tile_array[Tile_K*2];

    // One int32 has four 8-bit integers
    // Padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/4 + 8*7];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    // One int32 has two 16-bit integers
    wmmaSparseTile_16b8b8v<LoadType, VecType, Tile_K * VecLength / 2, Tile_K> sparse_tile_loader(
        row_offset_vec, lane_id, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[4] = {};
    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile_8b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/4, dimN_index/4, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    // Tile_N / warps / four threads in x-dim of output matrix
    __align__(16) int output_fragment_0[Tile_N / Warps / 4] = {};
    __align__(16) int output_fragment_1[Tile_N / Warps / 4] = {};
    wmmaComputeUtils_16b8b8v<Tile_K * VecLength / 2> computer(values_tile, dense_tile, output_fragment_0, output_fragment_1, lane_id);

    int steps = nonzeros / Tile_K;
    int residue = nonzeros % Tile_K;

    if(steps > 0){
        sparse_tile_loader.Load(0);
        __syncthreads();
        dense_tile_loader.Prefetch(0);

        int i = 1;
        #pragma unroll
        for(; i < steps; i++){
            dense_tile_loader.LoadRowfromRegister(i-1);
            sparse_tile_loader.Load(i);
            __syncthreads();
            dense_tile_loader.Prefetch(i);
            computer.TileMAC(i-1);
            __syncthreads();
        }

        dense_tile_loader.LoadRowfromRegister(i-1);
        __syncthreads();
        computer.TileMAC(i-1);
    }
   
    if(residue > 0){
        sparse_tile_loader.Residue();
        __syncthreads();
        dense_tile_loader.ResidueLoad(residue);
        __syncthreads();
        computer.TileMACResidue();
    } 

    wmmaOutputTile_16b8b8v<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment_0, output_fragment_1, output_matrix);
    output_tile_storer.Store();
}

//16-bit 16-bit Tile_N = 64 with 4 warps
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_16b(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int dimN_index = blockIdx.y * Tile_N;
    const int lane_id = threadIdx.x;
    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices
    __shared__ int values_tile_array[Tile_K*VecLength];
    __shared__ int column_indices_tile_array[Tile_K*2];

    // One int32 has two 16-bit integers
    // Padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/2 + 8*7];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    // One int32 has two 16-bit integers
    wmmaSparseTile_16b8b<LoadType, VecType, Tile_K * VecLength / 2, Tile_K> sparse_tile_loader(
        row_offset_vec, threadIdx.x % 32, threadIdx.x / 32, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[4] = {};
    // Initialize the pointers to the dense rhs matrix
    // One int32 has two 16-bit integers
    wmmaDenseTile_16b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/2, dimN_index/2, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    // Tile_N / warps / four threads in x-dim of output matrix
    // 16-bit decomposes into two 8-bits, x2
    __align__(16) int output_fragment[Tile_N / Warps / 2] = {};
    wmmaComputeUtils_16b<Tile_K * VecLength / 2> computer(values_tile, dense_tile, output_fragment, lane_id);

    int steps = nonzeros / Tile_K;
    int residue = nonzeros % Tile_K;

    if(steps > 0){
        sparse_tile_loader.Load(0);
        __syncthreads();
        dense_tile_loader.Prefetch(0);

        int i = 1;
        #pragma unroll
        for(; i < steps; i++){
            dense_tile_loader.LoadRowfromRegister(i-1);
            sparse_tile_loader.Load(i);
            __syncthreads();
            dense_tile_loader.Prefetch(i);
            computer.TileMAC(i-1);
            __syncthreads();
        }

        dense_tile_loader.LoadRowfromRegister(i-1);
        __syncthreads();
        computer.TileMAC(i-1);
    }
   
    if(residue > 0){
        sparse_tile_loader.Residue();
        __syncthreads();
        dense_tile_loader.ResidueLoad(residue);
        __syncthreads();
        computer.TileMACResidue();
    } 

    wmmaOutputTile_16b<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment, output_matrix);
    output_tile_storer.Store();
}

//16-bit 16-bit Tile_N = 64 with 4 warps
template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_K, 
          int Tile_N, int Warps, int VecLength>
__global__ void wmmaSpmm_kernel_16b8v(
    int m_vec, int dimN, int dimK, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int dimN_index = blockIdx.y * Tile_N;
    const int lane_id = threadIdx.x;
    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m_vec) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec*2);
    int nonzeros = __ldg(row_offsets + m_index_vec*2 + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices
    __shared__ int values_tile_array[Tile_K*VecLength];
    __shared__ int column_indices_tile_array[Tile_K*2];

    // One int32 has two 16-bit integers
    // Padding to avoid bank conflict 
    __shared__ int dense_tile_array[Tile_N*Tile_K/2 + 8*7];

    // Pointers to the shared memory tiles
    int* values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;
    int* dense_tile = dense_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    // One int32 has two 16-bit integers
    wmmaSparseTile_16b8b8v<LoadType, VecType, Tile_K * VecLength / 2, Tile_K> sparse_tile_loader(
        row_offset_vec, lane_id, values, column_indices,
        values_tile, column_indices_tile
    );

    __align__(16) int rhs_prefetch[4] = {};
    // Initialize the pointers to the dense rhs matrix
    // One int32 has two 16-bit integers
    wmmaDenseTile_16b<LoadType, Tile_K, Tile_N> dense_tile_loader(
        dimN/2, dimN_index/2, lane_id, rhs_matrix, column_indices_tile, dense_tile, rhs_prefetch 
    );

    // Accumulator registers for the output values.
    // Tile_N / warps / four threads in x-dim of output matrix
    // 16-bit decomposes into two 8-bits, x2
    __align__(16) int output_fragment_0[Tile_N / Warps / 2] = {};
    __align__(16) int output_fragment_1[Tile_N / Warps / 2] = {};
    wmmaComputeUtils_16b8v<Tile_K * VecLength / 2> computer(values_tile, dense_tile, output_fragment_0, output_fragment_1, lane_id);

    int steps = nonzeros / Tile_K;
    int residue = nonzeros % Tile_K;

    if(steps > 0){
        sparse_tile_loader.Load(0);
        __syncthreads();
        dense_tile_loader.Prefetch(0);

        int i = 1;
        #pragma unroll
        for(; i < steps; i++){
            dense_tile_loader.LoadRowfromRegister(i-1);
            sparse_tile_loader.Load(i);
            __syncthreads();
            dense_tile_loader.Prefetch(i);
            computer.TileMAC(i-1);
            __syncthreads();
        }

        dense_tile_loader.LoadRowfromRegister(i-1);
        __syncthreads();
        computer.TileMAC(i-1);
    }
   
    if(residue > 0){
        sparse_tile_loader.Residue();
        __syncthreads();
        dense_tile_loader.ResidueLoad(residue);
        __syncthreads();
        computer.TileMACResidue();
    } 

    wmmaOutputTile_16b8v<OutType> output_tile_storer(lane_id, VecLength, m_index_vec, dimN_index, dimN, output_fragment_0, output_fragment_1, output_matrix);
    output_tile_storer.Store();
}

template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSpmm_8b_template(
    int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);

    wmmaSpmm_kernel_8b<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
        m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    return cudaGetLastError();
}

//8-bit Tile_N = 128 with 4 warps
cudaError_t wmmaSpmm_8b(int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    switch(vec_length){
        case 2:
            return wmmaSpmm_8b_template<int, short, 1, 16, 128, 32, 4, 2>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const short *>(values), rhs_matrix, output_matrix);
            break;
        case 4:
            return wmmaSpmm_8b_template<int, int, 1, 16, 128, 32, 4, 4>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, values, rhs_matrix, output_matrix);
            break;
        case 8:
            return wmmaSpmm_8b_template<int, long long, 1, 16, 128, 32, 4, 8>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        default:
            printf("Unsupported Vector Length!\n");
            return cudaGetLastError();
    }
}

//16-bit 8-bit Tile_N = 128 with 4 warps
template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSpmm_16b8b_template(
    int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    if(vec_length == 8)
        wmmaSpmm_kernel_16b8b8v<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    else
        wmmaSpmm_kernel_16b8b<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    return cudaGetLastError();
}
cudaError_t wmmaSpmm_16b8b(int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    switch(vec_length){
        case 2:
            return wmmaSpmm_16b8b_template<int, int, 1, 16, 128, 32, 4, 2>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const int *>(values), rhs_matrix, output_matrix);
            break;
        case 4:
            return wmmaSpmm_16b8b_template<int, long long, 1, 16, 128, 32, 4, 4>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        case 8:
            return wmmaSpmm_16b8b_template<int, long long, 1, 16, 128, 32, 4, 8>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        default:
            printf("Unsupported Vector Length!\n");
            return cudaGetLastError();
    }
}

//16-bit 16-bit Tile_N = 64 with 4 warps
template <typename IndexType, typename VecType, int Tile_M, int Tile_K, int Tile_N, int WarpWidth, int Warps, int VecLength>
cudaError_t wmmaSpmm_16b_template(
    int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const VecType* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(n) / Tile_N), 1);
    dim3 block_dim(WarpWidth * Warps, Tile_M, 1);
    if(vec_length == 8)
        wmmaSpmm_kernel_16b8v<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    else
        wmmaSpmm_kernel_16b<int, int, VecType, int, Tile_K, Tile_N, Warps, VecLength><<<grid_dim, block_dim>>>(
            m_vec, n, k, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
    return cudaGetLastError();
}

cudaError_t wmmaSpmm_16b(int m_vec, int vec_length, int n, int k, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const int* __restrict__ values,
    const int* __restrict__ rhs_matrix,
    int* __restrict__ output_matrix)
{
    switch(vec_length){
        case 2:
            return wmmaSpmm_16b_template<int, int, 1, 16, 64, 32, 4, 2>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const int *>(values), rhs_matrix, output_matrix);
            break;
        case 4:
            return wmmaSpmm_16b_template<int, long long, 1, 16, 64, 32, 4, 4>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        case 8:
            return wmmaSpmm_16b_template<int, long long, 1, 16, 64, 32, 4, 8>(m_vec, vec_length, n, k, row_indices, 
        		    row_offsets, column_indices, reinterpret_cast<const long long *>(values), rhs_matrix, output_matrix);
            break;
        default:
            printf("Unsupported Vector Length!\n");
            return cudaGetLastError();
    }
}

}