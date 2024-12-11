#include <cuda.h>
#include "cuda_fp16.h"
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include "spmm_utils/dense_tile.h"
#include "spmm_utils/sparse_tile.h"
#include "spmm_utils/compute_utils.h"
#include "spmm_utils/output_tile.h"
#include "spmm_utils/sspmm_dense_tile.h"
#include "spmm_utils/sspmm_sparse_tile.h"
#include "spmm_utils/sspmm_computer.h"
#include "spmm_utils/sspmm_output_tile.h"
#include <stdio.h>
#include <mma.h>
#include <float.h>
using namespace nvcuda;

namespace spmm{

template <typename LoadType, typename IndexType, typename VecType, 
    typename OutType, typename StoreType, int Tile_K, 
    int Tile_N, int BlockWidth, int VecLength=8>
__device__ void wmmaSpmmKernel8_(
	int M, int N, int K,
	const int* __restrict__ row_indices,
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ sparse_values,
	const half* __restrict__ dense_matrix,
	OutType* __restrict__ output_matrix)
{
	// Tile_M = 1
	// Tile_N = 64
	// Tile_K = 32
	// For the wmma based implementation, we have Tile_M = 1
	int m_index_vec = blockIdx.x;
	int n_index = blockIdx.y * Tile_N;
	const int tid = threadIdx.x;
	// const int lane_id = threadIdx.x % 4;
	// const int thread_group = threadIdx.x / 4;
	

	// Threads that work on different m-dim indices are independent
	// If we're out of bounds in the m-dimension we can just return
	if (m_index_vec >= M) return;
	m_index_vec = __ldg(row_indices + m_index_vec);////real row_id

	// Load the row offset and calculate the number of nonzeros in the row
	int row_offset_vec = __ldg(row_offsets + m_index_vec);///row offset 0
	int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;///nnz in this row

	// Shared memory tiles for the lhs values and indices
	__shared__ half values_tile_array[VecLength * Tile_K];//8*32
	__shared__ int column_indices_tile_array[Tile_K];

	// Pointers to the shared memory tiles
	half * values_tile = values_tile_array;
	int* column_indices_tile = column_indices_tile_array;

	// Initialize the pointers to the sparse matrix
	mmaSparseTile_SSPMM<LoadType, VecType, Tile_N, Tile_K, VecLength> sparse_tile_loader(
		N, row_offset_vec, threadIdx.x, sparse_values, column_indices,
		values_tile, column_indices_tile
	);

	// Register fragment for the dense matrix values
	constexpr int kDenseFragmentSize = Tile_K * Tile_N / 32;//(32*64)/32 = 64

	__align__(16) half dense_matrix_fragment[kDenseFragmentSize];
	// Initialize the pointers to the dense matrix
	mmaDenseTile_SSPMM<LoadType, Tile_N, Tile_K> dense_tile_loader(
		threadIdx.x, N, n_index, dense_matrix, column_indices_tile, dense_matrix_fragment
	);
	
	// Accumulator registers for the output values.
	constexpr int kOutputFragmentSize = 16;
	__align__(16) float output_fragment[kOutputFragmentSize] = {};
	mmaComputeUtils8_SSPMM<Tile_N, Tile_K> computer(dense_matrix_fragment, values_tile, output_fragment, tid);

	int compute_nnz = nonzeros;
	#pragma unroll 8
	for (; nonzeros > 0; nonzeros -= Tile_K){
		sparse_tile_loader.Load();
		// sparse_tile_loader.Residue(nonzeros);
		__syncthreads();
		// if(nonzeros <= Tile_K) break;// **************break
		#pragma unroll 4
		for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
			dense_tile_loader.Load2Rows(n_group_idx);
		}
		__threadfence_block();
		if(nonzeros <= Tile_K) break;// **************break
		#pragma unroll 4
		for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
			computer.TileMAC(n_group_idx);
		}
		__syncthreads();
		compute_nnz = compute_nnz - Tile_K;
	}
	asm("");
	// sparse_tile_loader.ZeroTiles();
	// sparse_tile_loader.Residue(nonzeros);
	// __syncthreads();

	#pragma unroll 4
	for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
		
		// dense_tile_loader.Load2Rows(n_group_idx);
		computer.ResTileMAC(n_group_idx, compute_nnz);
		compute_nnz -= 8;
		if (compute_nnz <= 0) break;
	}
	asm("");
	// dense_tile_loader.ResidueLoad(n_group_idx, nonzeros);
	// computer.TileMACResidue(n_group_idx);

	mmaOutputTile8_SSPMM<OutType, StoreType> output_tile_storer(
		threadIdx.x, m_index_vec,
		n_index, N, output_fragment, output_matrix);

	output_tile_storer.Store();
}


template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, typename StoreType, int Tile_N, 
          int Tile_K, int BlockWidth, int VecLength=8>
__global__ void wmmaSpmmKernel8(
    int m, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    wmmaSpmmKernel8_<LoadType, IndexType, VecType, OutType, StoreType, Tile_N, Tile_K, BlockWidth, VecLength>(
        m, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix
    );
}


template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, typename StoreType, int Tile_N, 
          int Tile_K, int BlockWidth, int VecLength=8>
__global__ void batchedWmmaSpmmKernel8(
    int m, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values_b,
    int values_stride,
    const half* __restrict__ rhs_matrix_b,
    int rhs_stride,
    OutType* __restrict__ output_matrix_b,
    int output_stride)
{
    // Get the entry index
    int entry_idx = blockIdx.z;
    const half* values = values_b + entry_idx * values_stride;
    const half* rhs_matrix = rhs_matrix_b + entry_idx * rhs_stride;
    OutType* output_matrix = output_matrix_b + entry_idx * output_stride;

    wmmaSpmmKernel8_<LoadType, IndexType, VecType, OutType, StoreType, Tile_N, Tile_K, BlockWidth, VecLength>(
        m, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix
    );
}


template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_N, 
          int Tile_K, int BlockWidth, int VecLength=4>
__device__ void wmmaSpmmKernel4_(
    int m, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int k_index = blockIdx.y * Tile_K;
    const int lane_id = threadIdx.x % 4;
    const int thread_group = threadIdx.x / 4;

    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices
    __shared__ float2 values_tile_array[VecLength * Tile_N];
    __shared__ int column_indices_tile_array[Tile_N];

    // Pointers to the shared memory tiles
    float2 * values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    wmmaSparseTile<LoadType, VecType, VecLength, Tile_N, BlockWidth> sparse_tile_loader(
        k, row_offset_vec, threadIdx.x, values, column_indices,
        values_tile, column_indices_tile
    );

    // Register fragment for the dense matrix values
    constexpr int kDenseFragmentSize = Tile_N / 4 * 8;

    __align__(16) half dense_matrix_fragment[kDenseFragmentSize];

    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile<LoadType, Tile_N, Tile_K, BlockWidth> dense_tile_loader(
        k, k_index, lane_id, thread_group, rhs_matrix, column_indices_tile, dense_matrix_fragment
    );


    // Accumulator registers for the output values.
    constexpr int kOutputFragmentSize = 8;
    __align__(16) float output_fragment[kOutputFragmentSize] = {};
    wmmaComputeUtils4<VecType, Tile_N> computer(values_tile, dense_matrix_fragment, output_fragment, lane_id, thread_group);

    //
    // Begin kernel main loop
    //

    constexpr int InnerSteps = Tile_N / 4;

    for (; nonzeros >= Tile_N; nonzeros -= Tile_N){
        sparse_tile_loader.Load();
        __syncthreads();
        #pragma unroll
        for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
            dense_tile_loader.LoadRow(n_group_idx);
        }
        __threadfence_block();
        #pragma unroll
        for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
            computer.TileMAC(n_group_idx);
        }
        __syncthreads();
    }
    
    sparse_tile_loader.ZeroTiles();
    __syncthreads();
    sparse_tile_loader.Residue(nonzeros);
    __syncthreads();

    int n_group_idx = 0;

    #pragma unroll
    for (; n_group_idx < InnerSteps; n_group_idx ++){
        if (nonzeros < 4) break;
        dense_tile_loader.LoadRow(n_group_idx);
        computer.TileMAC(n_group_idx);
        nonzeros -= 4;
    }
    asm("");

    dense_tile_loader.ResidueLoad(n_group_idx, nonzeros);
    computer.TileMACResidue(n_group_idx);

    wmmaOutputTile4<OutType> output_tile_storer(lane_id, thread_group, m_index_vec, k_index, k, output_fragment, output_matrix);
    output_tile_storer.Store();
}


template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_N, 
          int Tile_K, int BlockWidth, int VecLength=4>
__global__ void wmmaSpmmKernel4(
    int m, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    wmmaSpmmKernel4_<LoadType, IndexType, VecType, OutType, Tile_N, Tile_K, BlockWidth, VecLength>(
        m, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix
    );
}


template <typename LoadType, typename IndexType, typename VecType, 
          typename OutType, int Tile_N, 
          int Tile_K, int BlockWidth, int VecLength=4>
__global__ void batchedWmmaSpmmKernel4(
    int m, int k, int n, 
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values_b,
    int values_stride,
    const half* __restrict__ rhs_matrix_b,
    int rhs_stride,
    OutType* __restrict__ output_matrix_b,
    int output_stride)
{
    // Get the entry index
    int entry_idx = blockIdx.z;
    const half* values = values_b + entry_idx * values_stride;
    const half* rhs_matrix = rhs_matrix_b + entry_idx * rhs_stride;
    OutType* output_matrix = output_matrix_b + entry_idx * output_stride;

    wmmaSpmmKernel4_<LoadType, IndexType, VecType, OutType, Tile_N, Tile_K, BlockWidth, VecLength>(
        m, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix
    );

}


template <typename LoadType, typename IndexType, typename VecType, typename OutType, int Tile_N, int Tile_K, int BlockWidth, int VecLength=2>
__device__ void wmmaSpmmKernel2_(
    int m, int k, int n,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    // For the wmma based implementation, we have Tile_M = 1
    int m_index_vec = blockIdx.x;
    int k_index = blockIdx.y * Tile_K;
    const int lane_id = threadIdx.x % 4;
    const int thread_group = threadIdx.x / 4;

    // Threads that work on different m-dim indices are independent
    // If we're out of bounds in the m-dimension we can just return
    if (m_index_vec >= m) return;
    m_index_vec = __ldg(row_indices + m_index_vec);

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;

    // Shared memory tiles for the lhs values and indices
    __shared__ float values_tile_array[VecLength * Tile_N];
    __shared__ int column_indices_tile_array[Tile_N];

    // Pointers to the shared memory tiles
    float * values_tile = values_tile_array;
    int* column_indices_tile = column_indices_tile_array;

    // Initialize the pointers to the sparse lhs matrix
    wmmaSparseTile<LoadType, VecType, VecLength, Tile_N, BlockWidth> sparse_tile_loader(
        k, row_offset_vec, threadIdx.x, values, column_indices,
        values_tile, column_indices_tile
    );

    // Register fragment for the dense matrix values
    constexpr int kDenseFragmentSize = Tile_N / 4 * 8;

    __align__(16) half dense_matrix_fragment[kDenseFragmentSize];

    // Initialize the pointers to the dense rhs matrix
    wmmaDenseTile<LoadType, Tile_N, Tile_K, BlockWidth> dense_tile_loader(
        k, k_index, lane_id, thread_group, rhs_matrix, column_indices_tile, dense_matrix_fragment
    );

    // Accumulator registers for the output values.
    constexpr int kOutputFragmentSize = 4;
    __align__(16) float output_fragment[kOutputFragmentSize] = {};
    wmmaComputeUtils2<VecType, Tile_N> computer(values_tile, dense_matrix_fragment, output_fragment, lane_id, thread_group);

    //
    // Begin kernel main loop
    //

    constexpr int InnerSteps = Tile_N / 4;

    for (; nonzeros >= Tile_N; nonzeros -= Tile_N){
        sparse_tile_loader.Load();
        __syncthreads();
        #pragma unroll
        for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
            dense_tile_loader. LoadRow(n_group_idx);
        }
        __threadfence_block();
        #pragma unroll
        for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
            computer.TileMAC(n_group_idx);
        }
        __syncthreads();
    }

    sparse_tile_loader.ZeroTiles();
    __syncthreads();
    sparse_tile_loader.Residue(nonzeros);
    __syncthreads();

    int n_group_idx = 0;
    #pragma unroll
    for (; n_group_idx < InnerSteps; n_group_idx ++){
        if (nonzeros < 4) break;
        dense_tile_loader.LoadRow(n_group_idx);
        computer.TileMAC(n_group_idx);
        nonzeros -= 4;
    }
    asm("");

    dense_tile_loader.ResidueLoad(n_group_idx, nonzeros);
    computer.TileMACResidue(n_group_idx);

    wmmaOutputTile2<OutType> output_tile_storer(lane_id, thread_group, m_index_vec, k_index, k, output_fragment, output_matrix);
    output_tile_storer.Store();
}


template <typename LoadType, typename IndexType, typename VecType, typename OutType, int Tile_N, int Tile_K, int BlockWidth, int VecLength=2>
__global__ void wmmaSpmmKernel2(
    int m, int k, int n,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)
{
    wmmaSpmmKernel2_<LoadType, IndexType, VecType, OutType, Tile_N, Tile_K, BlockWidth, VecLength>(
        m, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix
    );
}


template <typename LoadType, typename IndexType, typename VecType, typename OutType, int Tile_N, int Tile_K, int BlockWidth, int VecLength=2>
__global__ void batchedWmmaSpmmKernel2(
    int m, int k, int n,
    const int* __restrict__ row_indices, 
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half* __restrict__ values_b,
    int values_stride,
    const half* __restrict__ rhs_matrix_b,
    int rhs_stride,
    OutType* __restrict__ output_matrix_b,
    int output_stride)
{
    // Get the entry index
    int entry_idx = blockIdx.z;
    const half* values = values_b + entry_idx * values_stride;
    const half* rhs_matrix = rhs_matrix_b + entry_idx * rhs_stride;
    OutType* output_matrix = output_matrix_b + entry_idx * output_stride;

    wmmaSpmmKernel2_<LoadType, IndexType, VecType, OutType, Tile_N, Tile_K, BlockWidth, VecLength>(
        m, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix
    );
}

}

torch::Tensor spmm_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length)
{
    int m_vec = row_offsets.size(-1) - 1;
    int m = m_vec * vec_length;
    int n = rhs_matrix.size(0);
    int k = rhs_matrix.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(rhs_matrix.device());

    auto output_matrix = torch::empty({m, k}, options);

    dim3 grid;
    dim3 block;

    grid.x = m_vec;
    grid.y = k / 64;

    block.x = 32;

    switch(vec_length){
        case 8:
            spmm::wmmaSpmmKernel8<float4, int, float4, half, float2, 32, 64, 32, 8><<<grid, block>>>(
                m_vec, k, n, row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(),
                reinterpret_cast<half *>(values.data<torch::Half>()),
                reinterpret_cast<half *>(rhs_matrix.data<torch::Half>()),
                reinterpret_cast<half *>(output_matrix.data<torch::Half>())
            );
            break;
        case 4:
            spmm::wmmaSpmmKernel4<float4, int, float2, half, 32, 64, 32, 4><<<grid, block>>>(
                m_vec, k, n, row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(),
                reinterpret_cast<half *>(values.data<torch::Half>()),
                reinterpret_cast<half *>(rhs_matrix.data<torch::Half>()),
                reinterpret_cast<half *>(output_matrix.data<torch::Half>())
            );
            break;
        case 2:
            spmm::wmmaSpmmKernel2<float4, int, float, half, 32, 64, 32, 2><<<grid, block>>>(
                m_vec, k, n, row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(),
                reinterpret_cast<half *>(values.data<torch::Half>()),
                reinterpret_cast<half *>(rhs_matrix.data<torch::Half>()),
                reinterpret_cast<half *>(output_matrix.data<torch::Half>())
            );
            break;
    }

    return output_matrix;

}

torch::Tensor batched_spmm_cuda(
    torch::Tensor row_indices,
    torch::Tensor row_offsets,
    torch::Tensor column_indices,
    torch::Tensor values,
    torch::Tensor rhs_matrix,
    int vec_length)
{
    int m_vec = row_offsets.size(-1) - 1;
    int m = m_vec * vec_length;
    int n = rhs_matrix.size(-2);
    int k = rhs_matrix.size(-1);
    int batch_size = rhs_matrix.numel() / (n * k);

    int nnz = column_indices.numel();

    int values_stride = nnz * vec_length;
    int rhs_stride = n * k;
    int output_stride = m * k;

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(rhs_matrix.device());

    auto output_matrix = torch::empty({batch_size, m, k}, options);

    dim3 grid;
    dim3 block;

    grid.x = m_vec;
    grid.y = k / 64;
    grid.z = batch_size;
    block.x = 32;

    switch(vec_length){
        case 8:
            spmm::batchedWmmaSpmmKernel8<float4, int, float4, half, float2, 32, 64, 32, 8><<<grid, block>>>(
                m_vec, k, n, row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(),
                reinterpret_cast<half *>(values.data<torch::Half>()),
                values_stride,
                reinterpret_cast<half *>(rhs_matrix.data<torch::Half>()),
                rhs_stride,
                reinterpret_cast<half *>(output_matrix.data<torch::Half>()),
                output_stride
            );
            break;
        case 4:
            spmm::batchedWmmaSpmmKernel4<float4, int, float2, half, 32, 64, 32, 4><<<grid, block>>>(
                m_vec, k, n, row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(),
                reinterpret_cast<half *>(values.data<torch::Half>()),
                values_stride,
                reinterpret_cast<half *>(rhs_matrix.data<torch::Half>()),
                rhs_stride,
                reinterpret_cast<half *>(output_matrix.data<torch::Half>()),
                output_stride
            );
            break;
        case 2:
            spmm::batchedWmmaSpmmKernel2<float4, int, float, half, 32, 64, 32, 2><<<grid, block>>>(
                m_vec, k, n, row_indices.data<int>(), row_offsets.data<int>(), column_indices.data<int>(),
                reinterpret_cast<half *>(values.data<torch::Half>()),
                values_stride,
                reinterpret_cast<half *>(rhs_matrix.data<torch::Half>()),
                rhs_stride,
                reinterpret_cast<half *>(output_matrix.data<torch::Half>()),
                output_stride
            );
            break;
    }

    return output_matrix;

}
