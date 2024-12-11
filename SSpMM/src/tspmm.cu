#include "../include/tspmm.cuh"
#include "spmm_utils/T_dense_tile.h"
#include "spmm_utils/T_sparse_tile.h"
#include "spmm_utils/T_computer.h"
#include "spmm_utils/T_output_tile.h"
#include <stdio.h>
#include <mma.h>
#include <float.h>
#include <cuda_runtime.h>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

namespace spmm{

template <typename LoadType, typename IndexType, typename VecType, 
	typename OutType, typename StoreType, int Tile_N, 
	int Tile_K, int BlockWidth, int VecLength=16>
__global__ void wmmaSpmmV16(
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
	m_index_vec = __ldg(row_indices + m_index_vec);////real row_id

	// Load the row offset and calculate the number of nonzeros in the row
	int row_offset_vec = __ldg(row_offsets + m_index_vec);///row offset 0
	int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;///nnz in this row

	// Shared memory tiles for the lhs values and indices
	__shared__ half values_tile_array[VecLength * Tile_N];//16*32
	__shared__ int column_indices_tile_array[Tile_N];

	// Pointers to the shared memory tiles
	half * values_tile = values_tile_array;
	int* column_indices_tile = column_indices_tile_array;
	// Initialize the pointers to the sparse lhs matrix
	wmmaSparseTileV16<LoadType, VecType, VecLength, Tile_N, BlockWidth> sparse_tile_loader(
		k,row_offset_vec, threadIdx.x, values, column_indices,
		values_tile, column_indices_tile
	);

	// Register fragment for the dense matrix values
	constexpr int kDenseFragmentSize = Tile_N / 4 * 8;//32/4*8=64

	__align__(16) half dense_matrix_fragment[kDenseFragmentSize];

	// Initialize the pointers to the dense rhs matrix
	wmmaDenseTileV16<LoadType, Tile_N, Tile_K, BlockWidth> dense_tile_loader(
		k, k_index, lane_id, thread_group, rhs_matrix, column_indices_tile, dense_matrix_fragment
	);

	// Accumulator registers for the output values.
	constexpr int kOutputFragmentSize = 16;
	__align__(16) float output_fragment1[kOutputFragmentSize] = {};
	__align__(16) float output_fragment2[kOutputFragmentSize] = {};
	wmmaComputeUtils16<VecType, Tile_N> computer(values_tile, dense_matrix_fragment, output_fragment1, output_fragment2, lane_id, thread_group);

	constexpr int InnerSteps = Tile_N / 4;//32/4=8
	#pragma unroll 8
	for (; nonzeros >= Tile_N; nonzeros -= Tile_N){
		sparse_tile_loader.Load();
		__syncthreads();
		#pragma unroll 8
		for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
			dense_tile_loader.LoadRow(n_group_idx);
		}
		__threadfence_block();
		#pragma unroll 8
		for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
			computer.TileMAC(n_group_idx);
		}
		__syncthreads();
	}
	// asm("");

	sparse_tile_loader.ZeroTiles();
	// __syncthreads();
	sparse_tile_loader.Residue(nonzeros);
	__syncthreads();

	int n_group_idx = 0;

	#pragma unroll 8
	for (; n_group_idx < InnerSteps; n_group_idx ++){
		if (nonzeros < 4) break;
		dense_tile_loader.LoadRow(n_group_idx);
		computer.TileMAC(n_group_idx);
		nonzeros -= 4;
	}
	// asm("");

	dense_tile_loader.ResidueLoad(n_group_idx, nonzeros);
	computer.TileMACResidue(n_group_idx);

	wmmaOutputTile16<OutType, StoreType> output_tile_storer(lane_id, thread_group, m_index_vec, 
		k_index, k, output_fragment1, output_fragment2, output_matrix);
	output_tile_storer.Store();
}

template <typename LoadType, typename IndexType, typename VecType, 
	typename OutType, typename StoreType, int Tile_N, 
	int Tile_K, int BlockWidth, int VecLength=8>
__global__ void wmmaSpmmV8(
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
	m_index_vec = __ldg(row_indices + m_index_vec);////real row_id

	// Load the row offset and calculate the number of nonzeros in the row
	int row_offset_vec = __ldg(row_offsets + m_index_vec);///row offset 0
	int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;///nnz in this row

	// For VecLength=8, we don't need the memory aligner

	// Shared memory tiles for the lhs values and indices
	__shared__ half values_tile_array[VecLength * Tile_N];//8*32
	__shared__ int column_indices_tile_array[Tile_N];

	// Pointers to the shared memory tiles
	half * values_tile = values_tile_array;
	int* column_indices_tile = column_indices_tile_array;

	// Initialize the pointers to the sparse lhs matrix
	wmmaSparseTile<LoadType, VecType, VecLength, Tile_N, BlockWidth> sparse_tile_loader(
		k, row_offset_vec, threadIdx.x, values, column_indices,
		values_tile, column_indices_tile
	);

	// Register fragment for the dense matrix values
	constexpr int kDenseFragmentSize = Tile_N / 4 * 8;//32/4*8=64

	__align__(16) half dense_matrix_fragment[kDenseFragmentSize];

	// Initialize the pointers to the dense rhs matrix
	wmmaDenseTile<LoadType, Tile_N, Tile_K, BlockWidth> dense_tile_loader(
		k, k_index, lane_id, thread_group, rhs_matrix, column_indices_tile, dense_matrix_fragment
	);

	// Accumulator registers for the output values.
	constexpr int kOutputFragmentSize = 16;
	__align__(16) float output_fragment[kOutputFragmentSize] = {};
	wmmaComputeUtils8<VecType, Tile_N> computer(values_tile, dense_matrix_fragment, output_fragment, lane_id, thread_group);

	constexpr int InnerSteps = Tile_N / 4;//32/4=8
	#pragma unroll 8
	for (; nonzeros >= Tile_N; nonzeros -= Tile_N){
		sparse_tile_loader.Load();
		__syncthreads();
		#pragma unroll 8
		for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
			dense_tile_loader.LoadRow(n_group_idx);
		}
		__threadfence_block();
		#pragma unroll 8
		for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
			computer.TileMAC(n_group_idx);
		}
		__syncthreads();
	}
	asm("");

	sparse_tile_loader.ZeroTiles();
	// __syncthreads();
	sparse_tile_loader.Residue(nonzeros);
	__syncthreads();

	int n_group_idx = 0;

	#pragma unroll 8
	for (; n_group_idx < InnerSteps; n_group_idx ++){
		if (nonzeros < 4) break;
		dense_tile_loader.LoadRow(n_group_idx);
		computer.TileMAC(n_group_idx);
		nonzeros -= 4;
	}
	asm("");

	dense_tile_loader.ResidueLoad(n_group_idx, nonzeros);
	computer.TileMACResidue(n_group_idx);

	wmmaOutputTile8<OutType, StoreType> output_tile_storer(lane_id, thread_group, m_index_vec, 
		k_index, k, output_fragment, output_matrix);
	output_tile_storer.Store();

}

template <typename LoadType, typename IndexType, typename VecType, 
	typename OutType, int Tile_N, 
	int Tile_K, int BlockWidth, int VecLength=4>
__global__ void wmmaSpmmV4(
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
	__shared__ half values_tile_array[VecLength * Tile_N];
	__shared__ int column_indices_tile_array[Tile_N];

	// Pointers to the shared memory tiles
	half * values_tile = values_tile_array;
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
	// __syncthreads();
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

template <typename LoadType, typename IndexType, typename VecType, typename OutType, int Tile_N, int Tile_K, int BlockWidth, int VecLength=2>
__global__ void wmmaSpmmV2(
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
	__shared__ half values_tile_array[VecLength * Tile_N];
	__shared__ int column_indices_tile_array[Tile_N];

	// Pointers to the shared memory tiles
	half * values_tile = values_tile_array;
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
	// __syncthreads();
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


/////////////////////////////////////////////////////   1          32           64           32
template <typename LoadType, typename IndexType, int Tile_M, int Tile_N, int Tile_K, int BlockWidth>
cudaError_t TwmmaSpmmEx(//fp16 * fp16 = fp32
	int m_vec, int vec_length, int k, int n, 
	const int* __restrict__ row_indices, 
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ values,
	const half* __restrict__ rhs_matrix,
	float* __restrict__ output_matrix)
{
	// int BX = (k + Tile_K - 1) / Tile_K;
	// int BY = (m_vec + Tile_M - 1) / Tile_M;
	// const int NSPLIT = 1024;
	// int split_num = (k + NSPLIT - 1) / NSPLIT;
	// dim3 grid_dim((BX + split_num - 1) / split_num, BY, split_num);

	dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(k) / Tile_K), 1);
	dim3 block_dim(BlockWidth, Tile_M, 1); // BlockWidth * Tile_M = 32 * 1 = 32threads = 1warps
	switch(vec_length){
		case 2:
		// printf("V=2\n");
		wmmaSpmmV2<float4, int, float, float, Tile_N, Tile_K, BlockWidth, 2><<<grid_dim, block_dim>>>(
			m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	case 4:
		// printf("V=4\n");
		wmmaSpmmV4<float4, int, float2, float, Tile_N, Tile_K, BlockWidth, 4><<<grid_dim, block_dim>>>(
			m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	case 8:
		// printf("V=8\n");
		// <LoadType, IndexType, VecType, OutType, StoreType, Tile_N, Tile_K, BlockWidth, VecLength=8>
		wmmaSpmmV8<float4, int, float4, float, float4, Tile_N, Tile_K, BlockWidth, 8><<<grid_dim, block_dim>>>(
			m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	case 16:
		// printf("V=16\n");
		// <LoadType, IndexType, VecType, OutType, StoreType, Tile_N, Tile_K, BlockWidth, VecLength>
		wmmaSpmmV16<float4, int, float4, float, float4, Tile_N, Tile_K, BlockWidth, 16><<<grid_dim, block_dim>>>(
			m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	default:
		printf("Unsupported Vector Length!\n");
	}

	return cudaGetLastError();
}
template <typename LoadType, typename IndexType, int Tile_M, int Tile_N, int Tile_K, int BlockWidth>
cudaError_t TwmmaSpmmEx(//fp16 * fp16 = fp16
	int m_vec, int vec_length, int k, int n, 
	const int* __restrict__ row_indices, 
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ values,
	const half* __restrict__ rhs_matrix,
	half* __restrict__ output_matrix)
{
	// int BX = (k + Tile_K - 1) / Tile_K;//k / Tile_K;
	// int BY = (m_vec + Tile_M - 1) / Tile_M;//m_vec / Tile_M;
	// const int NSPLIT = 1024;
	// int split_num = (m_vec + NSPLIT - 1) / NSPLIT;
	// int GY = (BY + split_num - 1) / split_num;
	// dim3 grid_dim(GY, BX, split_num);
	//dim3 grid_dim(BX/2, BY, 2);

	dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(k) / Tile_K), 1);
	dim3 block_dim(BlockWidth, Tile_M, 1);//Tile_M = 1
	switch(vec_length){
	case 2:
		// printf("V=2\n");
		wmmaSpmmV2<float4, int, float, half, Tile_N, Tile_K, BlockWidth, 2><<<grid_dim, block_dim>>>(
			m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	case 4:
		// printf("V=4\n");
		wmmaSpmmV4<float4, int, float2, half, Tile_N, Tile_K, BlockWidth, 4><<<grid_dim, block_dim>>>(
			m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	case 8:
		// printf("V=8\n");
		// <LoadType, IndexType, VecType, OutType, StoreType, Tile_N, Tile_K, BlockWidth, VecLength>
		wmmaSpmmV8<float4, int, float4, half, float2, Tile_N, Tile_K, BlockWidth, 8><<<grid_dim, block_dim>>>(
			m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	case 16:
		// printf("V=16\n");
		// <LoadType, IndexType, VecType, OutType, StoreType, Tile_N, Tile_K, BlockWidth, VecLength>
		wmmaSpmmV16<float4, int, float4, half, float2, Tile_N, Tile_K, BlockWidth, 16><<<grid_dim, block_dim>>>(
			m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	default:
		printf("Unsupported Vector Length!\n");
	}

	return cudaGetLastError();
}

// Function for mixed precision//fp16 * fp16 = fp32
cudaError_t TwmmaSpmm(int m_vec, int vec_length, int k, int n, 
	const int* __restrict__ row_indices,
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ values,
	const half* __restrict__ rhs_matrix,
	float* __restrict__ output_matrix)
{
	return TwmmaSpmmEx<float4, int, 1, 32, 64, 32>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
}

// Function for half precision//fp16 * fp16 = fp16
cudaError_t TwmmaSpmm(int m_vec, int vec_length, int k, int n, 
	const int* __restrict__ row_indices, 
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ values,
	const half* __restrict__ rhs_matrix,
	half* __restrict__ output_matrix)
{
	// <LoadType, IndexType, Tile_M, Tile_N, Tile_K, BlockWidth>
	return TwmmaSpmmEx<float4, int, 1, 32, 64, 32>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
}

// Function for single precision//error precision
cudaError_t TwmmaSpmm(int m_vec, int vec_length, int k, int n,
	const int* __restrict__ row_indices,
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const float* __restrict__ values,
	const float* __restrict__ rhs_matrix,
	float* __restrict__ output_matrix)
{
	printf("wmmaSpmm doesn't support float input.\n");
	return cudaSuccess;
}

}