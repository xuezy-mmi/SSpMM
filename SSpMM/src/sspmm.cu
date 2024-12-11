#include "../include/sspmm.cuh"
#include "spmm_utils/sspmm_dense_tile.h"
#include "spmm_utils/sspmm_sparse_tile.h"
#include "spmm_utils/sspmm_computer.h"
#include "spmm_utils/sspmm_output_tile.h"
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
__global__ void SpMM_Volta_V16(
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
	// m_index_vec = __ldg(row_indices + m_index_vec);////real row_id

	// Load the row offset and calculate the number of nonzeros in the row
	int row_offset_vec = __ldg(row_offsets + m_index_vec);///row offset 0
	int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;///nnz in this row

	// For VecLength=8, we don't need the memory aligner

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
	// const int kDenseFragmentSize = Tile_N / 4 * 8;//32/4*8=64

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
	// const int InnerSteps = Tile_N / 4;//32/4=8
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
	dense_tile_loader.ResidueLoad(n_group_idx, nonzeros);
	computer.TileMACResidue(n_group_idx);

	wmmaOutputTile16<OutType, StoreType> output_tile_storer(lane_id, thread_group, m_index_vec, 
		k_index, k, output_fragment1, output_fragment2, output_matrix);
	output_tile_storer.Store();
}


template <typename LoadType, typename IndexType, typename VecType, 
	typename OutType, typename StoreType, int Tile_N, 
	int Tile_K, int BlockWidth, int VecLength=8>
__global__ void SpMM_Volta_V8(
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
	// m_index_vec = __ldg(row_indices + m_index_vec);////real row_id

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
	// wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b[4];
	// Initialize the pointers to the dense rhs matrix
	wmmaDenseTile<LoadType, Tile_N, Tile_K, BlockWidth> dense_tile_loader(
		k, k_index, lane_id, thread_group, rhs_matrix, column_indices_tile, dense_matrix_fragment
	);

	// Accumulator registers for the output values.
	constexpr int kOutputFragmentSize = 16;
	__align__(16) float output_fragment[kOutputFragmentSize] = {};
	Transp_ComputeUtils8<VecType, Tile_N> computer(values_tile, dense_matrix_fragment, output_fragment, lane_id, thread_group);
	// wmmaComputeUtils8<VecType, Tile_N> computer(values_tile, dense_matrix_fragment, output_fragment, lane_id, thread_group);

	int compute_nnz = nonzeros;
	#pragma unroll 8
	for (; nonzeros > 0; nonzeros -= Tile_N){
	// for (; nonzeros >= Tile_K; nonzeros -= Tile_N){
		sparse_tile_loader.Load();
		// __syncthreads();
		#pragma unroll 8
		for (int n_group_idx = 0; n_group_idx < 8; n_group_idx ++){
			dense_tile_loader.LoadRow(n_group_idx);
		}
		// __threadfence_block();
		if(nonzeros <= Tile_N) break;
		#pragma unroll 8
		for (int n_group_idx = 0; n_group_idx < 8; n_group_idx ++){
			computer.TileMAC(n_group_idx);
		}
		// __syncthreads();
		compute_nnz = compute_nnz - Tile_N;
	}
	// sparse_tile_loader.ZeroTiles();
	// // __syncthreads();
	// sparse_tile_loader.Residue(nonzeros);
	// __syncthreads();

	
	#pragma unroll 8
	for (int n_group_idx = 0; n_group_idx < 8; n_group_idx ++){
		
		// if (nonzeros < 4) break;
		// dense_tile_loader.LoadRow(n_group_idx);
		// computer.TileMAC(n_group_idx);
		computer.ResTileMAC(n_group_idx, compute_nnz);
		// nonzeros -= 4;
		compute_nnz -= 4;
		if (compute_nnz <= 0) break;
	}
	// dense_tile_loader.ResidueLoad(n_group_idx, nonzeros);
	// computer.TileMACResidue(n_group_idx);
	Transp_OutputTile8<OutType, StoreType> output_tile_storer(
	// wmmaOutputTile8<OutType, StoreType> output_tile_storer(
		lane_id, thread_group, threadIdx.x, m_index_vec, 
		k_index, k, output_fragment, output_matrix);
	output_tile_storer.Store();
}


template <typename LoadType, typename IndexType, typename VecType, 
    typename OutType, typename StoreType, int Tile_K, 
    int Tile_N, int BlockWidth, int VecLength=8>
__global__ void SSpMM_V8(
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
	// m_index_vec = __ldg(row_indices + m_index_vec);////real row_id

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
	constexpr int kDenseFragmentSize = Tile_K * Tile_N / 32;

	__align__(16) half dense_matrix_fragment[kDenseFragmentSize];
	// Initialize the pointers to the dense matrix
	mmaDenseTile_SSPMM<LoadType, Tile_N, Tile_K> dense_tile_loader(
		threadIdx.x, N, n_index, dense_matrix, column_indices_tile, dense_matrix_fragment
	);
	
	// Accumulator registers for the output values.
	constexpr int kOutputFragmentSize = 16;
	__align__(16) float output_fragment[kOutputFragmentSize] = {};
	mmaComputeUtils8_SSPMM<Tile_N, Tile_K> computer(dense_matrix_fragment, values_tile, output_fragment, tid);

	// int compute_nnz = nonzeros;
	#pragma unroll 8
	for (; nonzeros > 0; nonzeros -= Tile_K){
		// // sparse_tile_loader.Load();
		// sparse_tile_loader.Residue(nonzeros);
		// __syncthreads();
		// // if(nonzeros <= Tile_K) break;// **************break
		// #pragma unroll 4
		// for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
		// 	dense_tile_loader.Load2Rows(n_group_idx);
		// }
		// __threadfence_block();
		// // if(nonzeros <= Tile_K) break;// **************break
		// #pragma unroll 4
		// for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
		// 	computer.TileMAC(n_group_idx);
		// }
		// __syncthreads();
		// // compute_nnz = compute_nnz - Tile_K;
		sparse_tile_loader.Residue(nonzeros);
		// dense_tile_loader.Load2Rows(0);
		// computer.TileMAC(0);
		dense_tile_loader.Load2RowsSelect(0,0);
		computer.TileMACSelect(0,0);
		// dense_tile_loader.Load2Rows(1);
		// computer.TileMAC(1);
		dense_tile_loader.Load2RowsSelect(1,1);
		computer.TileMACSelect(1,1);
		// dense_tile_loader.Load2Rows(2);
		// computer.TileMAC(2);
		dense_tile_loader.Load2RowsSelect(2,2);
		computer.TileMACSelect(2,2);
		// dense_tile_loader.Load2Rows(3);
		// computer.TileMAC(3);
		dense_tile_loader.Load2RowsSelect(3,3);
		computer.TileMACSelect(3,3);
	}

	// #pragma unroll 4
	// for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
	// 	computer.ResTileMAC(n_group_idx, compute_nnz);
	// 	compute_nnz -= 8;
	// 	if (compute_nnz <= 0) break;
	// }

	mmaOutputTile8_SSPMM<OutType, StoreType> output_tile_storer(
		threadIdx.x, m_index_vec,
		n_index, N, output_fragment, output_matrix);

	output_tile_storer.Store();
}

// template <typename LoadType, typename OutType, typename StoreType,
	// int Tile_M, int Tile_N, int Tile_K, int VecLength=16>
template <typename LoadType, typename IndexType, typename VecType, 
    typename OutType, typename StoreType, int Tile_K, 
    int Tile_N, int BlockWidth, int VecLength=16>
__global__ void SSpMM_V16(
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
	// m_index_vec = __ldg(row_indices + m_index_vec);////real row_id

	// Load the row offset and calculate the number of nonzeros in the row
	int row_offset_vec = __ldg(row_offsets + m_index_vec);///row offset 0
	int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;///nnz in this row

	// Shared memory tiles for the lhs values and indices
	__shared__ half values_tile_array[VecLength * Tile_K];//16*32
	__shared__ int column_indices_tile_array[Tile_K];

	// Pointers to the shared memory tiles
	half * values_tile = values_tile_array;
	int* column_indices_tile = column_indices_tile_array;

	// Initialize the pointers to the sparse matrix
	mmaSparseTile16_SSPMM<LoadType, Tile_N, Tile_K, VecLength> sparse_tile_loader(
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

	__align__(16) float output_fragment1[kOutputFragmentSize] = {};
	__align__(16) float output_fragment2[kOutputFragmentSize] = {};
	mmaComputeUtils16_SSPMM<Tile_N, Tile_K> computer(dense_matrix_fragment, values_tile, output_fragment1, output_fragment2, tid);

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
	// asm("");
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
	// // asm("");
	// dense_tile_loader.ResidueLoad(n_group_idx, nonzeros);
	// computer.TileMACResidue(n_group_idx);
	mmaOutputTile16_SSPMM<OutType, StoreType> output_tile_storer(
		threadIdx.x, m_index_vec,
		n_index, N, output_fragment1, output_fragment2, output_matrix);

	output_tile_storer.Store();

}

/////////////////////////////////////////////////////   1          32           64           32
template <typename LoadType, typename IndexType, int Tile_M, int Tile_N, int Tile_K, int BlockWidth>
cudaError_t SpMMex(//fp16 * fp16 = fp32
	int m_vec, int vec_length, int k, int n,
	const int* __restrict__ row_indices,
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ values,
	const half* __restrict__ rhs_matrix,
	float* __restrict__ output_matrix)
{
	dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(k) / Tile_K), 1);
	dim3 block_dim(BlockWidth, Tile_M, 1); // BlockWidth * Tile_M = 32 * 1 = 32threads = 1warps

	switch(vec_length){
	case 8:
		// printf("V=8\n");
		// cudaFuncSetAttribute(SpMM_Volta_V8<float4, int, float4, float, float4, Tile_N, Tile_K, BlockWidth, 8>,
		// 	cudaFuncAttributePreferredSharedMemoryCarveout, 50);

		SpMM_Volta_V8<float4, int, float4, float, float4, Tile_N, Tile_K, BlockWidth, 8><<<grid_dim, block_dim>>>(
			m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	case 16:
		// printf("V=16\n");
		// <LoadType, IndexType, VecType, OutType, StoreType, Tile_N, Tile_K, BlockWidth, VecLength>
		SpMM_Volta_V16<float4, int, float4, float, float4, Tile_N, Tile_K, BlockWidth, 16><<<grid_dim, block_dim>>>(
			m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	default:
		printf("Unsupported Vector Length!\n");
	}
	return cudaGetLastError();
}
template <typename LoadType, typename IndexType, int Tile_M, int Tile_N, int Tile_K, int BlockWidth>
cudaError_t SpMMex(//fp16 * fp16 = fp16
	int m_vec, int vec_length, int k, int n, 
	const int* __restrict__ row_indices, 
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ values,
	const half* __restrict__ rhs_matrix,
	half* __restrict__ output_matrix)
{

	dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(k) / Tile_K), 1);
	dim3 block_dim(BlockWidth, Tile_M, 1);//Tile_M = 1
	switch(vec_length){
	case 8:
		// cudaFuncSetAttribute(SpMM_Volta_V8<float4, int, float4, half, float2, Tile_N, Tile_K, BlockWidth, 8>,
		// 	cudaFuncAttributePreferredSharedMemoryCarveout, 50);
		// printf("V=8\n");
		// <LoadType, IndexType, VecType, OutType, StoreType, Tile_N, Tile_K, BlockWidth, VecLength>
		SpMM_Volta_V8<float4, int, float4, half, float2, Tile_N, Tile_K, BlockWidth, 8><<<grid_dim, block_dim>>>(
			m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	case 16:
		// printf("V=16\n");
		// <LoadType, IndexType, VecType, OutType, StoreType, Tile_N, Tile_K, BlockWidth, VecLength>
		SpMM_Volta_V16<float4, int, float4, half, float2, Tile_N, Tile_K, BlockWidth, 16><<<grid_dim, block_dim>>>(
			m_vec, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	default:
		printf("Unsupported Vector Length!\n");
	}

	return cudaGetLastError();
}

// Function for mixed precision//fp16 * fp16 = fp32
cudaError_t SpMM(int m_vec, int vec_length, int k, int n, 
	const int* __restrict__ row_indices,
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ values,
	const half* __restrict__ rhs_matrix,
	float* __restrict__ output_matrix)
{
	return SpMMex<float4, int, 1, 32, 64, 32>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
}

// Function for half precision//fp16 * fp16 = fp16
cudaError_t SpMM(int m_vec, int vec_length, int k, int n, 
	const int* __restrict__ row_indices, 
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ values,
	const half* __restrict__ rhs_matrix,
	half* __restrict__ output_matrix)
{
	// <LoadType, IndexType, Tile_M, Tile_N, Tile_K, BlockWidth>
	return SpMMex<float4, int, 1, 32, 64, 32>(m_vec, vec_length, k, n, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
}

// Function for single precision//error precision
cudaError_t SpMM(int m_vec, int vec_length, int k, int n,
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

template <typename LoadType, typename IndexType, int Tile_M, int Tile_K, int Tile_N, int BlockWidth>
cudaError_t SSpMM_ex(//fp16 * fp16 = fp32
	int m_vec, int vec_length, int N, int K,
	const int* __restrict__ row_indices,
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ values,
	const half* __restrict__ rhs_matrix,
	float* __restrict__ output_matrix)
{
	dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(N) / Tile_N), 1);
	dim3 block_dim(BlockWidth, Tile_M, 1); // BlockWidth * Tile_M = 32 * 1 = 32threads = 1warps

	switch(vec_length){
		//  <LoadType, OutType, StoreType, Tile_M, Tile_N, Tile_K, VecLength=16> // custom
		//  <LoadType, IndexType, VecType, OutType, StoreType, Tile_K, Tile_N, BlockWidth, VecLength=8> // unified
	// case 2:
	// 	SSpMM_V8<float4, int, float4, float, float4, Tile_K, Tile_N, BlockWidth, 8><<<grid_dim, block_dim>>>(
	// 		m_vec, N, K, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
	// 	break;
	// case 4:
	// 	SSpMM_V4<float4, int, float2, float, float4, Tile_K, Tile_N, BlockWidth, 4><<<grid_dim, block_dim>>>(
	// 		m_vec, N, K, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
	// 	break;
	case 8:
		cudaFuncSetAttribute(SSpMM_V8<float4, int, float4, half, float2, Tile_N, Tile_K, BlockWidth, 8>,
			cudaFuncAttributePreferredSharedMemoryCarveout, 50);
		SSpMM_V8<float4, int, float4, float, float4, Tile_K, Tile_N, BlockWidth, 8><<<grid_dim, block_dim>>>(
			m_vec, N, K, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	case 16:
		SSpMM_V16<float4, int, float4, float, float4, Tile_K, Tile_N, BlockWidth, 16><<<grid_dim, block_dim>>>(
			m_vec, N, K, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	default:
		printf("Unsupported Vector Length!\n");
	}

	return cudaGetLastError();
}
template <typename LoadType, typename IndexType, int Tile_M, int Tile_K, int Tile_N, int BlockWidth>
cudaError_t SSpMM_ex(//fp16 * fp16 = fp16
	int m_vec, int vec_length, int N, int K,
	const int* __restrict__ row_indices,
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ values,
	const half* __restrict__ rhs_matrix,
	half* __restrict__ output_matrix)
{
	dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(N) / Tile_N), 1);
	dim3 block_dim(BlockWidth, Tile_M, 1);//Tile_M = 1
	switch(vec_length){
	// case 2:
	// 	SSpMM_V8<float4, int, float4, half, float2, Tile_K, Tile_N, BlockWidth, 8><<<grid_dim, block_dim>>>(
	// 		m_vec, N, K, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
	// 	break;	
	// case 4:
	// 	SSpMM_V4<float4, int, float2, half, float2, Tile_K, Tile_N, BlockWidth, 4><<<grid_dim, block_dim>>>(
	// 		m_vec, N, K, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
	// 	break;
	case 8:
		SSpMM_V8<float4, int, float4, half, float2, Tile_K, Tile_N, BlockWidth, 8><<<grid_dim, block_dim>>>(
			m_vec, N, K, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	case 16:
		SSpMM_V16<float4, int, float4, half, float2, Tile_K, Tile_N, BlockWidth,16><<<grid_dim, block_dim>>>(
			m_vec, N, K, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;

	default:
		printf("Unsupported Vector Length!\n");
	}

	return cudaGetLastError();
}

// Function for mixed precision//fp16 * fp16 = fp32
cudaError_t SSpMM(int m_vec, int vec_length, int N, int K,
	const int* __restrict__ row_indices,
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ values,
	const half* __restrict__ rhs_matrix,
	float* __restrict__ output_matrix)
{
	return SSpMM_ex<float4, int, 1, 32, 64, 32>(m_vec, vec_length, N, K, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
}

// Function for half precision//fp16 * fp16 = fp16
cudaError_t SSpMM(int m_vec, int vec_length, int N, int K, 
	const int* __restrict__ row_indices, 
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ values,
	const half* __restrict__ rhs_matrix,
	half* __restrict__ output_matrix)
{
	// <LoadType, IndexType, Tile_M, Tile_N, Tile_K, BlockWidth>
	return SSpMM_ex<float4, int, 1, 32, 64, 32>(m_vec, vec_length, N, K, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
}

// Function for single precision//error precision
cudaError_t SSpMM(int m_vec, int vec_length, int N, int K,
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