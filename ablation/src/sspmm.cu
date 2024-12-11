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
	// __shared__ half values_tile_array[VecLength * Tile_K * 2];//8*32
	extern __shared__ half values_tile_array[];
	__shared__ int column_indices_tile_array[Tile_K * 2];

	// Pointers to the shared memory tiles
	half * values_tile = values_tile_array;
	int* column_indices_tile = column_indices_tile_array;

	// Initialize the pointers to the sparse matrix
	mmaSparseTile_SSPMM<LoadType, VecType, Tile_N, Tile_K, VecLength> sparse_tile_loader(
		N, row_offset_vec, threadIdx.x, sparse_values, column_indices,
		values_tile, column_indices_tile
	);

	// Register fragment for the dense matrix values
	constexpr int kDenseFragmentSize = 2 * Tile_K * Tile_N / 32;//(32*64)/32 = 64

	__align__(16) half dense_matrix_fragment[kDenseFragmentSize];
	// Initialize the pointers to the dense matrix
	mmaDenseTile_SSPMM<LoadType, Tile_N, Tile_K> dense_tile_loader(
		threadIdx.x, N, n_index, dense_matrix, column_indices_tile, dense_matrix_fragment
	);
	
	// Accumulator registers for the output values.
	constexpr int kOutputFragmentSize = 16;
	__align__(16) float output_fragment[kOutputFragmentSize] = {};
	mmaComputeUtils8_SSPMM<Tile_N, Tile_K> computer(dense_matrix_fragment, values_tile, output_fragment, tid);

	sparse_tile_loader.Load(nonzeros, 0);
	__syncthreads();
	#pragma unroll 4
	for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
		dense_tile_loader.Load2Rows(n_group_idx, 0);
	}
	__threadfence_block();
	int compute_nnz = nonzeros;
	nonzeros -= Tile_K;

	int sel = 0;
    int sel_next = 1;
	#pragma unroll 8
	for (; nonzeros > 0; nonzeros -= Tile_K){

		sparse_tile_loader.Load(nonzeros, sel_next);
		__syncthreads();
		dense_tile_loader.Load2Rows(0, sel_next);
		computer.TileMAC(0, sel);
		dense_tile_loader.Load2Rows(1, sel_next);
		computer.TileMAC(1, sel);
		dense_tile_loader.Load2Rows(2, sel_next);
		computer.TileMAC(2, sel);
		dense_tile_loader.Load2Rows(3, sel_next);
		computer.TileMAC(3, sel);
		// #pragma unroll 4
		// for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
		// 	dense_tile_loader.Load2Rows(n_group_idx, sel_next);
		// }
		// __threadfence_block();
		// #pragma unroll 4
		// for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
		// 	computer.TileMAC(n_group_idx, sel);
		// }
		// __syncthreads();
		compute_nnz -= Tile_K;
		sel ^= 1;
        sel_next ^= 1;
	}

	#pragma unroll 4
	for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
		// computer.ResTileMAC(n_group_idx, compute_nnz, sel);
		computer.TileMAC(n_group_idx, sel);
		compute_nnz -= 8;
		if (compute_nnz <= 0) break;
	}
	// #pragma unroll 4
	// for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
	// 	computer.TileMAC(n_group_idx, sel);
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
	// __shared__ half values_tile_array[VecLength * Tile_K * 2];//16*32
	extern __shared__ half values_tile_array[];
	__shared__ int column_indices_tile_array[Tile_K * 2];

	// Pointers to the shared memory tiles
	half * values_tile = values_tile_array;
	int* column_indices_tile = column_indices_tile_array;

	// Initialize the pointers to the sparse matrix
	mmaSparseTile16_SSPMM<LoadType, Tile_N, Tile_K, VecLength> sparse_tile_loader(
		N, row_offset_vec, threadIdx.x, sparse_values, column_indices,
		values_tile, column_indices_tile
	);

	// Register fragment for the dense matrix values
	constexpr int kDenseFragmentSize = 2 * Tile_K * Tile_N / 32;//(32*64)/32 = 64

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

	sparse_tile_loader.Load(nonzeros, 0);
	__syncthreads();
	#pragma unroll 4
	for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
		dense_tile_loader.Load2Rows(n_group_idx, 0);
	}
	__threadfence_block();
	int compute_nnz = nonzeros;
	nonzeros -= Tile_K;

	int sel = 0;
    int sel_next = 1;
	#pragma unroll 8
	for (; nonzeros > 0; nonzeros -= Tile_K){

		sparse_tile_loader.Load(nonzeros, sel_next);
		__syncthreads();
		dense_tile_loader.Load2Rows(0, sel_next);
		computer.TileMAC(0, sel);
		dense_tile_loader.Load2Rows(1, sel_next);
		computer.TileMAC(1, sel);
		dense_tile_loader.Load2Rows(2, sel_next);
		computer.TileMAC(2, sel);
		dense_tile_loader.Load2Rows(3, sel_next);
		computer.TileMAC(3, sel);
		// #pragma unroll 4
		// for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
		// 	dense_tile_loader.Load2Rows(n_group_idx, sel_next);
		// }
		// __threadfence_block();
		// // if(nonzeros <= Tile_K) break;// **************break
		// #pragma unroll 4
		// for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
		// 	computer.TileMAC(n_group_idx, sel);
		// }
		// __syncthreads();
		compute_nnz = compute_nnz - Tile_K;
		sel ^= 1;
        sel_next ^= 1;
	}
	#pragma unroll 4
	for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
		// computer.ResTileMAC(n_group_idx, compute_nnz, sel);
		computer.TileMAC(n_group_idx, sel);
		compute_nnz -= 8;
		if (compute_nnz <= 0) break;
	}
	// #pragma unroll 4
	// for (int n_group_idx = 0; n_group_idx < 4; n_group_idx ++){
	// 	computer.TileMAC(n_group_idx, sel);
	// }

	mmaOutputTile16_SSPMM<OutType, StoreType> output_tile_storer(
		threadIdx.x, m_index_vec,
		n_index, N, output_fragment1, output_fragment2, output_matrix);
	output_tile_storer.Store();
}
// template <typename OutType>
// __global__ void wmma_spmm_v8(
//     const int M, const int K, const int N,
//     const int* __restrict__ row_offsets,
//     const int* __restrict__ column_indices,
//     const half * __restrict__ a,
//     const half * __restrict__ b,
//     OutType * __restrict__ c)
// {
//     const int BM = 8;
//     const int BK = 32;
//     const int BN = 64;
//     const int vec_len = 8;
//     int m_vec = blockIdx.x;
//     int by = blockIdx.y;
//     const int tid = threadIdx.x;
//     const int lane_id = threadIdx.x % 4;
//     const int t_group = threadIdx.x / 4;
//     if (m_vec >= M || by >= N/BN) return;

// 	// Load the row offset and calculate the number of nonzeros in the row
// 	int row_offset_vec = __ldg(row_offsets + m_vec);///row offset 0
// 	int nonzeros = __ldg(row_offsets + m_vec + 1) - row_offset_vec;///nnz in this row
//     // if (tid == 0) {
// 	// 	printf("m_vec: %d, by: %d, row_offset_vec: %d, nonzeros: %d\n", m_vec, by, row_offset_vec, nonzeros);
// 	// }
	
//     int compute_nnz = nonzeros;
//     const int BPAD = 0;
//     const int stride = 8;//9 = (BN + BPAD) / 8 = (64+8)/8
//     __shared__ __align__(16) int col_index[BK*2];
//     __shared__ __align__(16) float4 smemA[BK * 2];
// 	__shared__ __align__(16) float4 smemB[BK * stride * 2];
//     int s_a_offset = BK;
//     int s_b_offset = BK * stride;

//     int * s_col_idx = col_index;
//     float4 * s_a = smemA;
//     float4 * s_b = smemB;// + 2 * s_a_offset;
//     half * s_a_ = reinterpret_cast<half *>(s_a);
//     half * s_b_ = reinterpret_cast<half *>(s_b);
//     wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::col_major> frag_a[2];
//     wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b[2][2];
//     wmma::fragment<wmma::accumulator, 8, 32, 16, OutType> frag_c[2];
//     wmma::fill_fragment(frag_c[0], 0.0);
//     wmma::fill_fragment(frag_c[1], 0.0);

//     int load_a_smem_m = 0;
//     int load_a_smem_k = tid;
//     int load_b_smem_k1 = t_group * 4 + 0;
//     int load_b_smem_k2 = t_group * 4 + 1;
//     int load_b_smem_k3 = t_group * 4 + 2;
//     int load_b_smem_k4 = t_group * 4 + 3;
//     int load_b_smem_n = lane_id * 2;//0 1 2 3

//     int load_a_gmem_m = m_vec * BM + load_a_smem_m;
//     int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k*vec_len, K);

//     float4 * s_sparse_ = s_a + load_a_smem_k;
//     float4 * s_dense_1  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k1 * stride + load_b_smem_n;
//     float4 * s_dense_2  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k2 * stride + load_b_smem_n;
//     float4 * s_dense_3  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k3 * stride + load_b_smem_n;
//     float4 * s_dense_4  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k4 * stride + load_b_smem_n;

// 	int * s_column_idx_ = reinterpret_cast<int *>(s_col_idx) + tid;
// 	const int * g_column_idx_ = reinterpret_cast<const int *>(column_indices) + row_offset_vec + tid;

//     const float4 * g_a_ = reinterpret_cast<const float4 *>(a + load_a_gmem_addr);
// 	const half kZeroValues[8] = {};
//     if(tid < nonzeros){
//         *(s_column_idx_) = __ldg(g_column_idx_);
//         *(s_sparse_) = __ldg(g_a_);
//     }
//     else{
//         *(s_column_idx_) = 1;
//         *(s_sparse_) = reinterpret_cast<const float4*>(kZeroValues)[0];
//     }
// 	// __syncthreads();

//     nonzeros -= BK;
//     g_column_idx_ += BK;
//     g_a_ += BK;
//     int load_b_gmem_n = by * BN + load_b_smem_n * 16;
//     int row_idx1 = *(s_col_idx + t_group * 4 + 0);//col_index[t_group * 4 + 0];
//     int row_idx2 = *(s_col_idx + t_group * 4 + 1);//col_index[t_group * 4 + 1];
//     int row_idx3 = *(s_col_idx + t_group * 4 + 2);//col_index[t_group * 4 + 2];
//     int row_idx4 = *(s_col_idx + t_group * 4 + 3);//col_index[t_group * 4 + 3];
// 	// __syncthreads();
// 	int load_b_gmem_addr1 = row_idx1 * N + load_b_gmem_n;
//     int load_b_gmem_addr2 = row_idx2 * N + load_b_gmem_n;
//     int load_b_gmem_addr3 = row_idx3 * N + load_b_gmem_n;
//     int load_b_gmem_addr4 = row_idx4 * N + load_b_gmem_n;
// 	// if(load_b_gmem_addr1 >= K * N || load_b_gmem_addr2 >= K * N || load_b_gmem_addr3 >= K * N || load_b_gmem_addr4 >= K * N){
// 	// 	printf("error is here\n");
// 	// }
	
//     const float4 * g_b_1 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr1);
//     const float4 * g_b_2 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr2);
//     const float4 * g_b_3 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr3);
//     const float4 * g_b_4 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr4);
	
//     *(s_dense_1) = __ldg(g_b_1);
//     *(s_dense_2) = __ldg(g_b_2);
//     *(s_dense_3) = __ldg(g_b_3);
//     *(s_dense_4) = __ldg(g_b_4);
	
//     *(s_dense_1+1) = __ldg(g_b_1+1);
//     *(s_dense_2+1) = __ldg(g_b_2+1);
//     *(s_dense_3+1) = __ldg(g_b_3+1);
//     *(s_dense_4+1) = __ldg(g_b_4+1);
// 	// __syncthreads();
// 	// __threadfence_block();
//     int smem_sel = 0;
//     int smem_sel_next = 1;
// 	// if(tid == 0 && m_vec == 0) printf("here is ok\n");
//     #pragma unroll 8
//     for(; nonzeros > 0; nonzeros -= BK){
//         if(tid < nonzeros){
//             *(s_column_idx_ + smem_sel_next * s_a_offset) = __ldg(g_column_idx_);
//             *(s_sparse_ + smem_sel_next * s_a_offset) = __ldg(g_a_);
//         }
//         else{
//             *(s_column_idx_ + smem_sel_next * s_a_offset) = 0;
// 			// const half kZeroValues[8] = {};	
//             *(s_sparse_ + smem_sel_next * s_a_offset) = reinterpret_cast<const float4*>(kZeroValues)[0];
//         }
// 		// __syncthreads();
// 		// __threadfence_block();
//         g_column_idx_ += BK;
//         g_a_ += BK;

//         row_idx1 = *(s_col_idx + t_group * 4 + 0 + smem_sel_next * BK);//col_index[t_group * 4 + 0 + smem_sel_next * BK];
//         row_idx2 = *(s_col_idx + t_group * 4 + 1 + smem_sel_next * BK);//col_index[t_group * 4 + 1 + smem_sel_next * BK];
//         row_idx3 = *(s_col_idx + t_group * 4 + 2 + smem_sel_next * BK);//col_index[t_group * 4 + 2 + smem_sel_next * BK];
//         row_idx4 = *(s_col_idx + t_group * 4 + 3 + smem_sel_next * BK);//col_index[t_group * 4 + 3 + smem_sel_next * BK];
// 		// __syncthreads();
		
//         load_b_gmem_addr1 = row_idx1 * N + load_b_gmem_n;
//         load_b_gmem_addr2 = row_idx2 * N + load_b_gmem_n;
//         load_b_gmem_addr3 = row_idx3 * N + load_b_gmem_n;
//         load_b_gmem_addr4 = row_idx4 * N + load_b_gmem_n;
//         g_b_1 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr1);
//         g_b_2 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr2);
//         g_b_3 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr3);
//         g_b_4 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr4);
//         *(s_dense_1   + smem_sel_next * s_b_offset) = __ldg(g_b_1  );
//         *(s_dense_2   + smem_sel_next * s_b_offset) = __ldg(g_b_2  );
//         *(s_dense_3   + smem_sel_next * s_b_offset) = __ldg(g_b_3  );
//         *(s_dense_4   + smem_sel_next * s_b_offset) = __ldg(g_b_4  );
//         *(s_dense_1+1 + smem_sel_next * s_b_offset) = __ldg(g_b_1+1);
//         *(s_dense_2+1 + smem_sel_next * s_b_offset) = __ldg(g_b_2+1);
//         *(s_dense_3+1 + smem_sel_next * s_b_offset) = __ldg(g_b_3+1);
//         *(s_dense_4+1 + smem_sel_next * s_b_offset) = __ldg(g_b_4+1);
// 		// __threadfence_block();
		
//         wmma::load_matrix_sync(frag_a[0]   , s_a_              + smem_sel*BK*BM, BM);
//         wmma::load_matrix_sync(frag_a[1]   , s_a_ + vec_len*16 + smem_sel*BK*BM, BM);
//         wmma::load_matrix_sync(frag_b[0][0], s_b_                     + smem_sel*BK*(BN+BPAD), BN+BPAD);
//         wmma::load_matrix_sync(frag_b[0][1], s_b_                + 32 + smem_sel*BK*(BN+BPAD), BN+BPAD);
//         wmma::load_matrix_sync(frag_b[1][0], s_b_ + 16*(BN+BPAD)      + smem_sel*BK*(BN+BPAD), BN+BPAD);
//         wmma::load_matrix_sync(frag_b[1][1], s_b_ + 16*(BN+BPAD) + 32 + smem_sel*BK*(BN+BPAD), BN+BPAD);

//         wmma::mma_sync(frag_c[0], frag_a[0], frag_b[0][0], frag_c[0]);
//         wmma::mma_sync(frag_c[0], frag_a[1], frag_b[1][0], frag_c[0]);
//         wmma::mma_sync(frag_c[1], frag_a[0], frag_b[0][1], frag_c[1]);
//         wmma::mma_sync(frag_c[1], frag_a[1], frag_b[1][1], frag_c[1]);

//         // compute_nnz -= BK;
//         smem_sel ^= 1;
//         smem_sel_next ^= 1;
// 		// if(compute_nnz  <= BK) break;
//     }
// 	// if(tid == 0 && m_vec == 0) printf("here is ok\n");

//     wmma::load_matrix_sync(frag_a[0]   , s_a_              + smem_sel*BK*BM, BM);
//     wmma::load_matrix_sync(frag_a[1]   , s_a_ + vec_len*16 + smem_sel*BK*BM, BM);
//     wmma::load_matrix_sync(frag_b[0][0], s_b_                     + smem_sel*BK*(BN+BPAD), BN+BPAD);
//     wmma::load_matrix_sync(frag_b[0][1], s_b_                + 32 + smem_sel*BK*(BN+BPAD), BN+BPAD);
//     wmma::load_matrix_sync(frag_b[1][0], s_b_ + 16*(BN+BPAD)      + smem_sel*BK*(BN+BPAD), BN+BPAD);
//     wmma::load_matrix_sync(frag_b[1][1], s_b_ + 16*(BN+BPAD) + 32 + smem_sel*BK*(BN+BPAD), BN+BPAD);

//     wmma::mma_sync(frag_c[0], frag_a[0], frag_b[0][0], frag_c[0]);
//     wmma::mma_sync(frag_c[0], frag_a[1], frag_b[1][0], frag_c[0]);
//     wmma::mma_sync(frag_c[1], frag_a[0], frag_b[0][1], frag_c[1]);
//     wmma::mma_sync(frag_c[1], frag_a[1], frag_b[1][1], frag_c[1]);

//     int store_c_gmem_m = m_vec * BM;
//     int store_c_gmem_n = by * BN;
//     int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
//     wmma::store_matrix_sync(&c[store_c_gmem_addr     ], frag_c[0], N, wmma::mem_row_major);
//     wmma::store_matrix_sync(&c[store_c_gmem_addr + 32], frag_c[1], N, wmma::mem_row_major);
// }

template <typename OutType>
__global__ void wmma_spmm_v8(
    const int M, const int K, const int N,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half * __restrict__ a,
    const half * __restrict__ b,
    OutType * __restrict__ c)
{
    const int BM = 8;
    const int BK = 32;
    const int BN = 64;
    const int vec_len = 8;
    int m_vec = blockIdx.x;
    int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane_id = threadIdx.x % 8;
    const int t_group = threadIdx.x / 8;
    if (m_vec >= M || by >= N/BN) return;

	// Load the row offset and calculate the number of nonzeros in the row
	int row_offset_vec = __ldg(row_offsets + m_vec);///row offset 0
	int nonzeros = __ldg(row_offsets + m_vec + 1) - row_offset_vec;///nnz in this row
    // if (tid == 0) {
	// 	printf("m_vec: %d, by: %d, row_offset_vec: %d, nonzeros: %d\n", m_vec, by, row_offset_vec, nonzeros);
	// }
	
    // int compute_nnz = nonzeros;
    const int BPAD = 0;
    const int stride = 8;//9 = (BN + BPAD) / 8 = (64+8)/8
    __shared__ __align__(16) int col_index[BK*2];
    __shared__ __align__(16) float4 smemA[BK * 2];
	__shared__ __align__(16) float4 smemB[BK * stride * 2];
    int s_a_offset = BK;
    int s_b_offset = BK * stride;

    int * s_col_idx = col_index;
    float4 * s_a = smemA;
    float4 * s_b = smemB;// + 2 * s_a_offset;
    half * s_a_ = reinterpret_cast<half *>(s_a);
    half * s_b_ = reinterpret_cast<half *>(s_b);
	wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::col_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b[2][2];
    wmma::fragment<wmma::accumulator, 8, 32, 16, OutType> frag_c[2];
    wmma::fill_fragment(frag_c[0], 0.0);
    wmma::fill_fragment(frag_c[1], 0.0);

    // int load_a_smem_m = 0;
    int load_a_smem_k = tid;
    int load_b_smem_k1 = t_group * 8 + 0;
    int load_b_smem_k2 = t_group * 8 + 1;
    int load_b_smem_k3 = t_group * 8 + 2;
    int load_b_smem_k4 = t_group * 8 + 3;
	int load_b_smem_k5 = t_group * 8 + 4;
    int load_b_smem_k6 = t_group * 8 + 5;
    int load_b_smem_k7 = t_group * 8 + 6;
    int load_b_smem_k8 = t_group * 8 + 7;
    int load_b_smem_n = lane_id;//0 1 2 3 4 5 6 7

    // int load_a_gmem_m = m_vec * BM + load_a_smem_m;
    // int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k*vec_len, K);
	int load_a_gmem_addr = vec_len * row_offset_vec + load_a_smem_k * vec_len;

    float4 * s_sparse_ = s_a + load_a_smem_k;
    float4 * s_dense_1  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k1 * stride + load_b_smem_n;
    float4 * s_dense_2  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k2 * stride + load_b_smem_n;
    float4 * s_dense_3  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k3 * stride + load_b_smem_n;
    float4 * s_dense_4  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k4 * stride + load_b_smem_n;
    float4 * s_dense_5  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k5 * stride + load_b_smem_n;
    float4 * s_dense_6  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k6 * stride + load_b_smem_n;
    float4 * s_dense_7  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k7 * stride + load_b_smem_n;
    float4 * s_dense_8  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k8 * stride + load_b_smem_n;

	int * s_column_idx_ = reinterpret_cast<int *>(s_col_idx) + tid;
	const int * g_column_idx_ = reinterpret_cast<const int *>(column_indices) + row_offset_vec + tid;

    const float4 * g_a_ = reinterpret_cast<const float4 *>(a + load_a_gmem_addr);
	const half kZeroValues[8] = {};
    if(tid < nonzeros){
        *(s_column_idx_) = __ldg(g_column_idx_);
        *(s_sparse_) = __ldg(g_a_);
    }
    else{
        *(s_column_idx_) = int(0);
        *(s_sparse_) = reinterpret_cast<const float4*>(kZeroValues)[0];
    }
	// __syncthreads();

    nonzeros -= BK;
    g_column_idx_ += BK;
    g_a_ += BK;
    int load_b_gmem_n = by * BN + load_b_smem_n * 8;
    int row_idx1 = *(s_col_idx + t_group * 8 + 0);//col_index[t_group * 4 + 0];
    int row_idx2 = *(s_col_idx + t_group * 8 + 1);//col_index[t_group * 4 + 1];
    int row_idx3 = *(s_col_idx + t_group * 8 + 2);//col_index[t_group * 4 + 2];
    int row_idx4 = *(s_col_idx + t_group * 8 + 3);//col_index[t_group * 4 + 3];
	int row_idx5 = *(s_col_idx + t_group * 8 + 4);//col_index[t_group * 4 + 4];
    int row_idx6 = *(s_col_idx + t_group * 8 + 5);//col_index[t_group * 4 + 5];
    int row_idx7 = *(s_col_idx + t_group * 8 + 6);//col_index[t_group * 4 + 6];
    int row_idx8 = *(s_col_idx + t_group * 8 + 7);//col_index[t_group * 4 + 7];

	// __syncthreads();
	int load_b_gmem_addr1 = row_idx1 * N + load_b_gmem_n;
    int load_b_gmem_addr2 = row_idx2 * N + load_b_gmem_n;
    int load_b_gmem_addr3 = row_idx3 * N + load_b_gmem_n;
    int load_b_gmem_addr4 = row_idx4 * N + load_b_gmem_n;
	int load_b_gmem_addr5 = row_idx5 * N + load_b_gmem_n;
    int load_b_gmem_addr6 = row_idx6 * N + load_b_gmem_n;
    int load_b_gmem_addr7 = row_idx7 * N + load_b_gmem_n;
    int load_b_gmem_addr8 = row_idx8 * N + load_b_gmem_n;

	// if(load_b_gmem_addr1 >= K * N || load_b_gmem_addr2 >= K * N || load_b_gmem_addr3 >= K * N || load_b_gmem_addr4 >= K * N){
	// 	printf("error is here\n");
	// }
	
    const float4 * g_b_1 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr1);
    const float4 * g_b_2 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr2);
    const float4 * g_b_3 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr3);
    const float4 * g_b_4 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr4);
	const float4 * g_b_5 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr5);
    const float4 * g_b_6 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr6);
    const float4 * g_b_7 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr7);
    const float4 * g_b_8 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr8);
	
    *(s_dense_1) = __ldg(g_b_1);
    *(s_dense_2) = __ldg(g_b_2);
    *(s_dense_3) = __ldg(g_b_3);
    *(s_dense_4) = __ldg(g_b_4);
	*(s_dense_5) = __ldg(g_b_5);
    *(s_dense_6) = __ldg(g_b_6);
    *(s_dense_7) = __ldg(g_b_7);
    *(s_dense_8) = __ldg(g_b_8);

	// __syncthreads();
	// __threadfence_block();
    int smem_sel = 0;
    int smem_sel_next = 1;
	// if(tid == 0 && m_vec == 0) printf("here is ok\n");
    #pragma unroll 8
    for(; nonzeros > 0; nonzeros -= BK){
        if(tid < nonzeros){
            *(s_column_idx_ + smem_sel_next * BK) = __ldg(g_column_idx_);
            *(s_sparse_     + smem_sel_next * s_a_offset) = __ldg(g_a_);
        }
        else{
            *(s_column_idx_ + smem_sel_next * BK) = int(0);
			// const half kZeroValues[8] = {};	
            *(s_sparse_     + smem_sel_next * s_a_offset) = reinterpret_cast<const float4*>(kZeroValues)[0];
        }
		// __syncthreads();
		// __threadfence_block();
        g_column_idx_ += BK;
        g_a_ += BK;

        row_idx1 = *(s_col_idx + t_group * 8 + 0 + smem_sel_next * BK);//col_index[t_group * 4 + 0 + smem_sel_next * BK];
        row_idx2 = *(s_col_idx + t_group * 8 + 1 + smem_sel_next * BK);//col_index[t_group * 4 + 1 + smem_sel_next * BK];
        row_idx3 = *(s_col_idx + t_group * 8 + 2 + smem_sel_next * BK);//col_index[t_group * 4 + 2 + smem_sel_next * BK];
        row_idx4 = *(s_col_idx + t_group * 8 + 3 + smem_sel_next * BK);//col_index[t_group * 4 + 3 + smem_sel_next * BK];
		row_idx5 = *(s_col_idx + t_group * 8 + 4 + smem_sel_next * BK);//col_index[t_group * 4 + 0 + smem_sel_next * BK];
        row_idx6 = *(s_col_idx + t_group * 8 + 5 + smem_sel_next * BK);//col_index[t_group * 4 + 1 + smem_sel_next * BK];
        row_idx7 = *(s_col_idx + t_group * 8 + 6 + smem_sel_next * BK);//col_index[t_group * 4 + 2 + smem_sel_next * BK];
        row_idx8 = *(s_col_idx + t_group * 8 + 7 + smem_sel_next * BK);//col_index[t_group * 4 + 3 + smem_sel_next * BK];

		// __syncthreads();
		
        load_b_gmem_addr1 = row_idx1 * N + load_b_gmem_n;
        load_b_gmem_addr2 = row_idx2 * N + load_b_gmem_n;
        load_b_gmem_addr3 = row_idx3 * N + load_b_gmem_n;
        load_b_gmem_addr4 = row_idx4 * N + load_b_gmem_n;
		load_b_gmem_addr5 = row_idx5 * N + load_b_gmem_n;
        load_b_gmem_addr6 = row_idx6 * N + load_b_gmem_n;
        load_b_gmem_addr7 = row_idx7 * N + load_b_gmem_n;
        load_b_gmem_addr8 = row_idx8 * N + load_b_gmem_n;

        g_b_1 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr1);
        g_b_2 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr2);
        g_b_3 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr3);
        g_b_4 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr4);
		g_b_5 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr5);
        g_b_6 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr6);
        g_b_7 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr7);
        g_b_8 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr8);

        *(s_dense_1   + smem_sel_next * s_b_offset) = __ldg(g_b_1  );
        *(s_dense_2   + smem_sel_next * s_b_offset) = __ldg(g_b_2  );
        *(s_dense_3   + smem_sel_next * s_b_offset) = __ldg(g_b_3  );
        *(s_dense_4   + smem_sel_next * s_b_offset) = __ldg(g_b_4  );
        *(s_dense_5   + smem_sel_next * s_b_offset) = __ldg(g_b_5  );
        *(s_dense_6   + smem_sel_next * s_b_offset) = __ldg(g_b_6  );
        *(s_dense_7   + smem_sel_next * s_b_offset) = __ldg(g_b_7  );
        *(s_dense_8   + smem_sel_next * s_b_offset) = __ldg(g_b_8  );
		// __threadfence_block();
		
        wmma::load_matrix_sync(frag_a[0]   , s_a_              + smem_sel*BK*BM, BM);
        wmma::load_matrix_sync(frag_a[1]   , s_a_ + vec_len*16 + smem_sel*BK*BM, BM);
        wmma::load_matrix_sync(frag_b[0][0], s_b_                     + smem_sel*BK*(BN+BPAD), BN+BPAD);
        wmma::load_matrix_sync(frag_b[0][1], s_b_                + 32 + smem_sel*BK*(BN+BPAD), BN+BPAD);
        wmma::load_matrix_sync(frag_b[1][0], s_b_ + 16*(BN+BPAD)      + smem_sel*BK*(BN+BPAD), BN+BPAD);
        wmma::load_matrix_sync(frag_b[1][1], s_b_ + 16*(BN+BPAD) + 32 + smem_sel*BK*(BN+BPAD), BN+BPAD);

        wmma::mma_sync(frag_c[0], frag_a[0], frag_b[0][0], frag_c[0]);
        wmma::mma_sync(frag_c[0], frag_a[1], frag_b[1][0], frag_c[0]);
        wmma::mma_sync(frag_c[1], frag_a[0], frag_b[0][1], frag_c[1]);
        wmma::mma_sync(frag_c[1], frag_a[1], frag_b[1][1], frag_c[1]);

        // compute_nnz -= BK;
        smem_sel ^= 1;
        smem_sel_next ^= 1;
		// if(compute_nnz  <= BK) break;
    }
	// if(tid == 0 && m_vec == 0) printf("here is ok\n");

    wmma::load_matrix_sync(frag_a[0]   , s_a_              + smem_sel*BK*BM, BM);
    wmma::load_matrix_sync(frag_a[1]   , s_a_ + vec_len*16 + smem_sel*BK*BM, BM);
    wmma::load_matrix_sync(frag_b[0][0], s_b_                     + smem_sel*BK*(BN+BPAD), BN+BPAD);
    wmma::load_matrix_sync(frag_b[0][1], s_b_                + 32 + smem_sel*BK*(BN+BPAD), BN+BPAD);
    wmma::load_matrix_sync(frag_b[1][0], s_b_ + 16*(BN+BPAD)      + smem_sel*BK*(BN+BPAD), BN+BPAD);
    wmma::load_matrix_sync(frag_b[1][1], s_b_ + 16*(BN+BPAD) + 32 + smem_sel*BK*(BN+BPAD), BN+BPAD);

    wmma::mma_sync(frag_c[0], frag_a[0], frag_b[0][0], frag_c[0]);
    wmma::mma_sync(frag_c[0], frag_a[1], frag_b[1][0], frag_c[0]);
    wmma::mma_sync(frag_c[1], frag_a[0], frag_b[0][1], frag_c[1]);
    wmma::mma_sync(frag_c[1], frag_a[1], frag_b[1][1], frag_c[1]);

    int store_c_gmem_m = m_vec * BM;
    int store_c_gmem_n = by * BN;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    wmma::store_matrix_sync(&c[store_c_gmem_addr     ], frag_c[0], N, wmma::mem_row_major);
    wmma::store_matrix_sync(&c[store_c_gmem_addr + 32], frag_c[1], N, wmma::mem_row_major);

}

template <typename OutType>
__global__ void wmma_spmm_v8x2(
    const int M, const int K, const int N,
    const int* __restrict__ row_offsets,
    const int* __restrict__ column_indices,
    const half * __restrict__ a,
    const half * __restrict__ b,
    OutType * __restrict__ c)
{
    const int BM = 16;
    const int BK = 32;
    const int BN = 64;
    const int vec_len = 16;
    int m_vec = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x % 8;
    int t_group = threadIdx.x / 8;
    if (m_vec >= M || by >= N/BN) return;

	// Load the row offset and calculate the number of nonzeros in the row
	int row_offset_vec = __ldg(row_offsets + m_vec);///row offset 0
	int nonzeros = __ldg(row_offsets + m_vec + 1) - row_offset_vec;///nnz in this row
    // int nnz_ld = nonzeros;
    int compute_nnz = nonzeros;
    const int BPAD = 0;
    const int stride = 8;//9 = (BN + BPAD) / 8 = (64+8)/8
    __shared__ int col_index[BK*2];
    __shared__ float4 smem[BK * 2 * 2 + BK * stride * 2];
    int s_a_offset = 2*BK;
    int s_b_offset = BK * stride;

    int * s_col_idx = col_index;
    float4 * s_a = smem;
    float4 * s_b = smem + 2 * s_a_offset;
    half * s_a_ = reinterpret_cast<half *>(s_a);
    half * s_b_ = reinterpret_cast<half *>(s_b);
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::col_major> frag_a[2][2];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b[2][2];
    wmma::fragment<wmma::accumulator, 8, 32, 16, OutType> frag_c[2][2];
    wmma::fill_fragment(frag_c[0][0], 0.0);
    wmma::fill_fragment(frag_c[0][1], 0.0);
    wmma::fill_fragment(frag_c[1][0], 0.0);
    wmma::fill_fragment(frag_c[1][1], 0.0);

    int load_a_smem_m = 0;
    int load_a_smem_k = tid;
	int load_b_smem_k1 = t_group * 8 + 0;
    int load_b_smem_k2 = t_group * 8 + 1;
    int load_b_smem_k3 = t_group * 8 + 2;
    int load_b_smem_k4 = t_group * 8 + 3;
	int load_b_smem_k5 = t_group * 8 + 4;
    int load_b_smem_k6 = t_group * 8 + 5;
    int load_b_smem_k7 = t_group * 8 + 6;
    int load_b_smem_k8 = t_group * 8 + 7;
    int load_b_smem_n = lane_id;

    // int load_a_gmem_m = m_vec * BM + load_a_smem_m;
    // int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k*vec_len, K);
	int load_a_gmem_addr = row_offset_vec * vec_len + load_a_smem_k * vec_len;

    float4 * s_sparse_ = s_a + 2*load_a_smem_k;
    float4 * s_dense_1  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k1 * stride + load_b_smem_n;
    float4 * s_dense_2  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k2 * stride + load_b_smem_n;
    float4 * s_dense_3  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k3 * stride + load_b_smem_n;
    float4 * s_dense_4  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k4 * stride + load_b_smem_n;
    float4 * s_dense_5  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k5 * stride + load_b_smem_n;
    float4 * s_dense_6  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k6 * stride + load_b_smem_n;
    float4 * s_dense_7  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k7 * stride + load_b_smem_n;
    float4 * s_dense_8  = reinterpret_cast<float4 *>(s_b) + load_b_smem_k8 * stride + load_b_smem_n;
    
	// int * s_column_idx_ = s_col_idx + tid;
    // const int * g_column_idx_ = column_indices + row_offset_vec + tid;
	int * s_column_idx_ = reinterpret_cast<int *>(s_col_idx) + tid;
	const int * g_column_idx_ = reinterpret_cast<const int *>(column_indices) + row_offset_vec + tid;

	const float4 * g_a_ = reinterpret_cast<const float4 *>(a + load_a_gmem_addr);
	const half kZeroValues[8] = {};
    if(tid < nonzeros){
        *(s_column_idx_) = __ldg(g_column_idx_);
        *(s_sparse_  ) = __ldg(g_a_  );
		*(s_sparse_+1) = __ldg(g_a_+1);
    }
    else{
        *(s_column_idx_) = int(0);
        *(s_sparse_  ) = reinterpret_cast<const float4*>(kZeroValues)[0];
		*(s_sparse_+1) = reinterpret_cast<const float4*>(kZeroValues)[0];
    }
    nonzeros -= BK;
    g_column_idx_ += BK;
    g_a_ += 2*BK;
    int load_b_gmem_n = by * BN + load_b_smem_n * 8;
    int row_idx1 = *(s_col_idx + t_group * 8 + 0);//col_index[t_group * 4 + 0];
    int row_idx2 = *(s_col_idx + t_group * 8 + 1);//col_index[t_group * 4 + 1];
    int row_idx3 = *(s_col_idx + t_group * 8 + 2);//col_index[t_group * 4 + 2];
    int row_idx4 = *(s_col_idx + t_group * 8 + 3);//col_index[t_group * 4 + 3];
	int row_idx5 = *(s_col_idx + t_group * 8 + 4);//col_index[t_group * 4 + 4];
    int row_idx6 = *(s_col_idx + t_group * 8 + 5);//col_index[t_group * 4 + 5];
    int row_idx7 = *(s_col_idx + t_group * 8 + 6);//col_index[t_group * 4 + 6];
    int row_idx8 = *(s_col_idx + t_group * 8 + 7);//col_index[t_group * 4 + 7];

	int load_b_gmem_addr1 = row_idx1 * N + load_b_gmem_n;
    int load_b_gmem_addr2 = row_idx2 * N + load_b_gmem_n;
    int load_b_gmem_addr3 = row_idx3 * N + load_b_gmem_n;
    int load_b_gmem_addr4 = row_idx4 * N + load_b_gmem_n;
	int load_b_gmem_addr5 = row_idx5 * N + load_b_gmem_n;
    int load_b_gmem_addr6 = row_idx6 * N + load_b_gmem_n;
    int load_b_gmem_addr7 = row_idx7 * N + load_b_gmem_n;
    int load_b_gmem_addr8 = row_idx8 * N + load_b_gmem_n;

    const float4 * g_b_1 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr1);
    const float4 * g_b_2 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr2);
    const float4 * g_b_3 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr3);
    const float4 * g_b_4 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr4);
	const float4 * g_b_5 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr5);
    const float4 * g_b_6 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr6);
    const float4 * g_b_7 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr7);
    const float4 * g_b_8 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr8);
    
    *(s_dense_1) = __ldg(g_b_1);
    *(s_dense_2) = __ldg(g_b_2);
    *(s_dense_3) = __ldg(g_b_3);
    *(s_dense_4) = __ldg(g_b_4);
	*(s_dense_5) = __ldg(g_b_5);
    *(s_dense_6) = __ldg(g_b_6);
    *(s_dense_7) = __ldg(g_b_7);
    *(s_dense_8) = __ldg(g_b_8);

    int smem_sel = 0;
    int smem_sel_next = 1;
    #pragma unroll 8
    for(; nonzeros > 0; nonzeros -= BK){
        // *(s_column_idx_ + smem_sel_next * s_a_offset) = __ldg(g_column_idx_);
        // *(s_sparse_ + smem_sel_next * s_a_offset) = __ldg(g_a_);
        if(tid < nonzeros){
            *(s_column_idx_ + smem_sel_next * BK) = __ldg(g_column_idx_);
            *(s_sparse_     + smem_sel_next * s_a_offset) = __ldg(g_a_  );
			*(s_sparse_ + 1 + smem_sel_next * s_a_offset) = __ldg(g_a_+1);
        }
        else{
            *(s_column_idx_ + smem_sel_next * BK) = 0;
            *(s_sparse_     + smem_sel_next * s_a_offset) = reinterpret_cast<const float4*>(kZeroValues)[0];
			*(s_sparse_ + 1 + smem_sel_next * s_a_offset) = reinterpret_cast<const float4*>(kZeroValues)[0];
        }
        g_column_idx_ += BK;
        g_a_ += 2*BK;
        
        row_idx1 = *(s_col_idx + t_group * 8 + 0 + smem_sel_next * BK);//col_index[t_group * 4 + 0 + smem_sel_next * BK];
        row_idx2 = *(s_col_idx + t_group * 8 + 1 + smem_sel_next * BK);//col_index[t_group * 4 + 1 + smem_sel_next * BK];
        row_idx3 = *(s_col_idx + t_group * 8 + 2 + smem_sel_next * BK);//col_index[t_group * 4 + 2 + smem_sel_next * BK];
        row_idx4 = *(s_col_idx + t_group * 8 + 3 + smem_sel_next * BK);//col_index[t_group * 4 + 3 + smem_sel_next * BK];
		row_idx5 = *(s_col_idx + t_group * 8 + 4 + smem_sel_next * BK);//col_index[t_group * 4 + 0 + smem_sel_next * BK];
        row_idx6 = *(s_col_idx + t_group * 8 + 5 + smem_sel_next * BK);//col_index[t_group * 4 + 1 + smem_sel_next * BK];
        row_idx7 = *(s_col_idx + t_group * 8 + 6 + smem_sel_next * BK);//col_index[t_group * 4 + 2 + smem_sel_next * BK];
        row_idx8 = *(s_col_idx + t_group * 8 + 7 + smem_sel_next * BK);//col_index[t_group * 4 + 3 + smem_sel_next * BK];
        load_b_gmem_addr1 = row_idx1 * N + load_b_gmem_n;
        load_b_gmem_addr2 = row_idx2 * N + load_b_gmem_n;
        load_b_gmem_addr3 = row_idx3 * N + load_b_gmem_n;
        load_b_gmem_addr4 = row_idx4 * N + load_b_gmem_n;
		load_b_gmem_addr5 = row_idx5 * N + load_b_gmem_n;
        load_b_gmem_addr6 = row_idx6 * N + load_b_gmem_n;
        load_b_gmem_addr7 = row_idx7 * N + load_b_gmem_n;
        load_b_gmem_addr8 = row_idx8 * N + load_b_gmem_n;
        g_b_1 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr1);
        g_b_2 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr2);
        g_b_3 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr3);
        g_b_4 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr4);
		g_b_5 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr5);
        g_b_6 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr6);
        g_b_7 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr7);
        g_b_8 = reinterpret_cast<const float4 *>(b + load_b_gmem_addr8);
        *(s_dense_1   + smem_sel_next * s_b_offset) = __ldg(g_b_1  );
        *(s_dense_2   + smem_sel_next * s_b_offset) = __ldg(g_b_2  );
        *(s_dense_3   + smem_sel_next * s_b_offset) = __ldg(g_b_3  );
        *(s_dense_4   + smem_sel_next * s_b_offset) = __ldg(g_b_4  );
        *(s_dense_5   + smem_sel_next * s_b_offset) = __ldg(g_b_5  );
        *(s_dense_6   + smem_sel_next * s_b_offset) = __ldg(g_b_6  );
        *(s_dense_7   + smem_sel_next * s_b_offset) = __ldg(g_b_7  );
        *(s_dense_8   + smem_sel_next * s_b_offset) = __ldg(g_b_8  );
        
        

        wmma::load_matrix_sync(frag_a[0][0], s_a_                  + smem_sel*BK*BM, BM);
		wmma::load_matrix_sync(frag_a[0][1], s_a_              + 8 + smem_sel*BK*BM, BM);
        wmma::load_matrix_sync(frag_a[1][0], s_a_ + vec_len*16     + smem_sel*BK*BM, BM);
        wmma::load_matrix_sync(frag_a[1][1], s_a_ + vec_len*16 + 8 + smem_sel*BK*BM, BM);
        wmma::load_matrix_sync(frag_b[0][0], s_b_                     + smem_sel*BK*(BN+BPAD), BN+BPAD);
        wmma::load_matrix_sync(frag_b[0][1], s_b_                + 32 + smem_sel*BK*(BN+BPAD), BN+BPAD);
        wmma::load_matrix_sync(frag_b[1][0], s_b_ + 16*(BN+BPAD)      + smem_sel*BK*(BN+BPAD), BN+BPAD);
        wmma::load_matrix_sync(frag_b[1][1], s_b_ + 16*(BN+BPAD) + 32 + smem_sel*BK*(BN+BPAD), BN+BPAD);

        wmma::mma_sync(frag_c[0][0], frag_a[0][0], frag_b[0][0], frag_c[0][0]);
        wmma::mma_sync(frag_c[0][0], frag_a[1][0], frag_b[1][0], frag_c[0][0]);
        wmma::mma_sync(frag_c[0][1], frag_a[0][0], frag_b[0][1], frag_c[0][0]);
        wmma::mma_sync(frag_c[0][1], frag_a[1][0], frag_b[1][1], frag_c[0][0]);
        wmma::mma_sync(frag_c[1][0], frag_a[0][1], frag_b[0][0], frag_c[0][0]);
        wmma::mma_sync(frag_c[1][0], frag_a[1][1], frag_b[1][0], frag_c[0][0]);
        wmma::mma_sync(frag_c[1][1], frag_a[0][1], frag_b[0][1], frag_c[0][0]);
        wmma::mma_sync(frag_c[1][1], frag_a[1][1], frag_b[1][1], frag_c[0][0]);
        __syncthreads();
        // compute_nnz -= BK;
        smem_sel ^= 1;
        smem_sel_next ^= 1;
		// if(compute_nnz  <= BK) break;
    }
	wmma::load_matrix_sync(frag_a[0][0], s_a_                  + smem_sel*BK*BM, BM);
	wmma::load_matrix_sync(frag_a[0][1], s_a_              + 8 + smem_sel*BK*BM, BM);
	wmma::load_matrix_sync(frag_a[1][0], s_a_ + vec_len*16     + smem_sel*BK*BM, BM);
	wmma::load_matrix_sync(frag_a[1][1], s_a_ + vec_len*16 + 8 + smem_sel*BK*BM, BM);
	wmma::load_matrix_sync(frag_b[0][0], s_b_                     + smem_sel*BK*(BN+BPAD), BN+BPAD);
	wmma::load_matrix_sync(frag_b[0][1], s_b_                + 32 + smem_sel*BK*(BN+BPAD), BN+BPAD);
	wmma::load_matrix_sync(frag_b[1][0], s_b_ + 16*(BN+BPAD)      + smem_sel*BK*(BN+BPAD), BN+BPAD);
	wmma::load_matrix_sync(frag_b[1][1], s_b_ + 16*(BN+BPAD) + 32 + smem_sel*BK*(BN+BPAD), BN+BPAD);

	wmma::mma_sync(frag_c[0][0], frag_a[0][0], frag_b[0][0], frag_c[0][0]);
	wmma::mma_sync(frag_c[0][0], frag_a[1][0], frag_b[1][0], frag_c[0][0]);
	wmma::mma_sync(frag_c[0][1], frag_a[0][0], frag_b[0][1], frag_c[0][0]);
	wmma::mma_sync(frag_c[0][1], frag_a[1][0], frag_b[1][1], frag_c[0][0]);
	wmma::mma_sync(frag_c[1][0], frag_a[0][1], frag_b[0][0], frag_c[0][0]);
	wmma::mma_sync(frag_c[1][0], frag_a[1][1], frag_b[1][0], frag_c[0][0]);
	wmma::mma_sync(frag_c[1][1], frag_a[0][1], frag_b[0][1], frag_c[0][0]);
	wmma::mma_sync(frag_c[1][1], frag_a[1][1], frag_b[1][1], frag_c[0][0]);

    int store_c_gmem_m = m_vec * BM;
    int store_c_gmem_n = by * BN;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    wmma::store_matrix_sync(&c[store_c_gmem_addr             ], frag_c[0][0], N, wmma::mem_row_major);
    wmma::store_matrix_sync(&c[store_c_gmem_addr         + 32], frag_c[0][1], N, wmma::mem_row_major);
    wmma::store_matrix_sync(&c[store_c_gmem_addr + 8 * N     ], frag_c[1][0], N, wmma::mem_row_major);
    wmma::store_matrix_sync(&c[store_c_gmem_addr + 8 * N + 32], frag_c[1][1], N, wmma::mem_row_major);

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
		// cudaFuncSetAttribute(SpMM_1688_V8<float4, int, float4, float, float4, Tile_N, Tile_K, BlockWidth, 8>,
			// cudaFuncAttributePreferredSharedMemoryCarveout, 50);

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
		// cudaFuncSetAttribute(SpMM_1688_V8<float4, int, float4, float, float4, Tile_N, Tile_K, BlockWidth, 8>,
			// cudaFuncAttributePreferredSharedMemoryCarveout, 50);
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
	unsigned int dsmem = vec_length * Tile_K * 2 * sizeof(half);
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
		
		cudaFuncSetAttribute(SSpMM_V8<float4, int, float4, float, float4, Tile_K, Tile_N, BlockWidth, 8>,
			cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
		// cudaFuncSetAttribute(SSpMM_V8<float4, int, float4, float, float4, Tile_K, Tile_N, BlockWidth, 8>,
			// cudaFuncAttributePreferredSharedMemoryCarveout, 50);
		SSpMM_V8<float4, int, float4, float, float4, Tile_K, Tile_N, BlockWidth, 8><<<grid_dim, block_dim, dsmem>>>(
			m_vec, N, K, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	case 16:
		// unsigned int dsmem = vec_length * Tile_K * 2 * sizeof(half);
		cudaFuncSetAttribute(SSpMM_V16<float4, int, float4, float, float4, Tile_K, Tile_N, BlockWidth, 16>,
			cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
		SSpMM_V16<float4, int, float4, float, float4, Tile_K, Tile_N, BlockWidth, 16><<<grid_dim, block_dim, dsmem>>>(
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
	unsigned int dsmem = vec_length * Tile_K * 2 * sizeof(half);
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
		// cudaFuncSetAttribute(SSpMM_V8<float4, int, float4, half, float2, Tile_K, Tile_N, BlockWidth, 8>,
			// cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
		cudaFuncSetAttribute(SSpMM_V8<float4, int, float4, half, float2, Tile_K, Tile_N, BlockWidth, 8>,
			cudaFuncAttributePreferredSharedMemoryCarveout, 25);
		SSpMM_V8<float4, int, float4, half, float2, Tile_K, Tile_N, BlockWidth, 8><<<grid_dim, block_dim, dsmem>>>(
			m_vec, N, K, row_indices, row_offsets, column_indices, values, rhs_matrix, output_matrix);
		break;
	case 16:
		// cudaFuncSetAttribute(SSpMM_V8<float4, int, float4, half, float2, Tile_K, Tile_N, BlockWidth,16>,
			// cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
		cudaFuncSetAttribute(SSpMM_V16<float4, int, float4, half, float2, Tile_K, Tile_N, BlockWidth,16>,
			cudaFuncAttributePreferredSharedMemoryCarveout, 25);
		SSpMM_V16<float4, int, float4, half, float2, Tile_K, Tile_N, BlockWidth,16><<<grid_dim, block_dim, dsmem>>>(
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


cudaError_t WMMA_SpMM_ex(//fp16 * fp16 = fp16
	int m_vec, int vec_length, int N, int K,
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ a,
	const half* __restrict__ b,
	float* __restrict__ c)
{
	int BM = vec_length, BN = 64, BK = 32;
    int APAD = 8, BPAD = 8;
	dim3 grid_dim(m_vec, ceil(static_cast<float>(N) / BN), 1);
	dim3 block_dim(32);
	switch(vec_length){
	case 8:
	{
		unsigned int dsmem = (4 * (BM + APAD) * BK + 2 * BK * (BN + BPAD)) * sizeof(half);
        cudaFuncSetAttribute(wmma_spmm_v8<float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dsmem);
		wmma_spmm_v8<float><<<grid_dim, block_dim,dsmem>>>(
			m_vec, K, N, row_offsets, column_indices, a, b, c);
		break;
	}
	case 16:
	{
		unsigned int dsmem = (2 * (BM + APAD) * BK + 2 * BK * (BN + BPAD)) * sizeof(half);
        cudaFuncSetAttribute(wmma_spmm_v8x2<float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dsmem);
		wmma_spmm_v8x2<float><<<grid_dim, block_dim, dsmem>>>(
			m_vec, K, N, row_offsets, column_indices, a, b, c);
		break;
	}
	default:
		printf("Unsupported Vector Length!\n");
	}
	return cudaGetLastError();
}

cudaError_t WMMA_SpMM_ex(//fp16 * fp16 = fp16
	int m_vec, int vec_length, int N, int K,
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ a,
	const half* __restrict__ b,
	half* __restrict__ c)
{
	int BM = vec_length, BN = 64, BK = 32;
    int APAD = 8, BPAD = 8;
	dim3 grid_dim(m_vec, ceil(static_cast<float>(N) / BN), 1);
	dim3 block_dim(32);
	switch(vec_length){
	case 8:
	{
		unsigned int dsmem = (4 * (BM + APAD) * BK + 2 * BK * (BN + BPAD)) * sizeof(half);
        cudaFuncSetAttribute(wmma_spmm_v8<half>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dsmem);
		wmma_spmm_v8<half><<<grid_dim, block_dim, dsmem>>>(
			m_vec, K, N, row_offsets, column_indices, a, b, c);
		break;
	}
	case 16:
	{
		unsigned int dsmem = (2 * (BM + APAD) * BK + 2 * BK * (BN + BPAD)) * sizeof(half);
        cudaFuncSetAttribute(wmma_spmm_v8x2<half>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dsmem);
		wmma_spmm_v8x2<half><<<grid_dim, block_dim, dsmem>>>(
			m_vec, K, N, row_offsets, column_indices, a, b, c);
		break;
	}
	default:
		printf("Unsupported Vector Length!\n");
	}
	return cudaGetLastError();
}
// Function for half precision//fp16 * fp16 = fp16
cudaError_t WMMA_SpMM(
	int m_vec, int vec_length, int N, int K, 
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ a,
	const half* __restrict__ b,
	half* __restrict__ c)
{
	return WMMA_SpMM_ex(m_vec, vec_length, N, K, row_offsets, column_indices, a, b, c);
}
cudaError_t WMMA_SpMM(
	int m_vec, int vec_length, int N, int K, 
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const half* __restrict__ a,
	const half* __restrict__ b,
	float* __restrict__ c)
{
	// printf("wmmaSpmm doesn't support float input.\n");
	return WMMA_SpMM_ex(m_vec, vec_length, N, K, row_offsets, column_indices, a, b, c);
}
cudaError_t WMMA_SpMM(
	int m_vec, int vec_length, int N, int K, 
	const int* __restrict__ row_offsets,
	const int* __restrict__ column_indices,
	const float* __restrict__ a,
	const float* __restrict__ b,
	float* __restrict__ c)
{
	printf("wmmaSpmm doesn't support float input.\n");
	return cudaSuccess;
}
}