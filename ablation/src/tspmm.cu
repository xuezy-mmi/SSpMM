#include "../include/tspmm.cuh"
#include <stdio.h>
#include <mma.h>
#include <float.h>
#include <cuda_runtime.h>
using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

namespace spmm{

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