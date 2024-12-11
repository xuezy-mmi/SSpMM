#ifndef SSPMM_SPARSE_TILE_H
#define SSPMM_SPARSE_TILE_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;
namespace spmm{
    __device__ __forceinline__ void Mul(int x1, int2 x2, int2 *out) {
        out[0].x = x1 * x2.x;
        out[0].y = x1 * x2.y;
    }

    __device__ __forceinline__ void Mul(int x1, int4 x2, int4 *out) {
        out[0].x = x1 * x2.x;
        out[0].y = x1 * x2.y;
        out[0].z = x1 * x2.z;
        out[0].w = x1 * x2.w;
    }

    __device__ __forceinline__ void Mul(int x1, int x2, int *out) {
        out[0] = x1 * x2;
    }

    template <typename LoadType, typename VecType, int VecLength, int Tile_N, int BlockWidth>
    struct wmmaSparseTile {
        //
        // Static members
        //
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);//one load_inst load x fp16
        // static constexpr int kThreadItemsN_ = Tile_N / BlockWidth;//32/32=1
        //
        // Member variables
        //
        // The number of columns in the rhs matrix
        const int rhs_columns_;
        // The sparse matrix value array.
        const VecType* values_;
        // The sparse matrix column indices for each value
        const int * column_idxs_;
        VecType* values_tile_base_;
        // shared memory tile for sparse marix values
        int *column_idxs_tile_base_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ wmmaSparseTile(
            int rhs_columns, int row_offset_vec, int thread_idx_x,
            const half* __restrict__ values,
            const int* __restrict__ column_idxs,
            half *values_tile, 
            int * column_idxs_tile):
            rhs_columns_(rhs_columns / kValuesPerLoad_),
            values_(reinterpret_cast<const VecType *>(values + VecLength * row_offset_vec + VecLength * thread_idx_x)),
            column_idxs_(reinterpret_cast<const int *>(column_idxs) + row_offset_vec + thread_idx_x),
            values_tile_base_(reinterpret_cast<VecType *>(values_tile) + thread_idx_x),
            column_idxs_tile_base_(reinterpret_cast<int *>(column_idxs_tile) + thread_idx_x){}

        // Load
        __device__ __forceinline__ void Load(){
            VecType *values_tile = values_tile_base_;//values_tile + tid
            int* column_idxs_tile = column_idxs_tile_base_;//column_idxs_tile + tid

			*(values_tile) = __ldg(values_);
			// *(column_idxs_tile) = __ldg(column_idxs_) * rhs_columns_;
			*(column_idxs_tile) = __ldg(column_idxs_);
			
			values_ += BlockWidth;
			column_idxs_ += BlockWidth;
        }

        // Zero Tile
        __device__ __forceinline__ void ZeroTiles(){
            VecType *values_tile = values_tile_base_;
            int *column_idxs_tile = column_idxs_tile_base_;

            const half kZeroValues[VecLength] = {};

			*(values_tile) = reinterpret_cast<const VecType*>(kZeroValues)[0];
			*(column_idxs_tile) = 0;
        }

        // Load Residual
        __device__ __forceinline__ void Residue(int residue){
            VecType* values_tile = values_tile_base_;
            int *column_idxs_tile = column_idxs_tile_base_;

			if (residue <= threadIdx.x) return;
			*(values_tile) = __ldg(values_);
			// *(column_idxs_tile) = __ldg(column_idxs_) * rhs_columns_;
			*(column_idxs_tile) = __ldg(column_idxs_);

			values_ += BlockWidth;
			column_idxs_ += BlockWidth;
			residue -= BlockWidth;
        }

    };



    template <typename LoadType, typename VecType, int VecLength, int Tile_N, int BlockWidth>
    struct wmmaSparseTileV16 {
        //
        // Static members
        //

        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);//one load_inst load x fp16
        // static constexpr int kThreadItemsN_ = Tile_N / BlockWidth;//32/32=1

        //
        // Member variables
        //

        // The number of columns in the rhs matrix
        const int rhs_columns_;
        // The sparse matrix value array.
        const VecType* values_;
        // The sparse matrix column indices for each value
        const int * column_idxs_;
        VecType* values_tile_base_;
        // shared memory tile for sparse marix values
        int *column_idxs_tile_base_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ wmmaSparseTileV16(
            int rhs_columns,
            int row_offset_vec,
            int thread_idx_x,
            const half * __restrict__ values,
            const int * __restrict__ column_idxs,
            half * values_tile,
            int * column_idxs_tile):
            rhs_columns_(rhs_columns / kValuesPerLoad_),
            values_(reinterpret_cast<const VecType *>(values + VecLength * row_offset_vec + VecLength * thread_idx_x)),
            column_idxs_(reinterpret_cast<const int *>(column_idxs + row_offset_vec + thread_idx_x)),
            values_tile_base_(reinterpret_cast<VecType *>(values_tile + VecLength * thread_idx_x)),
            column_idxs_tile_base_(reinterpret_cast<int *>(column_idxs_tile + thread_idx_x)){}

        // Load
        __device__ __forceinline__ void Load(){
            VecType *values_tile = values_tile_base_;//values_tile + tid
            int* column_idxs_tile = column_idxs_tile_base_;//column_idxs_tile + tid

			*(values_tile) = __ldg(values_);
			*(values_tile + 1) = __ldg(values_ + 1);
			// *(column_idxs_tile) = __ldg(column_idxs_) * rhs_columns_;
			*(column_idxs_tile) = __ldg(column_idxs_);
					
			values_ += 2 * BlockWidth;
			column_idxs_ += BlockWidth;
        }

        // Zero Tile
        __device__ __forceinline__ void ZeroTiles(){
            VecType *values_tile = values_tile_base_;
            int *column_idxs_tile = column_idxs_tile_base_;

            const half kZeroValues[VecLength] = {};

			*(values_tile) = reinterpret_cast<const VecType*>(kZeroValues)[0];
			*(values_tile + 1) = reinterpret_cast<const VecType*>(kZeroValues)[0];
			*(column_idxs_tile) = 0;
        }

        // Load Residual
        __device__ __forceinline__ void Residue(int residue){
            VecType* values_tile = values_tile_base_;
            int *column_idxs_tile = column_idxs_tile_base_;

			if (residue <= threadIdx.x) return;

			*(values_tile) = __ldg(values_);
			*(values_tile + 1) = __ldg(values_ + 1);
			// *(column_idxs_tile) = __ldg(column_idxs_) * rhs_columns_;
			*(column_idxs_tile) = __ldg(column_idxs_);

			values_ += 2 * BlockWidth;
			column_idxs_ += BlockWidth;
			residue -= BlockWidth;
        }

    };

    template <typename LoadType, typename VecType, int Tile_N, int Tile_K, int VecLength>
    struct mmaSparseTile_SSPMM {
        //
        // Static members
        //
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);//one load_inst load x fp16
        // static constexpr int kThreadItemsN_ = Tile_N / BlockWidth;//32/32=1
        //
        // Member variables
        //

        // The number of columns in the rhs matrix
        const int rhs_columns_;
        // The sparse matrix value array.
        const VecType* values_;
        // The sparse matrix column indices for each value
        const int * column_idxs_;
        VecType* values_tile_base_; 
        // shared memory tile for sparse marix values
        int *column_idxs_tile_base_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ mmaSparseTile_SSPMM(
            int rhs_columns,
            int row_offset_vec,
            int thread_idx_x,
            const half* __restrict__ values,//LHS matrix
            const int* __restrict__ column_idxs,
            half *values_tile, 
            int * column_idxs_tile):
            rhs_columns_(rhs_columns / kValuesPerLoad_),
            values_(reinterpret_cast<const VecType *>(values + VecLength * row_offset_vec + VecLength * thread_idx_x)),
            column_idxs_(reinterpret_cast<const int *>(column_idxs + row_offset_vec + thread_idx_x)),
            values_tile_base_(reinterpret_cast<VecType *>(values_tile + VecLength * thread_idx_x)),
            column_idxs_tile_base_(reinterpret_cast<int *>(column_idxs_tile) + thread_idx_x){}

        // Load
        __device__ __forceinline__ void Load(){
            VecType *values_tile = values_tile_base_;//values_tile + tid
            int* column_idxs_tile = column_idxs_tile_base_;//column_idxs_tile + tid

			*(values_tile) = __ldg(values_);
			*(column_idxs_tile) = __ldg(column_idxs_);
			
			values_ += 32;
			column_idxs_ += 32;
        }

        // Zero Tile
        __device__ __forceinline__ void ZeroTiles(){
            VecType *values_tile = values_tile_base_;
            int *column_idxs_tile = column_idxs_tile_base_;

            const half kZeroValues[VecLength] = {};

			*(values_tile) = reinterpret_cast<const VecType*>(kZeroValues)[0];
			*(column_idxs_tile) = 0;
        }

        // Load Residual
        __device__ __forceinline__ void Residue(int residue){
            VecType* values_tile = values_tile_base_;
            int* column_idxs_tile = column_idxs_tile_base_;
            const half kZeroValues[VecLength] = {};
			if (threadIdx.x < residue){
			    *(values_tile) = __ldg(values_);
			    *(column_idxs_tile) = __ldg(column_idxs_);
            }
            else{
                *(values_tile) = reinterpret_cast<const VecType*>(kZeroValues)[0];
			    *(column_idxs_tile) = 0;
            }
			values_ += 32;
			column_idxs_ += 32;
        }

    };

    template <typename LoadType, int Tile_N, int Tile_K, int VecLength>
    struct mmaSparseTile16_SSPMM {
        //
        // Static members
        //
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);//one load_inst load x fp16
        // static constexpr int kThreadItemsN_ = Tile_N / BlockWidth;//32/32=1
        //
        // Member variables
        //

        // The number of columns in the rhs matrix
        const int rhs_columns_;
        // The sparse matrix value array.
        const float4* values_;
        // The sparse matrix column indices for each value
        const int * column_idxs_;
        float4* values_tile_base_; 
        // shared memory tile for sparse marix values
        int *column_idxs_tile_base_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ mmaSparseTile16_SSPMM(
            int rhs_columns,
            int row_offset_vec,
            int thread_idx_x,
            const half* __restrict__ values,//LHS matrix
            const int* __restrict__ column_idxs,
            half *values_tile, 
            int * column_idxs_tile):
            rhs_columns_(rhs_columns / kValuesPerLoad_),
            values_(reinterpret_cast<const float4 *>(values + VecLength * row_offset_vec + VecLength * thread_idx_x)),
            column_idxs_(reinterpret_cast<const int *>(column_idxs + row_offset_vec + thread_idx_x)),
            values_tile_base_(reinterpret_cast<float4 *>(values_tile + VecLength * thread_idx_x)),
            column_idxs_tile_base_(reinterpret_cast<int *>(column_idxs_tile + thread_idx_x)){}

        // Load
        __device__ __forceinline__ void Load(){
            float4 *values_tile = values_tile_base_;//values_tile + tid
            int* column_idxs_tile = column_idxs_tile_base_;//column_idxs_tile + tid

			*(values_tile) = __ldg(values_);
            *(values_tile + 1) = __ldg(values_ + 1);

			*(column_idxs_tile) = __ldg(column_idxs_);
			
			values_ += 2*32;
			column_idxs_ += 32;
        }

        // Zero Tile
        __device__ __forceinline__ void ZeroTiles(){
            float4 *values_tile = values_tile_base_;
            int *column_idxs_tile = column_idxs_tile_base_;

            const half kZeroValues[VecLength] = {};

			*(values_tile  ) = reinterpret_cast<const float4*>(kZeroValues)[0];
            *(values_tile+1) = reinterpret_cast<const float4*>(kZeroValues)[0];
			*(column_idxs_tile) = 0;
        }

        // Load Residual
        __device__ __forceinline__ void Residue(int residue){
            float4* values_tile = values_tile_base_;
            int* column_idxs_tile = column_idxs_tile_base_;
            const half kZeroValues[VecLength] = {};
			if (residue <= threadIdx.x) return;
            if(threadIdx.x < residue){
			    *(values_tile) = __ldg(values_);
                *(values_tile+1) = __ldg(values_+1);
			    *(column_idxs_tile) = __ldg(column_idxs_);
            }
            else{
                *(values_tile) = reinterpret_cast<const float4*>(kZeroValues)[0];
                *(values_tile+1) = reinterpret_cast<const float4*>(kZeroValues)[0];
			    *(column_idxs_tile) = 0;
            }
			values_ += 2*32;
			column_idxs_ += 32;
        }

    };

}
#endif