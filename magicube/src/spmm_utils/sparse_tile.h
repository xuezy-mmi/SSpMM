#ifndef SPMM_SPARSE_TILE_H
#define SPMM_SPARSE_TILE_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

    //8-bit 4 warps
    template <typename LoadType, typename VecType, int ValuesBlockWidth, int BlockWidth>
    struct wmmaSparseTile_8b{

        const int in_warp_tid_;
        const int warp_id_;
        // The sparse matrix value array.
        const int * values_;
        // The sparse matrix column indices for each value
        const int * column_idxs_;
        int * values_tile_base_;
        // shared memory tile for sparse marix values
        int * column_idxs_tile_base_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ wmmaSparseTile_8b(
            int row_offset_vec, int in_warp_tid, int warp_id,
            const VecType * __restrict__ values,
            const int * __restrict__ column_idxs,
            int *values_tile, int * column_idxs_tile):
            in_warp_tid_(in_warp_tid),
            warp_id_(warp_id),
            values_(reinterpret_cast<const int *>(values + row_offset_vec) + in_warp_tid),
            column_idxs_(reinterpret_cast<const int *>(column_idxs + row_offset_vec) + in_warp_tid),
            values_tile_base_(reinterpret_cast<int *>(values_tile) + in_warp_tid),
            column_idxs_tile_base_(reinterpret_cast<int *>(column_idxs_tile) + in_warp_tid){}
        
        // Load
        __device__ __forceinline__ void Load(int step){
            int * values_tile = values_tile_base_ + (step % 2) * ValuesBlockWidth;
            int * column_idxs_tile = column_idxs_tile_base_ + (step % 2) * BlockWidth;

	    if(warp_id_ == 0 && in_warp_tid_ < ValuesBlockWidth)
                *(values_tile) = __ldg(values_);
	    else if(warp_id_ == 1 && in_warp_tid_ < BlockWidth)
                *(column_idxs_tile) = __ldg(column_idxs_);
            values_ += ValuesBlockWidth;
            column_idxs_ += BlockWidth;
        }

        // Load Residual
        __device__ __forceinline__ void Residue(){
            int * values_tile = values_tile_base_;
            int * column_idxs_tile = column_idxs_tile_base_;

	    if(warp_id_ == 0 && in_warp_tid_ < ValuesBlockWidth)
                *(values_tile) = __ldg(values_);
	    else if(warp_id_ == 1 && in_warp_tid_ < BlockWidth)
                *(column_idxs_tile) = __ldg(column_idxs_);
            asm(""); // without this, it is said that the loop cannot be unrolled.
        }
    };

    // 4 warps
    // TODO: Same as wmmaSparseTile_8b?
    template <typename LoadType, typename VecType, int ValuesBlockWidth, int BlockWidth>
    struct wmmaSparseTile_16b8b{

        const int in_warp_tid_;
        const int warp_id_;
        // The sparse matrix value array.
        const int * values_;
        // The sparse matrix column indices for each value
        const int * column_idxs_;
        int * values_tile_base_;
        // shared memory tile for sparse marix values
        int * column_idxs_tile_base_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ wmmaSparseTile_16b8b(
            int row_offset_vec, int in_warp_tid, int warp_id,
            const VecType * __restrict__ values,
            const int * __restrict__ column_idxs,
            int *values_tile, int * column_idxs_tile):
            in_warp_tid_(in_warp_tid),
            warp_id_(warp_id),
            values_(reinterpret_cast<const int *>(values + row_offset_vec) + in_warp_tid),
            column_idxs_(reinterpret_cast<const int *>(column_idxs + row_offset_vec) + in_warp_tid),
            values_tile_base_(reinterpret_cast<int *>(values_tile) + in_warp_tid),
            column_idxs_tile_base_(reinterpret_cast<int *>(column_idxs_tile) + in_warp_tid){}
        
        // Load
        __device__ __forceinline__ void Load(int step){
            int * values_tile = values_tile_base_ + (step % 2) * ValuesBlockWidth;
            int * column_idxs_tile = column_idxs_tile_base_ + (step % 2) * BlockWidth;

	    if(warp_id_ == 0 && in_warp_tid_ < ValuesBlockWidth)
                *(values_tile) = __ldg(values_);
	    else if(warp_id_ == 1 && in_warp_tid_ < BlockWidth)
                *(column_idxs_tile) = __ldg(column_idxs_);
            values_ += ValuesBlockWidth;
            column_idxs_ += BlockWidth;
        }

        // Load Residual
        __device__ __forceinline__ void Residue(){
            int * values_tile = values_tile_base_;
            int * column_idxs_tile = column_idxs_tile_base_;

	    if(warp_id_ == 0 && in_warp_tid_ < ValuesBlockWidth)
                *(values_tile) = __ldg(values_);
	    else if(warp_id_ == 1 && in_warp_tid_ < BlockWidth)
                *(column_idxs_tile) = __ldg(column_idxs_);
            asm(""); // without this, it is said that the loop cannot be unrolled.
        }
    };

    // 4 warps
    // TODO: Same as wmmaSparseTile_8b?
    template <typename LoadType, typename VecType, int ValuesBlockWidth, int BlockWidth>
    struct wmmaSparseTile_16b8b8v{

        const int lane_id_;
        // The sparse matrix value array.
        const int * values_;
        // The sparse matrix column indices for each value
        const int * column_idxs_;
        int * values_tile_base_;
        // shared memory tile for sparse marix values
        int * column_idxs_tile_base_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ wmmaSparseTile_16b8b8v(
            int row_offset_vec, int lane_id,
            const VecType * __restrict__ values,
            const int * __restrict__ column_idxs,
            int *values_tile, int * column_idxs_tile):
            lane_id_(lane_id),
            values_(reinterpret_cast<const int *>(values + row_offset_vec * 2) + lane_id), //scaleA = 2
            values_tile_base_(reinterpret_cast<int *>(values_tile) + lane_id),
            column_idxs_(reinterpret_cast<const int *>(column_idxs + row_offset_vec) + lane_id - ValuesBlockWidth),
            column_idxs_tile_base_(reinterpret_cast<int *>(column_idxs_tile) + lane_id - ValuesBlockWidth){}
        
        // Load
        __device__ __forceinline__ void Load(int step){
            int * values_tile = values_tile_base_ + (step % 2) * ValuesBlockWidth;
            int * column_idxs_tile = column_idxs_tile_base_ + (step % 2) * BlockWidth;

	    if(lane_id_ < ValuesBlockWidth)
                *(values_tile) = __ldg(values_);
	    else if((lane_id_ - ValuesBlockWidth) < BlockWidth)
                *(column_idxs_tile) = __ldg(column_idxs_);
            values_ += ValuesBlockWidth;
            column_idxs_ += BlockWidth;
        }

        // Load Residual
        __device__ __forceinline__ void Residue(){
            int * values_tile = values_tile_base_;
            int * column_idxs_tile = column_idxs_tile_base_;

	    if(lane_id_ < ValuesBlockWidth)
                *(values_tile) = __ldg(values_);
	    else if((lane_id_ - ValuesBlockWidth) < BlockWidth)
                *(column_idxs_tile) = __ldg(column_idxs_);
            asm(""); // without this, it is said that the loop cannot be unrolled.
        }
    };

}
#endif
