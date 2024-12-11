#ifndef SPMM_DENSE_TILE_H
#define SPMM_DENSE_TILE_H

#include <cuda_fp16.h>

namespace spmm {
    //Tile_N = 128 threads_per_block = 128
    template <typename LoadType, int Tile_K, int Tile_N>
    struct wmmaDenseTile_8b{

        const int rhs_cols_;
        const int lane_id_;
        const int ints_per_row_;
        const LoadType *matrix_base_;
        const int *row_offsets_base_;
        LoadType *dense_tile_;
        int *rhs_prefetch_;

        __device__ __forceinline__ wmmaDenseTile_8b(
	    int rhs_cols,
            int offset, 
            int lane_id, 
            const int* __restrict__ matrix, 
            const int *row_offsets,
            int * dense_tile,
            int * rhs_prefetch):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            ints_per_row_(Tile_N/8),
            //ints_per_row_(Tile_N/4),
            matrix_base_(reinterpret_cast<const LoadType *>(matrix + offset)),
            row_offsets_base_(row_offsets),
            dense_tile_(reinterpret_cast<LoadType *>(dense_tile)),
            rhs_prefetch_(rhs_prefetch){}
        

        __device__ __forceinline__ void LoadRowfromRegister(int step){
            for(int i=0; i<4; i++){
                //const int pad_offset = i;
                *(dense_tile_ + i*72 + lane_id_ % 64 + (lane_id_ / 64) * 288) = rhs_prefetch_[i];
            }
        }

        __device__ __forceinline__ void Prefetch(int step){
            const int *row_offsets = row_offsets_base_ + (lane_id_ % 64) / ints_per_row_ + (step % 2) * Tile_K;
            const int global_offset = lane_id_ % ints_per_row_ + (lane_id_ / 64) * ints_per_row_;
            for(int i=0; i<4; i++){
                rhs_prefetch_[i] = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + global_offset);
            }
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int residue){
            const int *row_offsets = row_offsets_base_ + (lane_id_ % 64) / ints_per_row_;
            const int global_offset = lane_id_ % ints_per_row_ + (lane_id_ / 64) * ints_per_row_;
            const int steps = residue / 4;
            const int res_residue = residue % 4;

	    int i = 0;
            for(; i<steps; i++){
                *(dense_tile_ + i*72 + lane_id_ % 64 + (lane_id_ / 64) * 288)  = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + global_offset);
            }

            if(res_residue > 0){
                if(*(row_offsets + i*4) >= 0)
                    *(dense_tile_ + i*72 + lane_id_ % 64 + (lane_id_ / 64) * 288)  = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + global_offset);
            }
            //if(residue >= Tile_K){
            //    for(int i=0; i<4; i++){
            //        *(dense_tile_ + i*72 + lane_id_) = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
            //    }
	    //}else{
            //    const int steps = residue / 4;
            //    const int res_residue = residue % 4;
	    //    int i = 0;
            //    for(; i<steps; i++){
            //        *(dense_tile_ + i*72 + lane_id_) = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
            //    }

            //    if(res_residue > 0){
            //        if (*(row_offsets + i*4) >= 0)
            //            *(dense_tile_ + i*72 + lane_id_) = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + bank_id);
	    //    }
	    //}
        }
    };

    //Tile_N = 64 threads_per_block = 128
    template <typename LoadType, int Tile_K, int Tile_N>
    struct wmmaDenseTile_16b{

        const int rhs_cols_;
        const int lane_id_;
        const int ints_per_row_;
        const LoadType *matrix_base_;
        const int *row_offsets_base_;
        LoadType *dense_tile_;
        int *rhs_prefetch_;

        __device__ __forceinline__ wmmaDenseTile_16b(
	    int rhs_cols,
            int offset, 
            int lane_id, 
            const int* __restrict__ matrix, 
            const int *row_offsets,
            int * dense_tile,
            int * rhs_prefetch):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            ints_per_row_(Tile_N/4), // for 2 warps
            matrix_base_(reinterpret_cast<const LoadType *>(matrix + offset)),
            row_offsets_base_(row_offsets),
            dense_tile_(reinterpret_cast<LoadType *>(dense_tile)),
            rhs_prefetch_(rhs_prefetch){}
        

        __device__ __forceinline__ void LoadRowfromRegister(int step){
            for(int i=0; i<4; i++){
                //const int pad_offset = i;
                *(dense_tile_ + i*72 + lane_id_ % 64 + (lane_id_ / 64) * 288) = rhs_prefetch_[i];
            }
        }

        __device__ __forceinline__ void Prefetch(int step){
            const int *row_offsets = row_offsets_base_ + (lane_id_ % 64) / ints_per_row_ + (step % 2) * Tile_K;
            const int global_offset = lane_id_ % ints_per_row_ + (lane_id_ / 64) * ints_per_row_;
            for(int i=0; i<4; i++){
                rhs_prefetch_[i] = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + global_offset);
            }
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int residue){
            const int *row_offsets = row_offsets_base_ + (lane_id_ % 64) / ints_per_row_;
            const int global_offset = lane_id_ % ints_per_row_ + (lane_id_ / 64) * ints_per_row_;
            const int steps = residue / 4;
            const int res_residue = residue % 4;

	    int i = 0;
            for(; i<steps; i++){
                *(dense_tile_ + i*72 + lane_id_ % 64 + (lane_id_ / 64) * 288)  = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + global_offset);
            }

            if(res_residue > 0){
                if(*(row_offsets + i*4) >= 0)
                    *(dense_tile_ + i*72 + lane_id_ % 64 + (lane_id_ / 64) * 288)  = __ldg(matrix_base_ + *(row_offsets + i*4)*rhs_cols_ + global_offset);
            }
        }
    };


}
#endif