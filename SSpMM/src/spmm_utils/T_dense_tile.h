#ifndef T_DENSE_TILE_H
#define T_DENSE_TILE_H

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
namespace spmm {
    
    template <typename LoadType, int Tile_N, int Tile_K, int BlockWidth>
    struct wmmaDenseTile {
        //
        // Static members
        //

        // The number of values that will be loaded per-thread, per-load
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);
        static constexpr int kTotalStep = Tile_N / 4 - 1;

        //
        // Member variables
        //

        // The number of columns in the rhs matrix
        const int rhs_columns_;
        const int lane_id_;
        // The dense matrix pointer in global memory
        const LoadType *matrix_base_;
        // The loaded dense matrix row offset in shared memory
        const int* row_offsets_base_;
        // The register file fragment to load the dense values into.
        LoadType *matrix_fragment_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ wmmaDenseTile(
            int rhs_columns, int offset,
            int lane_id, int thread_group,
            const half* __restrict__ matrix,
            const int *row_offsets,
            half * matrix_fragment):
            rhs_columns_(rhs_columns / kValuesPerLoad_),
            lane_id_(lane_id),
            matrix_base_(reinterpret_cast<const LoadType *>(matrix + offset) + thread_group),
            row_offsets_base_(row_offsets + lane_id),
            matrix_fragment_(reinterpret_cast<LoadType *>(matrix_fragment)){}

        // Load a pair of odd and even row groups
        __device__ __forceinline__ void LoadRow(int row_group_idx){
            const int *row_offsets = row_offsets_base_ + row_group_idx * 4;

            // *(matrix_fragment_ + row_group_idx) = __ldg(matrix_base_ + *(row_offsets));
            *(matrix_fragment_ + row_group_idx) = __ldg(matrix_base_ + rhs_columns_ * (*(row_offsets)));
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int row_group_idx, int residue){
            if (lane_id_ >= residue) return;
            const int *row_offsets = row_offsets_base_ + row_group_idx * 4;
            // *(matrix_fragment_ + kTotalStep) = __ldg(matrix_base_ +  *(row_offsets));
            *(matrix_fragment_ + kTotalStep) = __ldg(matrix_base_ + rhs_columns_ * (*(row_offsets)));
        }
    };

    template <typename LoadType, int Tile_N, int Tile_K, int BlockWidth>
    struct wmmaDenseTileV16 {
        //
        // Static members
        //

        // The number of values that will be loaded per-thread, per-load
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);
        static constexpr int kTotalStep = Tile_N / 4 - 1;

        //
        // Member variables
        //

        // The number of columns in the rhs matrix
        const int rhs_columns_;
        const int lane_id_;
        // The dense matrix pointer in global memory
        const LoadType *matrix_base_;
        // The loaded dense matrix row offset in shared memory
        const int* row_offsets_base_;
        // The register file fragment to load the dense values into.
        LoadType *matrix_fragment_;

// k, k_index, lane_id, thread_group, rhs_matrix, column_indices_tile, dense_matrix_fragment
        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ wmmaDenseTileV16(
            int rhs_columns, int offset,
            int lane_id, int thread_group,
            const half* __restrict__ matrix,
            const int *row_offsets,
            half * matrix_fragment):
            rhs_columns_(rhs_columns / kValuesPerLoad_),
            lane_id_(lane_id),
            matrix_base_(reinterpret_cast<const LoadType *>(matrix + offset) + thread_group),
            row_offsets_base_(row_offsets + lane_id),
            matrix_fragment_(reinterpret_cast<LoadType *>(matrix_fragment)){}

        // Load a pair of odd and even row groups
        __device__ __forceinline__ void LoadRow(int row_group_idx){
            const int *row_offsets = row_offsets_base_ + row_group_idx * 4;

            // *(matrix_fragment_ + row_group_idx) = __ldg(matrix_base_ +  *(row_offsets));
            *(matrix_fragment_ + row_group_idx) = __ldg(matrix_base_ + rhs_columns_ * (*(row_offsets)));
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int row_group_idx, int residue){
            if (lane_id_ >= residue) return;
            const int *row_offsets = row_offsets_base_ + row_group_idx * 4;
            // *(matrix_fragment_ + kTotalStep) = __ldg(matrix_base_ +  *(row_offsets));
            *(matrix_fragment_ + kTotalStep) = __ldg(matrix_base_ + rhs_columns_ * (*(row_offsets)));
        }
    };
    

}

#endif