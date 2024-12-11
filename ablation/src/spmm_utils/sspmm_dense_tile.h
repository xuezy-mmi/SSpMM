#ifndef SSPMM_DENSE_TILE_H
#define SSPMM_DENSE_TILE_H
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


    // template <typename LoadType, int Tile_M, int Tile_N, int Tile_K>
    // struct mmaDenseTile_Turing {
    //     //
    //     // Static members
    //     //

    //     // The number of values that will be loaded per-thread, per-load
    //     static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);//8
    //     static constexpr int kTotalStep = Tile_K / 8 - 1;//32/8-1=3
    //     //
    //     // Member variables
    //     //
    //     // The number of columns in the rhs matrix
    //     const int rhs_columns_;
    //     const int tid_;
    //     // const int lane_id_;
    //     const int tx_;
    //     const int ty_;
    //     // The dense matrix pointer in global memory
    //     const float4 *matrix_base_;
    //     // The loaded dense matrix row offset in shared memory
    //     const int* row_offsets_base_;
    //     // The register file fragment to load the dense values into.
    //     float4 *matrix_fragment_;

    //     // Constructor. Set the initial pointer offsets
    //     __device__ __forceinline__ mmaDenseTile_Turing(
    //         int tid,
    //         int rhs_columns,
    //         int col_offset,
    //         const half* __restrict__ matrix,
    //         const int *row_offsets,
    //         half * matrix_fragment):
    //         rhs_columns_(rhs_columns / kValuesPerLoad_),
    //         tid_(tid),
    //         // lane_id_(tid % 8),
    //         tx_((tid%4) * 2 + (tid%8) / 4),
    //         ty_((tid/8)),
    //         matrix_base_(reinterpret_cast<const float4 *>(matrix + col_offset)),
    //         row_offsets_base_(row_offsets + tx_),
    //         matrix_fragment_(reinterpret_cast<float4 *>(matrix_fragment)){}

    //     // Load a pair of odd and even row groups
    //     __device__ __forceinline__ void LoadRow(int row_group_idx){// row_group_idx = 0 1 2 3
    //         const int row_offsets = *(row_offsets_base_ + row_group_idx * 8);

    //         *(matrix_fragment_ + 2*row_group_idx    ) = __ldg(matrix_base_ + ty_ + rhs_columns_ * (row_offsets)    );
    //         *(matrix_fragment_ + 2*row_group_idx + 1) = __ldg(matrix_base_ + ty_ + rhs_columns_ * (row_offsets) + 4);
    //         // register shuffle
    //         half * matrix_half_reg = reinterpret_cast<half *>(matrix_fragment_ + 2 * row_group_idx);
    //         int src_line = (tid_ + 4) % 8 + (tid_ / 8) * 8;
    //         #pragma unroll 8
    //         for(int i = 0; i < 8; i++){
    //             half temp;
    //             if(tid_ % 8 < 4) temp = matrix_half_reg[2*i+1];
    //             else temp = matrix_half_reg[2*i];
    //             temp = __shfl_sync(0xffffffff, temp, src_line, 32);
    //             if(tid_ % 8 < 4) matrix_half_reg[2*i+1] = temp;
    //             else matrix_half_reg[2*i] = temp;
    //         }
    //     }

    //     // Load the residual and compute the matrix product
    //     __device__ __forceinline__ void ResidueLoad(int row_group_idx, int residue){

    //         if (tx_ >= residue){
    //             const half kZeroValues[16] = {};
    //             *(matrix_fragment_ + 2*kTotalStep) = reinterpret_cast<const float4*>(kZeroValues)[0];
    //             *(matrix_fragment_ + 2*kTotalStep + 1) = reinterpret_cast<const float4*>(kZeroValues)[0];
    //         }
    //         const int row_offsets = *(row_offsets_base_ + row_group_idx * 8);

    //         *(matrix_fragment_ + 2*kTotalStep    ) = __ldg(matrix_base_ + ty_ + rhs_columns_ * (row_offsets)    );
    //         *(matrix_fragment_ + 2*kTotalStep + 1) = __ldg(matrix_base_ + ty_ + rhs_columns_ * (row_offsets) + 1);
    //         // register shuffle
    //         half * matrix_half_reg = reinterpret_cast<half *>(matrix_fragment_ + 2 * kTotalStep);
    //         int src_line = (tid_ + 4) % 8 + (tid_ / 8) * 8;
    //         #pragma unroll 8
    //         for(int i = 0; i < 8; i++){
    //             half temp;
    //             if(tid_ % 8 < 4) temp = matrix_half_reg[2*i+1];
    //             else temp = matrix_half_reg[2*i];
    //             temp = __shfl_sync(0xffffffff, temp, src_line, 32);
    //             if(tid_ % 8 < 4) matrix_half_reg[2*i+1] = temp;
    //             else matrix_half_reg[2*i] = temp;
    //         }
    //     }
    // };
    template <typename LoadType, int Tile_N, int Tile_K>
    struct mmaDenseTile_SSPMM {
        //
        // Static members
        //

        // The number of values that will be loaded per-thread, per-load
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);//8
        static constexpr int kTotalStep = Tile_K / 8 - 1;//32/8-1=3
        static constexpr int kReg_B_offset = 8;//Tile_K * Tile_N / 32 / 8;//64/8=8
        //
        // Member variables
        //
        // The number of columns in the rhs matrix
        const int rhs_columns_;
        const int tid_;
        const int lane_id_;
        const int thread_group_;
        const int residue_;
        // const int tx_;
        // const int ty_;
        // The dense matrix pointer in global memory
        const float4 * matrix_base_;
        // The loaded dense matrix row offset in shared memory
        const int * row_offsets_base_;
        // The register file fragment to load the dense values into.
        float4 *matrix_fragment_;
        // float4* dense_tile_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ mmaDenseTile_SSPMM(
            int tid,
            int rhs_columns,
            int col_offset,
            const half* __restrict__ matrix,
            const int *row_offsets,
            half * matrix_fragment):
            rhs_columns_(rhs_columns / kValuesPerLoad_),
            tid_(tid),
            lane_id_(tid % 4),
            thread_group_(tid / 4),
            residue_(tid % 8),
            // tx_((tid%4) * 2 + (tid%8) / 4),
            // ty_((tid/8)),
            matrix_base_(reinterpret_cast<const float4 *>(matrix + col_offset)),
            row_offsets_base_(row_offsets),//32 = 2 * 16
            matrix_fragment_(reinterpret_cast<float4 *>(matrix_fragment)){}

        // Load a pair of odd and even row groups
        __device__ __forceinline__ void Load2Rows(int row_group_idx, int sel){// row_group_idx = 0 1 2 3
            // const int row_offsets = *(row_offsets_base_ + row_group_idx * 8);
            const int row_offset0 = *(row_offsets_base_ + 2 * lane_id_     + row_group_idx * 8 + sel * Tile_K);
            const int row_offset1 = *(row_offsets_base_ + 2 * lane_id_ + 1 + row_group_idx * 8 + sel * Tile_K);
            *(matrix_fragment_ + 2*row_group_idx     + sel * kReg_B_offset) = __ldg(matrix_base_ + thread_group_ + rhs_columns_ * (row_offset0));
            *(matrix_fragment_ + 2*row_group_idx + 1 + sel * kReg_B_offset) = __ldg(matrix_base_ + thread_group_ + rhs_columns_ * (row_offset1));
            // *********register shuffle********* //
            half * matrix_half_reg = reinterpret_cast<half *>(matrix_fragment_ + 2 * row_group_idx + sel * kReg_B_offset);//2 * float4 -> 16 * half
            half temp[16];
            #pragma unroll 16
            for(int i = 0; i < 16; i++){
                temp[i] = matrix_half_reg[i];
            }
            #pragma unroll 8
            for(int i = 0; i < 8; i++){
                matrix_half_reg[2*i  ] = temp[i  ];
                matrix_half_reg[2*i+1] = temp[8+i];
            }
            
            // int src_line = (tid_ + 4) % 8 + (tid_ / 8) * 8;
            // #pragma unroll 8
            // for(int i = 0; i < 8; i++){
            //     half temp;
            //     if(tid_ % 8 < 4) temp = matrix_half_reg[2*i+1];
            //     else temp = matrix_half_reg[2*i];
            //     temp = __shfl_sync(0xffffffff, temp, src_line, 32);
            //     if(tid_ % 8 < 4) matrix_half_reg[2*i+1] = temp;
            //     else matrix_half_reg[2*i] = temp;
            // }
        }
        __device__ __forceinline__ void Load8Rows(int row_group_idx, int sel){// row_group_idx = 0 1 2 3
            // const int row_offsets = *(row_offsets_base_ + row_group_idx * 8);
            const int row_offset0 = *(row_offsets_base_ + 2 * lane_id_     + row_group_idx * 8 + sel * Tile_K);
            const int row_offset1 = *(row_offsets_base_ + 2 * lane_id_ + 1 + row_group_idx * 8 + sel * Tile_K);
            *(matrix_fragment_ + 2*row_group_idx     + sel * kReg_B_offset) = __ldg(matrix_base_ + thread_group_ + rhs_columns_ * (row_offset0));
            *(matrix_fragment_ + 2*row_group_idx + 1 + sel * kReg_B_offset) = __ldg(matrix_base_ + thread_group_ + rhs_columns_ * (row_offset1));
            // *********register shuffle********* //
            half * matrix_half_reg = reinterpret_cast<half *>(matrix_fragment_ + 2 * row_group_idx + sel * kReg_B_offset);//2 * float4 -> 16 * half
            half temp[16];
            #pragma unroll 16
            for(int i = 0; i < 16; i++){
                temp[i] = matrix_half_reg[i];
            }
            #pragma unroll 8
            for(int i = 0; i < 8; i++){
                matrix_half_reg[2*i  ] = temp[i  ];
                matrix_half_reg[2*i+1] = temp[8+i];
            }
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int row_group_idx, int residue){
            const half kZeroValues[8] = {};
            if (2*lane_id_ >= residue) return;
            else if(2*lane_id_ + 1 == residue){
                const int row_offset0 = *(row_offsets_base_ + 2 * lane_id_ + row_group_idx * 8);
                *(matrix_fragment_ + 2*kTotalStep    ) = __ldg(matrix_base_ + thread_group_ + rhs_columns_ * (row_offset0));
                *(matrix_fragment_ + 2*kTotalStep + 1) = reinterpret_cast<const float4*>(kZeroValues)[0];
            }
            else{
                const int row_offset0 = *(row_offsets_base_     + row_group_idx * 8);
                const int row_offset1 = *(row_offsets_base_ + 1 + row_group_idx * 8);
                *(matrix_fragment_ + 2*kTotalStep    ) = __ldg(matrix_base_ + thread_group_ + rhs_columns_ * (row_offset0));
                *(matrix_fragment_ + 2*kTotalStep + 1) = __ldg(matrix_base_ + thread_group_ + rhs_columns_ * (row_offset1));
            }
        }

        __device__ __forceinline__ void LoadRow(int row_group_idx){
            const int row_offset = *(row_offsets_base_ + lane_id_ + row_group_idx * 4);

            *(matrix_fragment_ + row_group_idx) = __ldg(matrix_base_ + thread_group_ + rhs_columns_ * row_offset);
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoadRow(int row_group_idx, int residue){
            if (lane_id_ >= residue) return;
            const int row_offset = *(row_offsets_base_ + lane_id_ + row_group_idx * 4);

            *(matrix_fragment_ + kTotalStep) = __ldg(matrix_base_ + thread_group_ + rhs_columns_ * row_offset);
        }

    };

}

#endif