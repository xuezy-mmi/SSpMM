#ifndef SSPMM_COMPUTER_H
#define SSPMM_COMPUTER_H

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

namespace spmm{

    template <typename VecType, int Tile_N>
    struct wmmaComputeUtils16 {

        //
        // Static members
        //

        static constexpr int kTotalStep = Tile_N / 4 - 1;
        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile values
        const float2* lhs_tile_;
        // Register file fragment storing the rhs tile
        const half* rhs_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_1;
        float* output_fragment_2;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils16(
            const half * lhs_tile,//shared half [vec_length*Tile_N]
            const half* rhs_fragment,//reg half [64]
            float* output_fragment1,//reg float [32]
            float* output_fragment2,//reg float [32]
            int lane_id, int thread_group):
            lhs_tile_(reinterpret_cast<const float2 *>(lhs_tile) + lane_id * 4 + thread_group / 4),
            rhs_fragment_(rhs_fragment),
            output_fragment_1(output_fragment1),
            output_fragment_2(output_fragment2){}


        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            float lhs_fragment1[2];
            float lhs_fragment2[2];

            float2 *lhs_fragment1_float2 = reinterpret_cast<float2 *>(lhs_fragment1);
            float2 *lhs_fragment2_float2 = reinterpret_cast<float2 *>(lhs_fragment2);
            *(lhs_fragment1_float2) = *(lhs_tile_ + n_group_idx * 16);
            *(lhs_fragment2_float2) = *(lhs_tile_ + 2 + n_group_idx * 16);
            const int* lhs_fragment1_int = reinterpret_cast<const int *>(lhs_fragment1);
            const int* lhs_fragment2_int = reinterpret_cast<const int *>(lhs_fragment2);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * n_group_idx);

            #pragma unroll 2
            for (int i = 0; i < 2; i++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_1[0 + 8 * i]), "+f"(output_fragment_1[1 + 8 * i]),
                    "+f"(output_fragment_1[2 + 8 * i]), "+f"(output_fragment_1[3 + 8 * i]),
                    "+f"(output_fragment_1[4 + 8 * i]), "+f"(output_fragment_1[5 + 8 * i]),
                    "+f"(output_fragment_1[6 + 8 * i]), "+f"(output_fragment_1[7 + 8 * i]):
                    "r"(lhs_fragment1_int[0]), "r"(lhs_fragment1_int[1]),
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i])
                );
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_2[0 + 8 * i]), "+f"(output_fragment_2[1 + 8 * i]),
                    "+f"(output_fragment_2[2 + 8 * i]), "+f"(output_fragment_2[3 + 8 * i]),
                    "+f"(output_fragment_2[4 + 8 * i]), "+f"(output_fragment_2[5 + 8 * i]),
                    "+f"(output_fragment_2[6 + 8 * i]), "+f"(output_fragment_2[7 + 8 * i]):
                    "r"(lhs_fragment2_int[0]), "r"(lhs_fragment2_int[1]),
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i])
                );
            }
        }

        // Compute Residue
        __device__ __forceinline__ void TileMACResidue(int n_group_idx){
            float lhs_fragment1[2];
            float lhs_fragment2[2];
            
            float2 *lhs_fragment1_float2 = reinterpret_cast<float2 *>(lhs_fragment1);
            float2 *lhs_fragment2_float2 = reinterpret_cast<float2 *>(lhs_fragment2);
            *(lhs_fragment1_float2) = *(lhs_tile_ + n_group_idx * 16);
            *(lhs_fragment2_float2) = *(lhs_tile_ + 2 + n_group_idx * 16);
            const int* lhs_fragment1_int = reinterpret_cast<const int *>(lhs_fragment1);
            const int* lhs_fragment2_int = reinterpret_cast<const int *>(lhs_fragment2);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * kTotalStep);

            #pragma unroll 2
            for (int i = 0; i < 2; i++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_1[0 + 8 * i]), "+f"(output_fragment_1[1 + 8 * i]),
                    "+f"(output_fragment_1[2 + 8 * i]), "+f"(output_fragment_1[3 + 8 * i]),
                    "+f"(output_fragment_1[4 + 8 * i]), "+f"(output_fragment_1[5 + 8 * i]),
                    "+f"(output_fragment_1[6 + 8 * i]), "+f"(output_fragment_1[7 + 8 * i]):
                    "r"(lhs_fragment1_int[0]), "r"(lhs_fragment1_int[1]),
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i])
                );
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_2[0 + 8 * i]), "+f"(output_fragment_2[1 + 8 * i]),
                    "+f"(output_fragment_2[2 + 8 * i]), "+f"(output_fragment_2[3 + 8 * i]),
                    "+f"(output_fragment_2[4 + 8 * i]), "+f"(output_fragment_2[5 + 8 * i]),
                    "+f"(output_fragment_2[6 + 8 * i]), "+f"(output_fragment_2[7 + 8 * i]):
                    "r"(lhs_fragment2_int[0]), "r"(lhs_fragment2_int[1]),
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i])
                );
            }
        }
    };


    template <typename VecType, int Tile_N>
    struct wmmaComputeUtils8 {

        //
        // Static members
        //

        static constexpr int kTotalStep = Tile_N / 4 - 1;//32/4-1=7
        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile values
        const float2* lhs_tile_;
        // Register file fragment storing the rhs tile
        const half* rhs_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils8(
            const half * lhs_tile,//shared float4 [Tile_N]
            const half* rhs_fragment,//reg half [64]
            float* output_fragment,//reg float [16]
            int lane_id, int thread_group):
            lhs_tile_(reinterpret_cast<const float2 *>(lhs_tile) + lane_id * 2 + thread_group / 4),
            rhs_fragment_(rhs_fragment),
            output_fragment_(output_fragment){}


        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            float lhs_fragment[2];
            
            float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
            *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 8);
            int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * n_group_idx);
            
            #pragma unroll 2
            for (int i = 0; i < 2; i++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 8 * i]), "+f"(output_fragment_[1 + 8 * i]),
                    "+f"(output_fragment_[2 + 8 * i]), "+f"(output_fragment_[3 + 8 * i]),
                    "+f"(output_fragment_[4 + 8 * i]), "+f"(output_fragment_[5 + 8 * i]),
                    "+f"(output_fragment_[6 + 8 * i]), "+f"(output_fragment_[7 + 8 * i]):
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1]),
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }

        // Compute Residue
        __device__ __forceinline__ void TileMACResidue(int n_group_idx){
            float lhs_fragment[2];
            
            float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
            *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 8);
            int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * kTotalStep);

            #pragma unroll 2
            for (int i = 0; i < 2; i++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 8 * i]), "+f"(output_fragment_[1 + 8 * i]),
                    "+f"(output_fragment_[2 + 8 * i]), "+f"(output_fragment_[3 + 8 * i]),
                    "+f"(output_fragment_[4 + 8 * i]), "+f"(output_fragment_[5 + 8 * i]),
                    "+f"(output_fragment_[6 + 8 * i]), "+f"(output_fragment_[7 + 8 * i]):
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1]),
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }
    };

    // template <typename VecType, int Tile_N, int A_PAD, int B_PAD>
    // struct Transp_ComputeUtils8 {
    //     //
    //     // Static members
    //     //
    //     static constexpr int kTotalStep = Tile_N / 4 - 1;//32/4-1=7
    //     //
    //     // Member variables
    //     //
    //     // Shared memory buffer storing the lhs tile values
    //     const float2* lhs_tile_;
    //     // Register file fragment storing the rhs tile
    //     const half* rhs_fragment_;
    //     // Register file fragment to accumulate results into.
    //     float* output_fragment_;
    //     // Constructor
    //     __device__ __forceinline__ Transp_ComputeUtils8(
    //         const half * lhs_tile,//shared float4 [Tile_N]
    //         const half* rhs_fragment,//reg half [64]
    //         float* output_fragment,//reg float [16]
    //         int lane_id, int thread_group):
    //         lhs_tile_(reinterpret_cast<const float2 *>(lhs_tile) + lane_id * 2 + thread_group / 4),
    //         rhs_fragment_(rhs_fragment),
    //         output_fragment_(output_fragment){}


    //     // Compute
    //     __device__ __forceinline__ void TileMAC(int n_group_idx, int step = 0){
    //         int offset = step % 2;
    //         float lhs_fragment[2];
            
    //         float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
    //         *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 8);
    //         int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
    //         const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * n_group_idx);
            
    //         #pragma unroll 2
    //         for (int i = 0; i < 2; i++){
    //             asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
    //                 "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
    //                 "{%8, %9}, \t"
    //                 "{%10, %11}, \t"
    //                 "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
    //                 "+f"(output_fragment_[0 + 8 * i]), "+f"(output_fragment_[1 + 8 * i]),
    //                 "+f"(output_fragment_[2 + 8 * i]), "+f"(output_fragment_[3 + 8 * i]),
    //                 "+f"(output_fragment_[4 + 8 * i]), "+f"(output_fragment_[5 + 8 * i]),
    //                 "+f"(output_fragment_[6 + 8 * i]), "+f"(output_fragment_[7 + 8 * i]):
    //                 // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1]),
    //                 "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i]),
    //                 "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
    //             );
    //         }
    //     }

    //     // Compute Residue
    //     __device__ __forceinline__ void TileMACResidue(int n_group_idx){
    //         float lhs_fragment[2];

    //         float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
    //         *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 8);
    //         int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
    //         const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * kTotalStep);

    //         #pragma unroll 2
    //         for (int i = 0; i < 2; i++){
    //             asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
    //                 "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
    //                 "{%8, %9}, \t"
    //                 "{%10, %11}, \t"
    //                 "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
    //                 "+f"(output_fragment_[0 + 8 * i]), "+f"(output_fragment_[1 + 8 * i]),
    //                 "+f"(output_fragment_[2 + 8 * i]), "+f"(output_fragment_[3 + 8 * i]),
    //                 "+f"(output_fragment_[4 + 8 * i]), "+f"(output_fragment_[5 + 8 * i]),
    //                 "+f"(output_fragment_[6 + 8 * i]), "+f"(output_fragment_[7 + 8 * i]):
    //                 // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1]),
    //                 "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i]),
    //                 "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
    //             );
    //         }
    //     }
    // };

    template <typename VecType, int Tile_N>
    struct Transp_ComputeUtils8 {

        //
        // Static members
        //

        static constexpr int kTotalStep = Tile_N / 4 - 1;//32/4-1=7
        //
        // Member variables
        //
        const int lane_id_;
        // Shared memory buffer storing the lhs tile values
        const float2* lhs_tile_;
        // Register file fragment storing the rhs tile
        const half* rhs_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;

        // Constructor
        __device__ __forceinline__ Transp_ComputeUtils8(
            const half * lhs_tile,//shared float4 [Tile_N]
            const half* rhs_fragment,//reg half [64]
            float* output_fragment,//reg float [16]
            int lane_id, int thread_group):
            lane_id_(lane_id),
            lhs_tile_(reinterpret_cast<const float2 *>(lhs_tile) + lane_id * 2 + thread_group / 4),
            rhs_fragment_(rhs_fragment),
            output_fragment_(output_fragment){}

        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            float lhs_fragment[2];
            int * lhs_fragment_int;
            float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
            *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 8);
            lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * n_group_idx);
            
            #pragma unroll 2
            for (int i = 0; i < 2; i++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 8 * i]), "+f"(output_fragment_[1 + 8 * i]),
                    "+f"(output_fragment_[2 + 8 * i]), "+f"(output_fragment_[3 + 8 * i]),
                    "+f"(output_fragment_[4 + 8 * i]), "+f"(output_fragment_[5 + 8 * i]),
                    "+f"(output_fragment_[6 + 8 * i]), "+f"(output_fragment_[7 + 8 * i]):
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1]),
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i]),
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }

        // Compute
        __device__ __forceinline__ void ResTileMAC(int n_group_idx, int residue){
            int * lhs_fragment_int;
			if(lane_id_ >= residue){
				half kZeroValues[4] = {};
				lhs_fragment_int = reinterpret_cast<int *>(kZeroValues);
            }
			else{
				float lhs_fragment[2];
				float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
				*(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 8);
				lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
			}
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * n_group_idx);
            
            #pragma unroll 2
            for (int i = 0; i < 2; i++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 8 * i]), "+f"(output_fragment_[1 + 8 * i]),
                    "+f"(output_fragment_[2 + 8 * i]), "+f"(output_fragment_[3 + 8 * i]),
                    "+f"(output_fragment_[4 + 8 * i]), "+f"(output_fragment_[5 + 8 * i]),
                    "+f"(output_fragment_[6 + 8 * i]), "+f"(output_fragment_[7 + 8 * i]):
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1]),
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i]),
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }

        // Compute Residue
        __device__ __forceinline__ void TileMACResidue(int n_group_idx){
            float lhs_fragment[2];

            float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
            *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 8);
            int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * kTotalStep);

            #pragma unroll 2
            for (int i = 0; i < 2; i++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 8 * i]), "+f"(output_fragment_[1 + 8 * i]),
                    "+f"(output_fragment_[2 + 8 * i]), "+f"(output_fragment_[3 + 8 * i]),
                    "+f"(output_fragment_[4 + 8 * i]), "+f"(output_fragment_[5 + 8 * i]),
                    "+f"(output_fragment_[6 + 8 * i]), "+f"(output_fragment_[7 + 8 * i]):
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1]),
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i]),
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }
    };

    template <int Tile_N, int Tile_K>
    struct mmaComputeUtils8_SSPMM {

        //
        // Static members
        //
        static constexpr int kTotalStep = Tile_K / 8 - 1;//32/8-1=3
        //
        // Member variables
        //
        const int lane_id_;
        // Shared memory buffer storing the rhs tile values
        const half* sparse_tile_;
        // Register file fragment storing the lhs tile
        const half* dense_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;
        // float* residue_fragment_;

        // Constructor
        __device__ __forceinline__ mmaComputeUtils8_SSPMM(
            const half * dense_fragment,//reg half [16*4]
            const half* sparse_tile,//shared half [32*8]
            float* output_fragment,//reg float/half [16]
            int tid):
            lane_id_(tid%4),
            sparse_tile_(sparse_tile + tid/4 + (tid%4)*16),
            dense_fragment_(dense_fragment),
            output_fragment_(output_fragment){}


        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            half sparse_fragment[2];
            sparse_fragment[0] = *(sparse_tile_ + n_group_idx * 64);
            sparse_fragment[1] = *(sparse_tile_ + n_group_idx * 64 + 8);
            int * sparse_fragment_int = reinterpret_cast<int *>(sparse_fragment);

            const int * dense_fragment_int = reinterpret_cast<const int *>(dense_fragment_ + n_group_idx * 16);
            
            #pragma unroll 4
            for (int i = 0; i < 4; i++){
                asm ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment_[0 + 4 * i]), "+f"(output_fragment_[1 + 4 * i]),
                    "+f"(output_fragment_[2 + 4 * i]), "+f"(output_fragment_[3 + 4 * i]):
                    "r"(dense_fragment_int[2*i]), "r"(dense_fragment_int[2*i+1]),
                    "r"(sparse_fragment_int[0])
					// "f"(output_fragment_[0 + 4 * i]), "f"(output_fragment_[1 + 4 * i]),
					// "f"(output_fragment_[2 + 4 * i]), "f"(output_fragment_[3 + 4 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }
                // Compute
        __device__ __forceinline__ void TileMACSelect(int a_group_idx, int b_group_idx){
            half sparse_fragment[2];
            sparse_fragment[0] = *(sparse_tile_ + a_group_idx * 64);
            sparse_fragment[1] = *(sparse_tile_ + a_group_idx * 64 + 8);
            int * sparse_fragment_int = reinterpret_cast<int *>(sparse_fragment);

            const int * dense_fragment_int = reinterpret_cast<const int *>(dense_fragment_ + b_group_idx * 16);
            
            #pragma unroll 4
            for (int i = 0; i < 4; i++){
                asm ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment_[0 + 4 * i]), "+f"(output_fragment_[1 + 4 * i]),
                    "+f"(output_fragment_[2 + 4 * i]), "+f"(output_fragment_[3 + 4 * i]):
                    "r"(dense_fragment_int[2*i]), "r"(dense_fragment_int[2*i+1]),
                    "r"(sparse_fragment_int[0])
					// "f"(output_fragment_[0 + 4 * i]), "f"(output_fragment_[1 + 4 * i]),
					// "f"(output_fragment_[2 + 4 * i]), "f"(output_fragment_[3 + 4 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }

        // Compute
        __device__ __forceinline__ void ResTileMAC(int n_group_idx, int residue){
            half sparse_fragment[2];
            sparse_fragment[1] = __float2half(0.0f);
            if(2*lane_id_ >= residue){
                sparse_fragment[0] = __float2half(0.0f);
                // sparse_fragment[1] = 0.0;
            }
            else if(2*lane_id_ >= residue-1){
                sparse_fragment[0] = *(sparse_tile_ + n_group_idx * 64);
                // sparse_fragment[1] = 0.0;
            }
            else{
                sparse_fragment[0] = *(sparse_tile_ + n_group_idx * 64);
                sparse_fragment[1] = *(sparse_tile_ + n_group_idx * 64 + 8);
            }
            int * sparse_fragment_int = reinterpret_cast<int *>(sparse_fragment);

            const int * dense_fragment_int = reinterpret_cast<const int *>(dense_fragment_ + n_group_idx * 16);
            
            #pragma unroll 4
            for (int i = 0; i < 4; i++){
                asm ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment_[0 + 4 * i]), "+f"(output_fragment_[1 + 4 * i]),
                    "+f"(output_fragment_[2 + 4 * i]), "+f"(output_fragment_[3 + 4 * i]):
                    "r"(dense_fragment_int[2*i]), "r"(dense_fragment_int[2*i+1]),
                    "r"(sparse_fragment_int[0])
					// "f"(output_fragment_[0 + 4 * i]), "f"(output_fragment_[1 + 4 * i]),
					// "f"(output_fragment_[2 + 4 * i]), "f"(output_fragment_[3 + 4 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }

        }
        // Compute Residue
        __device__ __forceinline__ void TileMACResidue(int n_group_idx){
            half sparse_fragment[2];
            sparse_fragment[0] = *(sparse_tile_ + n_group_idx * 64);
            sparse_fragment[1] = *(sparse_tile_ + n_group_idx * 64 + 8);
            int * sparse_fragment_int = reinterpret_cast<int *>(sparse_fragment);

            const int * dense_fragment_int = reinterpret_cast<const int *>(dense_fragment_ + kTotalStep * 16);

            #pragma unroll 4
            for (int i = 0; i < 4; i++){
                asm ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%7, %8, %9, %10}; ":
                    "=f"(output_fragment_[0 + 4 * i]), "=f"(output_fragment_[1 + 4 * i]),
                    "=f"(output_fragment_[2 + 4 * i]), "=f"(output_fragment_[3 + 4 * i]):
                    "r"(dense_fragment_int[2*i]), "r"(dense_fragment_int[2*i+1]),
                    "r"(sparse_fragment_int[0]),
					"f"(output_fragment_[0 + 4 * i]), "f"(output_fragment_[1 + 4 * i]),
					"f"(output_fragment_[2 + 4 * i]), "f"(output_fragment_[3 + 4 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }

        // __device__ __forceinline__ void TileMAC884(int n_group_idx){
        //     float lhs_fragment[2];
            
        //     float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
        //     *(lhs_fragment_float2) = *(reinterpret_cast<const float2 *>(sparse_tile_) + n_group_idx * 8);
        //     int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
        //     const int* rhs_fragment_int = reinterpret_cast<const int *>(dense_fragment_ + 8 * n_group_idx);
            
        //     #pragma unroll 2
        //     for (int i = 0; i < 2; i++){
        //         asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
        //             "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
        //             "{%8, %9}, \t"
        //             "{%10, %11}, \t"
        //             "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
        //             "+f"(output_fragment_[0 + 8 * i]), "+f"(output_fragment_[1 + 8 * i]),
        //             "+f"(output_fragment_[2 + 8 * i]), "+f"(output_fragment_[3 + 8 * i]),
        //             "+f"(output_fragment_[4 + 8 * i]), "+f"(output_fragment_[5 + 8 * i]),
        //             "+f"(output_fragment_[6 + 8 * i]), "+f"(output_fragment_[7 + 8 * i]):
        //             "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1]),
        //             "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i])
        //             // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
        //         );
        //     }
        // }

        // // Compute Residue
        // __device__ __forceinline__ void TileMACResidue884(int n_group_idx){
        //     float lhs_fragment[2];
            
        //     float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
        //     *(lhs_fragment_float2) = *(reinterpret_cast<const float2 *>(sparse_tile_) + n_group_idx * 8);
        //     int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
        //     const int* rhs_fragment_int = reinterpret_cast<const int *>(dense_fragment_ + 8 * kTotalStep);

        //     #pragma unroll 2
        //     for (int i = 0; i < 2; i++){
        //         asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
        //             "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
        //             "{%8, %9}, \t"
        //             "{%10, %11}, \t"
        //             "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
        //             "+f"(output_fragment_[0 + 8 * i]), "+f"(output_fragment_[1 + 8 * i]),
        //             "+f"(output_fragment_[2 + 8 * i]), "+f"(output_fragment_[3 + 8 * i]),
        //             "+f"(output_fragment_[4 + 8 * i]), "+f"(output_fragment_[5 + 8 * i]),
        //             "+f"(output_fragment_[6 + 8 * i]), "+f"(output_fragment_[7 + 8 * i]):
        //             "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1]),
        //             "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i])
        //             // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
        //         );
        //     }
        // }

    };

    template <int Tile_N, int Tile_K>
    struct mmaComputeUtils16_SSPMM {
        //
        // Static members
        //
        static constexpr int kTotalStep = Tile_K / 8 - 1;//32/8-1=3
        //
        // Member variables
        //
        const int lane_id_;
        // Shared memory buffer storing the rhs tile values
        const half* sparse_tile_;
        // Register file fragment storing the lhs tile
        const half* dense_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment1_;
        float* output_fragment2_;
        // float* residue_fragment_;
        // Constructor
        __device__ __forceinline__ mmaComputeUtils16_SSPMM(
            const half * dense_fragment,//reg half [16*4]
            const half* sparse_tile,//shared half [32*8]
            float* output_fragment1,//reg float/half [16]
            float* output_fragment2,
            int tid):
            lane_id_(tid%4),
            sparse_tile_(sparse_tile + tid/4 + (tid%4)*32),
            dense_fragment_(dense_fragment),
            output_fragment1_(output_fragment1),
            output_fragment2_(output_fragment2){}
        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            half sparse_fragment1[2];
            half sparse_fragment2[2];
            sparse_fragment1[0] = *(sparse_tile_ + n_group_idx * 64*2     );
            sparse_fragment1[1] = *(sparse_tile_ + n_group_idx * 64*2 + 16);
            sparse_fragment2[0] = *(sparse_tile_ + n_group_idx * 64*2 + 8 );
            sparse_fragment2[1] = *(sparse_tile_ + n_group_idx * 64*2 + 24);
            int * sparse_fragment1_int = reinterpret_cast<int *>(sparse_fragment1);
            int * sparse_fragment2_int = reinterpret_cast<int *>(sparse_fragment2);

            const int * dense_fragment_int = reinterpret_cast<const int *>(dense_fragment_ + n_group_idx * 16);
            
            #pragma unroll 4
            for (int i = 0; i < 4; i++){
                asm ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment1_[0 + 4 * i]), "+f"(output_fragment1_[1 + 4 * i]),
                    "+f"(output_fragment1_[2 + 4 * i]), "+f"(output_fragment1_[3 + 4 * i]):
                    "r"(dense_fragment_int[2*i]), "r"(dense_fragment_int[2*i+1]),
                    "r"(sparse_fragment1_int[0])
					// "f"(output_fragment_[0 + 4 * i]), "f"(output_fragment_[1 + 4 * i]),
					// "f"(output_fragment_[2 + 4 * i]), "f"(output_fragment_[3 + 4 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
                asm ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment2_[0 + 4 * i]), "+f"(output_fragment2_[1 + 4 * i]),
                    "+f"(output_fragment2_[2 + 4 * i]), "+f"(output_fragment2_[3 + 4 * i]):
                    "r"(dense_fragment_int[2*i]), "r"(dense_fragment_int[2*i+1]),
                    "r"(sparse_fragment2_int[0])
					// "f"(output_fragment_[0 + 4 * i]), "f"(output_fragment_[1 + 4 * i]),
					// "f"(output_fragment_[2 + 4 * i]), "f"(output_fragment_[3 + 4 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }
        // Compute
        __device__ __forceinline__ void ResTileMAC(int n_group_idx, int residue){
            half sparse_fragment1[2];
            half sparse_fragment2[2];
            sparse_fragment1[1] = __float2half(0.0f);
            sparse_fragment2[1] = __float2half(0.0f);
            if(2*lane_id_ >= residue){//even residue
                sparse_fragment1[0] = __float2half(0.0f);
                sparse_fragment2[0] = __float2half(0.0f);
            }
            else if(2*lane_id_ >= residue-1){//odd residue
                sparse_fragment1[0] = *(sparse_tile_ + n_group_idx * 64*2);
                sparse_fragment2[0] = *(sparse_tile_ + n_group_idx * 64*2 + 8);
            }
            else{
                sparse_fragment1[0] = *(sparse_tile_ + n_group_idx * 64*2);
                sparse_fragment1[1] = *(sparse_tile_ + n_group_idx * 64*2 + 16);
                sparse_fragment2[0] = *(sparse_tile_ + n_group_idx * 64*2 + 8 );
                sparse_fragment2[1] = *(sparse_tile_ + n_group_idx * 64*2 + 24);
            }
            int * sparse_fragment1_int = reinterpret_cast<int *>(sparse_fragment1);
            int * sparse_fragment2_int = reinterpret_cast<int *>(sparse_fragment2);

            const int * dense_fragment_int = reinterpret_cast<const int *>(dense_fragment_ + n_group_idx * 16);
            
            #pragma unroll 4
            for (int i = 0; i < 4; i++){
                asm ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment1_[0 + 4 * i]), "+f"(output_fragment1_[1 + 4 * i]),
                    "+f"(output_fragment1_[2 + 4 * i]), "+f"(output_fragment1_[3 + 4 * i]):
                    "r"(dense_fragment_int[2*i]), "r"(dense_fragment_int[2*i+1]),
                    "r"(sparse_fragment1_int[0])
					// "f"(output_fragment_[0 + 4 * i]), "f"(output_fragment_[1 + 4 * i]),
					// "f"(output_fragment_[2 + 4 * i]), "f"(output_fragment_[3 + 4 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
                asm ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment2_[0 + 4 * i]), "+f"(output_fragment2_[1 + 4 * i]),
                    "+f"(output_fragment2_[2 + 4 * i]), "+f"(output_fragment2_[3 + 4 * i]):
                    "r"(dense_fragment_int[2*i]), "r"(dense_fragment_int[2*i+1]),
                    "r"(sparse_fragment2_int[0])
					// "f"(output_fragment_[0 + 4 * i]), "f"(output_fragment_[1 + 4 * i]),
					// "f"(output_fragment_[2 + 4 * i]), "f"(output_fragment_[3 + 4 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }

        }
        // // Compute Residue
        // __device__ __forceinline__ void TileMACResidue(int n_group_idx){
        //     half sparse_fragment[2];
        //     sparse_fragment[0] = *(sparse_tile_ + n_group_idx * 64);
        //     sparse_fragment[1] = *(sparse_tile_ + n_group_idx * 64 + 8);
        //     int * sparse_fragment_int = reinterpret_cast<int *>(sparse_fragment);

        //     const int * dense_fragment_int = reinterpret_cast<const int *>(dense_fragment_ + kTotalStep * 16);

        //     #pragma unroll 4
        //     for (int i = 0; i < 4; i++){
        //         asm ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
        //             "{%0, %1, %2, %3}, \t"
        //             "{%4, %5}, \t"
        //             "{%6}, \t"
        //             "{%7, %8, %9, %10}; ":
        //             "=f"(output_fragment_[0 + 4 * i]), "=f"(output_fragment_[1 + 4 * i]),
        //             "=f"(output_fragment_[2 + 4 * i]), "=f"(output_fragment_[3 + 4 * i]):
        //             "r"(dense_fragment_int[2*i]), "r"(dense_fragment_int[2*i+1]),
        //             "r"(sparse_fragment_int[0]),
		// 			"f"(output_fragment_[0 + 4 * i]), "f"(output_fragment_[1 + 4 * i]),
		// 			"f"(output_fragment_[2 + 4 * i]), "f"(output_fragment_[3 + 4 * i])
        //             // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
        //         );
        //     }
        // }
    };

    template <int Tile_N, int Tile_K>
    struct mmaComputeUtils4_SSPMM {

        //
        // Static members
        //
        static constexpr int kTotalStep = Tile_K / 8 - 1;//32/8-1=3
        //
        // Member variables
        //
        const int tid_;
        const int lane_id_;
        // Shared memory buffer storing the rhs tile values
        const half* sparse_tile_;
        // Register file fragment storing the lhs tile
        const half* dense_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;
        // float* residue_fragment_;

        // Constructor
        __device__ __forceinline__ mmaComputeUtils4_SSPMM(
            const half * dense_fragment,//reg half [16*4]
            const half* sparse_tile,//shared half [32*8]
            float* output_fragment,//reg float/half [16]
            int tid):
            tid_(tid),
            lane_id_(tid%4),
            sparse_tile_(sparse_tile + tid/4 + (tid%4)*8),
            dense_fragment_(dense_fragment),
            output_fragment_(output_fragment){}


        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            half sparse_fragment[2];
            if(tid_ < 16){
                sparse_fragment[0] = *(sparse_tile_ + n_group_idx * 32);
                sparse_fragment[1] = *(sparse_tile_ + n_group_idx * 32 + 4);
            }
            else{
                sparse_fragment[0] = __float2half(0.0f);
                sparse_fragment[1] = __float2half(0.0f);
            }
            int * sparse_fragment_int = reinterpret_cast<int *>(sparse_fragment);

            const int * dense_fragment_int = reinterpret_cast<const int *>(dense_fragment_ + n_group_idx * 16);
            
            #pragma unroll 4
            for (int i = 0; i < 4; i++){
                asm ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment_[0 + 4 * i]), "+f"(output_fragment_[1 + 4 * i]),
                    "+f"(output_fragment_[2 + 4 * i]), "+f"(output_fragment_[3 + 4 * i]):
                    "r"(dense_fragment_int[2*i]), "r"(dense_fragment_int[2*i+1]),
                    "r"(sparse_fragment_int[0])
					// "f"(output_fragment_[0 + 4 * i]), "f"(output_fragment_[1 + 4 * i]),
					// "f"(output_fragment_[2 + 4 * i]), "f"(output_fragment_[3 + 4 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }
        // Compute
        __device__ __forceinline__ void ResTileMAC(int n_group_idx, int residue){
            half sparse_fragment[2];
            // sparse_fragment[1] = __float2half(0.0f);
            if(tid_ >= 16){
                sparse_fragment[0] = __float2half(0.0f);
                sparse_fragment[1] = __float2half(0.0f);
            }
            // else if(2*lane_id_ >= residue){
            //     sparse_fragment[0] = __float2half(0.0f);
            //     // sparse_fragment[1] = 0.0;
            // }
            else if(2*lane_id_ >= residue-1){
                sparse_fragment[0] = *(sparse_tile_ + n_group_idx * 32);
                // sparse_fragment[1] = 0.0;
            }
            else{
                sparse_fragment[0] = *(sparse_tile_ + n_group_idx * 32);
                sparse_fragment[1] = *(sparse_tile_ + n_group_idx * 32 + 4);
            }
            int * sparse_fragment_int = reinterpret_cast<int *>(sparse_fragment);

            const int * dense_fragment_int = reinterpret_cast<const int *>(dense_fragment_ + n_group_idx * 16);
            
            #pragma unroll 4
            for (int i = 0; i < 4; i++){
                asm ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment_[0 + 4 * i]), "+f"(output_fragment_[1 + 4 * i]),
                    "+f"(output_fragment_[2 + 4 * i]), "+f"(output_fragment_[3 + 4 * i]):
                    "r"(dense_fragment_int[2*i]), "r"(dense_fragment_int[2*i+1]),
                    "r"(sparse_fragment_int[0])
					// "f"(output_fragment_[0 + 4 * i]), "f"(output_fragment_[1 + 4 * i]),
					// "f"(output_fragment_[2 + 4 * i]), "f"(output_fragment_[3 + 4 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }

        }
    };

    template <int Tile_N, int Tile_K>
    struct mmaComputeUtils2_SSPMM {

        //
        // Static members
        //
        static constexpr int kTotalStep = Tile_K / 8 - 1;//32/8-1=3
        //
        // Member variables
        //
        const int tid_;
        const int lane_id_;
        // Shared memory buffer storing the rhs tile values
        const half* sparse_tile_;
        // Register file fragment storing the lhs tile
        const half* dense_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;
        // float* residue_fragment_;

        // Constructor
        __device__ __forceinline__ mmaComputeUtils2_SSPMM(
            const half * dense_fragment,//reg half [16*4]
            const half* sparse_tile,//shared half [32*8]
            float* output_fragment,//reg float/half [16]
            int tid):
            tid_(tid),
            lane_id_(tid%4),
            sparse_tile_(sparse_tile + tid/4 + (tid%4)*4),
            dense_fragment_(dense_fragment),
            output_fragment_(output_fragment){}


        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            half sparse_fragment[2];
            if(tid_ < 8){
                sparse_fragment[0] = *(sparse_tile_ + n_group_idx * 16);
                sparse_fragment[1] = *(sparse_tile_ + n_group_idx * 16 + 2);
            }
            else{
                sparse_fragment[0] = __float2half(0.0f);
                sparse_fragment[1] = __float2half(0.0f);
            }
            int * sparse_fragment_int = reinterpret_cast<int *>(sparse_fragment);

            const int * dense_fragment_int = reinterpret_cast<const int *>(dense_fragment_ + n_group_idx * 16);
            
            #pragma unroll 4
            for (int i = 0; i < 4; i++){
                asm ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment_[0 + 4 * i]), "+f"(output_fragment_[1 + 4 * i]),
                    "+f"(output_fragment_[2 + 4 * i]), "+f"(output_fragment_[3 + 4 * i]):
                    "r"(dense_fragment_int[2*i]), "r"(dense_fragment_int[2*i+1]),
                    "r"(sparse_fragment_int[0])
					// "f"(output_fragment_[0 + 4 * i]), "f"(output_fragment_[1 + 4 * i]),
					// "f"(output_fragment_[2 + 4 * i]), "f"(output_fragment_[3 + 4 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }
        // Compute
        __device__ __forceinline__ void ResTileMAC(int n_group_idx, int residue){
            half sparse_fragment[2];
            // sparse_fragment[1] = __float2half(0.0f);
            if(tid_ >= 8){
                sparse_fragment[0] = __float2half(0.0f);
                sparse_fragment[1] = __float2half(0.0f);
            }
            // else if(2*lane_id_ >= residue){
            //     sparse_fragment[0] = __float2half(0.0f);
            //     // sparse_fragment[1] = 0.0;
            // }
            else if(2*lane_id_ >= residue-1){
                sparse_fragment[0] = *(sparse_tile_ + n_group_idx * 16);
                // sparse_fragment[1] = 0.0;
            }
            else{
                sparse_fragment[0] = *(sparse_tile_ + n_group_idx * 16);
                sparse_fragment[1] = *(sparse_tile_ + n_group_idx * 16 + 2);
            }
            int * sparse_fragment_int = reinterpret_cast<int *>(sparse_fragment);

            const int * dense_fragment_int = reinterpret_cast<const int *>(dense_fragment_ + n_group_idx * 16);
            
            #pragma unroll 4
            for (int i = 0; i < 4; i++){
                asm ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment_[0 + 4 * i]), "+f"(output_fragment_[1 + 4 * i]),
                    "+f"(output_fragment_[2 + 4 * i]), "+f"(output_fragment_[3 + 4 * i]):
                    "r"(dense_fragment_int[2*i]), "r"(dense_fragment_int[2*i+1]),
                    "r"(sparse_fragment_int[0])
					// "f"(output_fragment_[0 + 4 * i]), "f"(output_fragment_[1 + 4 * i]),
					// "f"(output_fragment_[2 + 4 * i]), "f"(output_fragment_[3 + 4 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }

        }
    };

}

#endif