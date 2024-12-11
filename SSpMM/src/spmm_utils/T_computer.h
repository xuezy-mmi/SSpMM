#ifndef T_COMPUTER_H
#define T_COMPUTER_H
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

            #pragma unroll
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

            #pragma unroll
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
            
            #pragma unroll
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

            #pragma unroll
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
                );
            }
        }
    };


    // Compute Tile for k=4
    template <typename VecType, int Tile_N>
    struct wmmaComputeUtils4 {
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
        // Register file fragment to accumulate results into
        float* output_fragment_;
        int thread_group_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils4(
            const half * lhs_tile,
            const half* rhs_fragment,
            float* output_fragment,
            int lane_id, int thread_group):
            lhs_tile_(reinterpret_cast<const float2 *>(lhs_tile) + lane_id),
            rhs_fragment_(rhs_fragment),
            output_fragment_(output_fragment),
            thread_group_(thread_group){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            float lhs_fragment[2];
            float waste[4];
            float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
            if (thread_group_ < 4) *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 4);
            int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * n_group_idx);

            #pragma unroll
            for (int i = 0; i < 2; i ++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 4 * i]), "+f"(output_fragment_[1 + 4 * i]),
                    "+f"(output_fragment_[2 + 4 * i]), "+f"(output_fragment_[3 + 4 * i]),
                    "+f"(waste[0]), "+f"(waste[1]), "+f"(waste[2]), "+f"(waste[3]):
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i]),
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }

        // Compute Residue
        __device__ __forceinline__ void TileMACResidue(int n_group_idx){
            float lhs_fragment[2];
            float waste[4];
            float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
            if (thread_group_ < 4) *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 4);
            int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * kTotalStep);

            #pragma unroll
            for (int i = 0; i < 2; i ++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 4 * i]), "+f"(output_fragment_[1 + 4 * i]),
                    "+f"(output_fragment_[2 + 4 * i]), "+f"(output_fragment_[3 + 4 * i]),
                    "+f"(waste[0]), "+f"(waste[1]), "+f"(waste[2]), "+f"(waste[3]):
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i]),
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }
    };

    // Compute Tile for k=2
    template <typename VecType, int Tile_N>
    struct wmmaComputeUtils2 {
        //
        // Static members
        //
        static constexpr int kTotalStep = Tile_N / 4 - 1;

        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile value
        const float* lhs_tile_;
        // Register file fragment storing the rhs tile
        const half* rhs_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;
        int thread_group_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils2(
            const half* lhs_tile,
            const half* rhs_fragment,
            float* output_fragment,
            int lane_id, int thread_group):
            lhs_tile_(reinterpret_cast<const float *>(lhs_tile) + lane_id),
            rhs_fragment_(rhs_fragment),
            output_fragment_(output_fragment),
            thread_group_(thread_group){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            float lhs_fragment[2];
            float waste[6];
            if (thread_group_ < 4) *(lhs_fragment) = *(lhs_tile_ + n_group_idx * 4);
            half* lhs_fragment_half = reinterpret_cast<half *>(lhs_fragment);
            *(lhs_fragment_half + 2) = *(lhs_fragment_half + 1);

            int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * n_group_idx);

            #pragma unroll
            for (int i = 0; i < 2; i ++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %2, %1, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %2, %1, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 2 * i]), "+f"(output_fragment_[1 + 2 * i]),
                    "+f"(waste[0]), "+f"(waste[1]), "+f"(waste[2]), "+f"(waste[3]), "+f"(waste[4]), "+f"(waste[5]):
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i]),
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }

        // Compute Residue
        __device__ __forceinline__ void TileMACResidue(int n_group_idx){
            float lhs_fragment[2];
            float waste[6];
            if (thread_group_ < 4) *(lhs_fragment) = *(lhs_tile_ + n_group_idx * 4);
            half* lhs_fragment_half = reinterpret_cast<half *>(lhs_fragment);
            *(lhs_fragment_half + 2) = *(lhs_fragment_half + 1);

            int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * kTotalStep);

            #pragma unroll
            for (int i = 0; i < 2; i ++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %2, %1, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %2, %1, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 2 * i]), "+f"(output_fragment_[1 + 2 * i]),
                    "+f"(waste[0]), "+f"(waste[1]), "+f"(waste[2]), "+f"(waste[3]), "+f"(waste[4]), "+f"(waste[5]):
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i]),
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }
    };


}

#endif