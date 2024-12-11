#ifndef SPMM_COMPUTE_UTILS_H
#define SPMM_COMPUTE_UTILS_H

namespace spmm{
    // Tile_N=128 8-bit 4 warps
    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_8b{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_8b(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_(output_fragment){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[1];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            if(lane_id_ % 32 < ValuesBlockWidth)
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth + (step % 2) * ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;

            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[4 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[1];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            if(lane_id_ % 32 < ValuesBlockWidth)
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;

            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[4 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                );
            }
        }
    };

    // Tile_N=128 16-bit 8-bit 4 warps
    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_16b8b{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_16b8b(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_(output_fragment){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[1];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            if(lane_id_ % 32 < ValuesBlockWidth)
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth + (step % 2) * ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;

            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[4 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[1];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            if(lane_id_ % 32 < ValuesBlockWidth)
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;

            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[0 + i]), "+r"(output_fragment_[4 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                );
            }
        }
    };

    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_16b8b8v{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_0_;
        int* output_fragment_1_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_16b8b8v(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment_0,
            int* output_fragment_1,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_0_(output_fragment_0),
            output_fragment_1_(output_fragment_1){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[2];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32 + (step % 2) * ValuesBlockWidth];
	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32 + (step % 2) * ValuesBlockWidth];
            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[0 + i]), "+r"(output_fragment_0_[4 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                );
            }

            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[0 + i]), "+r"(output_fragment_1_[4 + i]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment_transpose[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[2];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32];
	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32];

            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[0 + i]), "+r"(output_fragment_0_[4 + i]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                );
            }

            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[0 + i]), "+r"(output_fragment_1_[4 + i]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment_transpose[i])
                );
            }
        }
    };

    // Tile_N=64 16-bit 16-bit 4 warps
    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_16b{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_16b(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_(output_fragment){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[1];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            if(lane_id_ % 32 < ValuesBlockWidth)
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth + (step % 2) * ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;

            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[i%2*4 + i/2]), "+r"(output_fragment_[i%2*4 + 2 + i/2]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[1];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

            if(lane_id_ % 32 < ValuesBlockWidth)
	        lhs_fragment[0] = lhs_tile_[lane_id_ % ValuesBlockWidth];
	    else
		lhs_fragment[0] = 0;

            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_[i%2*4 + i/2]), "+r"(output_fragment_[i%2*4 + 2 + i/2]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                );
            }
        }
    };

    // Tile_N=64 16-bit 16-bit 4 warps 8v
    template <int ValuesBlockWidth>
    struct wmmaComputeUtils_16b8v{

        // Shared memory buffers
        const int* lhs_tile_;
        const int* dense_tile_;
        // Register file fragment to accumulate results into
        int* output_fragment_0_;
        int* output_fragment_1_;
        int lane_id_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils_16b8v(
            const int* lhs_tile,
            const int* dense_tile,
            int* output_fragment_0,
            int* output_fragment_1,
            int lane_id):
            lhs_tile_(lhs_tile),
            lane_id_(lane_id),
            dense_tile_(dense_tile),
            output_fragment_0_(output_fragment_0),
            output_fragment_1_(output_fragment_1){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(int step){
            int lhs_fragment[2];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32 + (step % 2) * ValuesBlockWidth]; // ValuesBlockWidth = 64
            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[i%2*4 + i/2]), "+r"(output_fragment_0_[i%2*4 + 2 + i/2]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                );
            }

	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32 + (step % 2) * ValuesBlockWidth];
            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[i%2*4 + i/2]), "+r"(output_fragment_1_[i%2*4 + 2 + i/2]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment_transpose[i])
                );
            }
        }

        __device__ __forceinline__ void TileMACResidue(){
            int lhs_fragment[2];
            int rhs_fragment[4];
            int rhs_fragment_transpose[4];
	    int chunk_id = lane_id_ % 4;

	    // TODO:
            // This is a conflict-free design for Tile_N=128 with four warps.
	    // Should change these magic numbers for other Tile_N.
	    int base_offset = chunk_id * 72 + (lane_id_ % 64) / 4 + (lane_id_ / 64) * 288;

            #pragma unroll
	    for(int i=0; i<4; i++){
	        rhs_fragment[i] = *(dense_tile_ + base_offset + i*16); 
	    }

            unsigned char *rhs_fragment_char = reinterpret_cast<unsigned char *>(rhs_fragment); 
            unsigned char *rhs_fragment_transpose_char = reinterpret_cast<unsigned char *>(rhs_fragment_transpose);

            #pragma unroll
	    for(int i=0; i<4; i++){
	        for(int j=0; j<4; j++){
	            *(rhs_fragment_transpose_char + j*4 + i) = *(rhs_fragment_char + j + i*4);
	        }
	    }

	    lhs_fragment[0] = lhs_tile_[lane_id_ % 32]; // ValuesBlockWidth = 64
            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_0_[i%2*4 + i/2]), "+r"(output_fragment_0_[i%2*4 + 2 + i/2]):
                    "r"(lhs_fragment[0]),
                    "r"(rhs_fragment_transpose[i])
                );
            }

	    lhs_fragment[1] = lhs_tile_[lane_id_ % 32 + 32];
            #pragma unroll
            for (int i = 0; i < 4; i ++){
                asm("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 \t"
                    "{%0, %1}, \t"
                    "{%2}, \t"
                    "{%3}, \t"
                    "{%0, %1}; ":
                    "+r"(output_fragment_1_[i%2*4 + i/2]), "+r"(output_fragment_1_[i%2*4 + 2 + i/2]):
                    "r"(lhs_fragment[1]),
                    "r"(rhs_fragment_transpose[i])
                );
            }

        }
    };


}
#endif