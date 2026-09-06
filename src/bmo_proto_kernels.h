// Path-B per-element tier mask + warp-ballot prefix (Jetson fused matvec when packing_version >= 5).
#pragma once

#include <cuda_runtime.h>

// Per-element packed mask: 4 tiers per uint8 (2 bits each), row-major over rows*cols.
// row_c16[r] must match production formula:
//   (r * blocks_per_row - row_c2[r]/block_size - row_c4[r]/block_size - row_c8[r]/block_size) * block_size
void launch_fused_dequant_matvec_proto(
    const void * pw,
    const void * pm_elem,
    const void * fp16_vals,
    const int32_t * row_c2,
    const int32_t * row_c4,
    const int32_t * row_c8,
    const int32_t * row_c16,
    int rows,
    int cols,
    int block_size,
    int n_2bit_bytes,
    int n_4bit_bytes,
    float scale_low,
    float scale_int4,
    float scale_int8,
    float zp_low,
    float zp_int4,
    float zp_int8,
    const float * x,
    float * y,
    int rows_per_block,
    cudaStream_t stream);
