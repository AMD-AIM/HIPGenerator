# HipKittens GEMM Optimization Guide for AMD MI350X (gfx950)

## 1. Hardware Specifications (MI350X / CDNA4)

| Component | Value | Optimization Impact |
|-----------|-------|---------------------|
| Compute Units (CUs) | 256 | More CUs = more parallelism |
| SIMDs per CU | 4 | Each runs 1 wavefront |
| Wavefront Size | 64 threads | Fixed, affects tile sizing |
| Max Waves per CU | 32 | Occupancy limit |
| Max Workgroup Size | 1024 threads | Block size constraint |
| VGPRs per SIMD | 65,536 (32-bit) | Register pressure limit |
| **LDS per CU** | **160 KB** | Critical for shared memory tiles |
| L2 Cache | 4 MB | Tile reuse benefit |
| XCDs | 32 | Cross-die scheduling matters |
| CUs per XCD | 8 | Locality optimization |

## 2. Recommended Tile Configuration (from Official HipKittens)

### Working Configuration (8192x8192 @ 1457 TFLOPS)

```cpp
constexpr int BLOCK_SIZE       = 256;   // Output tile: 256x256
constexpr int HALF_BLOCK_SIZE  = 128;   // Split into 128x128 sub-tiles
constexpr int K_STEP           = 64;    // K dimension step
constexpr int WARPS_M          = 2;     // Warps along M dimension
constexpr int WARPS_N          = 4;     // Warps along N dimension
constexpr int NUM_WARPS        = 8;     // Total warps = 2x4 = 8
constexpr int NUM_THREADS      = 512;   // 8 warps * 64 threads
constexpr int DOT_SLICE        = 32;    // For mma_ABt slicing
```

### Memory Layout

- **Shared Tiles A**: `st_bf<128, 64, st_16x32_s>` - 128 rows × 64 cols
- **Shared Tiles B**: `st_bf<128, 64, st_16x32_s>` - 128 rows × 64 cols  
- **Double Buffered**: `As[2][2]`, `Bs[2][2]` = 4 tiles each
- **Total LDS**: ~160KB (uses MAX_SHARED_MEMORY)

### Register Tiles

- **A Register**: `rt_bf<64, 64, row_l, rt_16x32_s>` - 64×64 bf16
- **B Register**: `rt_bf<32, 64, row_l, rt_16x32_s>` - 32×64 bf16
- **Accumulator**: `rt_fl<64, 32, col_l, rt_16x16_s>` - 64×32 float32 (2×2 array)

## 3. Key Optimization Techniques

### 3.1 XCD-Aware Scheduling

```cpp
// Swizzle workgroup IDs for better L2 locality within XCD
wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, 64);

// Group WGM consecutive row blocks together
const int WGM = 8;  // 8 row blocks per group
```

**Why**: MI350X has 32 XCDs with 8 CUs each. Keeping related workgroups on the same XCD improves L2 cache hit rate.

### 3.2 Double Buffering with Tic-Toc

```cpp
int tic = 0, toc = 1;
// Load next tiles while computing current
G::load(As[toc][0], g_a, {0, 0, row*2, tile+1}, ...);
// Compute with current tiles
mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
// Swap buffers
tic ^= 1; toc ^= 1;
```

**Why**: Overlaps memory loads with compute, hiding memory latency.

### 3.3 MFMA Instruction Scheduling (CRITICAL)

```cpp
// Main loop MUST use step=2 (8 mma operations per iteration)
for (int tile = 0; tile < num_tiles - 2; tile += 2) {
    // Phase 1: Process tiles from buffer 0, prefetch buffer 1
    load(B_tile_0, st_subtile_b0);
    load(A_tile, st_subtile_a0);
    G::load(As[1][1], g_a, {0, 0, row*2+1, tile+1}, ...);  // Prefetch
    asm volatile("s_waitcnt lgkmcnt(8)");  // Wait for LDS loads
    __builtin_amdgcn_s_barrier();

    asm volatile("s_waitcnt lgkmcnt(0)");  // Ensure all LDS ready
    __builtin_amdgcn_s_setprio(1);  // Raise priority
    mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
    __builtin_amdgcn_s_setprio(0);  // Lower priority  
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);  // CRITICAL: Scheduling barrier
    
    // ... 7 more mma operations following same pattern
}
```

**Key Scheduling Elements**:
1. **Loop step=2**: Process 8 mma operations together for better overlap
2. **s_waitcnt vmcnt(N)**: Precise control over pending global loads
3. **s_waitcnt lgkmcnt(N)**: Precise control over pending LDS operations
4. **sched_barrier(0)**: Prevent instruction reordering at critical points
5. **Conditional barriers**: `if (warp_row == 1)` prologue, `if (warp_row == 0)` epilogue

**Performance Impact**: These scheduling optimizations increased speedup from 0.80x to **1.09x** on 8192×8192!

### 3.4 Readfirstlane Hoisting

```cpp
// Hoist address calculations outside main loop
i32x4 a_srsrc_base = make_srsrc(a_base, M * a_row_stride, a_row_stride);
uint32_t a_lds_00 = __builtin_amdgcn_readfirstlane(...);
```

**Why**: Reduces scalar register pressure and loop overhead.

### 3.5 Swizzled Memory Access

```cpp
uint32_t swizzled_offsets_A[memcpy_per_tile/2];
G::prefill_swizzled_offsets(As[0][0], g.a, swizzled_offsets_A);
G::load(As[0][0], g.a, coords, swizzled_offsets_A, ...);
```

**Why**: Swizzled access patterns avoid LDS bank conflicts.

## 4. Performance Analysis

### Experimental Results (load_inline Demo Kernels)

| Demo Version | 4096×4096 | 8192×8192 | Key Optimizations |
|-------------|-----------|-----------|-------------------|
| Simple baseline | 0.87x | 0.80x | Basic double buffering |
| + XCD swizzling | 0.88x | 0.87x | chiplet_transform_chunked |
| + readfirstlane | 0.88x | 0.89x | SGPR address hoisting |
| **+ Full scheduling** | **1.07x** | **1.09x** | **waitcnt + sched_barrier** |

### Official Results (8192×8192 BF16)

| Implementation | Time (ms) | TFLOPS | vs PyTorch |
|---------------|-----------|--------|------------|
| PyTorch (torch.matmul) | 0.975 | 1127 | 1.00x |
| AITER (hipBlasLt) | 0.745 | 1476 | 1.19x |
| **HipKittens Official** | **0.877** | **1253** | **1.11x** |
| **Our Optimized Demo** | **0.899** | **1224** | **1.09x** |

### Key Finding: Scheduling is CRITICAL

The biggest performance gain (0.80x → 1.09x = **36% improvement**) came from:
1. **Main loop step=2** - Processing 8 mma operations per iteration
2. **Precise waitcnt** - `vmcnt(4)`, `vmcnt(6)`, `lgkmcnt(8)` at specific points
3. **sched_barrier(0)** - Preventing unwanted instruction reordering
4. **Conditional barriers** - Using warp_row conditions for fine-grained sync

### Performance Bottlenecks Identified

1. **LDS Usage**: 156KB is near the 160KB limit, reducing occupancy
2. **Memory Bandwidth**: Large matrices benefit from tiled reuse
3. **B Transpose**: `B.t().contiguous()` adds overhead if not cached
4. **Instruction Scheduling**: Default compiler scheduling is suboptimal for GEMM

## 5. Recommended Prompt Optimizations

### For LLM Code Generation

1. **Always use the 256×256×64 tile configuration** - Validated to work
2. **Use st_16x32_s subtile structure** - Required for mma_ABt compatibility
3. **Include XCD swizzling** - Critical for multi-XCD GPUs
4. **Cache B transpose** - Avoid repeated `.t().contiguous()` in forward
5. **Use double buffering** - Essential for hiding memory latency

### Key HipKittens Types to Use

```cpp
// Global layout
using _gl_bf16 = gl<bf16, -1, -1, -1, -1>;

// Shared tiles (with st_16x32_s for swizzled access)
using ST_A = st_bf<128, 64, st_16x32_s>;
using ST_B = st_bf<128, 64, st_16x32_s>;

// Register tiles (with rt_16x32_s for MFMA compatibility)
using RT_A = rt_bf<64, 64, row_l, rt_16x32_s>;
using RT_B = rt_bf<32, 64, row_l, rt_16x32_s>;

// Accumulator (float32 with rt_16x16_s for output)
using RT_C = rt_fl<64, 32, col_l, rt_16x16_s>;
```

## 6. Constraints and Limitations

1. **Matrix size must be divisible by 256** (BLOCK_SIZE)
2. **K dimension must be divisible by 64** (K_STEP)
3. **mma_ABt expects B transposed**: B[K,N] → B^T[N,K]
4. **LDS ~160KB required** - Uses MAX_SHARED_MEMORY

## 7. Future Optimization Directions

1. **Smaller tiles for higher occupancy**: Trade compute efficiency for parallelism
2. **Producer-consumer pattern**: Separate load and compute warps
3. **Async copy**: Use hardware copy engine for memory loads
4. **Stream-K decomposition**: Better load balancing for non-square matrices

---

*Document generated from HipKittens official examples and MI350X hardware specifications.*
*Reference: https://arxiv.org/abs/2511.08083*

