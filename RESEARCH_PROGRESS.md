# HipKittens GEMM Kernel Generation Research Progress

## Iterative Optimization Framework

We have created an iterative optimization framework with 3 optimization levels:

### Level 1: Base Double Buffering (0.85x)
- **File**: `prompts/hipkittens_gemm_opt1.txt` (same as base)
- **Features**:
  - Shared memory double buffering
  - sched_barrier + setprio around MMA
  - 8-wave pingpong pattern (all warps do both load and compute)
- **Target**: 0.5x

### Level 2: Deeper Pipelining (0.85x)
- **File**: `prompts/hipkittens_gemm_opt2.txt`
- **Features**:
  - 2-tile prefetch (load tile k+2 while computing k)
  - Fine-grained waitcnt (vmcnt(4) instead of vmcnt(0))
- **Target**: 0.7x

### Level 3: 8c4p Producer-Consumer (0.75x)
- **File**: `prompts/hipkittens_gemm_opt3.txt`
- **Features**:
  - 4 producer warps (dedicated to global→shared loads)
  - 8 consumer warps (dedicated to shared→register→MMA)
  - 12 warps total, fully asynchronous overlap
  - Custom `warpgroupid()` function added
- **Target**: 0.95x
- **Note**: Currently slower than Level 1/2 due to:
  - Overhead of 12 warps vs 8 warps
  - More complex synchronization
  - May need larger matrices to show benefit

## Performance Results (4096x4096 GEMM)

| Level | Pattern | Speedup | Notes |
|-------|---------|---------|-------|
| Base (opt1) | 8-wave pingpong | 0.85x | Simplest, most robust |
| opt2 | Deeper pipeline | 0.85x | Same as base |
| opt3 | 8c4p producer-consumer | 0.75x | Overhead from extra warps |

## Key Findings

### warpgroupid Issue
- The HipKittens library does not define `kittens::warpgroupid()`
- We added a custom implementation:
```cpp
__device__ __forceinline__ int warpgroupid() { return threadIdx.x >> 8; }
```

### XCD Scheduling Complexity
- The reference kernel uses `chiplet_transform_chunked` for XCD-aware scheduling
- This requires careful tuning of chunk sizes and is complex for LLM to get right
- Simplified version works but may lose some L2 cache efficiency

### Producer-Consumer Trade-offs
- **Pros**: Better overlap between load and compute
- **Cons**: 
  - 12 warps = more register pressure
  - Extra synchronization overhead
  - Idle producer warps during compute phases

## Adaptive Optimization

Created unified adaptive prompt (`hipkittens_gemm_adaptive.txt`) that:
- Explains multiple optimization strategies
- Lets LLM choose based on matrix dimensions
- Provides concrete kernel templates

## Challenges Identified

1. **Fine-grained Interleave Complexity**: The reference kernel uses very precise scheduling:
   - Step 2 k-tiles per iteration
   - Interleaved load/compute within each step
   - Precise lgkmcnt(8)/vmcnt(6) values
   - This is hard for LLM to generate correctly without a very detailed template

2. **Producer-Consumer Overhead**: 12 warps (4 producer + 8 consumer) adds overhead that only pays off on very large matrices or fused kernels

3. **warpgroupid Missing**: HipKittens doesn't define `kittens::warpgroupid()`, requiring custom implementation

## Next Steps

1. **Large Matrix Testing**: Test all strategies on 8192x8192 or larger to see where 8c4p wins
2. **Fused Kernel Testing**: Test producer-consumer on GEMM+activation to see overlap benefits
3. **Interleave Template**: Create a more detailed interleave template if needed for >0.9x

## Files

```
prompts/
├── hipkittens_gemm.txt          # Base prompt
├── hipkittens_gemm_opt1.txt     # Level 1 (same as base)
├── hipkittens_gemm_opt2.txt     # Level 2 (deeper pipeline)
├── hipkittens_gemm_opt3.txt     # Level 3 (8c4p producer-consumer)
├── hipkittens_gemm_adaptive.txt # Unified adaptive prompt
└── config.json                  # Pattern matching config

optimize_iterative.py            # Framework script
```

