#pragma once
namespace osc_cuda {
__device__ inline void seed_masks(int w,int*call,int*refresh,int*active,int*needs){active[w]=call[w];needs[w]=refresh[w];}
}
