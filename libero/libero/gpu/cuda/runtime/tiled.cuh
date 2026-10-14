// Same-stream prepare -> disjoint private workers -> unique public commit.
// No global barrier, spinning counter or floating atomic accumulation.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
namespace tiled_execution {
// Caller preflight: tile >= 0, extent > 0, live >= 0, tile < gridDim.x.
struct Span { int begin, end; };
__device__ __forceinline__ Span span(int tile,int extent,int live) {
 const int64_t start=int64_t(tile)*int64_t(extent);
 const int begin=start<live?int(start):live;
 const int end=start+extent<live?int(start+extent):live;
 return {begin,end};
}
// Distinct slot per world/tile. Called only by that CTA's leader, after its
// local deterministic error reduction; prepare initializes every slot.
__device__ __forceinline__ void record(int w,int tile,int tiles,int code,int priority,
 int* errors,int* priorities) {
 const size_t slot=size_t(w)*size_t(tiles)+size_t(tile);
 errors[slot]=code; priorities[slot]=code?priority:INT_MAX;
}
// Collective coordinator entry. Inactive/admission flags are not modified.
__device__ __forceinline__ void clear(int w,int tiles,int* errors,int* priorities) {
 for(int t=int(threadIdx.x);t<tiles;t+=int(blockDim.x)) {
  const size_t slot=size_t(w)*size_t(tiles)+size_t(t);
  errors[slot]=0; priorities[slot]=INT_MAX;
 }
}
// Coordinator leader only, after same-stream worker kernel completion.
// Strict comparison preserves first tile for equal priorities. For contact
// errors use original contact ordinal as priority, never sorted queue ordinal.
__device__ __forceinline__ int stable_error(int w,int tiles,const int* errors,
 const int* priorities) {
 int code=0, best=INT_MAX;
 for(int t=0;t<tiles;++t) {
  const size_t slot=size_t(w)*size_t(tiles)+size_t(t);
  if(errors[slot] && (!code || priorities[slot]<best)) {
   code=errors[slot]; best=priorities[slot];
  }
 }
 return code;
}
// Coordinator collective; good must be uniform within this CTA. Never use
// private/public overlapping spans: callers validate aliasing before workers.
template<class T> __device__ __forceinline__ void copy_if_good(int w,int count,
 const T* private_output,T* public_output,bool good) {
 if(!good)return;
 const size_t base=size_t(w)*size_t(count);
 libero_vec::copy_block(private_output+base,public_output+base,size_t(count));
}
// Coordinator leader after a CTA barrier following all output copies.
// Semantic counters/epoch advance exactly once and only on success.
__device__ __forceinline__ void publish(int w,bool good,uint64_t* epoch,
 uint64_t* accepted_calls) {
 if(good) { if(epoch)++epoch[w]; if(accepted_calls)++accepted_calls[w]; }
}
} // namespace tiled_execution
