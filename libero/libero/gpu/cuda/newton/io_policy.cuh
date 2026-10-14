#pragma once
// Newton-only I/O selection. The production core chooses scalar for the two
// measured regression shapes (NV21/NV43); other entry points default to their
// established packet path. Arithmetic, capacity and commit guards stay local.
#include "../common/vector.cuh"
#ifndef LIBERO_NEWTON_SCALAR_IO
#define LIBERO_NEWTON_SCALAR_IO 0
#endif
namespace libero_newton_io {
template<class T> __device__ __forceinline__ void copy_block(const T* src,T* dst,size_t n){
#if LIBERO_NEWTON_SCALAR_IO
  for(size_t i=threadIdx.x;i<n;i+=blockDim.x)dst[i]=src[i];
#else
  libero_vec::copy_block(src,dst,n);
#endif
}
template<class T> __device__ __forceinline__ void copy_serial(const T* src,T* dst,size_t n){
#if LIBERO_NEWTON_SCALAR_IO
  for(size_t i=0;i<n;++i)dst[i]=src[i];
#else
  libero_vec::copy_serial(src,dst,n);
#endif
}
template<class T> __device__ __forceinline__ void fill_block(T* dst,size_t n,T value){
#if LIBERO_NEWTON_SCALAR_IO
  for(size_t i=threadIdx.x;i<n;i+=blockDim.x)dst[i]=value;
#else
  libero_vec::fill_block(dst,n,value);
#endif
}
template<class T> __device__ __forceinline__ void fill_serial(T* dst,size_t n,T value){
#if LIBERO_NEWTON_SCALAR_IO
  for(size_t i=0;i<n;++i)dst[i]=value;
#else
  libero_vec::fill_serial(dst,n,value);
#endif
}
} // namespace libero_newton_io
