#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace libero_cuda {
__device__ __forceinline__ double square(double x) { return x*x; }
__device__ __forceinline__ double clamp(double x, double lo, double hi) {
    return fmin(hi, fmax(lo, x));
}
__device__ __forceinline__ double dot3(const double* a, const double* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
__device__ __forceinline__ void cross3(const double* a, const double* b, double* out) {
    double x=a[1]*b[2]-a[2]*b[1], y=a[2]*b[0]-a[0]*b[2];
    double z=a[0]*b[1]-a[1]*b[0];
    out[0]=x; out[1]=y; out[2]=z;
}
__device__ __forceinline__ void copy3(const double* x, double* out) {
    const double a=x[0],b=x[1],c=x[2];out[0]=a;out[1]=b;out[2]=c;
}
// Quaternions use MuJoCo's (w,x,y,z) convention; rotate3 assumes unit length.
__device__ __forceinline__ void quat_mul(const double* a,const double* b,double* out) {
    double q[4]={a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
                 a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
                 a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
                 a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]};
    for(int i=0;i<4;++i) out[i]=q[i];
}
__device__ __forceinline__ void quat_rotate3(const double* q,const double* v,double* out) {
    double uv[3],uuv[3];cross3(q+1,v,uv);cross3(q+1,uv,uuv);
    for(int i=0;i<3;++i) out[i]=v[i]+2*(q[0]*uv[i]+uuv[i]);
}
__device__ __forceinline__ double dot6(const double* a,const double* b) {
    return dot3(a,b)+dot3(a+3,b+3);
}
// Spatial motion vectors are angular then linear, as in MuJoCo cvel.
__device__ __forceinline__ void cross_motion(const double* a,const double* b,double* out) {
    double angular[3],linear[3],extra[3];
    cross3(a,b,angular);cross3(a,b+3,linear);cross3(a+3,b,extra);
    for(int i=0;i<3;++i){out[i]=angular[i];out[i+3]=linear[i]+extra[i];}
}
__device__ __forceinline__ void cross_force(const double* a,const double* b,double* out) {
    double angular[3],linear[3],extra[3];
    cross3(a,b,angular);cross3(a+3,b+3,extra);cross3(a,b+3,linear);
    for(int i=0;i<3;++i){out[i]=angular[i]+extra[i];out[i+3]=linear[i];}
}
__device__ __forceinline__ int lane() { return threadIdx.x; }
__device__ __forceinline__ int width() { return blockDim.x; }
}  // namespace libero_cuda
