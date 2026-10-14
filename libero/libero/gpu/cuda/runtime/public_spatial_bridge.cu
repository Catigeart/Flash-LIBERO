#include <cublasdx.hpp>
#ifndef SPATIAL_NV
#error SPATIAL_NV must be the native model velocity dimension
#endif
namespace public_spatial_cuda {
__device__ void velocity(int w,int*active,int*status,double*jp,double*jr,
                        double*qvel,double*vp,double*vr) {
  __shared__ double a[3*SPATIAL_NV],b[SPATIAL_NV],c[3];
  __shared__ int allowed;
  if(threadIdx.x==0)allowed=active[w]==1&&status[w]==0;
  __syncthreads();if(!allowed)return;
  using Op=decltype(cublasdx::Size<3,1,SPATIAL_NV>()+cublasdx::Precision<double>()+
    cublasdx::Type<cublasdx::type::real>()+
    cublasdx::Arrangement<cublasdx::row_major,cublasdx::row_major,cublasdx::row_major>()+
    cublasdx::Alignment<8,8,8>()+cublasdx::Function<cublasdx::function::MM>()+
    cublasdx::SM<DX_SM>()+cublasdx::BlockDim<64>()+cublasdx::Block());
  for(int i=threadIdx.x;i<SPATIAL_NV;i+=blockDim.x)b[i]=qvel[w*SPATIAL_NV+i];
  for(int component=0;component<2;++component) {
    const double*matrix=component==0?jp:jr;
    double*out=component==0?vp:vr;
    for(int i=threadIdx.x;i<3*SPATIAL_NV;i+=blockDim.x)a[i]=matrix[w*3*SPATIAL_NV+i];
    for(int i=threadIdx.x;i<3;i+=blockDim.x)c[i]=0.;
    __syncthreads();Op().execute(1.,a,b,0.,c);__syncthreads();
    for(int i=threadIdx.x;i<3;i+=blockDim.x)out[w*3+i]=c[i];
    __syncthreads();
  }
}
}
