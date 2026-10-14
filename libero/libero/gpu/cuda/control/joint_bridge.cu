#include <cublasdx.hpp>
#ifndef JOINT_DIM
#error JOINT_DIM must be the cold native arm joint count
#endif
namespace joint_cuda {
__device__ void matvec(int w,int*active,int*error,double*matrix,double*input,double*out){
 __shared__ double a[JOINT_DIM*JOINT_DIM],b[JOINT_DIM],c[JOINT_DIM];
 __shared__ int allowed;
 if(threadIdx.x==0)allowed=active[w]==1&&error[w]==0;
 __syncthreads();if(!allowed)return;
 for(int i=threadIdx.x;i<JOINT_DIM*JOINT_DIM;i+=blockDim.x)a[i]=matrix[w*JOINT_DIM*JOINT_DIM+i];
 for(int i=threadIdx.x;i<JOINT_DIM;i+=blockDim.x){b[i]=input[w*JOINT_DIM+i];c[i]=0;}
 __syncthreads();
 using Op=decltype(cublasdx::Size<JOINT_DIM,1,JOINT_DIM>()+cublasdx::Precision<double>()+
 cublasdx::Type<cublasdx::type::real>()+
 cublasdx::Arrangement<cublasdx::row_major,cublasdx::row_major,cublasdx::row_major>()+
 cublasdx::Alignment<8,8,8>()+cublasdx::Function<cublasdx::function::MM>()+
 cublasdx::SM<DX_SM>()+cublasdx::BlockDim<64>()+cublasdx::Block());
 Op().execute(1.,a,b,0.,c);__syncthreads();
 for(int i=threadIdx.x;i<JOINT_DIM;i+=blockDim.x)out[w*JOINT_DIM+i]=c[i];
}
}
