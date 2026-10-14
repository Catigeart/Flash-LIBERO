#pragma once
#include <math.h>
namespace osc_config_cuda {
__device__ inline void reset_gains(int w,int*mask,double*base,double*gains){
 if(mask[w]!=1)return;for(int i=0;i<12;i++)gains[w*12+i]=base[i];
}
__device__ inline void gains(int w,int*active,int*policy,int*error,double*action,
 double*limits,double*gains,int mode,int adim){
 if(mode<1||active[w]!=1||policy[w]!=1||error[w])return;
 for(int i=0;i<6;i++){
  double kp=fmin(fmax(action[w*adim+(mode==2?6:0)+i],limits[i]),limits[6+i]);
  double damping=mode==2?fmin(fmax(action[w*adim+i],limits[12+i]),limits[18+i]):1.;
  double kd=2.*sqrt(kp)*damping;
  if(!isfinite(kp)||!isfinite(kd)){error[w]=2;return;}
  gains[w*12+i]=kp;gains[w*12+6+i]=kd;
 }
}
}
