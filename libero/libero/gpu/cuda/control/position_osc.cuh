#pragma once
namespace position_cuda {
// Original robosuite OSC.set_goal use_ori=False sets this exact fixed matrix
// at each policy goal update, independently of the current end-effector pose.
__device__ inline void orientation(int w,int*active,int*policy,int*error,double*goal){
 if(active[w]!=1||policy[w]!=1||error[w])return;
 const double downward[9]={0.,1.,0.,1.,0.,0.,0.,0.,-1.};
 for(int i=0;i<9;i++)goal[w*9+i]=downward[i];
}
}
