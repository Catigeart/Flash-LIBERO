// Diagnostic helper candidate ONLY; not integrated with any Newton runtime.
// One uniform 64-thread block/world. Every original k column completes before
// the next. Each i owns its original data cell and vector element. NV<=57 in
// the actual corpus; no row/contact/rank-update order is changed by this helper.
#include <cuda_runtime.h>

extern "C" __global__ void n_cooperative_rank_update_r01(
    double* data, double* vec, const int* plus, const int* active,
    const int* status, int* rank_output, int nv, int nworld) {
  int w=blockIdx.x;
  int lane=threadIdx.x;
  if (w>=nworld || active[w]!=1 || status[w]!=0) return;
  // Cold launch contract requires nv in [1,57], blockDim.x=64. No fallback
  // or truncation is allowed; dimension/launch validation lives in the future
  // isolated diagnostic loader, never a device silent success.
  __shared__ double c;
  __shared__ double cinv;
  __shared__ double s;
  __shared__ int nonzero;
  __shared__ int rank;
  if (lane==0) rank=nv;
  __syncthreads();
  int hb=w*nv*nv;
  int vb=w*nv;
  for (int k=0;k<nv;++k) {
    if (lane==0) {
      double x=vec[vb+k];
      nonzero=(x!=0.0);
      if (nonzero) {
        double Lkk=data[hb+k*nv+k];
        double tmp=__dmul_rn(Lkk,Lkk);
        double xx=__dmul_rn(x,x);
        tmp=plus[w]!=0 ? __dadd_rn(tmp,xx) : __dsub_rn(tmp,xx);
        if (tmp<1.0e-15) {tmp=1.0e-15;--rank;}
        double r=__dsqrt_rn(tmp);
        c=__ddiv_rn(r,Lkk);
        cinv=__ddiv_rn(1.0,c);
        s=__ddiv_rn(x,Lkk);
        data[hb+k*nv+k]=r;
      }
    }
    __syncthreads();
    if (nonzero && lane>k && lane<nv) {
      int i=lane;
      double current=data[hb+i*nv+k];
      double sv=__dmul_rn(s,vec[vb+i]);
      double updated=plus[w]!=0 ? __dadd_rn(current,sv) : __dsub_rn(current,sv);
      data[hb+i*nv+k]=__dmul_rn(updated,cinv);
      // Original two i-loops can interleave across i: no other element reads
      // this i's updated cell or vec. Within i the data write precedes vec.
      vec[vb+i]=__dsub_rn(__dmul_rn(c,vec[vb+i]),__dmul_rn(s,data[hb+i*nv+k]));
    }
    __syncthreads();
  }
  if (lane==0) rank_output[w]=rank;
}
