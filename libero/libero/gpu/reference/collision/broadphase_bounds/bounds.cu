// C02: one CUDA lane owns one world.  Source order is MuJoCo 2.3.7's
// engine_collision_driver.c:689-747 and engine_util_solve.c:713-805.
// Build with -fmad=true --prec-div=true --prec-sqrt=true --ftz=false.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>

struct C02Output {
  int status, entered, count, iter, reason;
  double cen[3], cov[9], eigval[3], frame[9], quat[4], aabb[6*38];
  int valid[38], axis[500], stop[500];
  double offdiag[500], cosine[500];
};

struct C02Workspace {
  int n, g, b;
  int *geom_bodyid, *body_geomadr, *body_geomnum, *geom_contype, *geom_conaffinity;
  double *geom_rbound, *geom_margin, *geom_xpos, *local_cov;
  int *active;
  C02Output *output;
  cudaStream_t stream;
  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;
};

__device__ __forceinline__ double c02_div(double x, double y) { return __ddiv_rn(x, y); }
__device__ __forceinline__ double c02_sqrt(double x) { return __dsqrt_rn(x); }

__device__ void c02_quat2mat(double* res, const double* q) {
  if (q[0]==1 && q[1]==0 && q[2]==0 && q[3]==0) {
    res[0]=1; res[1]=0; res[2]=0; res[3]=0; res[4]=1;
    res[5]=0; res[6]=0; res[7]=0; res[8]=1;
  } else {
    const double q00=q[0]*q[0], q01=q[0]*q[1], q02=q[0]*q[2], q03=q[0]*q[3];
    const double q11=q[1]*q[1], q12=q[1]*q[2], q13=q[1]*q[3];
    const double q22=q[2]*q[2], q23=q[2]*q[3], q33=q[3]*q[3];
    res[0]=q00+q11-q22-q33; res[4]=q00-q11+q22-q33; res[8]=q00-q11-q22+q33;
    res[1]=2*(q12-q03); res[2]=2*(q13+q02); res[3]=2*(q12+q03);
    res[5]=2*(q23-q01); res[6]=2*(q13-q02); res[7]=2*(q23+q01);
  }
}

__device__ void c02_mulquat(double* res, const double* a, const double* b) {
  double x[4] = {
    a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
    a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
    a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
    a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]
  };
  for (int k=0;k<4;k++) res[k]=x[k];
}

__device__ bool c02_normalize4(double* x) {
  double norm=c02_sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3]);
  if(!isfinite(norm)) return false;
  if (norm < 1e-15) { x[0]=1; x[1]=x[2]=x[3]=0; }
  else { double inv=c02_div(1.0,norm); for (int k=0;k<4;k++) x[k]*=inv; }
  return true;
}

__device__ void c02_mat_t_mat(double* res, const double* a, const double* b) {
  for(int k=0;k<9;k++) res[k]=0;
  for(int i=0;i<3;i++) for(int j=0;j<3;j++) {
    double t=a[3*i+j];
    if(t) for(int k=0;k<3;k++) res[3*j+k]+=b[3*i+k]*t;
  }
}

__device__ void c02_mat_mat(double* res, const double* a, const double* b) {
  for(int k=0;k<9;k++) res[k]=0;
  for(int i=0;i<3;i++) for(int j=0;j<3;j++) {
    double t=a[3*i+j];
    if(t) for(int k=0;k<3;k++) res[3*i+k]+=b[3*j+k]*t;
  }
}

__device__ int c02_eig3(C02Output* out, const double* cov) {
  double D[9], tmp[9], *eig=out->eigval, *frame=out->frame, *quat=out->quat;
  double tau,t,c;
  quat[0]=1; quat[1]=quat[2]=quat[3]=0;
  int iter, rk,ck,rotk;
  out->reason=3;
  for(iter=0;iter<500;iter++) {
    c02_quat2mat(frame,quat);
    for(int k=0;k<9;k++) if(!isfinite(frame[k])) return -1;
    c02_mat_t_mat(tmp,frame,cov);
    for(int k=0;k<9;k++) if(!isfinite(tmp[k])) return -1;
    c02_mat_mat(D,tmp,frame);
    for(int k=0;k<9;k++) if(!isfinite(D[k])) return -1;
    eig[0]=D[0]; eig[1]=D[4]; eig[2]=D[8];
    if(fabs(D[1])>fabs(D[2]) && fabs(D[1])>fabs(D[5])) {rk=0;ck=1;rotk=2;}
    else if(fabs(D[2])>fabs(D[5])) {rk=0;ck=2;rotk=1;}
    else {rk=1;ck=2;rotk=0;}
    out->axis[iter]=rotk;
    out->offdiag[iter]=D[3*rk+ck];
    out->stop[iter]=-1;
    if(fabs(D[3*rk+ck])<1e-12) {out->reason=1;out->stop[iter]=1;break;}
    tau=c02_div(D[4*ck]-D[4*rk],2*D[3*rk+ck]);
    if(!isfinite(tau) || !isfinite(1+tau*tau)) return -1;
    if(tau>=0) t=c02_div(1.0,tau+c02_sqrt(1+tau*tau));
    else t=c02_div(-1.0,-tau+c02_sqrt(1+tau*tau));
    c=c02_div(1.0,c02_sqrt(1+t*t));
    if(!isfinite(t)||!isfinite(c)) return -1;
    out->cosine[iter]=c;
    if(c>1.0-1e-12) {out->reason=2;out->stop[iter]=2;break;}
    tmp[1]=tmp[2]=tmp[3]=0;
    tmp[rotk+1]=(tau>=0 ? -c02_sqrt(0.5-0.5*c) : c02_sqrt(0.5-0.5*c));
    if(rotk==1) tmp[rotk+1]=-tmp[rotk+1];
    tmp[0]=c02_sqrt(1.0-tmp[rotk+1]*tmp[rotk+1]);
    if(!c02_normalize4(tmp)) return -1;
    c02_mulquat(quat,quat,tmp);
    if(!c02_normalize4(quat)) return -1;
  }
  for(int j=0;j<3;j++) {
    int j1=j%2;
    if(eig[j1]<eig[j1+1]) {
      t=eig[j1]; eig[j1]=eig[j1+1]; eig[j1+1]=t;
      tmp[0]=0.707106781186548;
      tmp[1]=tmp[2]=tmp[3]=0;
      tmp[(j1+2)%3+1]=tmp[0];
      c02_mulquat(quat,quat,tmp);
      if(!c02_normalize4(quat)) return -1;
    }
  }
  c02_quat2mat(frame,quat);
  return iter;
}

__global__ void c02_kernel(C02Workspace w) {
  int world=blockIdx.x*blockDim.x+threadIdx.x;
  if(world>=w.n) return;
  C02Output* o=w.output+world;
  if(w.active[world]==0) {o->status=1; return;}
  if(w.active[world]!=1) {o->status=2;return;}
  const double* x=w.geom_xpos+(size_t)world*w.g*3;
  int cnt=0;
  double cen[3]={0,0,0}, cov[9]={0,0,0,0,0,0,0,0,0};
  for(int i=0;i<w.g;i++) {
    if(w.geom_bodyid[i]) {
      for(int k=0;k<3;k++) if(!isfinite(x[3*i+k])) {o->status=3;return;}
      for(int k=0;k<3;k++) cen[k]+=x[3*i+k];
      cnt++;
    }
  }
  if(cnt==0) {o->status=0;o->entered=0;o->count=0;return;}
  for(int k=0;k<3;k++) cen[k]=c02_div(cen[k],double(cnt));
  for(int i=0;i<w.g;i++) if(w.geom_bodyid[i]) {
    double dif[3]={x[3*i]-cen[0],x[3*i+1]-cen[1],x[3*i+2]-cen[2]};
    double D00=dif[0]*dif[0],D01=dif[0]*dif[1],D02=dif[0]*dif[2];
    double D11=dif[1]*dif[1],D12=dif[1]*dif[2],D22=dif[2]*dif[2];
    cov[0]+=D00;cov[1]+=D01;cov[2]+=D02;
    cov[3]+=D01;cov[4]+=D11;cov[5]+=D12;
    cov[6]+=D02;cov[7]+=D12;cov[8]+=D22;
  }
  for(int k=0;k<9;k++) cov[k]=c02_div(cov[k],double(cnt));
  for(int k=0;k<3;k++) if(!isfinite(cen[k])) {o->status=4;return;}
  for(int k=0;k<9;k++) if(!isfinite(cov[k])) {o->status=4;return;}
  o->count=cnt; o->entered=1;
  for(int k=0;k<3;k++) o->cen[k]=cen[k];
  for(int k=0;k<9;k++) o->cov[k]=cov[k];
  o->iter=c02_eig3(o,cov);
  if(o->iter<0) {o->status=5;return;}
  for(int k=0;k<3;k++) if(!isfinite(o->eigval[k])) {o->status=5;return;}
  for(int k=0;k<9;k++) if(!isfinite(o->frame[k])) {o->status=5;return;}
  for(int k=0;k<4;k++) if(!isfinite(o->quat[k])) {o->status=5;return;}
  for(int k=0;k<6;k++) o->aabb[k]=0;
  for(int body=0;body<w.b;body++) {
    int valid=0, start=w.body_geomadr[body], num=w.body_geomnum[body];
    for(int i=0;i<num;i++) {
      int geom=start+i;
      if(w.geom_contype[geom] || w.geom_conaffinity[geom]) {valid=1;break;}
    }
    o->valid[body]=valid;
    if(body==0) continue;
    if(num==0) {for(int k=0;k<6;k++) o->aabb[6*body+k]=0;continue;}
    for(int i=0;i<num;i++) {
      int geom=start+i;
      for(int j=0;j<3;j++) {
        double pos=x[3*geom]*o->frame[3*j]+x[3*geom+1]*o->frame[3*j+1]
                   +x[3*geom+2]*o->frame[3*j+2];
        double lo=pos-w.geom_rbound[geom]-w.geom_margin[geom];
        double hi=pos+w.geom_rbound[geom]+w.geom_margin[geom];
        if(!isfinite(pos)||!isfinite(lo)||!isfinite(hi)) {o->status=6;return;}
        int k=6*body+2*j;
        if(i==0) {o->aabb[k]=lo;o->aabb[k+1]=hi;}
        else {
          if(!(o->aabb[k]<=lo)) o->aabb[k]=lo;
          if(!(o->aabb[k+1]>=hi)) o->aabb[k+1]=hi;
        }
      }
    }
  }
  o->status=0;
}

__global__ void c02_eig_kernel(C02Workspace w) {
  int world=blockIdx.x*blockDim.x+threadIdx.x;
  if(world>=w.n)return;
  C02Output* o=w.output+world;
  if(w.active[world]==0) {o->status=1;return;}
  if(w.active[world]!=1) {o->status=2;return;}
  const double* cov=w.local_cov+9*world;
  for(int k=0;k<9;k++) if(!isfinite(cov[k])) {o->status=4;return;}
  for(int k=0;k<9;k++)o->cov[k]=cov[k];
  o->iter=c02_eig3(o,cov);
  if(o->iter<0) {o->status=5;return;}
  o->entered=1;
  for(int k=0;k<3;k++) if(!isfinite(o->eigval[k])) {o->status=5;return;}
  for(int k=0;k<9;k++) if(!isfinite(o->frame[k])) {o->status=5;return;}
  for(int k=0;k<4;k++) if(!isfinite(o->quat[k])) {o->status=5;return;}
  o->status=0;
}

extern "C" {
static int c02_alloc(void** p,size_t n) {return cudaMalloc(p,n)==cudaSuccess ? 0 : -1;}
static int c02_copy(C02Workspace* w,void* dst,const void* src,size_t n) {
  return cudaMemcpyAsync(dst,src,n,cudaMemcpyHostToDevice,w->stream)==cudaSuccess?0:-1;
}
const char* c02_error_string(int code) {return cudaGetErrorString((cudaError_t)code);}
void c02_destroy(C02Workspace* w);
C02Workspace* c02_create(int n,int g,int b,const int* geom_bodyid,const int* body_geomadr,
  const int* body_geomnum,const double* geom_rbound,const double* geom_margin,
  const int* geom_contype,const int* geom_conaffinity) {
  if(n<1||n>128||g<0||g>1024||b<1||b>38) return nullptr;
  C02Workspace* w=(C02Workspace*)calloc(1,sizeof(C02Workspace));
  if(!w) return nullptr;
  w->n=n;w->g=g;w->b=b;
  if(cudaStreamCreateWithFlags(&w->stream,cudaStreamNonBlocking)!=cudaSuccess) {free(w);return nullptr;}
  if(c02_alloc((void**)&w->geom_bodyid,g*4)||c02_alloc((void**)&w->body_geomadr,b*4)||
     c02_alloc((void**)&w->body_geomnum,b*4)||c02_alloc((void**)&w->geom_rbound,g*8)||
     c02_alloc((void**)&w->geom_margin,g*8)||c02_alloc((void**)&w->geom_contype,g*4)||
     c02_alloc((void**)&w->geom_conaffinity,g*4)||c02_alloc((void**)&w->geom_xpos,(size_t)n*g*3*8)||
     c02_alloc((void**)&w->active,n*4)||c02_alloc((void**)&w->local_cov,(size_t)n*9*8)||
     c02_alloc((void**)&w->output,n*sizeof(C02Output))) {c02_destroy(w);return nullptr;}
  if(c02_copy(w,w->geom_bodyid,geom_bodyid,g*4)||c02_copy(w,w->body_geomadr,body_geomadr,b*4)||
     c02_copy(w,w->body_geomnum,body_geomnum,b*4)||c02_copy(w,w->geom_rbound,geom_rbound,g*8)||
     c02_copy(w,w->geom_margin,geom_margin,g*8)||c02_copy(w,w->geom_contype,geom_contype,g*4)||
     c02_copy(w,w->geom_conaffinity,geom_conaffinity,g*4)||cudaStreamSynchronize(w->stream)!=cudaSuccess) {
    c02_destroy(w);return nullptr;
  }
  return w;
}
int c02_upload(C02Workspace* w,const double* xpos,const int* active) {
  if(!w) return -1;
  if(c02_copy(w,w->geom_xpos,xpos,(size_t)w->n*w->g*3*8)||
     c02_copy(w,w->active,active,w->n*4)) return -2;
  return cudaStreamSynchronize(w->stream)==cudaSuccess?0:-3;
}
int c02_seed(C02Workspace* w,const C02Output* output) {
  if(c02_copy(w,w->output,output,w->n*sizeof(C02Output))) return -1;
  return cudaStreamSynchronize(w->stream)==cudaSuccess?0:-2;
}
int c02_upload_cov(C02Workspace* w,const double* cov,const int* active) {
  if(c02_copy(w,w->local_cov,cov,(size_t)w->n*9*8)||c02_copy(w,w->active,active,w->n*4)) return -1;
  return cudaStreamSynchronize(w->stream)==cudaSuccess?0:-2;
}
int c02_run_eig(C02Workspace* w) {
  c02_eig_kernel<<<(w->n+31)/32,32,0,w->stream>>>(*w);
  return cudaGetLastError()==cudaSuccess?0:-1;
}
int c02_run(C02Workspace* w) {
  c02_kernel<<<(w->n+31)/32,32,0,w->stream>>>(*w);
  return cudaGetLastError()==cudaSuccess ? 0:-1;
}
int c02_capture_graph(C02Workspace* w) {
  if(w->graph_exec) return -1;
  if(cudaStreamBeginCapture(w->stream,cudaStreamCaptureModeGlobal)!=cudaSuccess) return -2;
  c02_kernel<<<(w->n+31)/32,32,0,w->stream>>>(*w);
  if(cudaStreamEndCapture(w->stream,&w->graph)!=cudaSuccess) return -3;
  return cudaGraphInstantiate(&w->graph_exec,w->graph,0)==cudaSuccess ? 0:-4;
}
int c02_graph(C02Workspace* w) {return cudaGraphLaunch(w->graph_exec,w->stream)==cudaSuccess ? 0:-1;}
int c02_sync(C02Workspace* w) {return cudaStreamSynchronize(w->stream)==cudaSuccess ? 0:-1;}
int c02_download(C02Workspace* w,C02Output* output) {
  if(cudaMemcpyAsync(output,w->output,w->n*sizeof(C02Output),
                     cudaMemcpyDeviceToHost,w->stream)!=cudaSuccess) return -2;
  return c02_sync(w)?-1:0;
}
void c02_destroy(C02Workspace* w) {
  if(!w)return;
  if(w->graph_exec)cudaGraphExecDestroy(w->graph_exec);
  if(w->graph)cudaGraphDestroy(w->graph);
  cudaFree(w->geom_bodyid);cudaFree(w->body_geomadr);cudaFree(w->body_geomnum);
  cudaFree(w->geom_rbound);cudaFree(w->geom_margin);cudaFree(w->geom_contype);
  cudaFree(w->geom_conaffinity);cudaFree(w->geom_xpos);cudaFree(w->local_cov);
  cudaFree(w->active);cudaFree(w->output);
  cudaStreamDestroy(w->stream);free(w);
}
size_t c02_output_size() {return sizeof(C02Output);}
}
