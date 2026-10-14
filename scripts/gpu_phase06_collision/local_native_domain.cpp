// CPU-only source-control/write-domain witness; never a device detector.
#include "cpu_native_source.h"
extern "C" int p06c_source_domain(
 double* size, double* rbound, int* type, int* dataid,
 float* vert, int* vertadr, int* vertnum, int* graphadr, int* graph,
 double* xpos, double* xmat, int g1, int g2, double margin,
 int maxiter, double tolerance, int multiccd, int nmesh, int nvert, int ngraph,
 const unsigned long long* before, unsigned long long* after,
 int* diag, int* fw, int* mask, int* error) {
 static_assert(sizeof(double)==8 && sizeof(int)==4 && sizeof(mjContact)==104,"private source ABI");
 mjContact con[50];
 for (int i=0;i<50;i++) {
  memcpy(&con[i].dist,before+10*i,8);
  memcpy(con[i].pos,before+10*i+1,24);
  memcpy(con[i].frame,before+10*i+4,48);
  for (int k=6;k<9;k++) con[i].frame[k]=0;
 }
 memset(diag,0,16*sizeof(int)); memset(mask,0,500*sizeof(int)); *fw=0; *error=0;
 mjModel m={size,rbound,type,dataid,vert,vertadr,vertnum,graphadr,graph,
   {maxiter,multiccd?1:0,tolerance},error,diag,fw,nvert,ngraph,nmesh,4096,mask,con};
 mjData d={xpos,xmat};
 int n=type[g1]==6 && type[g2]==6 ? mjc_BoxBox(&m,&d,con,g1,g2,margin) :
                                        mjc_Convex(&m,&d,con,g1,g2,margin);
 diag[4]=n;diag[5]=*fw;
 for (int i=0;i<50;i++) {
  memcpy(after+10*i,&con[i].dist,8);
  memcpy(after+10*i+1,con[i].pos,24);
  memcpy(after+10*i+4,con[i].frame,48);
 }
 return n;
}
