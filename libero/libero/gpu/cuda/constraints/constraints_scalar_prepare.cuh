// S05 cold scalar-parameter prototype. It is deliberately not selected by
// hot impedance until the complete static/options content lease is available.
#pragma once
namespace cf_s {
__device__ void scalar_parameter_prepare(int w,const double* ref,const double* ip,
  const int* kind,double* normalized,int* friction_route,int records,
  double timestep,int refsafe){
  if(w)return; // one cold model preparation, shared across resident worlds
  for(int id=threadIdx.x;id<records;id+=blockDim.x){
    double r0=ref[id*2],r1=ref[id*2+1];
    if((r0>0.)!=(r1>0.)){r0=.02;r1=1.;}
    if(refsafe&&r0>0.)r0=fmax(r0,2.*timestep);
    const double p0=fmin(.9999,fmax(.0001,ip[id*5]));
    const double p1=fmin(.9999,fmax(.0001,ip[id*5+1]));
    const double width=fmax(0.,ip[id*5+2]);
    const double mid=fmin(.9999,fmax(.0001,ip[id*5+3]));
    const double power=fmax(1.,ip[id*5+4]);
    const double k=kind[id]!=1?(r0>0.?1./fmax(1e-15,p1*p1*r0*r0*r1*r1):-r0/fmax(1e-15,p1*p1)):0.;
    const double b=r1>0.?2./fmax(1e-15,p1*r0):-r1/fmax(1e-15,p1);
    double imp=0.;int route=0;
    if(p0==p1||width<=1e-15)imp=.5*(p0+p1);
    else {imp=p0;route=2;} // source scalar friction pos==margin==0
    double* output=normalized+id*12;
    output[0]=r0;output[1]=r1;output[2]=p0;output[3]=p1;
    output[4]=width;output[5]=mid;output[6]=power;
    output[7]=k;output[8]=b;output[9]=imp;output[10]=0.;output[11]=0.;
    friction_route[id]=route;
  }
}
}
