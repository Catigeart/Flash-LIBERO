// Apache-2.0 MuJoCo 2.3.7 BoxBox edge tail, engine_collision_box.c:977-1332.
// SAT exit is the exact binary64 input boundary. No CPU tail values enter.
#include <cuda_runtime.h>
#include <math.h>
#define mjMINVAL 1e-15
#define mjMAXCONPAIR 50
typedef double mjtNum;
struct Contact { double dist, pos[3], frame[6]; };
__device__ __forceinline__ void mju_zero3(double* a){a[0]=0;a[1]=0;a[2]=0;}
__device__ __forceinline__ void mju_zero(double* a,int n){for(int i=0;i<n;i++)a[i]=0;}
__device__ __forceinline__ void mju_copy3(double* a,const double* b){for(int i=0;i<3;i++)a[i]=b[i];}
__device__ __forceinline__ void mju_copy(double* a,const double* b,int n){for(int i=0;i<n;i++)a[i]=b[i];}
__device__ __forceinline__ void mju_scl3(double* a,const double* b,double s){for(int i=0;i<3;i++)a[i]=b[i]*s;}
__device__ __forceinline__ void mju_add3(double* a,const double* b,const double* c){for(int i=0;i<3;i++)a[i]=b[i]+c[i];}
__device__ __forceinline__ void mju_sub3(double* a,const double* b,const double* c){for(int i=0;i<3;i++)a[i]=b[i]-c[i];}
__device__ __forceinline__ void mju_addToScl3(double* a,const double* b,double s){for(int i=0;i<3;i++)a[i]+=b[i]*s;}
__device__ __forceinline__ double mju_dot3(const double* a,const double* b){return (a[0]*b[0]+a[1]*b[1])+a[2]*b[2];}
__device__ __forceinline__ double mju_abs(double x){return fabs(x);}
__device__ __forceinline__ void mju_transpose(double* a,const double* b,int n,int){for(int i=0;i<n;i++)for(int j=0;j<n;j++)a[i*n+j]=b[j*n+i];}
__device__ __forceinline__ void mju_rotVecMat(double* a,const double* v,const double* m){for(int i=0;i<3;i++)a[i]=(m[i*3]*v[0]+m[i*3+1]*v[1])+m[i*3+2]*v[2];}
__device__ __forceinline__ void mju_rotVecMatT(double* a,const double* v,const double* m){for(int i=0;i<3;i++)a[i]=(m[i]*v[0]+m[i+3]*v[1])+m[i+6]*v[2];}
// mju_dot(n=3) in engine_util_blas.c adds the initial zero at the end.
__device__ __forceinline__ double dot_mat(const double* a,const double* b){volatile double z=0;return z+((a[0]*b[0]+a[1]*b[1])+a[2]*b[2]);}
__device__ __forceinline__ void mju_mulMatMatT(double* r,const double* a,const double* b,int,int,int){for(int i=0;i<3;i++)for(int j=0;j<3;j++)r[3*i+j]=dot_mat(a+3*i,b+3*j);}

#define rotaxis(vecres, vecin) { (vecres)[0]=(vecin)[i0]*f0; (vecres)[1]=(vecin)[i1]*f1; (vecres)[2]=(vecin)[i2]*f2; }
#define rotmatx(matres, matin) { mju_scl3((matres)+0,(matin)+i0*3,f0); mju_scl3((matres)+3,(matin)+i1*3,f1); mju_scl3((matres)+6,(matin)+i2*3,f2); }

// status: 0 inactive/padding, 1 not entered, 2 edge, 3 tiny rnorm,
// 4 invalid count, 5 invalid active, 6 invalid SAT tuple, 7 numeric domain.
__device__ int edge_tail(const double* pair_f,const double* sat_f,const int* sat_i,
                         double* raw,int* stage_i,double* stage_d,
                         int* trace_i,double* trace_d,int* frame_written,double* frame0,
                         int wrong_d0){
  const double margin=pair_f[0],*size1=pair_f+1,*pos1=pair_f+4,*mat1=pair_f+7;
  const double *size2=pair_f+16,*pos2=pair_f+19,*mat2=pair_f+22;
  const double *pos21=sat_f,*pos12=sat_f+3,*rot=sat_f+6,*rott=sat_f+15;
  const double *rotabs=sat_f+24,*rottabs=sat_f+33,*clnorm=sat_f+42;
  double tmp1[3],tmp2[3],rotmore[9],p[3],r[9],s[3],rt[9];
  double points[50][3],depth[50],pts[6][3],ppts2[4][2],pu[4][3],axi[3][3];
  double linesu[4][6],lines[4][6],rnorm[3];
  double c1,c2,a,b,c,d,lx,ly,hz,l,x,y,u,v,llx,lly,innorm,margin2;
  int i0,i1,i2,i,j,q,code,q1,q2,n,m,k,cle1,cle2,in,ax1,ax2,pax1,pax2,clface,nl,nf;
  double f0,f1,f2;
  Contact con[50];
  code=sat_i[0];cle1=sat_i[1];cle2=sat_i[2];in=sat_i[3];
  margin2=margin*margin;
  n=0;
edgeedge:


  stage_i[0]=code;stage_i[1]=code-12;stage_i[2]=(code-12)/3;stage_i[3]=(code-12)%3;stage_i[4]=cle1;stage_i[5]=cle2;trace_i[0]=in;trace_i[1]=code;
  code -= 12;

  q1 = code / 3;
  q2 = code % 3;



  if (q2 == 0)
    ax1 = 1, ax2 = 2;
  if (q2 == 1)
    ax1 = 0, ax2 = 2;
  if (q2 == 2)
    ax1 = 1, ax2 = 0;
  if (q1 == 0)
    pax1 = 1, pax2 = 2;
  if (q1 == 1)
    pax1 = 0, pax2 = 2;
  if (q1 == 2)
    pax1 = 1, pax2 = 0;

  // printf("%lf %lf   %lf %lf\n",rot[ 3*q1+ ax1],rot [3*q1+ ax2],rott[3*q2+pax1],rott[3*q2+pax2]);
  // printf("%lf %lf\n",mju_dot3(clnorm,rott+3*ax1),mju_dot3(clnorm,rott+3*ax2));

  if (rotabs [3 * q1 + ax1] < rotabs [3 * q1 + ax2]) {
    ax1 = ax2;
    ax2 = 3 - q2 - ax1;
  }
  if (rottabs[3 * q2 + pax1] < rottabs[3 * q2 + pax2]) {
    pax1 = pax2;
    pax2 = 3 - q1 - pax1;
  }

  if (cle1 & (1 << pax2))
    clface = pax2;
  else
    clface = pax2 + 3;

  stage_i[6]=ax1;stage_i[7]=ax2;stage_i[8]=pax1;stage_i[9]=pax2;stage_i[10]=clface;

  // printf("%lf - %d %d %d %d   %d %d     %d %d %d %d %d\n",
  //        penetration,cle1,cle2,code,in,q1,q2,clface,ax1,ax2,pax1,pax2);


  mju_zero(rotmore, 9);
  if (clface == 0)
    rotmore[2] = -1, rotmore[4] = +1, rotmore[6] = +1;
  else if (clface == 1)
    rotmore[0] = +1, rotmore[5] = -1, rotmore[7] = +1;
  else if (clface == 2)
    rotmore[0] = +1, rotmore[4] = +1, rotmore[8] = +1;
  else if (clface == 3)
    rotmore[2] = +1, rotmore[4] = +1, rotmore[6] = -1;
  else if (clface == 4)
    rotmore[0] = +1, rotmore[5] = +1, rotmore[7] = -1;
  else if (clface == 5)
    rotmore[0] = -1, rotmore[4] = +1, rotmore[8] = -1;


  i0 = 0;
  i1 = 1;
  i2 = 2;
  f0 = f1 = f2 = 1;

  if (clface == 0) {
    i0 = 2;
    f0 = -1;
    i2 = 0;
  } else if (clface == 1) {
    i1 = 2;
    f1 = -1;
    i2 = 1;
  } else if (clface == 2) {
  } else if (clface == 3) {
    i0 = 2;
    i2 = 0;
    f2 = -1;
  } else if (clface == 4) {
    i1 = 2;
    i2 = 1;
    f2 = -1;
  } else if (clface == 5) {
    f0 = -1;
    f2 = -1;
  }

  // mju_rotVecMat(p,pos21,rotmore);
  // mju_rotVecMat(rnorm,clnorm,rotmore);
  rotaxis(p, pos21);
  rotaxis(rnorm, clnorm);

  // print("rnorm",rnorm);

  // mju_mulMatMat(r,rotmore,rot,3,3,3);
  rotmatx(r, rot);

  mju_rotVecMatT(tmp1, size1, rotmore);
  for (i = 0; i < 3; i++)
    s[i] = mju_abs(tmp1[i]);

  mju_transpose(rt, r, 3, 3);


  lx = s[0];
  ly = s[1];
  hz = s[2];
  p[2] -= hz;


  n = 0;
  mju_copy3(points[n], p);
  mju_addToScl3(points[n], rt + 3 * ax1, size2[ax1] * ((cle2 & (1 << ax1)) ? 1 : -1));
  mju_addToScl3(points[n], rt + 3 * ax2, size2[ax2] * ((cle2 & (1 << ax2)) ? 1 : -1));
  mju_copy3(points[n + 1], points[n]);
  mju_addToScl3(points[n], rt + 3 * q2, size2[q2]);
  n = 1;
  mju_addToScl3(points[n], rt + 3 * q2, -size2[q2]);
  n = 2;


  mju_copy3(points[n], p);
  mju_addToScl3(points[n], rt + 3 * ax1, size2[ax1] * ((cle2 & (1 << ax1)) ? -1 : 1));
  mju_addToScl3(points[n], rt + 3 * ax2, size2[ax2] * ((cle2 & (1 << ax2)) ? 1 : -1));
  mju_copy3(points[n + 1], points[n]);
  mju_addToScl3(points[n], rt + 3 * q2, size2[q2]);
  n = 3;
  mju_addToScl3(points[n], rt + 3 * q2, -size2[q2]);
  n = 4;


  mju_copy3(axi[0], points[0]);
  mju_sub3(axi[1], points[1], points[0]);
  mju_sub3(axi[2], points[2], points[0]);


  if(!isfinite(rnorm[2])){stage_i[60]=1;return 0;} stage_i[11]=(fabs(rnorm[2])<mjMINVAL);for(int z=0;z<9;z++)stage_d[z]=rotmore[z];for(int z=0;z<3;z++){stage_d[9+z]=p[z];stage_d[30+z]=s[z];stage_d[33+z]=rnorm[z];}for(int z=0;z<9;z++){stage_d[12+z]=r[z];stage_d[21+z]=rt[z];}stage_d[36]=lx;stage_d[37]=ly;stage_d[38]=hz;for(int z=0;z<12;z++)stage_d[39+z]=points[z/3][z%3];
  if (fabs(rnorm[2]) < mjMINVAL)
    return 0;

  innorm = (1 / rnorm[2]) * (in ? -1 : 1);
  // printf("%lf\n",innorm);

  for (i = 0; i < 4; i++)
  {
    c1 = -points[i][2] * (1 / rnorm[2]);

    mju_copy3(pu[i], points[i]);

    mju_addToScl3(points[i], rnorm, c1);

    // ppts[i][0]=points[i][0];
    // ppts[i][1]=points[i][1];
    ppts2[i][0] = points[i][0];
    ppts2[i][1] = points[i][1];
  }


  mju_copy3(pts[0], points[0]);
  mju_sub3(pts[1], points[1], points[0]);
  mju_sub3(pts[2], points[2], points[0]);

  m = 3;
  k = 0;
  n = 0;

  stage_d[51]=innorm;for(int z=0;z<12;z++)stage_d[52+z]=pu[z/3][z%3];

  if (m > 1) {
    mju_copy3(lines[k] + 0, pts[0]);
    mju_copy3(lines[k] + 3, pts[1]);
    mju_copy3(linesu[k] + 0, axi[0]);
    mju_copy3(linesu[k++] + 3, axi[1]);
  }
  if (m > 2) {
    mju_copy3(lines[k] + 0, pts[0]);
    mju_copy3(lines[k] + 3, pts[2]);
    mju_copy3(linesu[k] + 0, axi[0]);
    mju_copy3(linesu[k++] + 3, axi[2]);

    mju_add3(lines[k] + 0, pts[0], pts[1]);
    mju_copy3(lines[k] + 3, pts[2]);
    mju_add3(linesu[k] + 0, axi[0], axi[1]);
    mju_copy3(linesu[k++] + 3, axi[2]);

    mju_add3(lines[k] + 0, pts[0], pts[2]);
    mju_copy3(lines[k] + 3, pts[1]);
    mju_add3(linesu[k] + 0, axi[0], axi[2]);
    mju_copy3(linesu[k++] + 3, axi[1]);
  }

  stage_i[12]=k;trace_i[5]=k;for(int z=0;z<24;z++){stage_d[64+z]=lines[z/6][z%6];stage_d[88+z]=linesu[z/6][z%6];}
  for (i = 0; i < k; i++) {
    for (q = 0; q < 2; q++) {
      a = lines[i][0 + q];
      b = lines[i][3 + q];
      c = lines[i][1 - q];
      d = lines[i][4 - q];

      if(!isfinite(b)){stage_i[60]=1;return 0;} int eb=4*i+2*q;for(int t=0;t<2;t++){int e=eb+t;trace_i[8+e]=5;trace_i[80+e]=n;trace_d[8*e]=a;trace_d[8*e+1]=b;trace_d[8*e+2]=c;trace_d[8*e+3]=d;}
      if (fabs(b) > mjMINVAL) {
        for (j = -1; j <= 1; j += 2) {
          int e=eb+(j+1)/2;trace_i[8+e]=6;trace_i[80+e]=n;
          if (n < mjMAXCONPAIR) {
            l = s[q] * j;
            trace_d[8*e+4]=l;
            c1 = (l - a) * (1 / b);
            trace_d[8*e+5]=c1;
            if (c1 < 0 || c1 > 1)
              {stage_i[20]++;trace_i[8+e]=2;continue;}
            c2 = c + d * c1;
            trace_d[8*e+6]=c2;
            if (fabs(c2) > s[1 - q])
              {stage_i[21]++;trace_i[8+e]=3;continue;}

            trace_d[8*e+7]=(linesu[i][2]+linesu[i][5]*c1)*innorm;
            if ((linesu[i][2] + linesu[i][5]*c1)*innorm > margin)
              {stage_i[22]++;trace_i[8+e]=4;continue;}

            mju_scl3(points[n], linesu[i], 0.5);
            mju_addToScl3(points[n], linesu[i] + 3, 0.5 * c1);
            points[n][0 + q] += 0.5 * l;
            points[n][1 - q] += 0.5 * c2;
            depth[n] = points[n][2] * innorm * 2;
            trace_i[8+e]=1;trace_i[56+e]=n;
            n++;
          }
        }
      }
    }
  }

  nl = n;

  stage_i[13]=nl;trace_i[2]=nl;
  a = pts[1][0];
  b = pts[2][0];
  c = pts[1][1];
  d = pts[2][1];
  c1 = a * d - b * c;

  double d0=c1;trace_d[192]=c1;trace_d[194]=margin2;trace_d[195]=innorm;
  for (i = 0; i < 4; i++) {
    {int e=16+i;trace_i[8+e]=6;trace_i[80+e]=n;trace_d[8*e]=c1;trace_d[8*e+5]=c1;}
    if (n < mjMAXCONPAIR) {
      llx = i / 2 ? lx : -lx;
      lly = i % 2 ? ly : -ly;

      x = llx - pts[0][0];
      y = lly - pts[0][1];

      if(!isfinite(c1)||c1==0){stage_i[60]=1;return 0;}
      u = (x * d - y * b) * (1 / (wrong_d0 ? d0 : c1));
      v = (y * a - x * c) * (1 / (wrong_d0 ? d0 : c1));

      trace_d[8*(16+i)+1]=u;trace_d[8*(16+i)+2]=v;
      if (nl == 0) {
        if ((u < 0 || u > 1) && (v < 0 || v > 1))
          {stage_i[23]++;trace_i[8+16+i]=7;continue;}
      } else {
        if ((u < 0 || u > 1 || v < 0 || v > 1))
          {stage_i[24]++;trace_i[8+16+i]=8;continue;}
      }

      trace_i[32+16+i]=(u<0?1:0)|(u>1?2:0)|(v<0?4:0)|(v>1?8:0);
      if (u < 0)
        u = 0;
      if (u > 1)
        u = 1;
      if (v < 0)
        v = 0;
      if (v > 1)
        v = 1;


      trace_d[8*(16+i)+3]=u;trace_d[8*(16+i)+4]=v;
      mju_scl3(tmp1, pu[0], 1 - u - v);
      mju_addToScl3(tmp1, pu[1], u);
      mju_addToScl3(tmp1, pu[2], v);

      points[n][0] = llx;
      points[n][1] = lly;
      points[n][2] = 0;

      mju_sub3(tmp2, points[n], tmp1);

      c1 = mju_dot3(tmp2, tmp2);
      if(!isfinite(c1)||c1<0){stage_i[60]=1;return 0;} trace_d[8*(16+i)+5]=c1;trace_d[8*(16+i)+6]=tmp1[2];
      if (tmp1[2] > 0)
        if (c1 > margin2)
          {stage_i[25]++;trace_i[8+16+i]=9;continue;}

      mju_add3(points[n], points[n], tmp1);
      mju_scl3(points[n], points[n], 0.5);

      depth[n] = sqrt(c1) * (tmp1[2] < 0 ? -1 : 1);
      trace_i[8+16+i]=1;trace_i[56+16+i]=n;trace_d[8*(16+i)+7]=depth[n];
      n++;
    }
  }

  nf = n;

  stage_i[14]=nf;trace_i[3]=nf;trace_d[193]=c1;
  for (i = 0; i < 4; i++) {
    {int e=20+i;trace_i[8+e]=6;trace_i[80+e]=n;}
    if (n < mjMAXCONPAIR) {
      x = ppts2[i][0];
      y = ppts2[i][1];

      {int e=20+i;trace_d[8*e]=x;trace_d[8*e+1]=y;trace_d[8*e+2]=pu[i][2];trace_d[8*e+3]=innorm;trace_i[32+e]=(x<-lx?1:0)|(x>lx?2:0)|(y<-ly?4:0)|(y>ly?8:0);}
      if (nl == 0) {
        if (nf == 0) {
        } else {
          if (x < -lx || x > lx)
            if (y < -ly || y > ly)
              {stage_i[26]++;trace_i[8+20+i]=10;continue;}
        }
      } else {
        if (x < -lx || x > lx || y < -ly || y > ly)
          {stage_i[27]++;trace_i[8+20+i]=11;continue;}
      }

      for (c1 = 0, j = 0; j < 2; j++)
        if (ppts2[i][j] < -s[j])
          c1 += (ppts2[i][j] + s[j]) * (ppts2[i][j] + s[j]);
        else if (ppts2[i][j] > s[j])
          c1 += (ppts2[i][j] - s[j]) * (ppts2[i][j] - s[j]);

      c1 += pu[i][2] * innorm * pu[i][2] * innorm;

      if(!isfinite(c1)||c1<0){stage_i[60]=1;return 0;} trace_d[8*(20+i)+4]=c1;
      if (pu[i][2] > 0)
        if (c1 > margin2)
          {stage_i[28]++;trace_i[8+20+i]=12;continue;}


      tmp1[0] = ppts2[i][0] * 0.5;
      tmp1[1] = ppts2[i][1] * 0.5;
      tmp1[2] = 0;

      for (j = 0; j < 2; j++) {
        if (ppts2[i][j] < -s[j])
          tmp1[j] = -s[j] * 0.5;
        else if (ppts2[i][j] > s[j])
          tmp1[j] = +s[j] * 0.5;
      }
      mju_addToScl3(tmp1, pu[i], 0.5);
      mju_copy3(points[n], tmp1);

      trace_d[8*(20+i)+6]=tmp1[0];trace_d[8*(20+i)+7]=tmp1[1];
      depth[n] = sqrt(c1) * (pu[i][2] < 0 ? -1 : 1);
      trace_i[8+20+i]=1;trace_i[56+20+i]=n;trace_d[8*(20+i)+5]=depth[n];
      n++;
    }
  }

  stage_i[15]=n;trace_i[4]=n;for(int z=0;z<n*3;z++)stage_d[112+z]=points[z/3][z%3];for(int z=0;z<n;z++)stage_d[262+z]=depth[z];
  mju_mulMatMatT(r, mat1, rotmore, 3, 3, 3);

  mju_rotVecMat(tmp1, rnorm, r);

  mju_scl3(con[0].frame, tmp1, in ? -1 : 1);
  mju_zero3(con[0].frame + 3);


  *frame_written=1;for(int z=0;z<6;z++)frame0[z]=con[0].frame[z];
  for (i = 0; i < n; i++) {
    con[i].dist = depth[i];
    points[i][2] += hz;

    mju_rotVecMat(tmp2, points[i], r);

    mju_add3(con[i].pos, tmp2, pos1);

    mju_copy(con[i].frame, con[0].frame, 6);
  }

  if(n>0)for(int z=0;z<6;z++)stage_d[312+z]=con[0].frame[z];
  for(int z=0;z<n;z++){raw[10*z]=con[z].dist;for(int h=0;h<3;h++)raw[10*z+1+h]=con[z].pos[h];for(int h=0;h<6;h++)raw[10*z+4+h]=con[z].frame[h];} return n;

}

extern "C" __global__ void box_edge_contacts(const double* pair_f,const double* sat_f,
 const int* sat_i,const int* active,const int* counts,int N,int P,
 int* out_count,double* raw,int* state,int* stage_i,double* stage_d,
 int* trace_i,double* trace_d,int* frame_written,double* frame0,int wrong_d0){
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=N*P)return;
  int w=t/P,slot=t%P;
  out_count[t]=0;state[t]=0;frame_written[t]=0;
  for(int i=0;i<500;i++)raw[t*500+i]=0.0;
  for(int i=0;i<6;i++)frame0[t*6+i]=0.0;
  for(int i=0;i<64;i++)stage_i[t*64+i]=0;
  for(int i=0;i<320;i++)stage_d[t*320+i]=0.0;
  for(int i=0;i<128;i++)trace_i[t*128+i]=0;
  for(int i=0;i<256;i++)trace_d[t*256+i]=0.0;
  for(int i=0;i<24;i++)trace_i[t*128+56+i]=-1;
  if(active[w]==0)return;
  if(active[w]!=1){state[t]=5;return;}
  if(counts[w]<0 || counts[w]>P){state[t]=4;return;}
  if(slot>=counts[w])return;
  const int* si=sat_i+t*6;
  int route=si[4],code=si[0];
  if(si[5]!=0 || route<0 || route>2 ||
     (route==2 && (code<12 || code>20 || si[1]<0 || si[1]>7 || si[2]<0 || si[2]>7 || si[3]<0 || si[3]>1))){state[t]=6;return;}
  if(route!=2){state[t]=1;return;}
  state[t]=2;
  for(int i=0;i<31;i++)if(!isfinite(pair_f[t*31+i])){state[t]=7;stage_i[t*64+60]=1;return;}
  for(int i=0;i<45;i++)if(!isfinite(sat_f[t*45+i])){state[t]=7;stage_i[t*64+60]=1;return;}
  int n=edge_tail(pair_f+t*31,sat_f+t*45,si,raw+t*500,
    stage_i+t*64,stage_d+t*320,trace_i+t*128,trace_d+t*256,
    frame_written+t,frame0+t*6,wrong_d0);
  int numeric_bad=stage_i[t*64+60] || n<0 || n>50;
  if(!numeric_bad){
    for(int i=0;i<320;i++)if(!isfinite(stage_d[t*320+i]))numeric_bad=1;
    for(int i=0;i<256;i++)if(!isfinite(trace_d[t*256+i]))numeric_bad=1;
    for(int i=0;i<10*n;i++)if(!isfinite(raw[t*500+i]))numeric_bad=1;
    if(frame_written[t])for(int i=0;i<6;i++)if(!isfinite(frame0[t*6+i]))numeric_bad=1;
  }
  if(numeric_bad){
    state[t]=7;out_count[t]=0;frame_written[t]=0;
    for(int i=0;i<500;i++)raw[t*500+i]=0.0;
    for(int i=0;i<6;i++)frame0[t*6+i]=0.0;
    for(int i=0;i<64;i++)stage_i[t*64+i]=0;
    stage_i[t*64+60]=1;
    for(int i=0;i<320;i++)stage_d[t*320+i]=0.0;
    for(int i=0;i<128;i++)trace_i[t*128+i]=0;
    for(int i=0;i<256;i++)trace_d[t*256+i]=0.0;
    return;
  }
  out_count[t]=n;
  if(stage_i[t*64+11])state[t]=3;
}
extern "C" int launch_box_edge(const double* pair_f,const double* sat_f,const int* sat_i,
 const int* active,const int* counts,int N,int P,int* out_count,double* raw,int* state,
 int* stage_i,double* stage_d,int* trace_i,double* trace_d,int* frame_written,double* frame0,
 int wrong_d0,cudaStream_t stream){
 box_edge_contacts<<<(N*P+127)/128,128,0,stream>>>(pair_f,sat_f,sat_i,active,counts,N,P,
 out_count,raw,state,stage_i,stage_d,trace_i,trace_d,frame_written,frame0,wrong_d0);
 return (int)cudaGetLastError();
}
