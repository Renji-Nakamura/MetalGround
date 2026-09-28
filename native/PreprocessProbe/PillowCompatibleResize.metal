#include <metal_stdlib>
using namespace metal;
constant int PB=22;
constant int ROUNDING=1<<(PB-1);
inline uchar clip8(int s){ return uchar(clamp(s>>PB,0,255)); }
struct R{ uint sw,sh,dw,dh,k; };

kernel void h1(device const uchar* src[[buffer(0)]],device uchar* dst[[buffer(1)]],device const int2* bd[[buffer(2)]],device const int* cf[[buffer(3)]],constant R& p[[buffer(4)]],uint2 g[[thread_position_in_grid]]){
 if(g.x>=p.dw||g.y>=p.dh)return; int2 b=bd[g.x]; uint co=g.x*p.k; int sr=ROUNDING,sg=ROUNDING,sb=ROUNDING;
 for(int k=0;k<b.y;k++){int w=cf[co+uint(k)];uint i=(g.y*p.sw+uint(b.x+k))*4;sb+=int(src[i])*w;sg+=int(src[i+1])*w;sr+=int(src[i+2])*w;}
 uint o=(g.y*p.dw+g.x)*4;dst[o]=clip8(sr);dst[o+1]=clip8(sg);dst[o+2]=clip8(sb);dst[o+3]=255;
}
struct LV{uint sw,sh,dw,dh,rh,oy,k,fill;};
kernel void v1_letter(device const uchar* src[[buffer(0)]],device uchar* dst[[buffer(1)]],device const int2* bd[[buffer(2)]],device const int* cf[[buffer(3)]],constant LV& p[[buffer(4)]],uint2 g[[thread_position_in_grid]]){
 if(g.x>=p.dw||g.y>=p.dh)return;uint o=(g.y*p.dw+g.x)*4;if(g.y<p.oy||g.y>=p.oy+p.rh){uchar f=uchar(p.fill);dst[o]=f;dst[o+1]=f;dst[o+2]=f;dst[o+3]=255;return;}
 uint y=g.y-p.oy;int2 b=bd[y];uint co=y*p.k;int sr=ROUNDING,sg=ROUNDING,sb=ROUNDING;for(int k=0;k<b.y;k++){int w=cf[co+uint(k)];uint i=(uint(b.x+k)*p.sw+g.x)*4;sr+=int(src[i])*w;sg+=int(src[i+1])*w;sb+=int(src[i+2])*w;}dst[o]=clip8(sr);dst[o+1]=clip8(sg);dst[o+2]=clip8(sb);dst[o+3]=255;
}
kernel void h2(device const uchar* src[[buffer(0)]],device uchar* dst[[buffer(1)]],device const int2* bd[[buffer(2)]],device const int* cf[[buffer(3)]],constant R& p[[buffer(4)]],uint2 g[[thread_position_in_grid]]){
 if(g.x>=p.dw||g.y>=p.dh)return;int2 b=bd[g.x];uint co=g.x*p.k;int sr=ROUNDING,sg=ROUNDING,sb=ROUNDING;for(int k=0;k<b.y;k++){int w=cf[co+uint(k)];uint i=(g.y*p.sw+uint(b.x+k))*4;sr+=int(src[i])*w;sg+=int(src[i+1])*w;sb+=int(src[i+2])*w;}uint o=(g.y*p.dw+g.x)*4;dst[o]=clip8(sr);dst[o+1]=clip8(sg);dst[o+2]=clip8(sb);dst[o+3]=255;
}
struct FV{uint sw,sh,dw,dh,k;};
kernel void v2_norm(device const uchar* src[[buffer(0)]],device float* dst[[buffer(1)]],device const int2* bd[[buffer(2)]],device const int* cf[[buffer(3)]],constant FV& p[[buffer(4)]],constant float3& mean[[buffer(5)]],constant float3& stdv[[buffer(6)]],uint2 g[[thread_position_in_grid]]){
 if(g.x>=p.dw||g.y>=p.dh)return;int2 b=bd[g.y];uint co=g.y*p.k;int sr=ROUNDING,sg=ROUNDING,sb=ROUNDING;for(int k=0;k<b.y;k++){int w=cf[co+uint(k)];uint i=(uint(b.x+k)*p.sw+g.x)*4;sr+=int(src[i])*w;sg+=int(src[i+1])*w;sb+=int(src[i+2])*w;}float3 rgb=float3(float(clip8(sr)),float(clip8(sg)),float(clip8(sb)))/255.0;float3 v=(rgb-mean)/stdv;uint i=g.y*p.dw+g.x,pl=p.dw*p.dh;dst[i]=v.r;dst[pl+i]=v.g;dst[2*pl+i]=v.b;
}
