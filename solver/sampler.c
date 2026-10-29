// Simulador Monte Carlo para push/fold de 3 jugadores.
// Reparte manos al azar (BTN, SB, BB + board) y acumula, por cada trío de
// clases de mano (169x169x169), las equities heads-up y 3-way.
// Uso: ./sampler <num_repartos> <semilla> <archivo_salida>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct { uint32_t n, ebs, ebg, esg, tb, ts; } Cell;

static uint64_t rs;
static inline uint64_t rng(void){ rs^=rs>>12; rs^=rs<<25; rs^=rs>>27; return rs*2685821657736338717ULL; }
static inline uint32_t rbound(uint32_t n){ return (uint32_t)(((rng()>>32)*(uint64_t)n)>>32); }

static inline int straight_top(int m){
  for(int t=12;t>=4;t--) if(((m>>(t-4))&0x1F)==0x1F) return t;
  if((m&0x100F)==0x100F) return 3;
  return -1;
}

static inline uint32_t eval7(const int *c){
  int cnt[13]={0}, sm[4]={0}, sc[4]={0}, rm=0;
  for(int i=0;i<7;i++){int r=c[i]>>2,s=c[i]&3; cnt[r]++; sm[s]|=1<<r; sc[s]++; rm|=1<<r;}
  for(int s=0;s<4;s++) if(sc[s]>=5){
    int t=straight_top(sm[s]); if(t>=0) return (8u<<20)|t;
    uint32_t v=5u<<20; int k=0; for(int r=12;r>=0&&k<5;r--) if(sm[s]>>r&1){v|=(uint32_t)r<<(4*(4-k));k++;}
    return v;
  }
  int quad=-1,trip=-1,trip2=-1,p1=-1,p2=-1;
  for(int r=12;r>=0;r--){
    if(cnt[r]==4) quad=r;
    else if(cnt[r]==3){ if(trip<0) trip=r; else if(trip2<0) trip2=r; }
    else if(cnt[r]==2){ if(p1<0) p1=r; else if(p2<0) p2=r; }
  }
  if(quad>=0){ int k=0; for(int r=12;r>=0;r--) if(r!=quad&&cnt[r]){k=r;break;} return (7u<<20)|(quad<<4)|k; }
  if(trip>=0){ int p=trip2>p1?trip2:p1; if(p>=0) return (6u<<20)|(trip<<4)|p; }
  int t=straight_top(rm); if(t>=0) return (4u<<20)|t;
  if(trip>=0){ uint32_t v=(3u<<20)|(trip<<8); int k=0; for(int r=12;r>=0&&k<2;r--) if(r!=trip&&cnt[r]){v|=(uint32_t)r<<(4*(1-k));k++;} return v; }
  if(p1>=0&&p2>=0){ int k=0; for(int r=12;r>=0;r--) if(r!=p1&&r!=p2&&cnt[r]){k=r;break;} return (2u<<20)|(p1<<8)|(p2<<4)|k; }
  if(p1>=0){ uint32_t v=(1u<<20)|(p1<<12); int k=0; for(int r=12;r>=0&&k<3;r--) if(r!=p1&&cnt[r]){v|=(uint32_t)r<<(4*(2-k));k++;} return v; }
  uint32_t v=0; int k=0; for(int r=12;r>=0&&k<5;r--) if(cnt[r]){v|=(uint32_t)r<<(4*(4-k));k++;} return v;
}

// Índice en la grilla 13x13: fila/columna 0 = As. Pares en diagonal,
// suited sobre la diagonal, offsuit bajo la diagonal.
static inline int cls(int a,int b){
  int ra=a>>2, rb=b>>2, sa=a&3, sb=b&3;
  if(ra<rb){int t=ra;ra=rb;rb=t;}
  int ga=12-ra, gb=12-rb;
  if(ra==rb) return ga*13+ga;
  if(sa==sb) return ga*13+gb;
  return gb*13+ga;
}

static const int P[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};

int main(int argc,char**argv){
  if(argc<4){fprintf(stderr,"uso: %s repartos semilla salida\n",argv[0]);return 1;}
  long long N=atoll(argv[1]); rs=strtoull(argv[2],0,10)*0x9E3779B97F4A7C15ULL+1;
  size_t ncell=169UL*169*169;
  Cell *cells=calloc(ncell,sizeof(Cell));
  int deck[52]; for(int i=0;i<52;i++) deck[i]=i;
  for(long long it=0;it<N;it++){
    for(int i=0;i<11;i++){ int j=i+rbound(52-i); int t=deck[i]; deck[i]=deck[j]; deck[j]=t; }
    int c7[3][7]; uint32_t v[3]; int c[3];
    for(int p=0;p<3;p++){
      c7[p][0]=deck[2*p]; c7[p][1]=deck[2*p+1];
      for(int b=0;b<5;b++) c7[p][2+b]=deck[6+b];
      v[p]=eval7(c7[p]); c[p]=cls(deck[2*p],deck[2*p+1]);
    }
    int e2[3][3];
    for(int a=0;a<3;a++) for(int b=0;b<3;b++) e2[a][b]= v[a]>v[b]?2:(v[a]==v[b]?1:0);
    uint32_t m=v[0]; if(v[1]>m)m=v[1]; if(v[2]>m)m=v[2];
    int w=(v[0]==m)+(v[1]==m)+(v[2]==m); int sh[3];
    for(int p=0;p<3;p++) sh[p]= v[p]==m ? 6/w : 0;
    for(int q=0;q<6;q++){
      int a=P[q][0],b=P[q][1],g=P[q][2];
      Cell *x=&cells[((size_t)c[a]*169+c[b])*169+c[g]];
      x->n++; x->ebs+=e2[a][b]; x->ebg+=e2[a][g]; x->esg+=e2[b][g]; x->tb+=sh[a]; x->ts+=sh[b];
    }
  }
  FILE*f=fopen(argv[3],"wb"); fwrite(cells,sizeof(Cell),ncell,f); fclose(f);
  return 0;
}
