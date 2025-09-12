#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <cstdio>

// 模仿torch_neighborlist.py实现一个cuda版本简单的基于 cell list 的近邻搜索
// ---------------- 参数 ----------------
struct Box{
    float lx,ly,lz, lx2,ly2,lz2;
};
__constant__ Box d_box;
__constant__ float d_cut2;          // cutoff*cutoff
__constant__ int   d_pbc[3];        // 扩胞层数
__constant__ int   d_ncell[3];      // cell 网格维度
__constant__ int   d_stride[3];     // stride for cell index
__constant__ int   d_MN;            // 统一邻居维度

// ---------------- 工具 ----------------
__device__ inline void mic(float &dx, float &dy, float &dz){
    if(dx >  d_box.lx2) dx -= d_box.lx;
    if(dx < -d_box.lx2) dx += d_box.lx;
    if(dy >  d_box.ly2) dy -= d_box.ly;
    if(dy < -d_box.ly2) dy += d_box.ly;
    if(dz >  d_box.lz2) dz -= d_box.lz;
    if(dz < -d_box.lz2) dz += d_box.lz;
}

// ---------------- step-1: 生成 padded 原子 ----------------
__global__ void ker_gen_padded(
    int N,
    const float *x, const float *y, const float *z,
    int *padded_N,        // 输出：padded 原子总数
    float *px, float *py, float *pz,
    int *padded_rindex)   // 对应原始原子索引
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= N) return;

    int cnt = 0;
    for(int ia=-d_pbc[0]; ia<=d_pbc[0]; ++ia)
    for(int ib=-d_pbc[1]; ib<=d_pbc[1]; ++ib)
    for(int ic=-d_pbc[2]; ic<=d_pbc[2]; ++ic)
    {
        int off = (ia*d_pbc[1]*d_pbc[2] + ib*d_pbc[2] + ic) * N + n;
        px[off] = x[n] + ia*d_box.lx;
        py[off] = y[n] + ib*d_box.ly;
        pz[off] = z[n] + ic*d_box.lz;
        padded_rindex[off] = n;
        ++cnt;
    }
    if(n==0) *padded_N = cnt*N;   // 每个线程都写相同值，没关系
}

// ---------------- step-2: 建 cell list ----------------
__global__ void ker_build_cell_list(
    int padded_N,
    const float *px, const float *py, const float *pz,
    int *cell_head,     // size = ncell，初始化为 -1
    int *cell_next)     // size = padded_N
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= padded_N) return;

    int cx = floorf(px[n] / d_cut2);   // 用 cutoff 做格子边长
    int cy = floorf(py[n] / d_cut2);
    int cz = floorf(pz[n] / d_cut2);
    cx = (cx%d_ncell[0] + d_ncell[0]) % d_ncell[0];
    cy = (cy%d_ncell[1] + d_ncell[1]) % d_ncell[1];
    cz = (cz%d_ncell[2] + d_ncell[2]) % d_ncell[2];
    int c = cx*d_stride[0] + cy*d_stride[1] + cz*d_stride[2];

    int old = atomicExch(&cell_head[c], n);
    cell_next[n] = old;
}

// ---------------- step-3: 查近邻，写固定 (N,MN) 矩阵 ----------------
__global__ void ker_query_neigh(
    int N,
    const float *x, const float *y, const float *z,
    const int *cell_head, const int *cell_next,
    const float *px, const float *py, const float *pz,
    int *g_NN, int *g_NL)   // g_NL 列主序 [MN,N]
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if(n1 >= N) return;

    float x1 = x[n1], y1 = y[n1], z1 = z[n1];
    int cnt = 0;

    int cx = floorf(x1 / d_cut2);
    int cy = floorf(y1 / d_cut2);
    int cz = floorf(z1 / d_cut2);

#pragma unroll
    for(int dx=-1; dx<=1; ++dx)
    for(int dy=-1; dy<=1; ++dy)
    for(int dz=-1; dz<=1; ++dz)
    {
        int nx = (cx+dx + d_ncell[0]) % d_ncell[0];
        int ny = (cy+dy + d_ncell[1]) % d_ncell[1];
        int nz = (cz+dz + d_ncell[2]) % d_ncell[2];
        int c = nx*d_stride[0] + ny*d_stride[1] + nz*d_stride[2];

        for(int n2=cell_head[c]; n2!=-1; n2=cell_next[n2])
        {
            float xx = px[n2] - x1;
            float yy = py[n2] - y1;
            float zz = pz[n2] - z1;
            mic(xx,yy,zz);
            float r2 = xx*xx + yy*yy + zz*zz;
            if(r2<d_cut2 && r2>1e-4f){          // 排除自身
                if(cnt<d_MN) g_NL[cnt*N + n1] = n2; // 截断
                ++cnt;
            }
        }
    }
    g_NN[n1] = min(cnt, d_MN);
}

// ---------------- 主机封装 ----------------
void build_neighbor_cuda(
    int N, int MN,
    const float *h_x, const float *h_y, const float *h_z,
    const std::vector<float>& box,   // {lx,ly,lz}
    int *h_NN, int *h_NL)            // 输出
{
    float cut = 11.0f;
    float lx=box[0], ly=box[1], lz=box[2];
    Box h_box{lx,ly,lz, lx/2,ly/2,lz/2};

    // 1. 估算扩胞层数（直接抄 PyTorch 公式）
    float volume = lx*ly*lz;
    int pa = ceilf(cut * (ly*lz) / volume);
    int pb = ceilf(cut * (lz*lx) / volume);
    int pc = ceilf(cut * (lx*ly) / volume);
    int pbc[3] = {pa,pb,pc};

    // 2. cell 网格
    int ncx = ceilf(lx/cut), ncy = ceilf(ly/cut), ncz = ceilf(lz/cut);
    int ncell[3] = {ncx,ncy,ncz};
    int stride[3] = {ncy*ncz, ncz, 1};

    cudaMemcpyToSymbol(d_box,    &h_box,    sizeof(Box));
    cudaMemcpyToSymbol(d_cut2,   &cut,      sizeof(float));
    cudaMemcpyToSymbol(d_pbc,    pbc,       3*sizeof(int));
    cudaMemcpyToSymbol(d_ncell,  ncell,     3*sizeof(int));
    cudaMemcpyToSymbol(d_stride, stride,    3*sizeof(int));
    cudaMemcpyToSymbol(d_MN,     &MN,       sizeof(int));

    // 3. 设备内存
    float *d_x,*d_y,*d_z;
    cudaMalloc(&d_x,N*sizeof(float));
    cudaMalloc(&d_y,N*sizeof(float));
    cudaMalloc(&d_z,N*sizeof(float));
    cudaMemcpy(d_x,h_x,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_z,h_z,N*sizeof(float),cudaMemcpyHostToDevice);

    int padded_N = N * (2*pa+1)*(2*pb+1)*(2*pc+1);
    float *d_px,*d_py,*d_pz;
    int *d_pidx;
    cudaMalloc(&d_px,padded_N*sizeof(float));
    cudaMalloc(&d_py,padded_N*sizeof(float));
    cudaMalloc(&d_pz,padded_N*sizeof(float));
    cudaMalloc(&d_pidx,padded_N*sizeof(int));

    int *d_padded_N;
    cudaMalloc(&d_padded_N,sizeof(int));

    // 4. 生成 padded
    ker_gen_padded<<<(N+127)/128,128>>>(N,d_x,d_y,d_z,d_padded_N,d_px,d_py,d_pz,d_pidx);
    cudaDeviceSynchronize();

    // 5. 建 cell list
    int *d_cell_head, *d_cell_next;
    int tot_cell = ncx*ncy*ncz;
    cudaMalloc(&d_cell_head, tot_cell*sizeof(int));
    cudaMalloc(&d_cell_next, padded_N*sizeof(int));
    cudaMemset(d_cell_head,0xff,tot_cell*sizeof(int)); // -1
    ker_build_cell_list<<<(padded_N+127)/128,128>>>(padded_N,d_px,d_py,d_pz,d_cell_head,d_cell_next);
    cudaDeviceSynchronize();

    // 6. 查近邻
    int *d_NN,*d_NL;
    cudaMalloc(&d_NN,N*sizeof(int));
    cudaMalloc(&d_NL,MN*N*sizeof(int));
    ker_query_neigh<<<(N+127)/128,128>>>(N,d_x,d_y,d_z,d_cell_head,d_cell_next,d_px,d_py,d_pz,d_NN,d_NL);
    cudaMemcpy(h_NN,d_NN,N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_NL,d_NL,MN*N*sizeof(int),cudaMemcpyDeviceToHost);

    // 7. 收尾
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz); cudaFree(d_pidx);
    cudaFree(d_cell_head); cudaFree(d_cell_next);
    cudaFree(d_NN); cudaFree(d_NL);
}

// ---------------- 简单测试 ----------------
int main(){
    const int N=1000, MN=64;
    std::vector<float> x(N),y(N),z(N);
    for(int i=0;i<N;++i){ x[i]=rand()*20.f/RAND_MAX; y[i]=rand()*20.f/RAND_MAX; z[i]=rand()*20.f/RAND_MAX; }
    std::vector<int> NN(N), NL(MN*N);
    build_neighbor_cuda(N,MN,x.data(),y.data(),z.data(),{20,20,20},NN.data(),NL.data());
    for(int i=0;i<5;++i){
        printf("atom %d  nnei %d : ",i,NN[i]);
        for(int j=0;j<10;++j) printf("%d ",NL[j*N+i]);
        printf("...\n");
    }
    return 0;
}
