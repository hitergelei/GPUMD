/*  neighbor_pad.cu
 *  统一近邻列表维度：输出 g_NL[N*MN] 每行固定 MN 个元素，不足补 -1，多余截断。
 *  可直接被 pybind/torch-extension 封装，供 PyTorch 使用。
 */
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

// -------------- 参数结构体 --------------
struct Box
{
    float lx, ly, lz;   // box length
    float lx2, ly2, lz2;// half box (for MIC)
};

// -------------- 最小 MIC --------------
__device__ void apply_mic(const Box& b, float* dx, float* dy, float* dz)
{
    if (*dx >  b.lx2) *dx -= b.lx;
    if (*dx < -b.lx2) *dx += b.lx;
    if (*dy >  b.ly2) *dy -= b.ly;
    if (*dy < -b.ly2) *dy += b.ly;
    if (*dz >  b.lz2) *dz -= b.lz;
    if (*dz < -b.lz2) *dz += b.lz;
}

// -------------- step-1: 统计 + 截断 --------------
__global__ void ker_count_trim(
    int N, int MN,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    Box box, float cutoff2,
    int* tmp_NN,      // 真实邻居数 ≤MN
    int* tmp_NL)      // 临时缓冲区 [MN,N] 列主序
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 >= N) return;

    float x1 = x[n1], y1 = y[n1], z1 = z[n1];
    int cnt = 0;
    for (int n2 = 0; n2 < N; ++n2)
    {
        float dx = x[n2] - x1;
        float dy = y[n2] - y1;
        float dz = z[n2] - z1;
        apply_mic(box, &dx, &dy, &dz);
        float r2 = dx*dx + dy*dy + dz*dz;
        if (n2 != n1 && r2 < cutoff2)
        {
            if (cnt < MN) tmp_NL[cnt * N + n1] = n2;
            ++cnt;
        }
    }
    tmp_NN[n1] = min(cnt, MN);
}

// -------------- step-2: 规整化，写 -1 --------------
__global__ void ker_pad(
    int N, int MN,
    const int* __restrict__ tmp_NN,
    const int* __restrict__ tmp_NL,
    int* g_NL)        // 最终输出 [MN,N] 列主序
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 >= N) return;
    int real = tmp_NN[n1];
    for (int j = 0; j < MN; ++j)
    {
        if (j < real)
            g_NL[j * N + n1] = tmp_NL[j * N + n1];
        else
            g_NL[j * N + n1] = -1;
    }
}

// -------------- 主机接口 --------------
void build_neighbor_padded(
    int N, int MN,
    const float* h_x,
    const float* h_y,
    const float* h_z,
    const std::vector<float>& box6,  // {lx,ly,lz,lx2,ly2,lz2}
    int*    h_NN,   // 输出真实邻居数，长度 N
    int*    h_NL)   // 输出近邻矩阵，长度 N*MN
{
    // 设备内存
    float *d_x, *d_y, *d_z;
    int   *d_tmp_NN, *d_tmp_NL, *d_NL;
    cudaMalloc(&d_x,  N*sizeof(float));
    cudaMalloc(&d_y,  N*sizeof(float));
    cudaMalloc(&d_z,  N*sizeof(float));
    cudaMalloc(&d_tmp_NN, N*sizeof(int));
    cudaMalloc(&d_tmp_NL, N*MN*sizeof(int));
    cudaMalloc(&d_NL,     N*MN*sizeof(int));

    cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, N*sizeof(float), cudaMemcpyHostToDevice);

    Box box{box6[0], box6[1], box6[2], box6[3], box6[4], box6[5]};
    float cut2 = 11.0f * 11.0f;

    int block = 128;
    int grid  = (N + block - 1) / block;

    ker_count_trim<<<grid, block>>>(N, MN, d_x, d_y, d_z, box, cut2,
                                      d_tmp_NN, d_tmp_NL);
    ker_pad<<<grid, block>>>(N, MN, d_tmp_NN, d_tmp_NL, d_NL);

    cudaMemcpy(h_NN, d_tmp_NN, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_NL, d_NL,     N*MN*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_tmp_NN); cudaFree(d_tmp_NL); cudaFree(d_NL);
}

// -------------- 简单 CPU 测试 --------------
int main()
{
    const int N = 1000, MN = 64;
    std::vector<float> x(N), y(N), z(N);
    for (int i = 0; i < N; ++i){ x[i] = rand()*1.0f/RAND_MAX*20; /*...*/ }

    std::vector<float> box6{20,20,20, 10,10,10}; // lx,ly,lz, lx2,ly2,lz2
    std::vector<int> NN(N), NL(N*MN);

    build_neighbor_padded(N, MN, x.data(),y.data(),z.data(), box6,
                          NN.data(), NL.data());

    // 打印前 5 个原子
    for (int i = 0; i < 5; ++i){
        printf("atom %d  real neighbors %d : ", i, NN[i]);
        for (int j = 0; j < 10; ++j) printf("%d ", NL[j*N + i]);
        printf("...\n");
    }
    return 0;
}
