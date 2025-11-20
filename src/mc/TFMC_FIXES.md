# tfMC 实现修复说明

## 修复日期
2025年11月20日

## 修复内容

### 1. 编译错误修复

**问题**: 
```cpp
GPU_CHECK(gpuMemset(com_displacement.data(), 0, 4 * sizeof(double)));
```

**原因**: 
- `gpuMemset` 是在 `gpu_macro.cuh` 中定义的宏，映射到 `cudaMemset`
- 但 `GPU_CHECK` 宏不存在，应该使用 `CHECK` 宏（定义在 `error.cuh` 中）

**修复**:
```cpp
CHECK(cudaMemset(com_displacement.data(), 0, 4 * sizeof(double)));
```

### 2. 转动移除完整实现

**问题**: 之前只有简化版本，未实现完整的惯性张量和角速度计算

**LAMMPS 原理** (从 fix_tfmc.cpp):
```cpp
// 1. 计算质心
group->xcm(igroup, masstotal, cm);

// 2. 计算角动量: L = Σ[m_i × (r_i - r_cm) × d_i]
for (int i = 0; i < nlocal; i++) {
  if (mask[i] & groupbit) {
    domain->unmap(x[i], image[i], unwrap);
    dx = unwrap[0] - cm[0];
    dy = unwrap[1] - cm[1];
    dz = unwrap[2] - cm[2];
    p[0] += massone * (dy * xd[i][2] - dz * xd[i][1]);
    p[1] += massone * (dz * xd[i][0] - dx * xd[i][2]);
    p[2] += massone * (dx * xd[i][1] - dy * xd[i][0]);
  }
}

// 3. 计算惯性张量
group->inertia(igroup, cm, inertia);

// 4. 计算角速度: ω = I^(-1) × L
group->omega(angmom, inertia, omega);

// 5. 移除转动: x_i -= ω × (r_i - r_cm)
for (int i = 0; i < nlocal; i++) {
  if (mask[i] & groupbit) {
    x[i][0] -= omega[1] * dz - omega[2] * dy;
    x[i][1] -= omega[2] * dx - omega[0] * dz;
    x[i][2] -= omega[0] * dy - omega[1] * dx;
  }
}
```

**CUDA 实现**:

#### 新增 Kernel 1: 计算质心
```cpp
static __global__ void compute_center_of_mass_kernel(
  const int N,
  const double* __restrict__ mass,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  double* cm_x,
  double* cm_y,
  double* cm_z,
  double* total_mass)
```
- 并行计算 Σ(m_i × r_i) 和 Σ(m_i)
- 使用 shared memory reduction
- 结果通过 atomicAdd 累加

#### 新增 Kernel 2: 计算惯性张量
```cpp
static __global__ void compute_inertia_tensor_kernel(
  const int N,
  const double* __restrict__ mass,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const double cm_x,
  const double cm_y,
  const double cm_z,
  double* inertia)
```
- 计算 6 个独立分量: Ixx, Iyy, Izz, Ixy, Ixz, Iyz
- 公式:
  * I_xx = Σ[m_i × (y_i² + z_i²)]
  * I_yy = Σ[m_i × (x_i² + z_i²)]
  * I_zz = Σ[m_i × (x_i² + y_i²)]
  * I_xy = -Σ[m_i × x_i × y_i]
  * I_xz = -Σ[m_i × x_i × z_i]
  * I_yz = -Σ[m_i × y_i × z_i]

#### 保留 Kernel: 计算角动量
```cpp
static __global__ void compute_angular_momentum_kernel(...)
```
- L = Σ[m_i × (r_i - r_cm) × d_i]
- 已经实现，无需修改

#### 新增 Kernel 3: 移除转动
```cpp
static __global__ void remove_rotation_kernel(
  const int N,
  const double omega_x,
  const double omega_y,
  const double omega_z,
  const double cm_x,
  const double cm_y,
  const double cm_z,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  double* dx,
  double* dy,
  double* dz)
```
- 每个原子减去 ω × (r_i - r_cm)
- 完全并行化

#### CPU 端: 惯性张量求逆
```cpp
// 使用代数余子式方法求逆 3×3 矩阵
double det = I[0][0] * (I[1][1] * I[2][2] - I[1][2] * I[2][1]) - ...;
double invI[3][3];
invI[0][0] = (I[1][1] * I[2][2] - I[1][2] * I[2][1]) / det;
// ... 其他分量
```

## 完整流程

`remove_rotation()` 方法的执行流程：

```
1. 检查 fix_rotation 标志
2. 分配临时 GPU 数组（static，只分配一次）:
   - cm_data(4): 质心坐标 + 总质量
   - angmom_data(3): 角动量
   - inertia_data(6): 惯性张量
3. 在 GPU 上计算质心
4. 拷贝到 CPU，归一化
5. 在 GPU 上计算角动量（基于位移向量）
6. 在 GPU 上计算惯性张量
7. 拷贝角动量和惯性张量到 CPU
8. CPU 上求逆惯性张量
9. CPU 上计算角速度: ω = I^(-1) × L
10. GPU 上从位移中移除转动分量
```

## 与 LAMMPS 的一致性

| 特性 | LAMMPS fix_tfmc | GPUMD mc_ensemble_tfmc | 
|------|-----------------|------------------------|
| 位移生成 | ✅ 拒绝采样 | ✅ CUDA 并行实现 |
| 质量缩放 | ✅ (m_min/m_i)^0.25 | ✅ 完全一致 |
| COM 移除 | ✅ MPI reduction | ✅ GPU reduction |
| 角动量计算 | ✅ 串行循环 | ✅ GPU 并行 |
| 惯性张量 | ✅ group->inertia() | ✅ GPU 并行计算 |
| 矩阵求逆 | ✅ group->omega() | ✅ CPU 代数余子式 |
| 转动移除 | ✅ 串行应用 | ✅ GPU 并行 |

## 性能考虑

### GPU 优化
- 所有密集计算在 GPU 上执行
- 最小化 GPU↔CPU 数据传输
- 只在必要时传输（求逆 3×3 矩阵）
- 使用 shared memory reduction

### 内存优化
- 临时数组声明为 static，避免重复分配
- 复用 displacements 数组
- 最小化内存占用

### 精度考虑
- 全部使用 double 精度
- 惯性张量求逆前检查行列式
- 防止奇异矩阵导致的数值不稳定

## 测试建议

### 测试 1: 基本功能
```bash
# run.in
mc tfmc 100 50 300.0 300.0 0.20 12345
```

### 测试 2: COM 固定
```bash
mc tfmc 100 50 300.0 300.0 0.20 12345 com 1 1 1
```

### 测试 3: 转动固定
```bash
mc tfmc 100 50 300.0 300.0 0.20 12345 rot
```

### 测试 4: 完全约束
```bash
mc tfmc 100 50 300.0 300.0 0.20 12345 com 1 1 1 rot
```

### 验证方法
1. 检查 COM 速度（应为零）
2. 检查角动量（应为零）
3. 与 LAMMPS fix_tfmc 结果对比
4. 能量守恒检查

## 编译

在 makefile 中添加：
```makefile
mc/mc_ensemble_tfmc.o: mc/mc_ensemble_tfmc.cu mc/mc_ensemble_tfmc.cuh
	$(NVCC) $(CFLAGS) -c mc/mc_ensemble_tfmc.cu -o mc/mc_ensemble_tfmc.o
```

链接时包含 `mc/mc_ensemble_tfmc.o`

## 总结

所有问题已修复：
1. ✅ 编译错误修复（cudaMemset + CHECK 宏）
2. ✅ 完整转动移除实现（与 LAMMPS 一致）
3. ✅ GPU 优化的并行实现
4. ✅ 数值稳定性保证
