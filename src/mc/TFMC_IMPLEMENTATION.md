# Time-stamped Force-bias Monte Carlo (tfMC) CUDA Implementation for GPUMD

## 实现概述

本实现基于LAMMPS的fix_tfmc，完整移植到GPUMD的CUDA框架中，支持GPU并行加速。

## 已实现的文件

1. **mc_ensemble_tfmc.cuh** - 头文件
   - 定义MC_Ensemble_TFMC类
   - 继承自MC_Ensemble基类
   - 声明所有公共和私有方法

2. **mc_ensemble_tfmc.cu** - 实现文件
   - CUDA kernel实现位移生成
   - CUDA kernel实现COM运动移除
   - CUDA kernel实现位移应用
   - 主要算法逻辑

3. **mc.cu** - 更新主MC驱动
   - 添加tfMC ensemble类型支持
   - 解析tfMC参数
   - 集成到MC框架

4. **TFMC_README.md** - 使用文档
   - 详细的算法说明
   - 使用示例
   - 参数说明

5. **example_tfmc_run.in** - 示例输入文件
   - 展示各种tfMC用法

## 核心算法实现

### 1. 位移生成 (generate_tfmc_displacements_kernel)

```cuda
对于每个原子i和每个方向j:
  1. 计算 d_i = d_max * (m_min/m_i)^0.25
  2. 计算 γ = F_j · d_i / (2kT)
  3. 使用拒绝采样生成 ξ ∈ [-1, 1]
  4. displacement = ξ · d_i
```

**接受概率**:
- ξ < 0: P_acc = [exp(2ξγ)·exp(γ) - exp(-γ)] / [exp(γ) - exp(-γ)]
- ξ > 0: P_acc = [exp(-γ) - exp(2ξγ)·exp(-γ)] / [exp(γ) - exp(-γ)]
- ξ = 0: P_acc = 1

### 2. COM运动移除

```cuda
1. compute_com_displacement_kernel: 计算质心位移
   - 并行reduction计算 Σ(m_i · Δr_i)
   - 计算总质量
   
2. remove_com_motion_kernel: 移除COM运动
   - 每个原子减去COM位移
   - 支持选择性移除 (x, y, z方向)
```

### 3. 转动移除 (部分实现)

```cuda
compute_angular_momentum_kernel: 计算角动量
- L = Σ[m_i · (r_i - r_cm) × Δr_i]
- 完整实现需要计算惯性张量和角速度
```

## 使用方法

### run.in 语法

```
mc tfmc N_md N_mc T_initial T_final d_max seed [com x y z] [rot] [group ...]
```

### 参数说明

- `N_md`: MD步数间隔
- `N_mc`: 每次MC试探次数  
- `T_initial`: 初始温度 (K)
- `T_final`: 最终温度 (K)
- `d_max`: 最大位移长度 (Å)
- `seed`: 随机数种子

### 可选关键字

- `com x y z`: 固定质心运动 (0或1)
- `rot`: 固定转动
- `group method_id group_id`: 仅对指定组应用MC

### 示例

```bash
# 基本使用
mc tfmc 100 50 300.0 300.0 0.20 12345

# 固定COM运动
mc tfmc 100 50 300.0 300.0 0.20 12345 com 1 1 1

# 固定转动
mc tfmc 100 50 300.0 300.0 0.20 12345 rot

# 指定组
mc tfmc 100 50 300.0 300.0 0.20 12345 group 0 1
```

## 编译说明

### 修改 makefile

添加tfMC目标文件:

```makefile
mc/mc_ensemble_tfmc.o: mc/mc_ensemble_tfmc.cu mc/mc_ensemble_tfmc.cuh
	$(NVCC) $(CFLAGS) -c mc/mc_ensemble_tfmc.cu -o mc/mc_ensemble_tfmc.o
```

在链接步骤中包含: `mc/mc_ensemble_tfmc.o`

## 特性和优势

1. **GPU并行化**: 所有关键计算在GPU上并行执行
2. **高效采样**: 使用cuRAND在GPU上直接生成随机数
3. **灵活配置**: 支持COM和转动约束，支持分组
4. **与MD集成**: 无缝集成到GPUMD的MCMD框架

## 技术细节

### 数据结构

```cpp
GPU_Vector<double> mass;              // 原子质量
GPU_Vector<double> displacements;     // 位移向量 (3N)
GPU_Vector<double> com_displacement;  // COM位移 (4)
GPU_Vector<curandState> curand_states; // 随机数状态
```

### Kernel配置

- Block size: 256 threads
- Grid size: (N-1)/256 + 1 blocks
- 所有kernel都检查边界条件

### 精度考虑

- 使用double精度进行力和位移计算
- 温度单位转换: kB = 8.617343e-5 eV/K

## 已知限制

1. **转动移除**: 仅部分实现，需要完整的惯性张量计算
2. **分组支持**: 基本实现，需要更多测试
3. **统计输出**: 未实现接受率等统计信息
4. **性能优化**: 大系统可能需要进一步优化

## 未来改进

1. 完整实现转动移除功能
2. 添加接受率统计和输出
3. 支持约束原子 (类似LAMMPS的FixAtoms)
4. 性能分析和优化
5. 更详细的错误检查和验证
6. 支持可变温度tfMC

## 测试建议

### 测试系统

1. **简单立方晶格**: 验证基本功能
2. **多组分系统**: 测试质量缩放
3. **约束系统**: 测试COM和转动固定
4. **大系统**: 测试性能和可扩展性

### 验证指标

1. 能量守恒 (NVE系综)
2. 温度分布 (NVT系综)
3. 结构稳定性
4. COM运动是否正确移除
5. 与LAMMPS结果对比

## 参考文献

1. K. M. Bal and E. C. Neyts, J. Chem. Theory Comput. 11, 4545 (2015)
2. LAMMPS fix_tfmc 源代码
3. GPUMD MC框架文档

## 联系与支持

如有问题或建议，请提交issue到GPUMD GitHub仓库。

---
实现日期: 2025年11月20日
实现者: 基于LAMMPS fix_tfmc改编
