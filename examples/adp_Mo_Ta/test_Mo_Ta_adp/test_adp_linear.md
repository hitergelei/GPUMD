# ADP 线性复杂度测试文档

## 修改内容总结

### 1. 新增功能

在 `adp.cu` 中添加了线性复杂度的近邻计算功能，现在支持两种近邻计算方法：

- **O(N²) 暴力方法** (`build_neighbor_ON2`): 适用于小盒子系统，直接遍历所有原子对
- **O(N) 线性方法** (使用标准的 `find_neighbor`): 基于 cell list 算法，适用于大系统

### 2. 新增参数控制

在 ADP 势能文件中可以通过以下参数控制近邻计算方式和元素映射：

```
# 基本格式（支持多元素映射）
potential adp <adp_file> [element1] [element2] ... [neighbor=xx] [spline=xx]

# 单元素系统
potential adp Mo.adp Mo neighbor=linear
potential adp hjchen-Mo.adp Mo neighbor=on2

# 双元素系统（按顺序指定元素）
potential adp U_Mo.alloy.adp_cor.txt U Mo neighbor=linear
potential adp U_Mo.alloy.adp_cor.txt Mo U neighbor=on2

# 单元素体系使用双元素势能文件（只使用第一个元素Mo）  
potential adp U_Mo.alloy.adp_cor.txt Mo neighbor=linear

# 同时指定样条插值和近邻算法
potential adp U_Mo.alloy.adp_cor.txt U Mo spline=lammps neighbor=linear
```

参数选项：
- **元素映射**：按顺序指定体系中的元素，映射到ADP文件中对应的元素
- `neighbor=linear` 或 `neighbor=on1` 或 `neighbor=cell`: 使用 O(N) 线性复杂度算法（默认）
- `neighbor=on2` 或 `neighbor=n2` 或 `neighbor=brute`: 强制使用 O(N²) 暴力算法
- `spline=lammps`: 使用LAMMPS兼容的样条插值（默认）
- `spline=natural`: 使用自然三次样条插值

### 3. 自动选择机制

- 默认情况下，程序会自动选择最优算法：
  - 对于大盒子：自动使用 O(N) 线性算法
  - 对于小盒子：自动使用 O(N²) 算法（避免 cell list 的开销）
- 用户可以通过 `neighbor` 参数强制选择特定算法

### 4. 输出信息

程序运行时会显示当前使用的近邻计算方法和元素映射：

```
Use 2-element ADP potential, rc = 6.196997.
ADP element mapping: Mo->Mo 
ADP spline mode: LAMMPS.
ADP neighbor mode: O(N) cell list
```

对于双元素系统：
```
Use 2-element ADP potential, rc = 6.196997.
ADP element mapping: U->U Mo->Mo 
ADP spline mode: LAMMPS.
ADP neighbor mode: O(N) cell list
```

### 5. 代码实现细节

1. **新增 `map_element_types` GPU内核**: 将用户指定的元素类型映射到ADP文件中的元素索引
2. **新增 `setup_element_mapping` 函数**: 建立用户元素到ADP文件元素的映射关系
3. **修改 `parse_options` 函数**: 支持解析元素名称和 `neighbor` 参数
4. **修改 `compute` 函数**: 根据用户设置和盒子大小智能选择算法，并处理元素映射
5. **保留向后兼容性**: 不影响现有代码，默认使用线性算法和自动元素映射

### 6. 性能预期

- **小系统 (N < 1000)**: 两种方法性能相近
- **中等系统 (1000 < N < 10000)**: 线性方法开始显示优势
- **大系统 (N > 10000)**: 线性方法显著优于 O(N²) 方法

### 7. 使用示例

```bash
# 单元素系统 - 默认使用自动选择
potential adp hjchen-Mo.adp Mo

# 单元素系统 - 强制使用线性算法
potential adp hjchen-Mo.adp Mo neighbor=linear

# 单元素系统 - 强制使用 O(N²) 算法
potential adp hjchen-Mo.adp Mo neighbor=on2

# 双元素系统 - 使用U-Mo合金势能
potential adp U_Mo.alloy.adp_cor.txt U Mo neighbor=linear

# 单元素体系使用双元素势能文件（只映射到Mo）
potential adp U_Mo.alloy.adp_cor.txt Mo neighbor=linear

# 同时设置样条插值和近邻算法
potential adp U_Mo.alloy.adp_cor.txt U Mo spline=lammps neighbor=linear

# model.xyz只有Mo原子，但使用U-Mo势能文件
potential adp U_Mo.alloy.adp_cor.txt Mo neighbor=linear
```

## 测试建议

1. **功能测试**: 使用相同输入分别测试不同元素映射，确保结果一致
2. **性能测试**: 在不同大小的系统上比较两种算法的运行时间
3. **内存测试**: 检查内存使用情况，特别是在大系统上
4. **参数测试**: 验证元素映射和参数解析功能正确工作
5. **多元素测试**: 测试单元素、双元素系统的各种组合

## 注意事项

1. 线性算法需要额外的内存存储 cell list 数据结构
2. 对于非常小的系统，O(N²) 算法可能更高效
3. 程序会自动处理周期性边界条件
4. 修改保持了与 LAMMPS ADP 实现的完全兼容性
5. **元素映射顺序很重要**：用户指定的元素顺序必须与model.xyz中的类型编号对应
6. **势能文件兼容性**：确保ADP文件包含所需的元素类型