
## 待修改和增加的功能

### 二元或者多元体系的扩展
```txt
https://www.gpumd.org/dev/gpumd/input_files/model_xyz.html
https://www.gpumd.org/dev/nep/input_files/train_test_xyz.html
model.xyz中，species:S:1这种是ase库给extxyz格式官方规定的写法，不能改动的。model.xyz体系如果是一元，那就是1，如果是2元，那就是2，所以，read_xyz.cu中或者其他那个地方需要针对二元adp势文件格式，做一个判断或者元素正确映射处理么？因为我现在的model.xyz只是支持单质的，而且，run.in中的potential写法支持，也只是写成potential adp Mo.adp，如果我的体系是二元或多元，这样写法肯定是有问题的

如果model.xyz假如是只有Mo，然后/home/hjchen/projects/GPUMD/examples/adp_U_Mo/U_Mo.alloy.adp_cor.txt我只用到了Mo，那这个就有问题啊，程序不知道到底该调用谁的相互作用势函数，对吧。是不是可以写成这样的命名支持： 
如果model.xyz中是二元（既有U，又有Mo），那么就potential adp U_Mo.alloy.adp_cor.txt U Mo之类？如果model.xyz中只有Mo元素，那么就是potential adp U_Mo.alloy.adp_cor.txt Mo这样？如果model.xyz中只有U元素，那么就是potential adp U_Mo.alloy.adp_cor.txt U这样？ 这样可行么？是否有其他建模和设置问题？没有的话，你尝试下，对adp.cu以及其他关联的.cu代码进行功能扩展

```

### 降低近邻列表的计算复杂度
> 在adp.cu中，用的是O(N2)复杂度，有没有可能降低复杂度，比如采用类似GPUMD中实现的线性复杂度？