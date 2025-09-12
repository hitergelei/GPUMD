import torch
from torch import nn
from ase import Atoms
import logging
import json
import time
from ase.io import read, write, Trajectory
from ase.build import bulk, make_supercell
from typing import List, Dict, Optional
from ase.neighborlist import NeighborList
from ase.calculators.eam import EAM
import matplotlib.pyplot as plt
import numpy as np
from torch_neigh import TorchNeighborList
# from adp_func import emb, emb_vec, rho, phi_AA, u11, w11




class AseDataReader:
    def __init__(self, cutoff=5.0):            
        self.cutoff = cutoff
        self.neighbor_list = TorchNeighborList(cutoff=self.cutoff)
    
    def __call__(self, atoms_obj: Atoms):
        atoms_data = {
            'num_atoms': torch.tensor([atoms_obj.get_global_number_of_atoms()], dtype=torch.int32),
            'elems':torch.tensor(atoms_obj.numbers),
            'coord': torch.tensor(atoms_obj.positions, dtype=torch.float),
        }
        atoms_data['image_idx'] = torch.zeros((atoms_data['num_atoms'],), dtype=torch.long)  # 默认初始化值是0
        if atoms_obj.pbc.any():
            atoms_data['cell'] = torch.tensor(atoms_obj.cell[:], dtype=torch.float)


        pairs, pair_diff, pair_dist = self.neighbor_list(atoms_obj)
        atoms_data['pairs'] = torch.from_numpy(pairs)   
        atoms_data['pair_diff'] = torch.from_numpy(pair_diff).float()
        atoms_data['num_pairs'] =torch.tensor([pairs.shape[0]], dtype=torch.int32)
        atoms_data['pair_dist'] = torch.from_numpy(pair_dist).float()

        # 当atoms有能量和力输出时，读取这些输出。这些数据对训练模型是必要的，但是对预测不是必要的。
        try:
            # energy = torch.tensor([atoms_obj.get_potential_energy()], dtype=torch.float32)  # old ase version
            energy = torch.tensor([atoms_obj.info['REF_energy']], dtype=torch.float)  # new ase version
            atoms_data['REF_energy'] = energy  # ase中新版本(ase 3.23.0b1之后)不再支持energy_key='energy'，而是使用'REF_energy'
        except (AttributeError, RuntimeError):
            pass

        try:
            # forces = torch.tensor(atoms_obj.get_forces(apply_constraint=False), dtype=torch.float32) # old ase version
            forces = torch.tensor(atoms_obj.arrays['REF_forces'], dtype=torch.float) # new ase version
            atoms_data['REF_forces'] = forces  # ase中新版本(ase 3.23.0b1之后)不再支持forces_key='forces'，而是使用'REF_forces'
        except (AttributeError, RuntimeError):
            pass


        return atoms_data






class AseDataset(torch.utils.data.Dataset):
    def __init__(self, ase_db, cutoff=5.0, **kwargs):
        super().__init__(**kwargs)
        
        # 可以使用一个数据集路径的字符串，或者ASE中的Trajectory或list[Atoms]来初始化这个类
        if isinstance(ase_db, str):
            # self.db = Trajectory(ase_db)
            self.db = read(ase_db, index=':', format='extxyz')   

        else:
            self.db = ase_db
        
        self.cutoff = cutoff
        self.atoms_reader = AseDataReader(cutoff)
        
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, idx):
        atoms = self.db[idx]
        atoms_data = self.atoms_reader(atoms)
        return atoms_data


# dataset = AseDataset('./extxyz_Ta_Ce/Ta_unitcell_and_surface.extxyz', cutoff=5.0)
# dataset[0]


# {k: [dic[k] for dic in atoms_data] for k in atoms_data[0]}.keys()
# dict_keys(['num_atoms', 'elems', 'coord', 'image_idx', 'cell', 'pairs', 'n_diff', 'num_pairs', 'energy', 'forces'])
def collate_atomsdata(atoms_data: List[dict], pin_memory=True) -> Dict:
    # convert from list of dicts to dict of lists
    dict_of_lists = {k: [dic[k] for dic in atoms_data] for k in atoms_data[0]}  # 例如，atoms_data[0] = {'num_atoms': tensor([192]), 'elems': tensor([, ...,]), 'coord': tensor([[0.0000, 0.0000, 0.0000], ...,]), 'image_idx': tensor([0, 0, 0, ..., 0, 0, 0]), 'cell': tensor([[ 5.0000,  0.0000,  0.0000], [ 0.0000,  5.0000,  0.0000}
    if pin_memory:
        pin = lambda x: x.pin_memory()
    else:
        pin = lambda x: x
    
    # concatenate tensors
    collated = {k: torch.cat(v) if v[0].shape else torch.stack(v) 
                for k, v in dict_of_lists.items()}  # 例如，len(dict_of_lists['energy']) = 16 （一个batch中有16个构型）  # dict_of_lists['image_idx'][0].shape
    
    # create image index for each atom   # 例如：len(atoms_data) = 15
    image_idx = torch.repeat_interleave(
        torch.arange(len(atoms_data)), collated['num_atoms'], dim=0
    )
    collated['image_idx'] = image_idx    # image_idx.shape = 3072 = 16(构型) * 192(原子) # collated['image_idx'] = tensor([ 0,  0,  0,  ..., 15, 15, 15])
    
    # shift index of edges (because of batching)    # 一个batch里面有16个结构。每个结构里目前都是192个原子
    # hjchen: 确保在批量化训练过程中每个构型的边（原子对）索引是唯一的。这样可以避免不同构型中的边索引冲突，并确保模型能够正确处理批量数据。
    if  'pairs' in collated:
        edge_offset = torch.zeros_like(collated['num_atoms'])  # 例如 torch.zeros_like(collated['num_atoms']).shape = torch.Size([16])
        edge_offset[1:] = collated['num_atoms'][:-1]  # 例如此时，第一次有：edge_offset = tensor([  0, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192])
        edge_offset = torch.cumsum(edge_offset, dim=0)  # # 计算每个图的边的起始索引偏移量
        edge_offset = torch.repeat_interleave(edge_offset, collated['num_pairs'])
        edge_idx = collated['pairs'] + edge_offset.unsqueeze(-1)  # collated['pairs'].shape = torch.Size([16638, 2]); edge_offset.unsqueeze(-1).shape = torch.Size([16638, 1])
        collated['pairs'] = edge_idx
    
    return collated

# 当输入的是split_file时，split_file是一个json文件，里面存储了数据集的划分情况，此时，我们就不需要再次划分训练集、验证集和测试集了
def split_data(dataset, split_file=None, val_ratio=0.1, test_ratio=None):
    if split_file:
        with open(split_file, 'r') as f:
            splits = json.load(f)
    else:
        n = len(dataset)
        indices = np.random.permutation(n)
        num_validation = int(n * val_ratio)
        num_train = n - num_validation
        if test_ratio is not None:
            num_test = int(n * test_ratio)
            num_train -= num_test
        splits = {
            'train': indices[:num_train].tolist(),
            'validation': indices[num_train:num_train + num_validation].tolist(),                    
        }
        if test_ratio is not None:
            splits['test'] = indices[num_train+num_validation:].tolist()
    
    with open('datasplits.json', 'w') as f:
        json.dump(splits, f)
    
    datasplits = {}
    for k, v in splits.items():
        datasplits[k] = torch.utils.data.Subset(dataset, v)
    return datasplits



# 接着我们要计算一下这些数据的平均值和标准差，用于数据归一化。
def get_normalization(dataset, per_atom=True):
    energies = []
    for sample in dataset:
        e = sample['REF_energy']
        if per_atom:
            e /= sample['num_atoms']
        energies.append(e)
    energies = torch.cat(energies)
    mean = torch.mean(energies).item()
    stddev = torch.std(energies).item()

    return mean, stddev



#--------------------------------------------------1. 模型定义----------------------------------------------------
# 定义嵌入能F(rho)或者emb(rho)，作为神经网络模型
# TODO: 1.考虑下使用dropout技术，以防止过拟合
# TODO: 2.激活函数的选择，是否需要使用ReLU或者其他激活函数？
# TODO: 3.隐藏层的个数，如何选择？ 通过超参数搜索来确定最佳的隐藏层个数？
class EmbeddingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EmbeddingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    # def forward(self, x):
    #     # 确保输入数据至少为一维
    #     if not x.dim() >= 1:
    #         x = x.unsqueeze(0)
    #     x = torch.tanh(self.fc1(x))
    #     x = self.fc2(x)
    #     return x

    def forward(self, x):
        # 请确保输入数据至少为一维（针对纯元素的ADP势，所以只有一维）
        if x.dim() == 0:
            x = x.unsqueeze(0)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


"""
EmbeddingNetwork 是一个自定义的网络类，它可能定义了至少一个线性层（可能还有更多的层，比如隐藏层）。每个线性层都有两个参数：权重和偏置。这些参数在网络的前向传播中用于计算输出。

现在，让我们看看您提供的参数输出：

1.第一个参数是一个形状为 torch.Size([10, 1]) 的张量，这很可能是 embedding_network 中某个线性层的权重。这个权重矩阵将输入映射到隐藏层。
2.第二个参数是一个形状为 torch.Size([10]) 的张量，这是与第一个权重矩阵对应的偏置向量。
3.第三个参数是一个形状为 torch.Size([1, 10]) 的张量，这很可能是 embedding_network 中另一个线性层的权重，将隐藏层的输出映射到最终的输出。
4.第四个参数是一个形状为 torch.Size([1]) 的张量，这是最终输出层的偏置。

这些参数的形状和数量取决于 EmbeddingNetwork 类的具体实现。您的 EmbeddingNetwork 类可能包含以下结构：
(a) 一个输入层到隐藏层的线性映射，其权重形状为 [hidden_size, input_size]（在这里是 [10, 1]）和偏置形状为 [hidden_size]（在这里是 [10]）。
(b) 一个隐藏层到输出层的线性映射，其权重形状为 [output_size, hidden_size]（在这里是 [1, 10]）和偏置形状为 [output_size]（在这里是 [1]）。
因此，当您调用 net.parameters() 时，PyTorch 返回了 embedding_network 中所有可训练的参数，这些参数包括权重和偏置，它们都是网络的一部分，并且在训练过程中会被优化。


Q：如果我想在net.parameters()是还要知道这个参数的名字是什么，怎么办？

for name, param in net.named_parameters():
    print(f"Name: {name}, Parameter: {param}")

"""


class EmbeddingNetwork_2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(EmbeddingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # 请确保输入数据至少为一维（针对纯元素的ADP势，所以只有一维）
        if x.dim() == 0:
            x = x.unsqueeze(0)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x




# TODO: 可考虑把custom_params当做参数传入，而不是写死在代码中！！！
class ADP_single_ele_model(nn.Module):
    """single element ADP model"""
    def __init__(self, 
                # element:str, 
                cutoff:float=5.0,
                normalization: bool=True,
                target_mean: float=0.0,
                target_stddev: float=1.0,
                atomwise_normalization: bool=True,
                **kwargs,
                ):
        
        super().__init__(**kwargs)
        # self.element = element

        # 以下是模型的参数(适用于纯元素的ADP势)
        custom_params = {'rho_phi_F_param':  torch.tensor([
                            2.7281,
                            3.5863051,
                            37.623623,
                            22.683228,
                            7.6616936,
                            5.5784864,
                            0.9215712,
                            1.7317773,
                            0.1413604,
                            0.24908023,
                            -6.270608,
                            2.2659059,
                            -0.18881902,
                            -3.2595265,
                            -5.8015256,
                            3.2561238,
                            1.1035414,
                            -0.95508283,
                            0.7645085,
                            -6.360732]), 
                         'dipole_param': torch.tensor([
                                                    -0.10194129,
                                                    -2.098797,
                                                    6.1936436, 
                                                    4.4990587,    # rc
                                                    6.757866      # h
                                                    ]),
                         'quadrupole_param': torch.tensor([
                                                    0.08105006,
                                                    -1.6661074,
                                                    -9.597149,
                                                    ])    # 少了'rc'和'h'，因为这两个参数是公有的，所以在dipole和quadrupole中都要保持一致
                        }
        # 通过register注册模型参数，让优化器可以追踪这些参数
        # 注册 rho_phi_F_param 参数
        for i, value in enumerate(custom_params['rho_phi_F_param']):
            self.register_parameter(f'rho_phi_F_param_{i}', nn.Parameter(value))
        
        # 注册 dipole_param 参数
        for i, value in enumerate(custom_params['dipole_param']):
            self.register_parameter(f'dipole_param_{i}', nn.Parameter(value))
        
        # 注册 quadrupole_param 参数
        for i, value in enumerate(custom_params['quadrupole_param']):
            self.register_parameter(f'quadrupole_param_{i}', nn.Parameter(value))


        self.cutoff = cutoff

        # 创建嵌入能F(rho)的神经网络模型实例
        # TODO: 如何优化这个F(rho)模型的参数？隐藏层个数、学习率等
        # TODO: 如何让每个原子共享这个F(rho)的神经模型的参数？
        self.embedding_network = EmbeddingNetwork(input_size=1, hidden_size=20, output_size=1)

        # 2个隐藏层的神经网络(有问题，报错)
        # self.embedding_network = EmbeddingNetwork_2(input_size=1, hidden_size1=10, hidden_size2=10, output_size=1)

        self.criterion = nn.MSELoss()






        # Normalisation constants
        self.register_buffer("normalization", torch.tensor(normalization))
        self.register_buffer("atomwise_normalization", torch.tensor(atomwise_normalization))
        self.register_buffer("normalize_stddev", torch.tensor(target_stddev))
        self.register_buffer("normalize_mean", torch.tensor(target_mean))

    # F(rho)的原来的实现
    # 《Machine learning enhanced empirical potentials for metals and alloys》
    #----F(rho)---eq (18)
    #TODO: 不要用numpy，而是用PyTorch的张量操作,否则无法被优化器追踪
    def emb_vec_analy(self, rho):
        self.rho_e = self.rho_phi_F_param_2
        self.Fn0 = self.rho_phi_F_param_10
        self.Fn1 = self.rho_phi_F_param_11
        self.Fn2 = self.rho_phi_F_param_12
        self.Fn3 = self.rho_phi_F_param_13
        self.F0 = self.rho_phi_F_param_14
        self.F1 = self.rho_phi_F_param_15
        self.F2 = self.rho_phi_F_param_16
        self.F3 = self.rho_phi_F_param_17
        self.Fe = self.rho_phi_F_param_19
        self.eta = self.rho_phi_F_param_18
        self.rho_s = self.rho_phi_F_param_3

        self.rho_n = 0.85 * self.rho_e    # 为什么要这么分？有必要这么分吗？一定要这么分吗？
        self.rho_0 = 1.15 * self.rho_e

        # 计算每个区间的条件
        condition_1 = rho < self.rho_n
        condition_2 = (rho >= self.rho_n) & (rho < self.rho_0)
        condition_3 = rho >= self.rho_0

        # 初始化结果数组
        result = torch.zeros_like(rho)

        # 当 rho < rho_n 时
        x1 = (rho[condition_1] / self.rho_n - 1)
        x2 = x1**2
        x3 = x1**3
        e10 = self.Fn0
        e11 = self.Fn1 * x1
        e12 = self.Fn2 * x2
        e13 = self.Fn3 * x3
        result[condition_1] = e10 + e11 + e12 + e13

        # 当 rho_n <= rho < rho_0 时
        t = (rho[condition_2] / self.rho_e - 1)
        x0 = t**0
        x1 = t**1
        x2 = t**2
        x3 = t**3
        e10 = self.F0 * x0
        e11 = self.F1 * x1
        e12 = self.F2 * x2
        e13 = self.F3 * x3
        result[condition_2] = e10 + e11 + e12 + e13

        # 当 rho >= rho_0 时
        x = rho[condition_3] / self.rho_s
        lnx = torch.log(x)
        result[condition_3] = self.Fe * (1 - self.eta * lnx) * (x)**self.eta

        return result

    # 考虑直接用神经网络架构代替F(rho)的形式

    # 创建嵌入能F(rho)的神经网络模型实例:self.embedding_network
    # 用PyTorch的张量操作,否则无法被优化器追踪
    def emb_vec_nn(self, density):
        """
        # 示例用法
        model = ExampleModel()
        density = tensor([38.4102478027], device='cuda:0', grad_fn=<UnsqueezeBackward0>)
        emb_ener_values = model.emb_vec_nn(density)
        print(emb_ener_values)
        """
    
        # 确保密度数据至少为一维（目前只支持纯元素体系，所以只有一维）
        if isinstance(density, torch.Tensor):
            density_tensor = density
        else:
            density_tensor = torch.tensor(density, dtype=torch.float32)
        # 确保 density_tensor 至少为一维
        if density_tensor.dim() == 0:
            density_tensor = density_tensor.unsqueeze(0)
        
        # 使用 embedding_network 计算嵌入能
        self.emb_ener_values = self.embedding_network(density_tensor)
        return self.emb_ener_values

    #-------------Zhou, Johnson and Wadley (zjw04) EAM-------------
    # also can be seen in paper "Machine learning enhanced empirical potentials for metals and alloys"  Eq (15)
    def rho(self, r):
        # TODO: 这些参数是否需要在__init__()时，初始化一下？
        self.fe = self.rho_phi_F_param_1
        self.beta = self.rho_phi_F_param_5
        self.re = self.rho_phi_F_param_0
        self.lambda_ = self.rho_phi_F_param_9

        term1 = self.fe * torch.exp(-self.beta *  (r / self.re - 1))
        # 使用np.power()速度更快，且支持传入为数组的参数
        term2 = 1 + torch.pow((r / self.re - self.lambda_), 20)   # term2 = 1 + (r / re - lambda_)**20 
        return term1 / term2

    #------Mo的φ(r)
    def phi_AA(self, r):
        self.A = self.rho_phi_F_param_6
        self.re = self.rho_phi_F_param_0
        self.alpha = self.rho_phi_F_param_4
        self.kappa = self.rho_phi_F_param_8  
        self.B = self.rho_phi_F_param_7
        self.beta = self.rho_phi_F_param_5
        self.lambda_ = self.rho_phi_F_param_9

        term1 = self.A * torch.exp(-self.alpha * (r / self.re - 1))
        # 使用np.power()速度更快，且支持传入为数组的参数
        term2 = 1 + torch.pow((r / self.re - self.kappa), 20)   # 1 + (r / re - kappa)**20
        left = term1 / term2

        term3 = self.B * torch.exp(-self.beta * (r / self.re - 1))
        term4 = 1 + torch.pow((r / self.re - self.lambda_), 20)  # 1 + (r/ re - lambda_)**20
        right = term3 / term4
        return left - right

    # 《Angular-dependent interatomic potential for the aluminum-hydrogen system》
    # ψ(r)-cutoff funciton  eq (7)
    def psi(self, x):
        # 使用PyTorch操作重写函数，适用于x为张量的输入
        return torch.where(x < 0, x**4 / (1 + x**4), torch.zeros_like(x))
    

    def u11(self, r):
        self.d1 = self.dipole_param_0
        self.d2 = self.dipole_param_1
        self.d3 = self.dipole_param_2
        self.rc = self.dipole_param_3
        self.h  = self.dipole_param_4
        # term1 = d1 * np.exp(-d2) + d3   #  Al-H论文和其他论文中公式不同，可能是作者d2后面漏乘了r， 此外quadrupole中也是漏乘了r
        # 比如在《Angular-dependent interatomic potential for tantalum》公式中就有r项。https://doi.org/10.1016/j.actamat.2006.06.034
        term1 = self.d1 * torch.exp(-self.d2 * r) + self.d3
        x = (r - self.rc) / self.h
        term2 = self.psi(x)
        return term1 * term2


    #-----w(r) - quadrupole functions eq (12)
    def w11(self, r):
        self.q1 = self.quadrupole_param_0
        self.q2 = self.quadrupole_param_1
        self.q3 = self.quadrupole_param_2

        self.rc = self.dipole_param_3
        self.h  = self.dipole_param_4

        # term1 = q1 * np.exp(-q2) + q3   # 该公式写法有问题，弃用
        term1 = self.q1 * torch.exp(-self.q2 * r) + self.q3
        x = (r - self.rc) / self.h
        term2 = self.psi(x)
        return term1 * term2
        




    def forward(self, input_dict: Dict[str, torch.Tensor], compute_forces: bool=True) -> Dict[str, torch.Tensor]:
        # access atoms properties needed （一个batch中有16个构型, 16 * 54atoms = 864）
        # 例如：num_atoms = tensor([54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54], device='cuda:0', dtype=torch.int32)
        num_atoms = input_dict['num_atoms']   # 例如，当按batch训练时，一个batch_size=16个构型，则此时input_dict['num_atoms'].shape = torch.Size([16])
        total_atoms = int(torch.sum(num_atoms))
        # 例如：num_pairs = tensor([1734, 1404, 1404, 1404, 1404, 1836, 1404, 1734, 1416, 1404, 1748, 1622, 1404, 1408, 1408, 1406], device='cuda:0', dtype=torch.int32)
        num_pairs = input_dict['num_pairs'] # 例如，当按batch训练时，一个batch_size=16个构型，则此时input_dict['num_pairs'].shape = torch.Size([16])
        edge = input_dict['pairs']    # input_dict['pairs'].shape = torch.Size([24140, 2])  
        edge_diff = input_dict['pair_diff']   # sum(num_pairs)的值其实就是edge_diff.shape的行数
        _edge_dist = input_dict['pair_dist']  # 测试用，结果应该是跟下面的edge_dist一样的
        image_idx = input_dict['image_idx']
        total_atoms = int(torch.sum(num_atoms))  # （一个batch中有16个构型, 16 * 54atoms = 864）


        if compute_forces:
            edge_diff.requires_grad_(True)
        edge_dist = torch.linalg.norm(edge_diff, dim=1)

        pairs, pair_diff, pair_dist = edge, edge_diff, edge_dist

        # print("\n<--------u11(rij)*dx, u11(rij)*dy, u11(rij)*dz-------->")
        # 这里的[:, None] 是一个索引操作，它在张量的第二维增加了一个新的维度
        # print(pair_diff * self.u11(pair_dist)[:, None])   # shape = torch.Size([24140, 3])
        # print((pair_diff * self.u11(pair_dist)[:, None]).shape)  # torch.Size([24140, 3])

        # 使用 PyTorch 的操作代替 NumPy 的操作
        mu_i_alpha = torch.cat((pairs, pair_diff * self.u11(pair_dist)[:, None]), dim=1)

        # print("\n<--------w11(rij)*rij^αβ (α=0,1,2; β=0,1,2；代表x,y,z这3个方向)-------->")
        # 使用PyTorch的操作代替np.einsum操作，从而确保张量操作在计算图中可以被优化器追踪，进而进行梯度计算和参数更新
        
        # ---> way1: 使用torch.bmm()方法
        # pair_diff_expanded = pair_diff.unsqueeze(2)  # shape: [24140, 3, 1]
        # result = torch.bmm(pair_diff_expanded, pair_diff_expanded.transpose(1, 2))  # shape: [24140, 3, 3]
        #------------------------------------------------------------
        # ---> way2: 可以使用 torch.einsum 代替原来的np.einsum('ij,ik->ijk', pair_diff, pair_diff)
        result = torch.einsum('ij,ik->ijk', pair_diff, pair_diff)
        #------------------------------------------------------------
        # print('result.shape = ', result.shape)  # result.shape =  torch.Size([24140, 3, 3])


        # 例如，当按batch训练时，一个batch_size=16个构型，则此时input_dict['num_atoms'].shape = torch.Size([16])
        # TODO: 一定要显性设置requires_grad=True么，否则在计算梯度时会出现错误？
        energy = torch.zeros_like(input_dict['num_atoms'], dtype=edge_diff.dtype, requires_grad=True)  # 一个batch中每个构型的能量

        # TODO 实现类似于node_scalar功能（构型中，每个单原子的属性，例如能量？） 例如，node_scalar.shape = torch.Size([3072])  # 3072 = 16(构型) * 192(原子)，其中， 一个batch中有16个构型，每个构型有192个原子
        # energy.index_add_(0, image_idx, node_scalar)  # node_scalar 决定了每个原子的属性，比如每个原子的能量

        # TODO: 这里，实现ADP势中构型的能量的计算！！！！（传统势似乎无法对单个原子能量进行分解？！！！！）


        # -------------way1----------------
        # 使用列表存储每个构型的 total_density
        # self.total_density = [torch.zeros(n, requires_grad=True) for n in num_atoms]

        # -------------way2----------------
        # 创建一个形状为 [batch_size, max(num_atoms)] 的张量，并用 0 填充
        batch_size = num_atoms.shape[0]
        device = num_atoms.device  # 获取 num_atoms 的设备
        max_atoms = int(num_atoms.max())

        #------------------------------ADP势中（一个batch的）结构的计算项--------------------------------
        # 创建一个形状为 [batch_size] 的张量来存储每个构型的 mu_energy之类的项
        batch_energy = torch.zeros(batch_size, device=device)
        embedding_energy = torch.zeros(batch_size, device=device)   # 公式的第1项：∑F(rho_i)
        embed_ener_per_atom = torch.zeros((batch_size, max_atoms), device=device)  # 测试打印绘图用：每个构型中的每个中心原子i的F(rho_i)的结果 
        pair_energy = torch.zeros(batch_size, device=device)        # 公式的第2项：∑Φ(rij) / 2
        
        mu_energy = torch.zeros(batch_size, device=device)          # 公式的第3项：∑mu_i^α^2 / 2
        lam_energy = torch.zeros(batch_size, device=device)         # 公式的第4项：∑lam_i^αβ^2 / 2
        trace_energy = torch.zeros(batch_size, device=device)       # 公式的第5项：-∑νi^2 / 6


        #--------------------------------------------------------------------------


        
        batch_total_density = torch.zeros((batch_size, max_atoms), device=device)  # shape = torch.Size([16, 54])
        # 创建掩码，标记有效的原子
        _mask = torch.arange(max_atoms, device=device).expand(batch_size, max_atoms) < num_atoms.unsqueeze(1)
        # 后面涉及到构型的原子电子密度累加时，则使用掩码进行电子密度项的累加
        # sum_density = (self.total_density * _mask.float()).sum(dim=1)


        # 参考：ase.calculators.eam.EAM 中的函数和 lammps 的 pair_adp.cpp 代码
        batch_mu = torch.zeros((batch_size, max_atoms, 3), device=device, requires_grad=True)  # dx, dy, dz 三个方向
        batch_lam_matrix_9 = torch.zeros((batch_size, max_atoms, 3, 3), device=device, requires_grad=True)  # 3x3 矩阵



        # _phi_AA_values = self.phi_AA(edge_dist)   # TODO: 如果一个batch中每个构型的原子数有不一致，得考虑用掩码方式进行计算
        # _rho_i_values = self.rho(edge_dist)   # TODO: 如果一个batch中每个构型的原子数有不一致，得考虑用掩码方式进行计算
  


        atom_offset = 0  # 初始化原子偏移量

        # 遍历batch_size
        for batch_i in range(batch_size):
            emb_energy_cfg = 0  # 初始化每个构型的embedding_energy    # 公式的第1项：∑F(rho_i)
            pair_energy_cfg = 0   # 初始化每个构型的 pair_energy            # 公式的第2项：∑Φ(rij) / 2
            mu_cfg = []     # 初始化每个构型的 mu                     # 公式的第3项：∑mu_i^α^2 / 2
            lam_cfg = []    # 初始化每个构型的 lam                    # 公式的第4项：∑lam_i^αβ^2 / 2
            trace_cfg = 0  # 初始化每个构型的 trace                  # 公式的第5项：-∑νi^2 / 6

            # 遍历每个构型的原子数 
            for atom_i in range(num_atoms[batch_i].item()):  # 例如一个构型中的原子数为54
                global_atom_i = atom_i + atom_offset  # 计算全局原子索引 
                # 例如：选择 atom_idx 索引为 0 的粒子对 
                idx_mask = mu_i_alpha[:, 0].to(torch.int) == global_atom_i   # # 使用 to(torch.int) 代替 astype(int)
                # print(f"原子 {atom_i} 的掩码: {idx_mask}")  # idx_mask.cpu().numpy() # 表示先将张量从 GPU 移动到 CPU,然后转换为 NumPy 数组
                phi_AA_values = self.phi_AA(pair_dist[idx_mask])  # 选择原子对的距离
                pair_energy_cfg += torch.sum(phi_AA_values) / 2.0  # 公式的第2项：∑Φ(rij) / 2
                rho_i_values = self.rho(pair_dist[idx_mask])  # ρ(rij)  对应ase中的self.electron_density[j_index](r[nearest][use])
                density = torch.sum(rho_i_values, dim=0)    # ρ(i)  选择对应原子的rho_i  

                # 将计算结果存储在 batch_total_density 中 （注意：索引值对应的密度值为0是掩码的原子）
                batch_total_density[batch_i, atom_i] = torch.sum(rho_i_values, dim=0)   # 测试用： 每个中心原子i对应的ρ(i)的结果。
                # batch_total_density[batch_i, atom_i] = torch.sum(self.rho(pair_dist[idx_mask]), dim=0)   



                #--------case1:神经网络形式的NN(rho)------------

                # 每个中心原子atoms_i共享一个F(rho)的神经网络模型参数
                NN_rho_val = self.embedding_network(density)  # 中心原子atoms_i的密度（一个标量值，然后经过张量处理），作为输入到神经网络架构的emb函数中: F(rho_i)
                emb_energy_cfg += NN_rho_val    # 公式的第1项：∑F(rho_i)
                embed_ener_per_atom[batch_i, atom_i] = NN_rho_val   # 用于测试ase的结果: 每个中心原子i对应的F(rho_i))
                

                #--------case2:解析式的F(rho)------------
                # emb_ener_atomic_val_ref = self.emb_vec_analy(density) # emb_vec_analy表示中心原子atom_i的密度值输入到解析式的emb函数中: F(rho_i)
                # emb_energy_cfg += emb_ener_atomic_val_ref    # 公式的第1项：∑F(rho_i)
                # embed_ener_per_atom[batch_i, atom_i] = emb_ener_atomic_val_ref   # 用于测试ase的结果: 每个中心原子i对应的F(rho_i))


                
                mu_arr = mu_i_alpha[idx_mask][:, 2:5]   # 对应x, y, z三个方向的mu_i^α (α=0,1,2分别表示x, y, z方向)
                mu_sum = torch.sum(mu_arr, dim=0)   # 对应x, y, z三个方向的mu_i^α (α=0,1,2分别表示x, y, z方向)的累加和
                mu_cfg.append(mu_sum)  # 好像没啥用？

                # 避免在需要梯度的张量上进行原地操作, 否则这会破坏计算图。这里需要创建一个新的张量来存储修改后的值，故使用 clone() 方法来避免原地操作。
                new_batch_mu = batch_mu.clone()   # 先克隆 batch_mu
                new_batch_mu[batch_i, atom_i] = mu_sum  # 然后更新克隆的张量  # 用于测试ase的结果: 每个中心原子i对应的mu_i^α (α=0,1,2分别表示x, y, z方向)的结果
                batch_mu = new_batch_mu  # 最后用新的张量替换原来的 batch_mu （注：只要张量的batch_mu.requires_grad 属性为 True， 就可以被优化器追踪并用于求导的。）
                

                rvec = pair_diff[idx_mask]
                r = pair_dist[idx_mask]
                qr = self.w11(r)
                lam = torch.einsum('i,ij,ik->ijk', qr, rvec, rvec)
                lam_sum = torch.sum(lam, dim=0) # 对应x, y, z三个方向的lam_i^αβ (α=0,1,2; β=0,1,2；代表x,y,z这3个方向)的累加和
                lam_cfg.append(lam_sum)  # 好像没啥用？
                # 同样，这里使用 clone() 方法来避免原地操作。
                new_batch_lam_matrix_9 = batch_lam_matrix_9.clone()
                new_batch_lam_matrix_9[batch_i, atom_i] = lam_sum  # 用于测试ase的结果: 每个中心原子i对应的lam_i^αβ (α=0,1,2; β=0,1,2；代表x,y,z这3个方向)的结果
                batch_lam_matrix_9 = new_batch_lam_matrix_9  # 最后用新的张量替换原来的 batch_lam_matrix_9


            # 更新原子偏移量 (注：在每次遍历完一个构型后，更新atom_offset，使得下一个构型的原子索引从之前的最大索引值开始。)
            atom_offset += num_atoms[batch_i].item()

            # 将每个构型的 embedding_energy 存储在相应的索引位置
            embedding_energy[batch_i] = emb_energy_cfg     # 单个结构的计算 OK

            # 将每个构型的 pair_energy 存储在相应的索引位置
            pair_energy[batch_i] = pair_energy_cfg         # 单个结构的计算 OK

            # 计算每个构型的mu_energy
            mu_energy[batch_i] = torch.sum(batch_mu[batch_i]**2) / 2.   # 公式的第3项：∑mu_i^α^2 / 2

            # 计算每个构型的lam_energy
            lam_energy[batch_i] = torch.sum(batch_lam_matrix_9[batch_i]**2) / 2.     # 公式的第4项：∑lam_i^αβ^2 / 2


            
            for atom_i in range(num_atoms[batch_i].item()):  # this is the atom to be embedded
                # 公式的第5项：-∑νi^2 / 6
                trace_cfg -= torch.sum(batch_lam_matrix_9[batch_i, atom_i].trace()**2) / 6

            # 计算每个构型的trace_energy
            trace_energy[batch_i] = trace_cfg

            
        # 计算一个batch_size中，每个构型的总能量
        batch_energy = pair_energy + embedding_energy + mu_energy + lam_energy + trace_energy
        energy = batch_energy

        # Apply (de-)normalization
        if self.normalization:
            normalizer = self.normalize_stddev
            energy = normalizer * energy
            mean_shift = self.normalize_mean
            if self.atomwise_normalization:
                mean_shift = input_dict["num_atoms"] * mean_shift
            energy = energy + mean_shift

        result_dict = {'energy': energy}


        # 力的计算 (能量对位置的导数)
        if compute_forces:
            grad_outputs : List[Optional[torch.Tensor]] = [torch.ones_like(energy)]    # for model deploy
            dE_ddiff = torch.autograd.grad(
                [energy,],
                [edge_diff,],
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=True,
            )
            dE_ddiff = torch.zeros_like(edge_diff) if dE_ddiff is None else dE_ddiff[0]   # for torch.jit.script
            assert dE_ddiff is not None
            
            # diff = R_j - R_i, so -dE/dR_j = -dE/ddiff, -dE/R_i = dE/ddiff
            i_forces = torch.zeros((total_atoms, 3), device=edge_diff.device, dtype=edge_diff.dtype)
            j_forces = torch.zeros_like(i_forces)
            i_forces.index_add_(0, edge[:, 0], dE_ddiff)
            j_forces.index_add_(0, edge[:, 1], dE_ddiff)
            forces = i_forces - j_forces
            
            result_dict['forces'] = forces  


        return result_dict
        










cutoff_chj = 6.5000000000000000e+00
net = ADP_single_ele_model(cutoff=cutoff_chj)
#-----------------------------测试
# 打印模型的所有参数
for name, param in net.named_parameters():
    print(f"{name}: {param}")
#-----------------------------测试

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 把网络放到GPU上
net.to(device)





# 设置模型优化器optimizer，学习率调度器scheduler和损失函数loss。
# scheduler的设置是必要的，因为当模型逐渐趋于合理时，我们希望学习率小一些以接近全局最优。
# learning_rate = 0.0005

learning_rate = 0.01  # hjchen

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# scheduler
# scheduler_fn = lambda step: 0.96 ** (step / 100000)             # 每训练10万步减小学习率

scheduler_fn = lambda step: 0.96 ** (step / 10000)             # 每训练1万步减小学习率

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)

criterion = torch.nn.MSELoss()



# 设置误差函数，用来计算RMSE和MAE等误差量。
# 误差计算的损失函数
def update_metrics(metrics, batch, outputs, e_loss, f_loss):
    # Accumulate counts of atoms and structures
    metrics['count_a'] += batch['REF_forces'].shape[0]
    metrics['count_s'] += batch['REF_energy'].shape[0]
    
    # Accumulate squared errors and absolute errors
    e_diff = outputs["energy"].detach() - batch["REF_energy"]
    f_diff = outputs["forces"].detach() - batch["REF_forces"]
    metrics['e_se'] += e_diff.square().sum()
    metrics['e_ae'] += e_diff.abs().sum()
    metrics['f_se'] += f_diff.square().sum()
    metrics['f_ae'] += f_diff.abs().sum()
    
    # Accumulate losses
    metrics['loss_e'] += e_loss.detach().item() * batch['REF_energy'].shape[0]  # 累积了能量损失乘以批次中结构的数量。
    metrics['loss_f'] += f_loss.detach().item() * batch['REF_forces'].shape[0]  #  累积了力损失乘以批次中原子的数量。



    # TODO: via hjchen: Accumulate per-atom energy squared errors
    # num_atoms = batch['REF_forces'].shape[0]
    # per_atom_e_diff = e_diff / num_atoms  # 假设能量均匀分配到每个原子
    # metrics['per_atom_e_se'] += per_atom_e_diff.square().sum()
    # metrics['per_atom_count'] += num_atoms

def reset_metrics(metrics):
    # Reset all metrics
    for k in metrics:
        metrics[k] = 0

def compute_final_metrics(metrics):
    # Compute final metrics such as RMSE, MAE, etc.
    final_metrics = {}
    final_metrics['e_rmse'] = torch.sqrt(metrics['e_se'] / metrics['count_s'])
    final_metrics['e_mae'] = metrics['e_ae'] / metrics['count_s']
    final_metrics['f_rmse'] = torch.sqrt(metrics['f_se'] / metrics['count_a'] / 3)   # metrics['count_a']是每个Epoch中训练集（或验证集）的总原子数，除以3是因为每个原子有3个力分量
    final_metrics['f_mae'] = metrics['f_ae'] / metrics['count_a'] / 3
    final_metrics['loss_e'] = metrics['loss_e'] / metrics['count_s']
    final_metrics['loss_f'] = metrics['loss_f'] / metrics['count_a']
    final_metrics['total_loss'] = final_metrics['loss_e'] + final_metrics['loss_f']
    final_metrics['cost'] = metrics['cost']

    # TODO:  还得在AseDataReader方法中添加一个新的属性：REF_per_atom_energy，用于存储每个原子的能量值
    #  hjchen: Compute per-atom energy RMSE
    # final_metrics['per_atom_e_rmse'] = torch.sqrt(metrics['e_se'] / metrics['count_a'])


    # 把metrics从GPU转移到CPU
    for k, v in final_metrics.items():
        if isinstance(v, torch.Tensor):
            final_metrics[k] = v.item()
            
    return final_metrics

def log_metrics(metrics, batch_idx, total_batches):
    # Log metrics
    print(f'Batch {batch_idx}/{total_batches}: ' + ", ".join([f"{k}= {v:10.3g}" for k, v in metrics.items()]), flush=True)  # 设置flush=True 会立即将输出刷新到文件，而不是等待缓冲区填满后才刷新。













# --------------------------------------------------2. 模型训练----------------------------------------------------
forces_weight = 98
energy_weight = 2

def run_epoch(
    net, 
    dataloader, 
    criterion, 
    optimizer, 
    scheduler, 
    device,
    energy_weight,
    forces_weight,
    log_freq=10,
    is_training=True):
    # Initialize metrics
    metrics = {
        'count_a': 0, 'count_s': 0, 'cost': 0,
        'e_se': 0, 'e_ae': 0, 'f_se': 0, 'f_ae': 0, 
        'loss_e': 0, 'loss_f': 0,
        # 'per_atom_e_se': 0, 'per_atom_count': 0,
    }  # via hjchen: 初始化metrics字典时，添加per_atom_e_se和per_atom_count
    
    # Set network mode
    net.train(is_training)
    
    if is_training:
        print("Training:")

    # Loop over batches
    for batch_idx, batch_host in enumerate(dataloader):
        start = time.time()
        
        # Transfer to device
        batch = {k: v.to(device=device, non_blocking=True) for (k, v) in batch_host.items()}
        
        # Reset gradients if training
        if is_training:
            optimizer.zero_grad()
        
        # Forward pass
        outputs = net(batch, compute_forces=True)    # 前向传播计算输出
        
        print('outputs["energy"] = ', outputs["energy"])  # 只有第1轮的第一个batch_size是直接计算的，后面的batch_size都是通过误差反向传播等优化ADP参数后，计算的
        print('batch["REF_energy"] = ', batch["REF_energy"])

        # Calculate losses
        # TODO : 如果一个batch中size是16，但是最后一个batch的size是8，这样会不会影响训练效果？？？Loss此时是如何计算的？？？
        # TODO: 是否有必要通过ase对构型进行权重处理？？？然后训练时，有针对性的调整权重来训练？ NEP和mace中都有对数据集进行权重处理的代码
        e_loss = criterion(outputs["energy"], batch["REF_energy"]) * energy_weight    # outputs['energy'].shape = torch.Size([16])
        f_loss = criterion(outputs['forces'], batch['REF_forces']) * forces_weight
        total_loss = e_loss + f_loss
        
        # Update metrics
        update_metrics(metrics, batch, outputs, e_loss, f_loss)
        
        # Backward pass and optimization if training
        if is_training:
            total_loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Update training cost
        metrics['cost'] += time.time() - start
        
        # Logging during training
        if is_training and (batch_idx + 1) % log_freq == 0:
            log_metrics(compute_final_metrics(metrics), batch_idx + 1, len(dataloader))
            reset_metrics(metrics)

    # Compute final metrics for the epoch if validation
    if not is_training:
        final_metrics = compute_final_metrics(metrics)
        print("\nValidation:")
        log_metrics(final_metrics, batch_idx + 1, len(dataloader))
        return final_metrics
    






#----------------------------------------------------------------------------------------------------

# https://github.com/ACEsuit/mace/blob/main/mace/data/utils.py
# Since ASE version 3.23.0b1, using energy_key 'energy' is no longer safe, instead, use 'REF_energy'. forces and stress are also need to be changed like this.

# 还可以使用ase对训练集带权重的数据进行处理（mace/data/utils.py中也有处理），例如GPUMD中：
#https://github.com/zhyan0603/GPUMDkit/blob/main/Scripts/format_conversion/add_weight.py

# 切分数据集, 自定义的截断距离为5.0 （可调超参数？）
dataset_all = AseDataset('./extxyz_Mo/all_186_Mo_structs-with-54atoms_REF_prefix.extxyz', cutoff=cutoff_chj)

# dataset = dataset_all[0]   # 一个构型的数据
dataset = dataset_all   # 所有构型的数据

import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

"""
为了在打印张量时显示更多的有效位数，你可以使用 torch.set_printoptions 来设置打印选项。
你可以增加 precision 参数的值来显示更多的有效位数。
"""
torch.set_printoptions(precision=10)

_batch_size = 16

# hjchen: 我怀疑train_loader在每个Epoch中没有被打乱，所以我在这里创建train_loader时设置了shuffle=True， 在每个Epoch开始时重新创建train_loader
def create_data_loaders(dataset, batch_size, split_file):
    datasplits = split_data(dataset, split_file=split_file)  # 当有输入split_file时，会从split_file中读取数据集的划分
    train_loader = torch.utils.data.DataLoader(
        dataset=datasplits['train'],
        batch_size=batch_size,
        collate_fn=collate_atomsdata,
        shuffle=True  # 确保batch_size中训练数据被随机打乱
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=datasplits['validation'],
        batch_size=batch_size,
        collate_fn=collate_atomsdata,
        shuffle=False  # 验证集通常不需要打乱
    )
    return train_loader, val_loader

# 初始化数据加载器
# train_loader, val_loader = create_data_loaders(dataset, _batch_size, 'datasplits_chj.json')


# 我们也可以选择在训练代码中自动保存模型。当模型训练过久时，可能出现训练集的误差显著小于验证集，此时便表示我们的模型训练正在过拟合。
# 我们可以使用early-stopping早停技术来提前结束我们的模型训练。以下代码包含了如何自动保存最优模型，如何early-stopping。
class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, best_loss):
        if val_loss - best_loss > self.min_delta:
            self.counter +=1
            if self.counter >= self.patience:        # 当 val_loss > best_loss 达到 patience 次数时，早停条件成立
                self.early_stop = True
                
        return self.early_stop

#TODO: 弄一个600轮的基本差不多了，对于这个数据集，1000轮应该是足够的了.但是要设置每多少轮保存一次模型，以及多少轮early stop？防止断电重训以及过拟合等情况
max_epoch = 600   # for 186 data Mo


best_val_loss = np.inf       # 把初始最低loss设为无限大 
early_stop = EarlyStopping(patience=20, min_delta=0.001)  # 设置早停条件
metrics = []         # 收集metrics用来画图

for i in range(max_epoch):
    print(f"\nEpoch {i+1}/{max_epoch}:")
    
    # 每个epoch重新创建数据加载器，确保数据被打乱
    train_loader, val_loader = create_data_loaders(dataset, _batch_size, 'datasplits_chj.json')  # for 186 data Mo
    
    # run training
    run_epoch(net, train_loader, criterion, optimizer, scheduler, device, energy_weight, forces_weight, log_freq=10, is_training=True)
    
    # run validation
    final_metrics = run_epoch(net, val_loader, criterion, optimizer, scheduler, device, energy_weight, forces_weight, log_freq=10, is_training=False)

    # collect metrics
    metrics.append(final_metrics)

    # 每200轮保存一次模型
    if (i + 1) % 200 == 0:
        torch.save(net, f'model_epoch_{i+1}.pth')
        print(f"------------------>Model saved at epoch {i+1}")

    #-----------------------------------早停的代码似乎不好-----------------------------------
    # # early_stopping
    # if not early_stop(final_metrics['total_loss'], best_val_loss):
    #     if final_metrics['total_loss'] < best_val_loss:
    #         best_val_loss = final_metrics['total_loss']
    #         torch.save(net, 'best_model_chj.pt')
    # else:
    #     break       # 当早停条件成立时，跳出循环


    # # 检查早停条件
    # if early_stop(final_metrics['total_loss'], best_val_loss):
    #     print("Early stopping")
    #     break  # 当早停条件成立时，跳出循环

    # # 保存最佳模型
    # if final_metrics['total_loss'] < best_val_loss:
    #     best_val_loss = final_metrics['total_loss']
    #     torch.save(net, 'best_model_chj.pt')


#------------------------------------------------------------------------------
# datasplits = split_data(dataset, split_file='datasplits_chj.json')
# train_loader = torch.utils.data.DataLoader(
#     dataset=datasplits['train'],
#     batch_size= _batch_size,
#     collate_fn=collate_atomsdata,
# )
# val_loader = torch.utils.data.DataLoader(
#     datasplits['validation'],
#     batch_size= _batch_size,
#     collate_fn=collate_atomsdata,
# )

# mean, stddev = get_normalization(datasplits['train'], per_atom=True)
# print(f"target_mean={mean:.3f}, target_stddev={stddev:.3f}")


# # next(iter(train_loader))
# # next(iter(val_loader))
# #----------------------------------------------------------------------------------------------------




# criterion = torch.nn.MSELoss()



# max_epoch = 500

# for i in range(max_epoch):
#     print(f"\nEpoch {i+1}/{max_epoch}:")
    
#     # run training
#     run_epoch(net, train_loader, criterion, optimizer, scheduler, device, energy_weight, forces_weight, log_freq=10, is_training=True)
    
#     # run validation
#     final_metrics = run_epoch(net, val_loader, criterion, optimizer, scheduler, device, energy_weight, forces_weight, log_freq=10, is_training=False)

#---------------------------------------------------------------------------------

torch.save(net, 'trained_model_chj_temp.pth')


print("++++++++++++++++++++++++++++++++++++++++++++++++++++info++++++++++++++++++++++++++++++++++++++++++++")
for name, param in net.named_parameters():
    print(f"{name}: {param}")



