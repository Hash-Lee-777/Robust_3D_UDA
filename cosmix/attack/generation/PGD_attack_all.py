import MinkowskiEngine as ME
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
import utils.models as models
from utils.collation import CollateFN
from utils.datasets.synlidar import SynLiDARDataset
from utils.losses import CELoss, DICELoss, SoftDICELoss
import torch
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
import time
import os
from pytorch_lightning import Trainer
import pdb
import torch.optim as optim
import argparse
from attack import CrossEntropyAdvLoss, LogitsAdvLoss
from attack import L2Dist
import torch.nn as nn
import open3d as o3d
from datetime import datetime
from sklearn.neighbors import KDTree 
import random
import warnings
import argparse
from configs import get_config
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import loggers as pl_loggers
from attack.datasets.SynlidarPhase import AdvSynLiDARDataset

# generate adversarial training and validaiton dataset and save in local

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

def timeit(func):
    """
        recording time cost of func
    """
    def _wrap(*args,**kwargs):
        start_time = time.time()
        result = func(*args,**kwargs)
        elastic_time = time.time()-start_time
        print("[INFO] The execution time of the function '%s is %.6fs\n"%(
            func.__name__,elastic_time
        ))
        return result
    return _wrap

def load_model(checkpoint_path, model):
    """
        load pretrained model
    """
    def clean_state_dict(state):
        # clean state dict from names of PL
        for k in list(ckpt.keys()):
            if "model" in k:
                ckpt[k.replace("model.", "")] = ckpt[k]
            del ckpt[k]
        return state

    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
    ckpt = clean_state_dict(ckpt)
    model.load_state_dict(ckpt, strict=True)
    return model

class AdvAttacker:
    """
    class for cw attack
    """
    def __init__(
        self,
        model=None,
        attack_lr=1e-3,
        attack_weight=0.,
        attack_alpha=1,
        attack_iters=10,
        attack_eps=10,
        attack_sample_rate=1e-3,
        attack_dist_c=1e-2,
        attack_dist_type='l1',
        attack_bound=200,
        attack_save_dir='experiments/SAVEDIR/attackedLidar_Poss100',
        attack_save_prediction=True,        
        ):
        """
        FSGD attack by perturbing points.
        Args:
            model (torch.nn.Module): victim model
            attack_lr (float, optional): lr for optimization. Defaults to 1e-2.
            attack_weight(float): weight for opitmization. Defaults to 0.
            attack_alpha: perturb scope for attack. Defaults to 1.
            attack_sample_rate: sample rate for choosing the adversarial sample.
            attack_dist_c: hyperparams for l2 distance
            attack_eps: the max clamp perturbation
            attack_dist_type: ['l1','l2']
        """
        self.model = model
        self.attack_lr = attack_lr
        self.attack_weight = attack_weight
        self.attack_alpha = attack_alpha
        self.attack_iters = attack_iters
        self.attack_sample_rate = attack_sample_rate
        self.attack_dist_c = attack_dist_c
        self.attack_eps = attack_eps
        self.attack_dist_type = attack_dist_type
        self.attack_bound = attack_bound
        self.attack_save_dir = attack_save_dir
        self.attack_save_prediction = attack_save_prediction
         # 初始化记录的指标
        self.attack_stats = {
            'attack_lr': attack_lr,
            'attack_weight': attack_weight,
            'attack_alpha': attack_alpha,
            'attack_iters': attack_iters,
            'attack_eps': attack_eps,
            'attack_sample_rate': attack_sample_rate,
            'attack_dist_c': attack_dist_c,
            'attack_dist_type': attack_dist_type,
            'attack_bound': attack_bound,
            'attack_save_dir': attack_save_dir,
            'attack_save_prediction': attack_save_prediction
        }

    def __str__(self) -> str:
        # 返回攻击的指标信息
        return str(self.attack_stats)

    def init_model(self,model):
        self.model = model

    @timeit
    def post_filter(self,adv_coords,bound=1):
        """
        params:
            adv_coords: Tensor([N,4])
        """
        adv_filter_coords = None
        record_num = 0
        # 将坐标转换到CPU上以便处理
        adv_coords_cpu = adv_coords.detach().clone().cpu().numpy()
        adv_coords_cpu = adv_coords_cpu.astype(np.int32)
        back_coords = adv_coords_cpu.copy() # 保存原始坐标
        # 将坐标转换为字符串以便比较
        unique_coords = set()
        # 初始化一个列表，用于记录需要进行扰动的索引
        duplicated_indices = []

        # 初始化检查
        for i, coord in enumerate(adv_coords_cpu[:,1:]):
            coords_tuple = tuple(coord)
            if coords_tuple in unique_coords:
                duplicated_indices.append(i)
            else:
                unique_coords.add(coords_tuple)

        if not duplicated_indices:
            adv_filter_coords = torch.tensor(adv_coords_cpu, device=adv_coords.device)
            flag = False
            return flag,adv_filter_coords,record_num

        # 循环直到不存在重复点为止
        while duplicated_indices:
            # 对重复点进行处理
            record_num += 1
            for idx in duplicated_indices:
                # # 随机生成一个扰动并四舍五入为整数
                # delta = np.round(np.random.uniform(low=-bound, high=bound, size=(3,))).astype(np.int32)
                # 随机生成一个位置索引，只有一个元素为1，其余为0
                random_index = np.random.randint(3)
                delta = np.zeros((3,), dtype=np.int32)
                delta[random_index] = 1

                # 将扰动应用到重复点上
                adv_coords_cpu[idx, 1:] += delta

            # 重新检查更新后的点云是否还有重复点
            unique_coords.clear()
            duplicated_indices = []
            # 使用循环检查每个坐标是否重复，如果重复则记录其索引
            for i, coord in enumerate(adv_coords_cpu[:, 1:]):
                coords_tuple = tuple(coord)
                if coords_tuple in unique_coords:
                    duplicated_indices.append(i)
                else:
                    unique_coords.add(coords_tuple)

            # print(f"[INFO] post filter iteration time: {record_num}, duplicated numbers are {len(duplicated_indices)}")

        adv_filter_coords = torch.tensor(adv_coords_cpu, device=adv_coords.device)
        return True, adv_filter_coords,  record_num

    def _group_coords(self,coords):
        """
            对传入的coords进行分组
        """
        unique_values = torch.unique(coords[:, 0]) # [0,1,2,3]
        # save coordinates
        grouped_coords = []
        # walk throught all values in unique_values
        for value in unique_values:
            # 构造布尔索引
            bool_index = (coords[:, 0] == value)
            # 使用布尔索引提取子张量
            grouped_coordinates = coords[bool_index]
            # 存储到列表中
            grouped_coords.append(grouped_coordinates)
        return grouped_coords

    def _ungroup(self,tensors_list=None):
        """
            对传入的coords进行解分组
        """
        combined_coordinates = []
        for groups_idx in range(len(tensors_list)):
            grouped_coords = tensors_list[groups_idx]

            # 按0维拼接坐标、特征和标签
            combined_coordinates.append(grouped_coords)
            # 将列表转换为张量
        combined_coordinates = torch.cat(combined_coordinates, dim=0)
        return combined_coordinates

    def _duplicate_remove(self,adv_coords):
        """
        params:
            adv_coords: Tensor([N,4])
            return: remove_num,adv_coords,remaining_indices
        """
        adv_filter_coords = None
        record_num = 0
        adv_coords_cpu = adv_coords.detach().clone().cpu().numpy()
        adv_coords_cpu = adv_coords_cpu.astype(np.int32)
        unique_coords = set()
        removed_indices = set()  # 用于记录要删除的点的索引

        # 初始化检查
        for i, coord in enumerate(adv_coords_cpu[:,1:]):
            coords_tuple = tuple(coord)
            if coords_tuple in unique_coords:
                # if exists, recording related coord index
                removed_indices.add(i)
            else:
                unique_coords.add(coords_tuple)

        if not removed_indices:
            adv_filter_coords = torch.tensor(adv_coords_cpu, device=adv_coords.device)
            return False,adv_filter_coords,torch.arange(len(adv_coords_cpu))

        # 删除重复的点并记录剩下的点的索引
        adv_filter_coords = np.delete(adv_coords_cpu,list(removed_indices),axis=0)
        remaining_indices = torch.tensor([i for i in range(len(adv_coords_cpu)) if i not in removed_indices])
        return len(list(removed_indices)), adv_filter_coords,remaining_indices

    @timeit
    def post_filter_remove(self,adv_coords):
        grouped_coords = self._group_coords(adv_coords)
        remaining_indices = []
        remaining_coords = []
        num_record = 0
        for coords in grouped_coords:
            num_removed,filter_coords,indices=self._duplicate_remove(coords)
            remaining_coords.append(torch.from_numpy(filter_coords).int())
            remaining_indices.append(indices)
            num_record += num_removed
        coords = self._ungroup(remaining_coords)
        indices = self._ungroup(remaining_indices)
        return num_record,coords,indices

    def attackPGD_test(self,data,labels=None):
        adv_data,loss = None,None
        ori_coords = data.coordinates.clone().detach()
        max_ori_x = torch.max(torch.abs(ori_coords[:,1]))
        max_ori_y = torch.max(torch.abs(ori_coords[:,2]))
        max_ori_z = torch.max(torch.abs(ori_coords[:,3]))
        # 初始特征处理
        features_ori_x = ori_coords[:,1]/max_ori_x + 1.0
        features_ori_y = ori_coords[:,2]/max_ori_y + 1.0
        features_ori_z = ori_coords[:,3]/max_ori_z + 1.0
        features_ori_x = features_ori_x.unsqueeze(1)
        features_ori_y = features_ori_y.unsqueeze(1)
        features_ori_z = features_ori_z.unsqueeze(1)
        concact_ori_features = torch.cat((features_ori_x,features_ori_y,features_ori_z),dim=0)
        ori_features = data.features
        adv_coords = data.coordinates.clone().detach()
        # 成功进行语义分割的样本个数
        success_label_rate = [0., 0., 0.]
        # 干扰样本强度
        for _ in range(self.attack_iters):
            grad_X = None
            grad_Y = None
            grad_Z = None
            # 找到 x, y, z 列的绝对值最大值
            max_abs_x = torch.max(torch.abs(adv_coords[:, 1]))
            max_abs_y = torch.max(torch.abs(adv_coords[:, 2]))
            max_abs_z = torch.max(torch.abs(adv_coords[:, 3]))
            # 归一化 x, y, z 列的值到 0-1 范围内
            features_x_tensor = adv_coords[:, 1] / max_abs_x + 1.0
            features_y_tensor = adv_coords[:, 2] / max_abs_y + 1.0
            features_z_tensor = adv_coords[:, 3] / max_abs_z + 1.0
            features_x_tensor = features_x_tensor.unsqueeze(1)
            features_y_tensor = features_y_tensor.unsqueeze(1)
            features_z_tensor = features_z_tensor.unsqueeze(1)
            criterion = SoftDICELoss(ignore_label=-1)
            # Create a sparse tensor object
            st = ME.SparseTensor(coordinates=ori_coords, features=ori_features)
            st_X = ME.SparseTensor(coordinates=adv_coords, features=features_x_tensor)
            st_Y = ME.SparseTensor(coordinates=adv_coords, features=features_y_tensor)
            st_Z = ME.SparseTensor(coordinates=adv_coords, features=features_z_tensor)
            st_X.features.requires_grad = True
            st_Y.features.requires_grad = True
            st_Z.features.requires_grad = True
            concat_adv_features = torch.cat((st_X.features, st_Y.features, st_Z.features), dim=0)
            # Compute the loss function (sum of all elements in coordinates and features)
            out   = self.model(st)
            out_X = self.model(st_X)
            out_Y = self.model(st_Y)
            out_Z = self.model(st_Z)
            logits = out.F
            logits_X = out_X.F
            logits_Y = out_Y.F
            logits_Z = out_Z.F
            pred = torch.argmax(logits,dim=1)
            pred_X = torch.argmax(logits_X, dim=1)  # [num_points,1]
            pred_Y = torch.argmax(logits_Y, dim=1)
            pred_Z = torch.argmax(logits_Z, dim=1)
            # loss_ori = criterion(logits,labels)
            loss_X = criterion(logits_X, labels)
            loss_Y = criterion(logits_Y, labels)
            loss_Z = criterion(logits_Z, labels)
            prev_loss = loss_X + loss_Y + loss_Z
            prev_loss.requires_grad_(True)
            # 计算L2距离损失
            # dist_loss = F.mse_loss(concat_adv_features,concact_ori_features)
            # 计算L1距离损失
            # dist_loss = F.l1_loss(concat_adv_features,concact_ori_features)
            # 确定距离损失函数
            dist_loss = F.l1_loss(concat_adv_features,concact_ori_features) if self.attack_dist_type=='l1' else F.mse_loss(concat_adv_features,concact_ori_features)
            # 确定光滑度损失函数
            dist_loss.requires_grad_(True)
            # 计算总的损失函数
            cost = prev_loss + self.attack_dist_c*dist_loss
            cost.requires_grad_(True)
            opt = torch.optim.Adam([st_X.features, st_Y.features, st_Z.features], lr=self.attack_lr, weight_decay=self.attack_weight)
            ori_success_label_num = torch.sum(torch.eq(pred, labels)).item()
            success_label_rate[0] = torch.sum(torch.eq(pred_X, labels)).item()/labels.size()[0]
            success_label_rate[1] = torch.sum(torch.eq(pred_Y, labels)).item()/labels.size()[0]
            success_label_rate[2] = torch.sum(torch.eq(pred_Z, labels)).item()/labels.size()[0]
            # 反向传播
            opt.zero_grad()
            cost.backward()
            opt.step()
            # get gradient of X,Y,Z
            back_grad_X = st_X.features.grad.squeeze()
            back_grad_Y = st_Y.features.grad.squeeze()
            back_grad_Z = st_Z.features.grad.squeeze()

            # select apparent gradient
            max_grad_X, max_indices_X = torch.topk(torch.abs(back_grad_X),int(self.attack_sample_rate*back_grad_X.size(0)))
            max_grad_Y, max_indices_Y = torch.topk(torch.abs(back_grad_Y),int(self.attack_sample_rate*back_grad_Y.size(0)))
            max_grad_Z, max_indices_Z = torch.topk(torch.abs(back_grad_Z),int(self.attack_sample_rate*back_grad_Z.size(0)))

            min_grad_X, min_indices_X = torch.topk(torch.abs(back_grad_X),int(self.attack_sample_rate*back_grad_X.size(0)),largest=False)
            min_grad_Y, min_indices_Y = torch.topk(torch.abs(back_grad_Y),int(self.attack_sample_rate*back_grad_Y.size(0)),largest=False)
            min_grad_Z, min_indices_Z = torch.topk(torch.abs(back_grad_Z),int(self.attack_sample_rate*back_grad_Z.size(0)),largest=False)

            all_indices_X = torch.cat([max_indices_X,min_indices_X])
            all_indices_X = torch.unique(all_indices_X)

            all_indices_Y = torch.cat([max_indices_Y,min_indices_Y])
            all_indices_Y = torch.unique(all_indices_Y)

            all_indices_Z = torch.cat([max_indices_Z,min_indices_Z])
            all_indices_Z = torch.unique(all_indices_Z)

            grad_X = torch.zeros_like(back_grad_X)
            grad_Y = torch.zeros_like(back_grad_Y)
            grad_Z = torch.zeros_like(back_grad_Z)

            grad_X[all_indices_X] = back_grad_X[all_indices_X]
            grad_Y[all_indices_Y] = back_grad_Y[all_indices_Y]
            grad_Z[all_indices_Z] = back_grad_Z[all_indices_Z]

            grad_X = grad_X.sign() * self.attack_alpha
            grad_Y = grad_Y.sign() * self.attack_alpha
            grad_Z = grad_Z.sign() * self.attack_alpha

            adv_perturb_coords = adv_coords.clone().detach()
            adv_perturb_coords[:, 1] = torch.add(adv_coords[:, 1], grad_X)
            adv_perturb_coords[:, 2] = torch.add(adv_coords[:, 2], grad_Y)
            adv_perturb_coords[:, 3] = torch.add(adv_coords[:, 3], grad_Z)
            # 计算扰动，进行剪切
            eta = torch.clamp(adv_perturb_coords[:,1:]-ori_coords[:,1:],min=-self.attack_eps,max=self.attack_eps)

            # 对扰动的点做限制，限制扰动的点位于水平上的l1距离 |100|+|100|开外
            distance_bound = self.attack_bound
            # distances = torch.abs(adv_perturb_coords[:,1])+torch.abs(adv_perturb_coords[:,2])
            distances = (adv_perturb_coords[:,1]**2+adv_perturb_coords[:,2]**2+adv_perturb_coords[:,3]**2).sqrt()
            mask = distances<distance_bound
            eta[mask] = 0
            record_adv_stensor = adv_coords.clone().float().detach()
            adv_coords[:,1:] = adv_coords[:,1:]+eta
            # 后置处理
            flag,adv_filter_coords,record_num = self.post_filter(adv_coords,bound=1)
            adv_coords = adv_filter_coords
            # attack_sample_rate = 0.1 if flag else attack_sample_rate * 2
            # attack_sample_rate = 0.9 if attack_sample_rate > 1.0 else attack_sample_rate
            # 统计扰动点的数量
            num_perturbed_points = torch.sum((ori_coords[:, 1:] != adv_coords[:, 1:]).any(dim=1))
            # 计算扰动点占原始点数的比例
            perturbation_ratio = num_perturbed_points/ori_coords.size()[0]
            perturbation_ratio = perturbation_ratio.item()
            # print(f"[INFO] the process of attack iteration {iteration}: \n"
            #       f"the sample rate is {attack_sample_rate}\n"
            #       f"overall loss is {loss} \n"
            #       f"the post filter flat: {flag} \n"
            #       f"post filter time consume is {time_consume} seconds")
            print(f'[INFO] after attack the loss is {cost}, dict_loss is {dist_loss}')
        print("[INFO] after attack the pred success rate is {}\n"
              "the perturb ratio is {}".format(success_label_rate, perturbation_ratio))
        adv_data = adv_coords
        return adv_data,loss

    def attackPGD(self,data,labels=None):
        """
            attack mode: pgd attack
            projected gradient attack test
            data: ME.SparseTensor(coordinates=coordinates,features=features)
            labels : [num_points,1] Tensor
        """
        adv_data,loss = None,None
        ori_coords = data.coordinates.clone().detach()
        max_ori_x = torch.max(torch.abs(ori_coords[:,1]))
        max_ori_y = torch.max(torch.abs(ori_coords[:,2]))
        max_ori_z = torch.max(torch.abs(ori_coords[:,3]))
        # 初始特征处理
        features_ori_x = ori_coords[:,1]/max_ori_x + 1.0
        features_ori_y = ori_coords[:,2]/max_ori_y + 1.0
        features_ori_z = ori_coords[:,3]/max_ori_z + 1.0
        features_ori_x = features_ori_x.unsqueeze(1)
        features_ori_y = features_ori_y.unsqueeze(1)
        features_ori_z = features_ori_z.unsqueeze(1)
        concact_ori_features = torch.cat((features_ori_x,features_ori_y,features_ori_z),dim=0)
        ori_features = data.features
        adv_coords = data.coordinates.clone().detach()
        # 成功进行语义分割的样本个数
        success_label_rate = [0., 0., 0.]
        # 干扰样本强度
        for _ in range(self.attack_iters):
            # 找到 x, y, z 列的绝对值最大值
            max_abs_x = torch.max(torch.abs(adv_coords[:, 1]))
            max_abs_y = torch.max(torch.abs(adv_coords[:, 2]))
            max_abs_z = torch.max(torch.abs(adv_coords[:, 3]))
            # 归一化 x, y, z 列的值到 0-1 范围内
            features_x_tensor = adv_coords[:, 1] / max_abs_x + 1.0
            features_y_tensor = adv_coords[:, 2] / max_abs_y + 1.0
            features_z_tensor = adv_coords[:, 3] / max_abs_z + 1.0
            features_x_tensor = features_x_tensor.unsqueeze(1)
            features_y_tensor = features_y_tensor.unsqueeze(1)
            features_z_tensor = features_z_tensor.unsqueeze(1)
            criterion = SoftDICELoss(ignore_label=-1)
            # Create a sparse tensor object
            st = ME.SparseTensor(coordinates=ori_coords, features=ori_features)
            st_X = ME.SparseTensor(coordinates=adv_coords, features=features_x_tensor)
            st_Y = ME.SparseTensor(coordinates=adv_coords, features=features_y_tensor)
            st_Z = ME.SparseTensor(coordinates=adv_coords, features=features_z_tensor)
            st_X.features.requires_grad = True
            st_Y.features.requires_grad = True
            st_Z.features.requires_grad = True
            concat_adv_features = torch.cat((st_X.features, st_Y.features, st_Z.features), dim=0)
            # Compute the loss function (sum of all elements in coordinates and features)
            out   = self.model(st)
            out_X = self.model(st_X)
            out_Y = self.model(st_Y)
            out_Z = self.model(st_Z)
            logits = out.F
            logits_X = out_X.F
            logits_Y = out_Y.F
            logits_Z = out_Z.F
            pred = torch.argmax(logits,dim=1)
            pred_X = torch.argmax(logits_X, dim=1)  # [num_points,1]
            pred_Y = torch.argmax(logits_Y, dim=1)
            pred_Z = torch.argmax(logits_Z, dim=1)
            # loss_ori = criterion(logits,labels)
            loss_X = criterion(logits_X, labels)
            loss_Y = criterion(logits_Y, labels)
            loss_Z = criterion(logits_Z, labels)
            prev_loss = loss_X + loss_Y + loss_Z
            prev_loss.requires_grad_(True)
            # 计算L2距离损失
            # dist_loss = F.mse_loss(concat_adv_features,concact_ori_features)
            # 计算L1距离损失
            # dist_loss = F.l1_loss(concat_adv_features,concact_ori_features)
            # 确定距离损失函数
            dist_loss = F.l1_loss(concat_adv_features,concact_ori_features) if self.attack_dist_type=='l1' else F.mse_loss(concat_adv_features,concact_ori_features)
            # 确定光滑度损失函数
            dist_loss.requires_grad_(True)
            # 计算总的损失函数
            cost = prev_loss + self.attack_dist_c*dist_loss
            cost.requires_grad_(True)
            opt = torch.optim.Adam([st_X.features, st_Y.features, st_Z.features], lr=self.attack_lr, weight_decay=self.attack_weight)
            ori_success_label_num = torch.sum(torch.eq(pred, labels)).item()
            success_label_rate[0] = torch.sum(torch.eq(pred_X, labels)).item()/labels.size()[0]
            success_label_rate[1] = torch.sum(torch.eq(pred_Y, labels)).item()/labels.size()[0]
            success_label_rate[2] = torch.sum(torch.eq(pred_Z, labels)).item()/labels.size()[0]
            # 反向传播
            opt.zero_grad()
            cost.backward()
            opt.step()
            # 获取梯度
            grad_X = st_X.features.grad.squeeze()
            grad_Y = st_Y.features.grad.squeeze()
            grad_Z = st_Z.features.grad.squeeze()
            grad_X = grad_X.sign() * self.attack_alpha
            grad_Y = grad_Y.sign() * self.attack_alpha
            grad_Z = grad_Z.sign() * self.attack_alpha
            adv_perturb_coords = adv_coords.clone().detach()
            adv_perturb_coords[:, 1] = torch.add(adv_coords[:, 1], grad_X)
            adv_perturb_coords[:, 2] = torch.add(adv_coords[:, 2], grad_Y)
            adv_perturb_coords[:, 3] = torch.add(adv_coords[:, 3], grad_Z)
            # 计算扰动，进行剪切
            eta = torch.clamp(adv_perturb_coords[:,1:]-ori_coords[:,1:],min=-self.attack_eps,max=self.attack_eps)
            # 对扰动进行处理，选取采样点数范围内的梯度扰动
            # 计算需要扰动的点的数量
            num_points = eta.size(0)
            perturb_num = int(num_points*self.attack_sample_rate)
            # 生成采样的随机扰动
            idx = torch.randperm(num_points)[:perturb_num]
            perturb_eta = torch.zeros_like(eta)
            perturb_eta[idx] = eta[idx]
            eta = perturb_eta
            # 对扰动的点做限制，限制扰动的点位于水平上的l1距离 |100|+|100|开外
            distance_bound = self.attack_bound
            # distances = torch.abs(adv_perturb_coords[:,1])+torch.abs(adv_perturb_coords[:,2])
            distances = (adv_perturb_coords[:,1]**2+adv_perturb_coords[:,2]**2).sqrt()
            mask = distances<distance_bound
            eta[mask] = 0
            record_adv_stensor = adv_coords.clone().float().detach()
            adv_coords[:,1:] = adv_coords[:,1:]+eta
            # 后置处理
            flag,adv_filter_coords,record_num = self.post_filter(adv_coords,bound=1)
            adv_coords = adv_filter_coords
            # attack_sample_rate = 0.1 if flag else attack_sample_rate * 2
            # attack_sample_rate = 0.9 if attack_sample_rate > 1.0 else attack_sample_rate
            # 统计扰动点的数量
            num_perturbed_points = torch.sum((ori_coords[:, 1:] != adv_coords[:, 1:]).any(dim=1))
            # 计算扰动点占原始点数的比例
            perturbation_ratio = num_perturbed_points/ori_coords.size()[0]
            perturbation_ratio = perturbation_ratio.item()
            # print(f"[INFO] the process of attack iteration {iteration}: \n"
            #       f"the sample rate is {attack_sample_rate}\n"
            #       f"overall loss is {loss} \n"
            #       f"the post filter flat: {flag} \n"
            #       f"post filter time consume is {time_consume} seconds")
            print(f'[INFO] after attack the loss is {cost}, dict_loss is {dist_loss}')
        print("[INFO] after attack the pred success rate is {}\n"
              "the perturb ratio is {}".format(success_label_rate, perturbation_ratio))
        adv_data = adv_coords
        return adv_data,loss

class AdvPLTTrainer(pl.core.LightningModule):
    r"""
    Segmentation Module for MinkowskiEngine for training on one domain.
    """
    def __init__(self,
                 model,
                 training_dataset,
                 validation_dataset,
                 optimizer_name='SGD',
                 criterion='CELoss',
                 lr=1e-3,
                 batch_size=12,
                 weight_decay=1e-4,
                 momentum=0.98,
                 val_batch_size=6,
                 train_num_workers=10,
                 val_num_workers=10,
                 num_classes=19,
                 clear_cache_int=2,
                 scheduler_name=None,
                 attacker=None,
                 count_train_val=None
                 ):

        super().__init__()
        self.attacker = attacker
        for name, value in vars().copy().items():
            if name != "self":
                setattr(self, name, value)

        self.attacker.init_model(self.model)
        if criterion == 'CELoss':
            self.criterion = CELoss(ignore_label=self.training_dataset.ignore_label,
                                    weight=None)
        elif criterion == 'DICELoss':
            self.criterion = DICELoss(ignore_label=self.training_dataset.ignore_label)
        elif criterion == 'SoftDICELoss':
            if self.num_classes == 19:
                self.criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label, is_kitti=True)
            else:
                self.criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label)
        else:
            raise NotImplementedError

        self.ignore_label = self.training_dataset.ignore_label
        self.validation_phases = ['source_validation', 'target_validation']

        self.save_hyperparameters(ignore='model')
        self.count_train_val = count_train_val

        self.sequences = ["00","01","02","03","04","05","06","07","08","09","10","11","12"]
        # 对抗样本保存地址
        split_path = os.path.join(ABSOLUTE_PATH, 'utils/datasets/_splits/synlidar.pkl')
        self.get_indices(split_path=split_path)
        # 得到场景中的对应索引
        self.split = self.get_indices(split_path=split_path)
     
    def get_indices(self,split_path=None):
        split = torch.load(split_path)
        # 得到每个场景中对应的索引
        return split

    def training_step(self, batch, batch_idx):
        # Must clear cache at regular interval 
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()
        phase = 'source_training'
        labels = batch['labels'].long()
        # <================grouped stensor attack start=================>
        coordinates = batch["coordinates"].detach().contiguous()
        features = batch["features"].detach().contiguous()
        stensor = ME.SparseTensor(coordinates=coordinates,features=features)
        if isinstance(self.attacker,AdvAttacker):
            adv_coords,adv_loss = self.attacker.attackPGD(data=stensor,labels=labels)
        adv_stensor = ME.SparseTensor(coordinates=adv_coords,features=features)
        
        # must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()
        out = self.model(stensor).F
        adv_out = self.model(adv_stensor).F
        loss = self.criterion(out, labels)
        adv_loss = self.criterion(adv_out, labels)

        soft_pseudo = F.softmax(out, dim=-1)
        conf, preds = soft_pseudo.max(1)

        soft_adv_pseudo = F.softmax(adv_out, dim=-1)
        adv_conf, adv_preds = soft_adv_pseudo.max(1)
        iou_tmp = jaccard_score(labels.cpu().numpy(),preds.detach().cpu().numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=0)
        adv_iou_tmp = jaccard_score(labels.cpu().numpy(),adv_preds.detach().cpu().numpy(), average=None,
                                    labels=np.arange(0, self.num_classes),
                                    zero_division=0)

        present_labels, class_occurs = np.unique(labels.detach().cpu().numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        present_names = self.training_dataset.class2names[present_labels].tolist()
        present_names = [os.path.join(phase, p + '_iou') for p in present_names]
        results_dict = dict(zip(present_names, iou_tmp.tolist()))
        adv_results_dict = dict(zip(present_names, adv_iou_tmp.tolist()))

        results_dict[f'{phase}/loss'] = loss
        results_dict[f'{phase}/iou'] = np.mean(iou_tmp[present_labels])

        adv_results_dict[f'{phase}/adv_loss'] = adv_loss
        adv_results_dict[f'{phase}/adv_iou'] = np.mean(adv_iou_tmp[present_labels])
        for k, v in results_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v).float()
            else:
                v = v.float()
            self.log(
                name=k,
                value=v,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=True,
                batch_size=self.batch_size
            )
        for k, v in adv_results_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v).float()
            else:
                v = v.float()
            self.log(
                name=k,
                value=v,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=True,
                batch_size=self.batch_size
            )
        if self.attacker.attack_save_prediction and self.attacker.attack_save_dir is not None:
            coordinates = batch["coordinates"]
            idx = batch["idx"]
            prediction_dict = {
                "save_folder": self.attacker.attack_save_dir,
                "idx": idx,
                "coords": coordinates.cpu(),
                "adv_coords": adv_stensor.coordinates.cpu(),
                "labels": labels.cpu(),
                "preds": preds.cpu(),
                "adv_preds": adv_preds.cpu(),
                "conf": conf.cpu(),
                "adv_conf": adv_conf.cpu()
            }
            # pdb.set_trace()
            self.save_predictions(prediction_dict)
            # pdb.set_trace()
            torch.cuda.empty_cache()
      
        return loss
    
    def calculate_sequences(self,idx):
        i=0
        while i < len(self.count_train_val):
            if idx.item()<self.count_train_val[0]:
                return 0,"00"
            elif idx.item()<self.count_train_val[i+1]:
                return i,self.sequences[i+1]
            else:
                i+=1
            
    def save_predictions(self,prediction_dict):
        batch_idx = prediction_dict['idx']
        save_folder = prediction_dict["save_folder"]

        for b,idx in enumerate(batch_idx): 
            sequence_num,sequence_save = self.calculate_sequences(idx)
            # 计算在sequences中的具体frame num
            frame_num = self.split['train'][sequence_save][idx-self.count_train_val[sequence_num]]
            labels = prediction_dict["labels"]
            # original information about point cloud
            coords = prediction_dict["coords"]
            preds = prediction_dict["preds"]
            conf = prediction_dict["conf"]
            # adversarial information about point cloud
            adv_coords = prediction_dict["adv_coords"]
            adv_preds = prediction_dict["adv_preds"]
            adv_conf = prediction_dict["adv_conf"]

            s_idx = idx.item()
            b_idx = coords[:,0]==b
            points = coords[b_idx, 1:]
            adv_points = adv_coords[b_idx,1:]
            p = preds[b_idx]
            c = conf[b_idx]
            l = labels[b_idx]
            adv_p = adv_preds[b_idx]
            adv_c = adv_conf[b_idx]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(self.training_dataset.color_map[p + 1])

            adv_pcd = o3d.geometry.PointCloud()
            adv_pcd.points = o3d.utility.Vector3dVector(adv_points)
            adv_pcd.colors = o3d.utility.Vector3dVector(self.training_dataset.color_map[adv_p+1])

            iou_tmp = jaccard_score(p.cpu().numpy(), l.cpu().numpy(), average=None,
                                    labels=np.arange(0, self.num_classes),
                                    zero_division=0)
            adv_iou_tmp = jaccard_score(adv_p.cpu().numpy(), l.cpu().numpy(), average=None,
                                    labels=np.arange(0, self.num_classes),
                                    zero_division=0)

            present_labels, _ = np.unique(l.cpu().numpy(), return_counts=True)
            present_labels = present_labels[present_labels != self.ignore_label]
            iou_tmp = np.nanmean(iou_tmp[present_labels]) * 100
            adv_iou_tmp = np.nanmean(adv_iou_tmp[present_labels])*100

            # pdb.set_trace()

            # 保存原始点云用于可视化
            os.makedirs(os.path.join(save_folder, 'ori_preds',sequence_save), exist_ok=True)
            os.makedirs(os.path.join(save_folder, 'ori_labels',sequence_save), exist_ok=True)
            os.makedirs(os.path.join(save_folder, 'ori_pseudo',sequence_save), exist_ok=True)
            os.makedirs(os.path.join(save_folder, 'ori_labels_pt',sequence_save), exist_ok=True)

            # 保存对抗点云用于可视化
            os.makedirs(os.path.join(save_folder, 'adv_preds',sequence_save), exist_ok=True)
            os.makedirs(os.path.join(save_folder, 'adv_labels',sequence_save), exist_ok=True)
            os.makedirs(os.path.join(save_folder, 'adv_pseudo',sequence_save), exist_ok=True)
            os.makedirs(os.path.join(save_folder, 'adv_labels_pt',sequence_save), exist_ok=True)
            
            # 保存真实标签和预测结果用于后续处理
            torch.save(l,os.path.join(save_folder,'ori_labels_pt',sequence_save,f"{frame_num}_{int(iou_tmp)}.pt"))
            torch.save(adv_p,os.path.join(save_folder,'adv_labels_pt',sequence_save,f"{frame_num}_{int(adv_iou_tmp)}.pt"))

            o3d.io.write_point_cloud(os.path.join(save_folder, 'ori_preds',sequence_save, f"{frame_num}_{int(iou_tmp)}.ply"), pcd)
            pcd.colors = o3d.utility.Vector3dVector(self.training_dataset.color_map[l + 1])
            o3d.io.write_point_cloud(os.path.join(save_folder, 'ori_labels', f'{frame_num}.ply'), pcd)
            valid_pseudo = c > 0.85
            p[torch.logical_not(valid_pseudo)] = -1
            pcd.colors = o3d.utility.Vector3dVector(self.training_dataset.color_map[p + 1])
            o3d.io.write_point_cloud(os.path.join(save_folder, 'ori_pseudo',sequence_save, f'{frame_num}.ply'), pcd)

            o3d.io.write_point_cloud(os.path.join(save_folder, 'adv_preds',sequence_save, f'{frame_num}_{int(adv_iou_tmp)}.ply'), adv_pcd)
            adv_pcd.colors = o3d.utility.Vector3dVector(self.training_dataset.color_map[l + 1])
            o3d.io.write_point_cloud(os.path.join(save_folder, 'adv_labels',sequence_save, f'{frame_num}.ply'), adv_pcd)
            valid_pseudo = adv_c > 0.85
            p[torch.logical_not(valid_pseudo)] = -1
            adv_pcd.colors = o3d.utility.Vector3dVector(self.training_dataset.color_map[adv_p + 1])
            o3d.io.write_point_cloud(os.path.join(save_folder, 'adv_pseudo',sequence_save, f'{frame_num}.ply'), adv_pcd)
    
    def groups_stensor(self,coordinates,features,labels):
        coordinates = coordinates
        features = features
        labels = labels
        # 获取第一维中不同值的列表
        # unique_values = torch.unique(coordinates[:, 0]).nonzero().squeeze()
        unique_values = torch.unique(coordinates[:, 0])

        # 存储分组后的子张量和对应标签的列表
        grouped_stensors = []
        # 遍历不同的值，按照每个值进行分组
        for value in unique_values:
            # 构造布尔索引
            bool_index = (coordinates[:, 0] == value)
            # 使用布尔索引提取子张量
            grouped_coordinates = coordinates[bool_index]
            grouped_features = features[bool_index]
            grouped_labels = labels[bool_index]
            # 存储到列表中
            grouped_stensor = ME.SparseTensor(coordinates=grouped_coordinates,features=grouped_features)
            grouped_stensors.append([grouped_stensor,grouped_labels])

        return grouped_stensors

    def combine_stensor(self,adv_stensors):
        combined_coordinates = []
        combined_features = []
        combined_labels = []
        for groups_idx in range(len(adv_stensors)):
            grouped_stensor = adv_stensors[groups_idx][0]
            grouped_coordinates = grouped_stensor.coordinates
            grouped_features = grouped_stensor.features
            grouped_labels = adv_stensors[groups_idx][1]

            # 按0维拼接坐标、特征和标签
            combined_coordinates.append(grouped_coordinates)
            combined_features.append(grouped_features)
            combined_labels.append(grouped_labels)

            # 将列表转换为张量
        combined_coordinates = torch.cat(combined_coordinates, dim=0)
        combined_features = torch.cat(combined_features, dim=0)
        combined_labels = torch.cat(combined_labels, dim=0)

        # 创建组合后的SparseTensor对象
        combined_stensor = ME.SparseTensor(coordinates=combined_coordinates, features=combined_features)
        return combined_stensor,combined_labels
    
    def validation_step(self, batch, batch_idx):
        phase = 'source_validation'

        # input batch
        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])

        # must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(stensor).F.cpu()

        labels = batch['labels'].long().cpu()

        loss = self.criterion(out, labels)

        soft_pseudo = F.softmax(out, dim=-1)

        conf, preds = soft_pseudo.max(1)

        iou_tmp = jaccard_score(preds.detach().numpy(), labels.numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=0)

        present_labels, class_occurs = np.unique(labels.numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        present_names = self.training_dataset.class2names[present_labels].tolist()
        present_names = [os.path.join(phase, f'{p}_iou') for p in present_names]
        results_dict = dict(zip(present_names, iou_tmp.tolist()))

        results_dict[f'{phase}/loss'] = loss
        results_dict[f'{phase}/iou'] = np.mean(iou_tmp[present_labels])

        for k, v in results_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v).float()
            else:
                v = v.float()
            self.log(
                name=k,
                value=v,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=True,
                batch_size=self.val_batch_size,
                add_dataloader_idx=False)
        return results_dict

    def configure_optimizers(self):
        if self.scheduler_name is None:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            return optimizer
        else:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            if self.scheduler_name == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            elif self.scheduler_name == 'ExponentialLR':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            elif self.scheduler_name == 'CyclicLR':
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr/10000, max_lr=self.lr,
                                                              step_size_up=5, mode="triangular2")

            else:
                raise NotImplementedError

            return [optimizer], [scheduler]

def get_dataloader(dataset, batch_size, collate_fn=CollateFN(), shuffle=False, pin_memory=True):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      collate_fn=collate_fn,
                      shuffle=shuffle,
                      num_workers=3,
                      pin_memory=pin_memory)

def get_parser():
    parser = argparse.ArgumentParser()
    #读取config
    parser.add_argument(
        "--config_file",
        default="configs/attack/lidar2kitti_VP.yaml",
        type=str,
        help="Path to config file"
    )
    return parser.parse_args()

def attack_loading(config=None):
    """
    :func attack_loading: loading attacker, training dataset, validation dataset, test dataset
    return: pl_module,training_dataloader,validation_dataloader
    """
    synlidar2poss = np.array(['person', 'rider', 'car', 'trunk',
                          'plants', 'traffic-sign', 'pole', 'garbage-can',
                          'building', 'cone', 'fence', 'bike', 'ground'])
    synlidar2poss_color = np.array([(255, 255, 255),  # unlabelled
                                (250, 178, 50),  # person
                                (255, 196, 0),   # rider
                                (25, 25, 255),   # car
                                (107, 98, 56),   # trunk
                                (157, 234, 50),  # plants
                                (173,23,121),    # traffic-sign
                                (83,93,130),     # pole
                                (255,255,255),    # garbage-can
                                (233, 166, 250), # building
                                (255,255,255),    # cone
                                (255, 214, 251), # fense
                                (187,0,255),     # bike
                                (0, 0, 0),       # ground
                                ])/255.  # ground
    synlidar2kitti = np.array(['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
                               'bicyclist', 'motorcyclist',
                               'road', 'parking', 'sidewalk', 'other-ground',
                               'building', 'fence', 'vegetation', 'trunk',
                               'terrain', 'pole', 'traffic-sign'])
    synlidar2kitti_color = np.array([(255, 255, 255),  # unlabelled
                                     (25, 25, 255),  # car
                                     (187, 0, 255),  # bicycle
                                     (187, 50, 255),  # motorcycle
                                     (0, 247, 255),  # truck
                                     (50, 162, 168),  # other-vehicle
                                     (250, 178, 50),  # person
                                     (255, 196, 0),  # bicyclist
                                     (255, 196, 0),  # motorcyclist
                                     (0, 0, 0),  # road
                                     (148, 148, 148),  # parking
                                     (255, 20, 60),  # sidewalk
                                     (164, 173, 104),  # other-ground
                                     (233, 166, 250),  # building
                                     (255, 214, 251),  # fence
                                     (157, 234, 50),  # vegetation
                                     (107, 98, 56),  # trunk
                                     (78, 72, 44),  # terrain
                                     (83, 93, 130),  # pole
                                     (173, 23, 121)]) / 255.  # traffic-sign
    
    # training dataset for synlidar
    # config.attack.phase = train, 生成训练数据对应的对抗样本
    # config.attack.phase = validation, 生成评估数据对应的对抗样本
    training_dataset =  AdvSynLiDARDataset(
        dataset_path=config.dataset.dataset_path,
        version=config.dataset.version,
        phase='train',
        voxel_size=config.dataset.voxel_size,
        augment_data=config.dataset.augment_data,
        num_classes=config.model.out_classes,
        ignore_label=config.dataset.ignore_label,
        mapping_path=config.dataset.mapping_path,
        weights_path=None,
        attacked_indices=None,
        adv_coords_dict=None,
        adv_labels_dict=None,
        attacked_phase=config.attack.phase
    )

    print(training_dataset.count_train_val)
    # validation dataset for synlidar
    validation_dataset = SynLiDARDataset(
        dataset_path=config.dataset.dataset_path,
        version=config.dataset.version,
        phase=config.attack.phase,
        voxel_size=config.dataset.voxel_size,
        augment_data=False,
        num_classes=config.model.out_classes,
        ignore_label=config.dataset.ignore_label,
        mapping_path=config.dataset.mapping_path,
        weights_path=None
    )
    
    if config.dataset.target == 'SemanticKITTI':
        training_dataset.class2names = synlidar2kitti
        validation_dataset.class2names = synlidar2kitti
        training_dataset.color_map = synlidar2kitti_color
        validation_dataset.color_map = synlidar2kitti_color
        training_dataset.ignore_label = -1
        validation_dataset.ignore_label = -1
    else:
        training_dataset.class2names = synlidar2poss
        validation_dataset.class2names = synlidar2poss
        training_dataset.color_map = synlidar2poss_color
        validation_dataset.color_map = synlidar2poss_color
        training_dataset.ignore_label = -1
        validation_dataset.ignore_label = -1

    collation = CollateFN()

    training_dataloader = get_dataloader(training_dataset,
                                         collate_fn=collation,
                                         batch_size=config.pipeline.dataloader.batch_size,
                                         shuffle=True
                                         )
    # batch_size=1 对应的是一个场景
    validation_dataloader = get_dataloader(validation_dataset,
                                           collate_fn=collation,
                                           batch_size=config.pipeline.dataloader.batch_size,
                                           shuffle=False)
    
    Model = getattr(models, config.model.name)
    model = Model(config.model.in_feat_size, config.model.out_classes)
    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    # # <============模型加载预训练参数==================>
    model = load_model(checkpoint_path=config.attack.checkpoint_path, model=model)

    # <=================加载adv attacker================>
    attacker = AdvAttacker(
        attack_lr=config.attack.lr,
        attack_weight=config.attack.weight,
        attack_alpha=config.attack.alpha,
        attack_iters=config.attack.iters,
        attack_eps=config.attack.eps,
        attack_sample_rate=config.attack.sample_rate,
        attack_dist_c=config.attack.dist_c,
        attack_dist_type=config.attack.dist_type,
        attack_bound=config.attack.bound,
        attack_save_dir=config.attack.save_dir,
        attack_save_prediction=config.attack.save_prediction,
    )
    # attacker.logattacker()
    # <=========================初始化PLTTrainer========================>
    pl_module = AdvPLTTrainer(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model=model,
        criterion=config.pipeline.loss,
        optimizer_name=config.pipeline.optimizer.name,
        batch_size=config.pipeline.dataloader.batch_size,
        val_batch_size=config.pipeline.dataloader.batch_size,
        lr=config.pipeline.optimizer.lr,                           # 1e-3
        num_classes=config.model.out_classes,                      # kitti 19;poss 19
        train_num_workers=config.pipeline.dataloader.num_workers,  # 16
        val_num_workers=config.pipeline.dataloader.num_workers,    # 16
        clear_cache_int=config.pipeline.lightning.clear_cache_int,
        scheduler_name=None,
        attacker=attacker,
        count_train_val=training_dataset.count_train_val
    )

    print(f"[INFO] The adversarial attacker {attacker} initialized by user.")
    return pl_module,training_dataloader,validation_dataloader

def init_adv_training(config=None,pl_module=None,training_dataloader=None,validation_dataloader=None):
    """
    :func init_adv_training: training and save adversarial samples
    """
    # <========================初始化logger=============================>
    wandb_logger = WandbLogger(project=config.pipeline.wandb.project_name,
                               entity=config.pipeline.wandb.entity_name,
                               name=config.pipeline.wandb.run_name,
                               offline=config.pipeline.wandb.offline)

    loggers = [wandb_logger]

    run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())
    run_name = config.pipeline.wandb.run_name + '_' + run_time
    save_dir = os.path.join(config.pipeline.save_dir, run_name)

    checkpoint_callback = [ModelCheckpoint(dirpath=os.path.join(save_dir, 'checkpoints'), save_top_k=-1)]

    trainer = Trainer(max_epochs=config.pipeline.epochs,
                      gpus=config.pipeline.gpus,
                      accelerator="ddp",
                      default_root_dir=config.pipeline.save_dir,
                      weights_save_path=save_dir,
                      precision=config.pipeline.precision,
                      logger=loggers,
                      check_val_every_n_epoch=config.pipeline.lightning.check_val_every_n_epoch,
                      val_check_interval=config.pipeline.lightning.val_check_interval,
                      num_sanity_val_steps=1,
                      resume_from_checkpoint=None,
                      callbacks=checkpoint_callback,
                      )
    trainer.fit(pl_module,
                train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloader)

def test_cw_attack_all(config=None):
    # fix_random_seed
    os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    np.random.seed(config.pipeline.seed)
    torch.manual_seed(config.pipeline.seed)
    torch.cuda.manual_seed(config.pipeline.seed)
    torch.backends.cudnn.benchmark = True
    print('[INFO] traning settings preparation!\n')

    pl_module,training_dataloader,validation_dataloader = attack_loading(
        config=config
    )

    print('[INFO] start adversarial samples generation!\n')

    init_adv_training(
        config=config,
        pl_module=pl_module,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader
    )

    print('[INFO] end adversarial samples generation!\n')

    print('[INFO] traning settings preparation!\n')

    pl_module,training_dataloader,validation_dataloader = attack_loading(
        config=config
    )

    print('[INFO] start adversarial samples generation!\n')

    init_adv_training(
        config=config,
        pl_module=pl_module,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader
    )

    print('[INFO] end adversarial samples generation!\n')

if __name__ == "__main__":
    args = get_parser()
    config = get_config(args.config_file)
    # fix_random_seed
    os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    np.random.seed(config.pipeline.seed)
    torch.manual_seed(config.pipeline.seed)
    torch.cuda.manual_seed(config.pipeline.seed)
    torch.backends.cudnn.benchmark = True
    print('[INFO] traning settings preparation!\n')

    synlidar2poss = np.array(['person', 'rider', 'car', 'trunk',
                          'plants', 'traffic-sign', 'pole', 'garbage-can',
                          'building', 'cone', 'fence', 'bike', 'ground'])
    synlidar2poss_color = np.array([(255, 255, 255),  # unlabelled
                                (250, 178, 50),  # person
                                (255, 196, 0),   # rider
                                (25, 25, 255),   # car
                                (107, 98, 56),   # trunk
                                (157, 234, 50),  # plants
                                (173,23,121),    # traffic-sign
                                (83,93,130),     # pole
                                (255,255,255),    # garbage-can
                                (233, 166, 250), # building
                                (255,255,255),    # cone
                                (255, 214, 251), # fense
                                (187,0,255),     # bike
                                (0, 0, 0),       # ground
                                ])/255.  # ground
    synlidar2kitti = np.array(['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
                               'bicyclist', 'motorcyclist',
                               'road', 'parking', 'sidewalk', 'other-ground',
                               'building', 'fence', 'vegetation', 'trunk',
                               'terrain', 'pole', 'traffic-sign'])
    synlidar2kitti_color = np.array([(255, 255, 255),  # unlabelled
                                     (25, 25, 255),  # car
                                     (187, 0, 255),  # bicycle
                                     (187, 50, 255),  # motorcycle
                                     (0, 247, 255),  # truck
                                     (50, 162, 168),  # other-vehicle
                                     (250, 178, 50),  # person
                                     (255, 196, 0),  # bicyclist
                                     (255, 196, 0),  # motorcyclist
                                     (0, 0, 0),  # road
                                     (148, 148, 148),  # parking
                                     (255, 20, 60),  # sidewalk
                                     (164, 173, 104),  # other-ground
                                     (233, 166, 250),  # building
                                     (255, 214, 251),  # fence
                                     (157, 234, 50),  # vegetation
                                     (107, 98, 56),  # trunk
                                     (78, 72, 44),  # terrain
                                     (83, 93, 130),  # pole
                                     (173, 23, 121)]) / 255.  # traffic-sign

    # training dataset for synlidar
    # config.attack.phase = train, 生成训练数据对应的对抗样本
    # config.attack.phase = validation, 生成评估数据对应的对抗样本
    training_dataset =  AdvSynLiDARDataset(
        dataset_path=config.dataset.dataset_path,
        version=config.dataset.version,
        phase=config.attack.phase,
        voxel_size=config.dataset.voxel_size,
        augment_data=config.dataset.augment_data,
        num_classes=config.model.out_classes,
        ignore_label=config.dataset.ignore_label,
        mapping_path=config.dataset.mapping_path,
        weights_path=None,
        attacked_indices=None,
        adv_coords_dict=None,
        adv_labels_dict=None,
        attacked_phase=config.attack.phase,
    )

    print(f'[INFO] adversarial sample generation, selected {config.attack.phase} {training_dataset.count_train_val}\n')
    # validation dataset for synlidar
    validation_dataset = SynLiDARDataset(
        dataset_path=config.dataset.dataset_path,
        version=config.dataset.version,
        phase='validation',
        voxel_size=config.dataset.voxel_size,
        augment_data=False,
        num_classes=config.model.out_classes,
        ignore_label=config.dataset.ignore_label,
        mapping_path=config.dataset.mapping_path,
        weights_path=None
    )
    
    if config.dataset.target == 'SemanticKITTI':
        training_dataset.class2names = synlidar2kitti
        validation_dataset.class2names = synlidar2kitti
        training_dataset.color_map = synlidar2kitti_color
        validation_dataset.color_map = synlidar2kitti_color
        training_dataset.ignore_label = -1
        validation_dataset.ignore_label = -1
    else:
        training_dataset.class2names = synlidar2poss
        validation_dataset.class2names = synlidar2poss
        training_dataset.color_map = synlidar2poss_color
        validation_dataset.color_map = synlidar2poss_color
        training_dataset.ignore_label = -1
        validation_dataset.ignore_label = -1

    collation = CollateFN()

    training_dataloader = get_dataloader(training_dataset,
                                         collate_fn=collation,
                                         batch_size=config.pipeline.dataloader.batch_size,
                                         shuffle=True
                                         )
    # batch_size=1 对应的是一个场景
    validation_dataloader = get_dataloader(validation_dataset,
                                           collate_fn=collation,
                                           batch_size=config.pipeline.dataloader.batch_size,
                                           shuffle=False)
    
    Model = getattr(models, config.model.name)
    model = Model(config.model.in_feat_size, config.model.out_classes)
    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    # # <============模型加载预训练参数==================>
    model = load_model(checkpoint_path=config.attack.checkpoint_path, model=model)

    # <=================加载adv attacker================>
    attacker = AdvAttacker(
        attack_lr=config.attack.lr,
        attack_weight=config.attack.weight,
        attack_alpha=config.attack.alpha,
        attack_iters=config.attack.iters,
        attack_eps=config.attack.eps,
        attack_sample_rate=config.attack.sample_rate,
        attack_dist_c=config.attack.dist_c,
        attack_dist_type=config.attack.dist_type,
        attack_bound=config.attack.bound,
        attack_save_dir=config.attack.save_dir,
        attack_save_prediction=config.attack.save_prediction,
    )

    # <=========================初始化PLTTrainer========================>
    pl_module = AdvPLTTrainer(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model=model,
        criterion=config.pipeline.loss,
        optimizer_name=config.pipeline.optimizer.name,
        batch_size=config.pipeline.dataloader.batch_size,
        val_batch_size=config.pipeline.dataloader.batch_size,
        lr=config.pipeline.optimizer.lr,                           # 1e-3
        num_classes=config.model.out_classes,                      # kitti 19;poss 19
        train_num_workers=config.pipeline.dataloader.num_workers,  # 16
        val_num_workers=config.pipeline.dataloader.num_workers,    # 16
        clear_cache_int=config.pipeline.lightning.clear_cache_int,
        scheduler_name=None,
        attacker=attacker,
        count_train_val=training_dataset.count_train_val
    )

    print('[INFO] start adversarial samples generation!\n')

    # <========================初始化logger=============================>
    wandb_logger = WandbLogger(project=config.pipeline.wandb.project_name,
                               entity=config.pipeline.wandb.entity_name,
                               name=config.pipeline.wandb.run_name,
                               offline=config.pipeline.wandb.offline)

    loggers = [wandb_logger]

    run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())
    run_name = config.pipeline.wandb.run_name + '_' + run_time
    save_dir = os.path.join(config.attack.weights_dir, run_name)

    checkpoint_callback = [ModelCheckpoint(dirpath=os.path.join(save_dir, 'checkpoints'), save_top_k=-1)]

    trainer = Trainer(max_epochs=config.pipeline.epochs,
                      gpus=config.pipeline.gpus,
                      accelerator="ddp",
                      default_root_dir=config.pipeline.save_dir,
                      weights_save_path=save_dir,
                      precision=config.pipeline.precision,
                      logger=loggers,
                      check_val_every_n_epoch=config.pipeline.lightning.check_val_every_n_epoch,
                      val_check_interval=config.pipeline.lightning.val_check_interval,
                      num_sanity_val_steps=1,
                      resume_from_checkpoint=None,
                    #   callbacks=checkpoint_callback,
                      )
    trainer.fit(pl_module,
                train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloader)
   
    print('[INFO] end adversarial samples generation!\n')
    
    
