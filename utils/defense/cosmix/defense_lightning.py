"""
    defense lightning
    Author: Sdim-lemons
    Adding: Decoder used for defense
"""
import torch.nn as nn
import os
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import MinkowskiEngine as ME
from utils.losses import CELoss, SoftDICELoss, KPSLoss, CombinedLoss
import open3d as o3d
import pdb
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import cKDTree
import torch.cuda.amp as amp
from collections import namedtuple


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 定义命名元组
SPCDParams = namedtuple('SPCDParams', ['fine_points', 'coords_idx', 'labels_idx', 'conf_idx', 'feats_idx'])
TPCDParams = namedtuple('TPCDParams',['fine_points','coords_idx','labels_idx','pseudo_idx','conf_idx','feats_idx'])
BatchData = namedtuple('BatchData', ['coords', 'labels', 'features', 'pseudo_labels'])

# 采样 ['FPS','SRS']
def farthest_point_sampling(points, num_samples):
    """
    最远点采样方法
    
    参数:
    - points: np.ndarray, 形状为 (N, 3) 的点云数据
    - num_samples: int, 采样点数
    
    返回:
    - sampled_points: np.ndarray, 形状为 (num_samples, 3) 的采样点
    - sampled_indices: np.ndarray, 采样点的索引
    """
    device = points.device
    points = points.detach().cpu().numpy()
    N, _ = points.shape
    sampled_indices = np.zeros(num_samples, dtype=int)
    sampled_points = np.zeros((num_samples, 3))
    
    # 随机选择第一个点
    farthest_idx = np.random.randint(0, N)
    sampled_indices[0] = farthest_idx
    sampled_points[0] = points[farthest_idx]
    
    # 记录每个点到已选择的最近点的距离
    distances = np.linalg.norm(points - points[farthest_idx], axis=1)
    
    for i in range(1, num_samples):
        farthest_idx = np.argmax(distances)
        sampled_indices[i] = farthest_idx
        sampled_points[i] = points[farthest_idx]
        
        # 更新每个点到已选择的最近点的距离
        new_distances = np.linalg.norm(points - points[farthest_idx], axis=1)
        distances = np.minimum(distances, new_distances)
    
    return torch.from_numpy(sampled_points).to(device), torch.from_numpy(sampled_indices).to(device)

def simple_random_sampling(points, num_samples):
    """
    简单随机采样方法
    
    参数:
    - points: torch.Tensor, 形状为 (N, 3) 的点云数据
    - num_samples: int, 采样点数
    
    返回:
    - sampled_points: torch.Tensor, 形状为 (num_samples, 3) 的采样点
    - sampled_indices: torch.Tensor, 采样点的索引
    """
    device = points.device
    N, _ = points.shape
    sampled_indices = np.random.choice(N, num_samples, replace=False)
    sampled_points = points[sampled_indices]
    
    return sampled_points, torch.from_numpy(sampled_indices).to(device)

class SimMaskedAdaptationDefense(pl.core.LightningModule):
    
    def __init__(self,
                 student_model,
                 teacher_model,
                 decoder,
                 momentum_updater,
                 training_dataset,
                 source_validation_dataset,
                 target_validation_dataset,
                 optimizer_name="SGD",
                 source_criterion='SoftDICELoss',
                 target_criterion='SoftDiceLoss',
                 other_criterion=None,
                 source_weight=0.5,
                 target_weight=0.5,
                 rec_weight=1.,
                 filtering=None,
                 lr=1e-3,
                 train_batch_size=12,
                 val_batch_size=12,
                 weight_decay=1e-4,
                 momentum=0.98,
                 num_classes=19,
                 clear_cache_int=2,
                 scheduler_name=None,
                 update_every=1,
                 weighted_sampling=False,
                 target_confidence_th=0.95,
                 selection_perc=0.5,
                 save_mix=False,
                 decode_coords=False,
                 conf_update=False,
                 lambda_weight=0.1,
                 decoder_weight=0.1,
                 train_decoder=True
                 ):
        
        self.batch_counter = 0
        # for param in decoder.parameters():
        #     param.requires_grad = False
        self.decoder_weight = decoder_weight
        self.train_decoder = train_decoder

        super().__init__()
        for name, value in list(vars().items()):
            if name != "self":
                setattr(self, name, value)
                
        self.ignore_label = self.training_dataset.ignore_label
        self.source_cls_num_list = [1]*num_classes
        self.target_cls_num_list = [1]*num_classes
        
        # ########### source_LOSSES ##############
        if source_criterion == 'CELoss':
            self.source_criterion = CELoss(ignore_label=self.training_dataset.ignore_label, weight=None)
        elif source_criterion == 'SoftDICELoss':
            self.source_criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label)
        elif source_criterion == 'KPSLoss':
            self.source_criterion = KPSLoss(cls_num_list=self.source_cls_num_list)
        elif source_criterion == 'CombinedLoss':
            self.source_cls_num_list = [1]*num_classes
            self.target_cls_num_list = [1]*num_classes
            self.soft_dice_params = {
                "ignore_label": None,
                "powerize": True,
                "use_tmask": True,
                "neg_range": False,
                "eps": 0.05,
                "is_kitti": False
            }

            self.source_kps_params = {
                "cls_num_list": self.source_cls_num_list,
                "max_m": 0.5,
                "weighted": True,
                "weight": None,
                "s": 30,
                "is_kitti": False
            }

            self.target_kps_params = {
                "cls_num_list": self.target_cls_num_list,
                "max_m": 0.5,
                "weighted": True,
                "weight": None,
                "s": 30,
                "is_kitti": False
            }
            
            self.source_criterion = CombinedLoss(self.soft_dice_params, self.source_kps_params, self.lambda_weight) 
        else:
            raise NotImplementedError

        # ########### target_LOSSES ##############
        if target_criterion == 'CELoss':
            self.target_criterion = CELoss(ignore_label=self.training_dataset.ignore_label, weight=None)
        elif target_criterion == 'SoftDICELoss':
            self.target_criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label)
        elif target_criterion == 'KPSLoss':
            self.target_criterion = KPSLoss(cls_num_list=self.target_cls_num_list)
        elif target_criterion == 'CombinedLoss':
            self.target_criterion = CombinedLoss(self.soft_dice_params, self.target_kps_params, self.lambda_weight)
        else:
            raise NotImplementedError


        # in case, we will use an extra criterion
        self.other_criterion = other_criterion

        # ############ WEIGHTS ###############
        self.source_weight = source_weight
        self.target_weight = target_weight

        # ############ LABELS ###############
        self.ignore_label = self.training_dataset.ignore_label
        # self.target_pseudo_buffer = pseudo_buffer

        # init
        self.save_hyperparameters(ignore=['teacher_model', 'student_model', 'training_dataset', 'source_validation_dataset', 'target_validation_dataset','decoder'])
        

        # others
        self.validation_phases = ['source_validation', 'target_validation']
        # self.validation_phases = ['pseudo_target']

        self.class2mixed_names = self.training_dataset.class2names
        self.class2mixed_names = np.append(self.class2mixed_names, ["target_label"], axis=0)

        self.voxel_size = self.training_dataset.voxel_size

        # self.knn_search = KNN(k=self.propagation_size, transpose_mode=True)

        if self.training_dataset.weights is not None and self.weighted_sampling:
            tot = self.source_validation_dataset.weights.sum()
            self.sampling_weights = 1 - self.source_validation_dataset.weights/tot

        else:
            self.sampling_weights = None

    @property
    def momentum_pairs(self):
        """Defines base momentum pairs that will be updated using exponential moving average.
        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs (two element tuples).
        """
        return [(self.student_model, self.teacher_model)]

    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        self.last_step = 0

    # 随机选择点，如果数目不够则补充
    def random_sample(self, points, sub_num):
        """
        :param points: input points of shape [N, 3]
        :return: np.ndarray of N' points sampled from input points
        """

        num_points = points.shape[0]

        if sub_num is not None:
            if sub_num <= num_points:
                sampled_idx = np.random.choice(np.arange(num_points), sub_num, replace=False)
            else:
                over_idx = np.random.choice(np.arange(num_points), sub_num - num_points, replace=False)
                sampled_idx = np.concatenate([np.arange(num_points), over_idx])
        else:
            sampled_idx = np.arange(num_points)

        return sampled_idx

    @staticmethod
    def switch_off(labels, switch_classes):
        for s in switch_classes:
            class_idx = labels == s
            labels[class_idx] = -1

        return labels

    def mask_data(self, batch, is_oracle=False):
        # source
        batch_source_pts = batch['source_coordinates'].detach().cpu().numpy()
        batch_source_labels = batch['source_labels'].detach().cpu().numpy()
        batch_source_features = batch['source_features'].detach().cpu().numpy()

        # target
        batch_target_idx = batch['target_coordinates'][:, 0].detach().cpu().numpy()
        batch_target_pts = batch['target_coordinates'].detach().cpu().numpy()
        batch_target_features = batch['target_features'].detach().cpu().numpy()

        batch_size = int(np.max(batch_target_idx).item() + 1)

        if is_oracle:
            batch_target_labels = batch['target_labels'].detach().cpu().numpy()

        else:
            batch_target_labels = batch['pseudo_labels'].detach().cpu().numpy()

        new_batch = {'masked_target_pts': [],
                     'masked_target_labels': [],
                     'masked_target_features': [],
                     'masked_source_pts': [],
                     'masked_source_labels': [],
                     'masked_source_features': []}

        target_order = np.arange(batch_size)

        # 对每个batch的数据进行操作,对于每个batch的数据添加mix操作后的数据进行增强
        for b in range(batch_size):
            source_b_idx = batch_source_pts[:, 0] == b
            target_b = target_order[b]
            target_b_idx = batch_target_idx == target_b

            # source
            source_pts = batch_source_pts[source_b_idx, 1:] * self.voxel_size
            source_labels = batch_source_labels[source_b_idx]
            source_features = batch_source_features[source_b_idx]

            # target
            target_pts = batch_target_pts[target_b_idx, 1:] * self.voxel_size
            target_labels = batch_target_labels[target_b_idx]
            target_features = batch_target_features[target_b_idx]

            masked_target_pts, masked_target_labels, masked_target_features, masked_target_mask = self.mask(
                origin_pts=source_pts,
                origin_labels=source_labels,
                origin_features=source_features,
                dest_pts=target_pts,
                dest_labels=target_labels,
                dest_features=target_features
                )

            masked_source_pts, masked_source_labels, masked_source_features, masked_source_mask = self.mask(
                origin_pts=target_pts,
                origin_labels=target_labels,
                origin_features=target_features,
                dest_pts=source_pts,
                dest_labels=source_labels,
                dest_features=source_features,
                is_pseudo=True
                )

            if self.save_mix:
                os.makedirs('trial_viz_mix_paper', exist_ok=True)
                os.makedirs('trial_viz_mix_paper/s2t', exist_ok=True)
                os.makedirs('trial_viz_mix_paper/t2s', exist_ok=True)
                os.makedirs('trial_viz_mix_paper/source', exist_ok=True)
                os.makedirs('trial_viz_mix_paper/target', exist_ok=True)

                source_pcd = o3d.geometry.PointCloud()
                valid_source = source_labels != -1
                source_pcd.points = o3d.utility.Vector3dVector(source_pts[valid_source])
                source_pcd.colors = o3d.utility.Vector3dVector(self.source_validation_dataset.color_map[source_labels[valid_source]+1])

                target_pcd = o3d.geometry.PointCloud()
                target_pcd.points = o3d.utility.Vector3dVector(target_pts)
                target_pcd.colors = o3d.utility.Vector3dVector(self.source_validation_dataset.color_map[target_labels+1])

                s2t_pcd = o3d.geometry.PointCloud()
                s2t_pcd.points = o3d.utility.Vector3dVector(masked_target_pts)
                s2t_pcd.colors = o3d.utility.Vector3dVector(self.source_validation_dataset.color_map[masked_target_labels+1])

                t2s_pcd = o3d.geometry.PointCloud()
                valid_source = masked_source_labels != -1
                t2s_pcd.points = o3d.utility.Vector3dVector(masked_source_pts[valid_source])
                t2s_pcd.colors = o3d.utility.Vector3dVector(self.source_validation_dataset.color_map[masked_source_labels[valid_source]+1])

                o3d.io.write_point_cloud(f'trial_viz_mix_paper/source/{self.trainer.global_step}_{b}.ply', source_pcd)
                o3d.io.write_point_cloud(f'trial_viz_mix_paper/target/{self.trainer.global_step}_{b}.ply', target_pcd)
                o3d.io.write_point_cloud(f'trial_viz_mix_paper/s2t/{self.trainer.global_step}_{b}.ply', s2t_pcd)
                o3d.io.write_point_cloud(f'trial_viz_mix_paper/t2s/{self.trainer.global_step}_{b}.ply', t2s_pcd)

                os.makedirs('trial_viz_mix_paper/s2t_mask', exist_ok=True)
                os.makedirs('trial_viz_mix_paper/t2s_mask', exist_ok=True)
                os.makedirs('trial_viz_mix_paper/source_mask', exist_ok=True)
                os.makedirs('trial_viz_mix_paper/target_mask', exist_ok=True)

                source_pcd.paint_uniform_color([1, 0.706, 0])
                target_pcd.paint_uniform_color([0, 0.651, 0.929])

                s2t_pcd = o3d.geometry.PointCloud()
                s2t_pcd.points = o3d.utility.Vector3dVector(masked_target_pts)
                s2t_colors = np.zeros_like(masked_target_pts)
                s2t_colors[masked_target_mask] = [1, 0.706, 0]
                s2t_colors[np.logical_not(masked_target_mask)] = [0, 0.651, 0.929]
                s2t_pcd.colors = o3d.utility.Vector3dVector(s2t_colors)

                t2s_pcd = o3d.geometry.PointCloud()
                valid_source = masked_source_labels != -1
                t2s_pcd.points = o3d.utility.Vector3dVector(masked_source_pts[valid_source])
                t2s_colors = np.zeros_like(masked_source_pts[valid_source])
                masked_source_mask = masked_source_mask[valid_source]
                t2s_colors[masked_source_mask] = [0, 0.651, 0.929]
                t2s_colors[np.logical_not(masked_source_mask)] = [1, 0.706, 0]
                t2s_pcd.colors = o3d.utility.Vector3dVector(t2s_colors)

                o3d.io.write_point_cloud(f'trial_viz_mix_paper/source_mask/{self.trainer.global_step}_{b}.ply', source_pcd)
                o3d.io.write_point_cloud(f'trial_viz_mix_paper/target_mask/{self.trainer.global_step}_{b}.ply', target_pcd)
                o3d.io.write_point_cloud(f'trial_viz_mix_paper/s2t_mask/{self.trainer.global_step}_{b}.ply', s2t_pcd)
                o3d.io.write_point_cloud(f'trial_viz_mix_paper/t2s_mask/{self.trainer.global_step}_{b}.ply', t2s_pcd)

            _, _, _, masked_target_voxel_idx = ME.utils.sparse_quantize(
                coordinates=masked_target_pts,
                features=masked_target_features,
                labels=masked_target_labels,
                quantization_size=self.training_dataset.voxel_size,
                return_index=True)

            _, _, _, masked_source_voxel_idx = ME.utils.sparse_quantize(
                coordinates=masked_source_pts,
                features=masked_source_features,
                labels=masked_source_labels,
                quantization_size=self.training_dataset.voxel_size,
                return_index=True)

            masked_target_pts = masked_target_pts[masked_target_voxel_idx]
            masked_target_labels = masked_target_labels[masked_target_voxel_idx]
            masked_target_features = masked_target_features[masked_target_voxel_idx]

            masked_source_pts = masked_source_pts[masked_source_voxel_idx]
            masked_source_labels = masked_source_labels[masked_source_voxel_idx]
            masked_source_features = masked_source_features[masked_source_voxel_idx]

            masked_target_pts = np.floor(masked_target_pts/self.training_dataset.voxel_size)
            masked_source_pts = np.floor(masked_source_pts/self.training_dataset.voxel_size)

            batch_index = np.ones([masked_target_pts.shape[0], 1]) * b
            masked_target_pts = np.concatenate([batch_index, masked_target_pts], axis=-1)

            batch_index = np.ones([masked_source_pts.shape[0], 1]) * b
            masked_source_pts = np.concatenate([batch_index, masked_source_pts], axis=-1)

            new_batch['masked_target_pts'].append(masked_target_pts)
            new_batch['masked_target_labels'].append(masked_target_labels)
            new_batch['masked_target_features'].append(masked_target_features)
            new_batch['masked_source_pts'].append(masked_source_pts)
            new_batch['masked_source_labels'].append(masked_source_labels)
            new_batch['masked_source_features'].append(masked_source_features)

        for k, i in new_batch.items():
            if k in ['masked_target_pts', 'masked_target_features', 'masked_source_pts', 'masked_source_features']:
                new_batch[k] = torch.from_numpy(np.concatenate(i, axis=0)).to(self.device)
            else:
                new_batch[k] = torch.from_numpy(np.concatenate(i, axis=0))

        return new_batch

    def count_cls_num_list(self,labels,num_classes):
        # 使用 bincount 来计算每个类别的数量
        cls_counts = torch.bincount(labels+1, minlength=num_classes)
        # 将 bincount 的结果复制到 num_cls_list 中
        num_cls_list = (cls_counts+1).tolist()
        return num_cls_list

    # 伪标签则选择所有类别,真实标签根据权重选择类别
    def sample_classes(self, origin_classes, num_classes, is_pseudo=False):

        if not is_pseudo:
            if self.weighted_sampling and self.sampling_weights is not None:
                
                sampling_weights = self.sampling_weights[origin_classes] * (1/self.sampling_weights[origin_classes].sum())

                selected_classes = np.random.choice(origin_classes, num_classes,
                                                    replace=False, p=sampling_weights)

            else:
                selected_classes = np.random.choice(origin_classes, num_classes, replace=False)

        else:
            selected_classes = origin_classes

        return selected_classes

    def mask(self, origin_pts, origin_labels, origin_features,
             dest_pts, dest_labels, dest_features, is_pseudo=False):

        # to avoid when filtered labels are all -1
        if (origin_labels == -1).sum() < origin_labels.shape[0]:
            origin_present_classes = np.unique(origin_labels)
            origin_present_classes = origin_present_classes[origin_present_classes != -1]

            num_classes = int(self.selection_perc * origin_present_classes.shape[0])

            selected_classes = self.sample_classes(origin_present_classes, num_classes, is_pseudo)

            selected_idx = []
            selected_pts = []
            selected_labels = []
            selected_features = []

            if not self.training_dataset.augment_mask_data:
                for sc in selected_classes:
                    class_idx = np.where(origin_labels == sc)[0]

                    selected_idx.append(class_idx)
                    selected_pts.append(origin_pts[class_idx])
                    selected_labels.append(origin_labels[class_idx])
                    selected_features.append(origin_features[class_idx])

                if len(selected_pts) > 0:
                    # selected_idx = np.concatenate(selected_idx, axis=0)
                    selected_pts = np.concatenate(selected_pts, axis=0)
                    selected_labels = np.concatenate(selected_labels, axis=0)
                    selected_features = np.concatenate(selected_features, axis=0)

            else:

                for sc in selected_classes:
                    class_idx = np.where(origin_labels == sc)[0]

                    class_pts = origin_pts[class_idx]
                    num_pts = class_pts.shape[0]
                    sub_num = int(0.5 * num_pts)

                    # random subsample
                    random_idx = self.random_sample(class_pts, sub_num=sub_num)
                    class_idx = class_idx[random_idx]
                    class_pts = class_pts[random_idx]

                    # get transformation
                    voxel_mtx, affine_mtx = self.training_dataset.mask_voxelizer.get_transformation_matrix()

                    rigid_transformation = affine_mtx @ voxel_mtx
                    # apply transformations
                    homo_coords = np.hstack((class_pts, np.ones((class_pts.shape[0], 1), dtype=class_pts.dtype)))
                    class_pts = homo_coords @ rigid_transformation.T[:, :3]
                    class_labels = np.ones_like(origin_labels[class_idx]) * sc
                    class_features = origin_features[class_idx]

                    selected_idx.append(class_idx)
                    selected_pts.append(class_pts)
                    selected_labels.append(class_labels)
                    selected_features.append(class_features)

                if len(selected_pts) > 0:
                    # selected_idx = np.concatenate(selected_idx, axis=0)
                    selected_pts = np.concatenate(selected_pts, axis=0)
                    selected_labels = np.concatenate(selected_labels, axis=0)
                    selected_features = np.concatenate(selected_features, axis=0)

            if len(selected_pts) > 0:
                dest_idx = dest_pts.shape[0]
                dest_pts = np.concatenate([dest_pts, selected_pts], axis=0)
                dest_labels = np.concatenate([dest_labels, selected_labels], axis=0)
                dest_features = np.concatenate([dest_features, selected_features], axis=0)

                mask = np.ones(dest_pts.shape[0])
                mask[:dest_idx] = 0

            if self.training_dataset.augment_data:
                # get transformation
                voxel_mtx, affine_mtx = self.training_dataset.voxelizer.get_transformation_matrix()
                rigid_transformation = affine_mtx @ voxel_mtx
                # apply transformations
                homo_coords = np.hstack((dest_pts, np.ones((dest_pts.shape[0], 1), dtype=dest_pts.dtype)))
                dest_pts = homo_coords @ rigid_transformation.T[:, :3]

        return dest_pts, dest_labels, dest_features, mask.astype(bool)

        # 解码source data和target data用于数据增强

    # 置信度更新标签更新和利用解码器进行数据增强
    def decode_batch(
        self, 
        batch, 
        conf=0.90, 
        num_pts=2048,
        decode_coords=True,
        update_label=False,
    ):  
        # 同为False,保持原有操作
        if not decode_coords:
            return batch
        
        self.decoder.eval()
        self.teacher_model.eval()
        batch_size = torch.unique(batch["source_coordinates"][:, 0]).max() + 1 
        batch_size = batch_size.int().item()
        
        update_dict = {
            "source_coordinates": [],
            "source_labels": [],
            "source_features": [],
            "target_coordinates": [],
            "target_features": [],
            "target_labels": [],
            "pseudo_labels": [],
        }
        
        # 使用解码器解码点云, 返回(source_fine_points,target_fine_points),返回解码器解码点云
        def decode_single_batch(source_coords_decoded, target_coords_decoded):
            decode_loss = None
            
            source_fine_points = source_coords_decoded.detach().float()
            source_coords_batch = torch.unsqueeze(source_coords_decoded, 0)
            gt_decoded = source_coords_batch.float().contiguous()
            inputs = source_coords_batch.float().transpose(2, 1).contiguous()
            result_dict = self.decoder(x=inputs, gt=gt_decoded, is_training=False)
            source_fine_points = torch.squeeze(result_dict['out2'], 0).detach().float()
            if self.decode_batch:
                # train decoder
                out2, loss2, source_net_loss = self.decoder(
                    x=inputs,gt=gt_decoded,is_training=True,alpha=1
                )
            del source_coords_batch, gt_decoded, inputs, result_dict
            torch.cuda.empty_cache()
            
            target_fine_points = target_coords_decoded.detach().float()
            target_coords_batch = torch.unsqueeze(target_coords_decoded, 0)
            gt_decoded = target_coords_batch.float().contiguous()
            inputs = target_coords_batch.float().transpose(2, 1).contiguous()
            result_dict = self.decoder(x=inputs, gt=gt_decoded, is_training=False)
            target_fine_points = torch.squeeze(result_dict['out2'], 0).detach().float()
            if self.decode_batch:
                # train decoder
                out2, loss2, target_net_loss = self.decoder(
                    x=inputs,gt=gt_decoded,is_training=True,alpha=1
                )
            del target_coords_batch, gt_decoded, inputs, result_dict
            torch.cuda.empty_cache()
            
            if source_net_loss is not None and target_net_loss is not None:    
                decode_loss = source_net_loss + target_net_loss
            return source_fine_points, target_fine_points, decode_loss

        # 根据置信度更新点云标签
        def update_label_single_batch(source_params, target_params):
          
            source_updated_coords, source_updated_labels, source_updated_nums, source_updated_fixed_nums = self.update_label(
                source_params.fine_points, source_params.coords_idx, source_params.labels_idx, source_params.conf_idx, conf=conf,modify_labels=update_label
            )
            torch.cuda.empty_cache()

            target_updated_coords, target_updated_pseudo, target_updated_nums, target_updated_fixed_nums = self.update_label(
                target_params.fine_points, target_params.coords_idx, target_params.pseudo_idx, target_params.conf_idx, conf=conf,modify_labels=update_label
            )
            torch.cuda.empty_cache()

            source_coords_idx, source_feats_idx, source_labels_idx, source_voxel_idx = ME.utils.sparse_quantize(
                coordinates=source_updated_coords.cpu(),
                features=source_params.feats_idx.cpu(),
                labels=source_updated_labels.cpu().int(),
                ignore_label=-1,
                quantization_size=self.training_dataset.voxel_size,
                return_index=True
            )
            b_tensor = torch.full((source_coords_idx.shape[0], 1), b)
            source_coords_idx = torch.cat((b_tensor, source_coords_idx), dim=1).to(self.device)

            target_coords_idx, target_feats_idx, target_pseudo_idx, target_voxel_idx = ME.utils.sparse_quantize(
                coordinates=target_updated_coords.cpu(),
                features=target_params.feats_idx.cpu(),
                labels=target_updated_pseudo.cpu().int(),
                ignore_label=-1,
                quantization_size=self.training_dataset.voxel_size,
                return_index=True
            )
            b_tensor = torch.full((target_coords_idx.shape[0], 1), b)
            target_coords_idx = torch.cat((b_tensor, target_coords_idx), dim=1).to(self.device)
            target_labels_idx = target_params.labels_idx[target_voxel_idx]
            del target_params, source_params
            
            return \
            SPCDParams(source_updated_coords, source_coords_idx, source_labels_idx, source_updated_nums, source_feats_idx), \
            TPCDParams(target_updated_coords, target_coords_idx, target_labels_idx,  target_pseudo_idx, target_updated_nums, target_feats_idx)
        
        # (update_source_params,update_target_params,decode_loss)        
        def process_single_batch(b):
            source_b_idx = batch["source_coordinates"][:, 0] == b
            target_b_idx = batch["target_coordinates"][:, 0] == b
            source_coords_idx = batch["source_coordinates"][source_b_idx].to(self.device)
            source_labels_idx = batch["source_labels"][source_b_idx.cpu()].to(self.device)
            source_features_idx = batch["source_features"][source_b_idx].to(self.device)
            target_coords_idx = batch["target_coordinates"][target_b_idx].to(self.device)
            target_labels_idx = batch["target_labels"][target_b_idx.cpu()].to(self.device)
            target_features_idx = batch["target_features"][target_b_idx].to(self.device)
            target_pseudo_idx = batch["pseudo_labels"][target_b_idx.cpu()].to(self.device)
            
            # 计算当前模型的特征输出
            with torch.no_grad():
                with amp.autocast():
                    source_stensor = ME.SparseTensor(
                        coordinates=source_coords_idx.int(), features=source_features_idx
                    )
                    source_out = self.teacher_model(source_stensor).F
                    source_out = F.softmax(source_out.detach(), dim=-1)
                    
                    target_stensor = ME.SparseTensor(
                        coordinates=target_coords_idx.int(), features=target_features_idx
                    )
                    target_out = self.teacher_model(target_stensor).F
                    target_out = F.softmax(target_out.detach(), dim=-1)
        
            source_conf_idx, source_preds = source_out.max(1)
            target_conf_idx, target_preds = target_out.max(1)

            source_coords_decoded = source_coords_idx[..., 1:].float() * self.training_dataset.voxel_size
            source_coords_decoded, source_srs_idx = simple_random_sampling(source_coords_decoded, num_samples=num_pts)
            target_coords_decoded = target_coords_idx[..., 1:].float() * self.training_dataset.voxel_size
            target_coords_decoded, target_srs_idx = simple_random_sampling(target_coords_decoded, num_samples=num_pts)
            
            # 解码恢复后点云
            source_fine_points, target_fine_points, decode_loss = decode_single_batch(source_coords_decoded, target_coords_decoded)
            # 对点云进行标签更新
            source_params = SPCDParams(source_fine_points, source_coords_idx, source_labels_idx, source_conf_idx, source_features_idx)
            target_params = TPCDParams(target_fine_points, target_coords_idx, target_labels_idx, target_pseudo_idx, target_conf_idx, target_features_idx)
            updated_source_params, updated_target_params = update_label_single_batch(source_params, target_params)
            torch.cuda.empty_cache()
            return updated_source_params, updated_target_params,decode_loss

        total_decode_loss = 0.0
        for b in range(batch_size):
            updated_source_params, updated_target_params,decode_loss = process_single_batch(b)
            update_dict["source_coordinates"].append(updated_source_params.coords_idx)
            update_dict["source_features"].append(updated_source_params.feats_idx)
            update_dict["source_labels"].append(updated_source_params.labels_idx)
            update_dict["target_coordinates"].append(updated_target_params.coords_idx)
            update_dict["pseudo_labels"].append(updated_target_params.pseudo_idx)
            update_dict["target_features"].append(updated_target_params.feats_idx)
            update_dict["target_labels"].append(updated_target_params.labels_idx)
            if decode_loss is not None:
                total_decode_loss += decode_loss

        for key in update_dict:
            batch[key] = torch.cat(update_dict[key], dim=0)
        torch.cuda.empty_cache()
        return batch,total_decode_loss.cpu()

    def update_label(self, fine_points, coords_idx, labels_idx, conf_idx, conf=0.95, distance_threshold=1.5, k=6, modify_labels=True):
        coords_idx = coords_idx[..., 1:].float() * self.training_dataset.voxel_size
        coords_idx = coords_idx.cpu()
        fine_points = fine_points.detach().cpu()

        tree = cKDTree(coords_idx.numpy())
        distances, indices = tree.query(fine_points.numpy(), k=1)
        valid_mask = distances <= distance_threshold
        valid_indices = indices[valid_mask]

        updated_coords = coords_idx.clone()
        updated_labels = labels_idx.clone()
        updated_nums = 0
        updated_fixed_nums = 0
        
        # 标签更新
        for idx in range(len(coords_idx)):
            if idx in valid_indices:
                distances, neighbor_indices = tree.query(coords_idx[idx].cpu().numpy(), k=k)
                neighbor_confidences = conf_idx[neighbor_indices]
                neighbor_labels = labels_idx[neighbor_indices]

                valid_neighbors = neighbor_confidences > conf

                if modify_labels and valid_neighbors.any():
                    nearest_neighbor_label = neighbor_labels[valid_neighbors][0]
                    updated_labels[idx] = nearest_neighbor_label
                    updated_fixed_nums += 1
                elif modify_labels:
                    # updated_labels[idx] = -1
                    continue
                else: 
                    # 如果置信度大于conf,保留标签
                    updated_labels[idx] = conf_idx[idx] if conf_idx[idx]>conf else -1
                updated_nums += 1

        return updated_coords, updated_labels, updated_nums, updated_fixed_nums

    def training_step(self, batch, batch_idx):
        '''
        :param batch: training batch
        :param batch_idx: batch idx
        :return: None
        '''

        '''
        batch.keys():
            - source_coordinates
            - source_labels
            - source_features
            - source_idx
            - target_coordinates
            - target_labels
            - target_features
            - target_idx
        '''
        # refresh cls_num_list
        # self.cls_num_list = self.training_dataset.get_cls_num_list(self.kps_counter)
        # self.kps_counter = self.kps_counter + 1
        if isinstance(self.source_criterion,KPSLoss):
            self.source_cls_num_list = self.count_cls_num_list(batch['source_labels'],self.num_classes)
            self.target_cls_num_list = self.count_cls_num_list(batch['target_labels'],self.num_classes)

        # Must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        # target batch
        target_stensor = ME.SparseTensor(coordinates=batch['target_coordinates'].int(),
                                         features=batch['target_features'])

        target_labels = batch['target_labels'].long().cpu()

        source_stensor = ME.SparseTensor(coordinates=batch['source_coordinates'].int(),
                                         features=batch['source_features'])

        source_labels = batch['source_labels'].long().cpu()

        self.teacher_model.eval()
        with torch.no_grad():

            target_pseudo = self.teacher_model(target_stensor).F.cpu()

            if self.filtering == 'confidence':
                target_confidence_th = self.target_confidence_th
                target_pseudo = F.softmax(target_pseudo, dim=-1)
                target_conf, target_pseudo = target_pseudo.max(dim=-1)
                filtered_target_pseudo = -torch.ones_like(target_pseudo)
                valid_idx = target_conf > target_confidence_th
                filtered_target_pseudo[valid_idx] = target_pseudo[valid_idx]
                target_pseudo = filtered_target_pseudo.long()

            else:
                target_pseudo = F.softmax(target_pseudo, dim=-1)
                target_conf, target_pseudo = target_pseudo.max(dim=-1)

        batch['pseudo_labels'] = target_pseudo
        batch['source_labels'] = source_labels
        # teacher模型获取到的pseudo_labels用于 t -> s
        # pdb.set_trace()
        if self.batch_counter%20 == 0:
            batch = self.decode_batch(batch=batch,decode_coords=self.decode_coords,update_label=self.conf_update)
        self.batch_counter += 1
        
        masked_batch = self.mask_data(batch, is_oracle=False)

        s2t_stensor = ME.SparseTensor(coordinates=masked_batch["masked_target_pts"].int(),
                                      features=masked_batch["masked_target_features"])

        t2s_stensor = ME.SparseTensor(coordinates=masked_batch["masked_source_pts"].int(),
                                      features=masked_batch["masked_source_features"])

        s2t_labels = masked_batch["masked_target_labels"]
        t2s_labels = masked_batch["masked_source_labels"]

        s2t_out = self.student_model(s2t_stensor).F.cpu()
        t2s_out = self.student_model(t2s_stensor).F.cpu()

        s2t_loss = self.target_criterion(s2t_out, s2t_labels.long())
        t2s_loss = self.target_criterion(t2s_out, t2s_labels.long())

   
        final_loss = self.target_weight * s2t_loss \
                + self.source_weight * t2s_loss 

        results_dict = {'s2t_loss': s2t_loss.detach(),
                    't2s_loss': t2s_loss.detach()}

        with torch.no_grad():
            self.student_model.eval()
            target_out = self.student_model(target_stensor).F.cpu()
            _, target_preds = target_out.max(dim=-1)

            target_iou_tmp = jaccard_score(target_preds.numpy(), target_labels.numpy(), average=None,
                                            labels=np.arange(0, self.num_classes),
                                            zero_division=0.)
            present_labels, class_occurs = np.unique(target_labels.numpy(), return_counts=True)
            present_labels = present_labels[present_labels != self.ignore_label]
            present_names = self.training_dataset.class2names[present_labels].tolist()
            present_names = ['student/' + p + '_target_iou' for p in present_names]
            results_dict.update(dict(zip(present_names, target_iou_tmp.tolist())))
            results_dict['student/target_iou'] = np.mean(target_iou_tmp[present_labels])

        self.student_model.train()

        valid_idx = torch.logical_and(target_pseudo != -1, target_labels != -1)
        correct = (target_pseudo[valid_idx] == target_labels[valid_idx]).sum()
        pseudo_acc = correct / valid_idx.sum()

        results_dict['teacher/acc'] = pseudo_acc
        results_dict['teacher/confidence'] = target_conf.mean()

        ann_pts = (target_pseudo != -1).sum()
        results_dict['teacher/annotated_points'] = ann_pts/target_pseudo.shape[0]

        for k, v in results_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v).float()
            else:
                v = v.float()
            self.log(
                name=k,
                value=v,
                logger=True,
                sync_dist=True,
                rank_zero_only=True,
                batch_size=self.train_batch_size,
                add_dataloader_idx=False
            )

        return final_loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        """Performs the momentum update of momentum pairs using exponential moving average at the
        end of the current training step if an optimizer step was performed.
        Args:
            outputs (Dict[str, Any]): the outputs of the training step.
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.
            dataloader_idx (int): index of the dataloader.
        """
        if self.trainer.global_step > self.last_step and self.trainer.global_step % self.update_every == 0:
            # update momentum backbone and projector
            momentum_pairs = self.momentum_pairs
            for mp in momentum_pairs:
                self.momentum_updater.update(*mp)
            # update tau
            cur_step = self.trainer.global_step
            if self.trainer.accumulate_grad_batches:
                cur_step = cur_step * self.trainer.accumulate_grad_batches
            self.momentum_updater.update_tau(
                cur_step=cur_step,
                max_steps=len(self.trainer.train_dataloader) * self.trainer.max_epochs,
            )
        self.last_step = self.trainer.global_step
        #  # Add the condition to set self.decode_flag
        # self.decode_flag = True  if self.trainer.global_step%100==0 else False
        # print(f'\n[INFO] update decode_flag, value {self.decode_flag}')
  
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        phase = self.validation_phases[dataloader_idx]
        # input batch
        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])

        # must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.student_model(stensor).F.cpu()

        labels = batch['labels'].long().cpu()


        if phase == 'source_validation':
            loss = self.source_criterion(out, labels)
        else:
            loss = self.target_criterion(out, labels)

        soft_pseudo = F.softmax(out[:, :-1], dim=-1)

        conf, preds = soft_pseudo.max(1)

        iou_tmp = jaccard_score(preds.detach().numpy(), labels.numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=0.)

        present_labels, class_occurs = np.unique(labels.numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        present_names = self.training_dataset.class2names[present_labels].tolist()
        present_names = [os.path.join(phase, p + '_iou') for p in present_names]
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
                add_dataloader_idx=False
            )

    def configure_optimizers(self):
        if self.scheduler_name is None:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.student_model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.student_model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            return optimizer
        else:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.student_model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.student_model.parameters(),
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
            elif self.scheduler_name == 'OneCycleLR':
                steps_per_epoch = int(len(self.training_dataset) / self.train_batch_size)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                                                steps_per_epoch=steps_per_epoch,
                                                                epochs=self.trainer.max_epochs)

            else:
                raise NotImplementedError

            return [optimizer], [scheduler]

