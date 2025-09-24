# adversarial synlidar 
from utils.datasets.dataset import BaseDataset
import os
import torch
import yaml
import numpy as np
import pdb
import MinkowskiEngine as ME
import open3d as o3d
import time
import random
import glob

# <====================tools=====================>
def timeit(func):
    """
    func: time recording for function
    """
    def _wrap(*args,**kwargs):
        start_time = time.time()
        print(f"[INFO] the execution of function {func.__name__} start:")
        result = func(*args,**kwargs)
        elastic_time = time.time()-start_time
        print("[INFO] The execution time of the function '%s is %.6fs\n"%(
            func.__name__,elastic_time
        ))
        return result
    return _wrap
       
def get_sum_split(path: str='utils/datasets/_splits/synlidar.pkl',phase:str='train'):
    """
        calculate train split and evaluation split for Synlidar dataset
        phase in ['train','validation']
    """
    split_path = path
    split = torch.load(split_path)
    # calculate train split
    split_values = []
    count_val = 0
    split_list = [len(split[phase][key]) for key, values in split[phase].items()]
    for item in split_list:
        count_val = count_val + item
        split_values.append(count_val)
    return split,split_values

# ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.dirname(__file__)
current_dir = os.path.dirname(current_dir)
ABSOLUTE_PATH = os.path.dirname(current_dir)

def get_random_indices(dataset_length,prob:float=1.):
    
    if not 0. <= prob <= 1.:
        raise ValueError("Probability must be between 0 and 1")
    if prob == 0.:
        return None
    num_samples = int(dataset_length * prob)
    # Generate random indices
    random_indices = random.sample(range(dataset_length), num_samples)
    return random_indices

# return adversarial training dataset or original training dataset
# label: gt, adversarial
# coordinates: original, adversarial
class AdvSynLidarTrain(BaseDataset):
    def __init__(
            self,
            version: str = 'full',
            phase: str = 'train',
            dataset_path: str = '/data/SynLiDAR/sequences',
            adv_dataset_path: str = 'experiments/SAVEDIR/attackedLidar_lidar2kitti_all/train',
            adv_gt_label:bool = False,
            mapping_path: str = 'utils/datasets/_resources/synlidar_semantickitti.yaml',
            weights_path: str = 'utils/datasets/_weights/synlidar2semantickitti_correct.npy',
            voxel_size: float = 0.05,
            use_intensity: bool = False,
            augment_data: bool = False,
            sub_num: int = 80000,
            device: str = None,
            num_classes: int = 7,
            ignore_label: int = None,
            attacked_prob=0.,
            split_path = None
        ):
        if weights_path is not None:
            weights_path = os.path.join(ABSOLUTE_PATH, weights_path)
        # initialization, adversarial sample list
        self.adv_dataset_path = adv_dataset_path
        print(f'[INFO] attack synlidar data path {self.adv_dataset_path}, exits: {os.path.exists(self.adv_dataset_path)}')
        self.adv_gt_label = adv_gt_label
        self.split_path = split_path
        
        super().__init__(version=version,
                         phase=phase,
                         dataset_path=dataset_path,
                         voxel_size=voxel_size,
                         sub_num=sub_num,
                         use_intensity=use_intensity,
                         augment_data=augment_data,
                         device=device,
                         num_classes=num_classes,
                         ignore_label=ignore_label,
                         weights_path=weights_path)
        # 用于记录00-12中每一个场景的总数
        # self.count_train_val=[293, 1719, 2694, 3681, 5350, 6841, 11348, 13306, 14452, 16194, 17292, 18717, 19834]
        self.name = 'SynLiDARTrain'
        if self.version == 'full':
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            self.get_splits()
        elif self.version == 'mini':
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            self.get_splits()
        elif self.version == 'sequential':
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            self.get_splits()
        else:
            raise NotImplementedError

        with open(os.path.join(ABSOLUTE_PATH, mapping_path),'r') as file:
            self.maps = yaml.safe_load(file)

        self.pcd_path = []
        self.label_path = []

        self.adv_pcd_path = []
        self.adv_label_path = []
        self.adv_gt_path = []

        self.selected = []

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val
        # self.frames_record = []
        # self.sequences_record = []
        # frames,sequences用于随机对应的序列和帧序号
        for sequence, frames in self.split[self.phase].items():
            for frame in frames:
                pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(frame):06d}.bin')
                label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(frame):06d}.label')
                adv_pcd_path = os.path.join(self.adv_dataset_path,phase,'adv_preds',sequence,f'{int(frame)}_*.ply')
                adv_label_path = os.path.join(self.adv_dataset_path,phase,'adv_labels_pt',sequence,f'{int(frame)}_*.pt')
                ori_label_path = os.path.join(self.adv_dataset_path,phase,'ori_labels_pt',sequence,f'{int(frame)}_*.pt')
                self.pcd_path.append(pcd_path)
                self.label_path.append(label_path)
                self.adv_pcd_path.append(adv_pcd_path)
                self.adv_label_path.append(adv_label_path)
                self.adv_gt_path.append(ori_label_path)
    

        self.attacked_indices = get_random_indices(len(self.pcd_path),attacked_prob)
              
        if self.attacked_indices is None:
            print(f'[INFO] --> Selecting clean Synlidar {self.phase} dataset, total num {len(self.pcd_path)}!')
        else:
            print(f'[INFO] --> Selecting adversarial Synlidar {self.phase} dataset, total num {len(self.pcd_path)}, attacked indices {len(self.attacked_indices)}!')

    def get_splits(self):
        if self.version == 'full':
            split_path = os.path.join(ABSOLUTE_PATH, 'utils/datasets/_splits/synlidar.pkl')
        elif self.version == 'mini':
            split_path = os.path.join(ABSOLUTE_PATH, 'utils/datasets/_splits/synlidar_mini.pkl')
        elif self.version == 'sequential':
            split_path = os.path.join(ABSOLUTE_PATH, 'utils/datasets/_splits/synlidar_sequential.pkl')
        else:
            raise NotImplementedError
        # 如果默认的split路径不存在
        if self.split_path is None:
            self.split_path = split_path
        print(f'[INFO] using split path as {self.split_path}, exists: {os.path.isfile(self.split_path)}')
            
        if not os.path.isfile(self.split_path):
            self.split = {'train': {s: [] for s in self.sequences},
                          'validation': {s: [] for s in self.sequences}}
            if self.version != 'sequential':
                for sequence in self.sequences:
                    num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))
                    valid_frames = []

                    for v in np.arange(num_frames):
                        pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(v):06d}.bin')
                        label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(v):06d}.label')

                        if os.path.isfile(pcd_path) and os.path.isfile(label_path):
                            valid_frames.append(v)
                    if self.version == 'full':
                        train_selected = np.random.choice(valid_frames, int(num_frames/10), replace=False)
                    else:
                        train_selected = np.random.choice(valid_frames, int(num_frames/100), replace=False)

                    for t in train_selected:
                        valid_frames.remove(t)

                    validation_selected = np.random.choice(valid_frames, int(num_frames/100), replace=False)

                    self.split['train'][sequence].extend(train_selected)
                    self.split['validation'][sequence].extend(validation_selected)
                torch.save(self.split, split_path)
            else:
                for sequence in self.sequences:
                    num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))
                    total_for_sequence = int(num_frames/10)
                    print('--> TOTAL:', total_for_sequence)
                    train_selected = []
                    validation_selected = []

                    for v in np.arange(num_frames):
                        pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(v):06d}.bin')
                        label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(v):06d}.label')

                        if os.path.isfile(pcd_path) and os.path.isfile(label_path):
                            if len(train_selected) == 0:
                                train_selected.append(v)
                                last_added = v
                            elif len(train_selected) < total_for_sequence and (v-last_added) >= 5:
                                train_selected.append(v)
                                last_added = v
                                print(last_added)
                            else:
                                validation_selected.append(v)

                    validation_selected = np.random.choice(validation_selected, int(num_frames/100), replace=False)

                    self.split['train'][sequence].extend(train_selected)
                    self.split['validation'][sequence].extend(validation_selected)
                torch.save(self.split, split_path)
        # 如果默认的split路径存在
        else:
            # self.split = torch.load(split_path)
            self.split,self.count_train_val = get_sum_split(self.split_path,'train')
            print('[INFO] TRAIN SEQUENCES', self.split['train'].keys())
            print("[INFO] SEQUENCES TOTAL", self.count_train_val)        

    def __len__(self):
        return len(self.pcd_path)

    def calculate_sequences(self,idx):
        """
            calculate sequence num and frame num based on idx
        """
        i=0
        count_val = self.count_train_val
        sequence = None
        sequence_num = None
        while i < len(count_val):
            if idx.item()<count_val[0]:
                sequence_num = 0
                sequence = "00"
                break
            elif idx.item()<count_val[i+1]:
                sequence_num = i 
                sequence = self.sequences[i+1]
                break
            else:
                i+=1
        frame_num = self.split['train'][sequence][idx-self.count_train_val[sequence_num]]
        return sequence,frame_num

    def get_data(self, i:int):
        """
            获取原始点云数据
        """
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = np.fromfile(label_tmp, dtype=np.uint32)
        label = label.reshape((-1))
        label = self.remap_lut_val[label].astype(np.int32)
        points = pcd[:, :3]
        if self.use_intensity:
            colors = pcd[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
        data = {'points': points, 'colors': colors, 'labels': label}

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        return {"coordinates": points,
                "features": colors,
                "labels": labels,
                "idx": torch.tensor(i),
            }

    def clean_items(self ,i:int):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = np.fromfile(label_tmp, dtype=np.uint32)
        label = label.reshape((-1))
        label = self.remap_lut_val[label].astype(np.int32)
        points = pcd[:, :3]
        if self.use_intensity:
            colors = pcd[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
        data = {'points': points, 'colors': colors, 'labels': label}

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        # print(f'[INFO] the shape of points {points.shape}, the shape of labels {labels.shape}')

        sampled_idx = np.arange(points.shape[0])
        if self.phase == 'train' and self.augment_data:
            sampled_idx = self.random_sample(points)
            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]

            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            points = homo_coords @ rigid_transformation.T[:, :3]
        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        # print(f'[INFO] the shape of points {points.shape}, the shape of labels {labels.shape}')

        quantized_coords, feats, labels, voxel_idx = ME.utils.sparse_quantize(
            coordinates=points,
            features=colors,
            labels=labels,
            ignore_label=vox_ign_label,
            quantization_size=self.voxel_size,
            return_index=True
        )

        # print(f'[INFO] the shape of points {quantized_coords.shape}, the shape of labels {labels.shape}')

        if isinstance(quantized_coords, np.ndarray):
            quantized_coords = torch.from_numpy(quantized_coords)

        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)

        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        if isinstance(voxel_idx, np.ndarray):
            voxel_idx = torch.from_numpy(voxel_idx)

        if sampled_idx is not None:
            sampled_idx = sampled_idx[voxel_idx]
            sampled_idx = torch.from_numpy(sampled_idx)
        else:
            sampled_idx = None
        
        return {"coordinates": quantized_coords,
                "features": feats,
                "labels": labels,
                "sampled_idx": sampled_idx,
                "idx": torch.tensor(i),
            }

    def adv_items(self,i:int):
        try:
            pcd_pattern = self.adv_pcd_path[i]
            matching_files = glob.glob(pcd_pattern)
            pcd_tmp = matching_files[0]
            label_pattern = self.adv_gt_path[i] if self.adv_gt_label else self.adv_label_path[i]
            matching_files = glob.glob(label_pattern)
            label_tmp = matching_files[0]
            
            point_cloud = o3d.io.read_point_cloud(pcd_tmp)
            points = np.asarray(point_cloud.points)
            labels = torch.load(label_tmp).int()
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
            data = {'points': points, 'colors': colors, 'labels': labels}

            points = data['points']
            colors = data['colors']
            labels = data['labels']

            # print(f'[INFO] the shape of points {points.shape}, the shape of labels {labels.shape}')
            sampled_idx = np.arange(points.shape[0])

            """
            if self.phase == 'train' and self.augment_data:
                sampled_idx = self.random_sample(points)
                points = points[sampled_idx]
                colors = colors[sampled_idx]
                labels = labels[sampled_idx]

                voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

                rigid_transformation = affine_mtx @ voxel_mtx
                # Apply transformations

                homo_coords = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
                # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
                points = homo_coords @ rigid_transformation.T[:, :3]

            if self.ignore_label is None:
                vox_ign_label = -100
            else:
                vox_ign_label = self.ignore_label
            """

            # print(f'[INFO] before quantized the shape of points {points.shape}, the shape of labels {labels.shape}')

            quantized_coords, feats, labels, voxel_idx = ME.utils.sparse_quantize(
                coordinates=points,
                features=colors,
                labels=labels,
                ignore_label=self.ignore_label,
                # quantization_size=self.voxel_size,
                quantization_size=1,
                return_index=True
            )
            
            # print(f'[INFO] after quantized the shape of points {quantized_coords.shape}, the shape of labels {labels.shape}')

            if isinstance(quantized_coords, np.ndarray):
                quantized_coords = torch.from_numpy(quantized_coords)

            if isinstance(feats, np.ndarray):
                feats = torch.from_numpy(feats)

            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)

            if isinstance(voxel_idx, np.ndarray):
                voxel_idx = torch.from_numpy(voxel_idx)

            if sampled_idx is not None:
                sampled_idx = sampled_idx[voxel_idx]
                sampled_idx = torch.from_numpy(sampled_idx)
            else:
                sampled_idx = None
        
            return {"coordinates": quantized_coords,
                    "features": feats,
                    "labels": labels,
                    "sampled_idx": sampled_idx,
                    "idx": torch.tensor(i),
                }
        
        except Exception as e:
            # 打印异常信息
            print(f"[Error] Error happened in get adv sample, idx: {i}, error: {type(e).__name__}: {e}")
            return self.clean_items(i)
        
    def __getitem__(self, i: int):
        if self.attacked_indices is not None and i in self.attacked_indices:
            return self.adv_items(i)
        return self.clean_items(i)
    
# return adversarial validation dataset or original validation dataset
# label: gt,adversarial
# coordinates: original, adversarial
class AdvSynlidarVal(BaseDataset):
    def __init__(
            self,
            version: str = 'full',
            phase: str = 'validation',
            dataset_path: str = '/data/SynLiDAR/sequences',
            adv_dataset_path: str = 'experiments/SAVEDIR/attackedLidar_lidar2kitti_all/validation',
            adv_gt_label:bool = False,
            mapping_path: str = 'utils/datasets/_resources/synlidar_semantickitti.yaml',
            weights_path: str = 'utils/datasets/_weights/synlidar2semantickitti_correct.npy',
            voxel_size: float = 0.05,
            use_intensity: bool = False,
            augment_data: bool = False,
            sub_num: int = 80000,
            device: str = None,
            num_classes: int = 7,
            ignore_label: int = None,
            attacked_prob=0.,
            split_path=None
        ):
        if weights_path is not None:
            weights_path = os.path.join(ABSOLUTE_PATH, weights_path)
        # initialization, adversarial sample list
        self.adv_dataset_path = adv_dataset_path
        self.adv_gt_label = adv_gt_label
        self.split_path = split_path
        
        super().__init__(version=version,
                         phase=phase,
                         dataset_path=dataset_path,
                         voxel_size=voxel_size,
                         sub_num=sub_num,
                         use_intensity=use_intensity,
                         augment_data=augment_data,
                         device=device,
                         num_classes=num_classes,
                         ignore_label=ignore_label,
                         weights_path=weights_path)
        # 用于记录00-12中每一个场景的总数
        # self.count_train_val=[293, 1719, 2694, 3681, 5350, 6841, 11348, 13306, 14452, 16194, 17292, 18717, 19834]
        self.name = 'SynLiDARValidation'
        if self.version == 'full':
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            self.get_splits()
        elif self.version == 'mini':
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            self.get_splits()
        elif self.version == 'sequential':
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            self.get_splits()
        else:
            raise NotImplementedError

        with open(os.path.join(ABSOLUTE_PATH, mapping_path),'r') as file:
            self.maps = yaml.safe_load(file)

        self.pcd_path = []
        self.label_path = []

        self.adv_pcd_path = []
        self.adv_label_path = []
        self.adv_gt_path = []

        self.selected = []

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val
        # self.frames_record = []
        # self.sequences_record = []
        # frames,sequences用于随机对应的序列和帧序号
        for sequence, frames in self.split[self.phase].items():
            for frame in frames:
                pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(frame):06d}.bin')
                label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(frame):06d}.label')
                adv_pcd_path = os.path.join(self.adv_dataset_path,phase,'adv_preds',sequence,f'{int(frame)}_*.ply')
                adv_label_path = os.path.join(self.adv_dataset_path,phase,'adv_labels_pt',sequence,f'{int(frame)}_*.pt')
                ori_label_path = os.path.join(self.adv_dataset_path,phase,'ori_labels_pt',sequence,f'{int(frame)}_*.pt')
                self.pcd_path.append(pcd_path)
                self.label_path.append(label_path)
                self.adv_pcd_path.append(adv_pcd_path)
                self.adv_label_path.append(adv_label_path)
                self.adv_gt_path.append(ori_label_path)
              
        self.attacked_indices = get_random_indices(len(self.pcd_path),attacked_prob)
        if self.attacked_indices is None:
            print(f'[INFO] --> Selecting clean Synlidar {self.phase} dataset, total num {len(self.pcd_path)}!')
        else:
            print(f'[INFO] --> Selecting adversarial Synlidar {self.phase} dataset, total num {len(self.pcd_path)}, attacked indices {len(self.attacked_indices)}!')
    
    def get_splits(self):
        if self.version == 'full':
            split_path = os.path.join(ABSOLUTE_PATH, 'utils/datasets/_splits/synlidar.pkl')
        elif self.version == 'mini':
            split_path = os.path.join(ABSOLUTE_PATH, 'utils/datasets/_splits/synlidar_mini.pkl')
        elif self.version == 'sequential':
            split_path = os.path.join(ABSOLUTE_PATH, 'utils/datasets/_splits/synlidar_sequential.pkl')
        else:
            raise NotImplementedError

        # 如果默认的split路径不存在
        if self.split_path is None:
            self.split_path = split_path
            
        if not os.path.isfile(self.split_path):
            self.split = {'train': {s: [] for s in self.sequences},
                          'validation': {s: [] for s in self.sequences}}
            if self.version != 'sequential':
                for sequence in self.sequences:
                    num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))
                    valid_frames = []

                    for v in np.arange(num_frames):
                        pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(v):06d}.bin')
                        label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(v):06d}.label')

                        if os.path.isfile(pcd_path) and os.path.isfile(label_path):
                            valid_frames.append(v)
                    if self.version == 'full':
                        train_selected = np.random.choice(valid_frames, int(num_frames/10), replace=False)
                    else:
                        train_selected = np.random.choice(valid_frames, int(num_frames/100), replace=False)

                    for t in train_selected:
                        valid_frames.remove(t)

                    validation_selected = np.random.choice(valid_frames, int(num_frames/100), replace=False)

                    self.split['train'][sequence].extend(train_selected)
                    self.split['validation'][sequence].extend(validation_selected)
                torch.save(self.split, split_path)
            else:
                for sequence in self.sequences:
                    num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))
                    total_for_sequence = int(num_frames/10)
                    print('--> TOTAL:', total_for_sequence)
                    train_selected = []
                    validation_selected = []

                    for v in np.arange(num_frames):
                        pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(v):06d}.bin')
                        label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(v):06d}.label')

                        if os.path.isfile(pcd_path) and os.path.isfile(label_path):
                            if len(train_selected) == 0:
                                train_selected.append(v)
                                last_added = v
                            elif len(train_selected) < total_for_sequence and (v-last_added) >= 5:
                                train_selected.append(v)
                                last_added = v
                                print(last_added)
                            else:
                                validation_selected.append(v)

                    validation_selected = np.random.choice(validation_selected, int(num_frames/100), replace=False)

                    self.split['train'][sequence].extend(train_selected)
                    self.split['validation'][sequence].extend(validation_selected)
                torch.save(self.split, split_path)
        # 如果默认的split路径存在
        else:
            # self.split = torch.load(split_path)
            self.split,self.count_train_val = get_sum_split(self.split_path,'validation')
            print('[INFO] TRAIN SEQUENCES', self.split['validation'].keys())
            print("[INFO] SEQUENCES TOTAL", self.count_train_val)        

    def __len__(self):
        return len(self.pcd_path)

    def calculate_sequences(self,idx):
        """
            calculate sequence num and frame num based on idx
        """
        i=0
        count_val = self.count_train_val
        sequence = None
        sequence_num = None
        while i < len(count_val):
            if idx.item()<count_val[0]:
                sequence_num = 0
                sequence = "00"
                break
            elif idx.item()<count_val[i+1]:
                sequence_num = i 
                sequence = self.sequences[i+1]
                break
            else:
                i+=1
        frame_num = self.split['validation'][sequence][idx-self.count_train_val[sequence_num]]
        return sequence,frame_num

    def clean_items(self,i:int):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = np.fromfile(label_tmp, dtype=np.uint32)
        label = label.reshape((-1))
        label = self.remap_lut_val[label].astype(np.int32)
        points = pcd[:, :3]
        if self.use_intensity:
            colors = pcd[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
        data = {'points': points, 'colors': colors, 'labels': label}

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        # print(f'[INFO] the shape of points {points.shape}, the shape of labels {labels.shape}')

        sampled_idx = np.arange(points.shape[0])
        if self.phase == 'train' and self.augment_data:
            sampled_idx = self.random_sample(points)
            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]

            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            points = homo_coords @ rigid_transformation.T[:, :3]
        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        # print(f'[INFO] the shape of points {points.shape}, the shape of labels {labels.shape}')

        quantized_coords, feats, labels, voxel_idx = ME.utils.sparse_quantize(
            coordinates=points,
            features=colors,
            labels=labels,
            ignore_label=vox_ign_label,
            quantization_size=self.voxel_size,
            return_index=True
        )

        # print(f'[INFO] the shape of points {quantized_coords.shape}, the shape of labels {labels.shape}')

        if isinstance(quantized_coords, np.ndarray):
            quantized_coords = torch.from_numpy(quantized_coords)

        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)

        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        if isinstance(voxel_idx, np.ndarray):
            voxel_idx = torch.from_numpy(voxel_idx)

        if sampled_idx is not None:
            sampled_idx = sampled_idx[voxel_idx]
            sampled_idx = torch.from_numpy(sampled_idx)
        else:
            sampled_idx = None
        
        return {"coordinates": quantized_coords,
                "features": feats,
                "labels": labels,
                "sampled_idx": sampled_idx,
                "idx": torch.tensor(i),
            }

    def adv_items(self,i:int):
        try:
            pcd_pattern = self.adv_pcd_path[i]
            matching_files = glob.glob(pcd_pattern)
            pcd_tmp = matching_files[0]

            label_pattern = self.adv_gt_path[i] if self.adv_gt_label else self.adv_label_path[i]
            matching_files = glob.glob(label_pattern)
            label_tmp = matching_files[0]

            point_cloud = o3d.io.read_point_cloud(pcd_tmp)
            points = np.asarray(point_cloud.points)
            labels = torch.load(label_tmp).int()
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
            data = {'points': points, 'colors': colors, 'labels': labels}

            points = data['points']
            colors = data['colors']
            labels = data['labels']
            sampled_idx = np.arange(points.shape[0])
            
            # print(f'[INFO] before quantized the shape of points {points.shape}, the shape of labels {labels.shape}')

            quantized_coords, feats, labels, voxel_idx = ME.utils.sparse_quantize(
                coordinates=points,
                features=colors,
                labels=labels,
                ignore_label=self.ignore_label,
                quantization_size=1,
                return_index=True
            )
            
            # print(f'[INFO] after quantized the shape of points {quantized_coords.shape}, the shape of labels {labels.shape}')

            if isinstance(quantized_coords, np.ndarray):
                quantized_coords = torch.from_numpy(quantized_coords)

            if isinstance(feats, np.ndarray):
                feats = torch.from_numpy(feats)

            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)

            if isinstance(voxel_idx, np.ndarray):
                voxel_idx = torch.from_numpy(voxel_idx)

            if sampled_idx is not None:
                sampled_idx = sampled_idx[voxel_idx]
                sampled_idx = torch.from_numpy(sampled_idx)
            else:
                sampled_idx = None
        
            return {"coordinates": quantized_coords,
                    "features": feats,
                    "labels": labels,
                    "sampled_idx": sampled_idx,
                    "idx": torch.tensor(i),
                }
        except Exception as e:
            # 打印异常信息
            print(f"[Error] Error happened in adversarial sample shape, idx: {i} {type(e).__name__}: {e}")
            return self.clean_items(i)

    def get_data(self, i: int):
        """
            获取原始点云数据
        """
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = np.fromfile(label_tmp, dtype=np.uint32)
        label = label.reshape((-1))
        label = self.remap_lut_val[label].astype(np.int32)
        points = pcd[:, :3]
        if self.use_intensity:
            colors = pcd[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
        data = {'points': points, 'colors': colors, 'labels': label}

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        return {"coordinates": points,
                "features": colors,
                "labels": labels,
                "idx": torch.tensor(i),
            }
    
    def __getitem__(self, i: int):
        if self.attacked_indices is not None and i in self.attacked_indices:
            return self.adv_items(i)
        return self.clean_items(i)
    

# CollationDataset, mix training dataset and validation dataset
# using for generating adversarial sample
class CollationDataset(BaseDataset):
    """
        整合trainning dataset and validation dataset
    """
    def __init__(
            self,
            version: str = 'full',
            phase=None,
            dataset_path: str = '/data/SynLiDAR/sequences',
            mapping_path: str = '_resources/synlidar_semantickitti.yaml',
            weights_path: str = '_weights/synlidar2semantickitti_correct.npy',
            voxel_size: float = 0.05,
            use_intensity: bool = False,
            augment_data: bool = False,
            sub_num: int = 80000,
            device: str = None,
            num_classes: int = 7,
            ignore_label: int = None,
        ):
        """
        collation training dataset and validation dataset
        """
        if weights_path is not None:
            weights_path = os.path.join(ABSOLUTE_PATH, weights_path)
        # initialization
        super().__init__(version=version,
                         phase=None,
                         dataset_path=dataset_path,
                         voxel_size=voxel_size,
                         sub_num=sub_num,
                         use_intensity=use_intensity,
                         augment_data=augment_data,
                         device=device,
                         num_classes=num_classes,
                         ignore_label=ignore_label,
                         weights_path=weights_path,
                         )
        self.name = 'SynLiDARDataset'
        if self.version == 'full':
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            self.get_splits()
        elif self.version == 'mini':
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            self.get_splits()
        elif self.version == 'sequential':
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            self.get_splits()
        else:
            raise NotImplementedError
        
        with open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r') as mapping_file:
            self.maps = yaml.safe_load(mapping_file)

        self.train_pcd_path = []
        self.train_label_path = []
        self.val_pcd_path = []
        self.val_label_path = []

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val

        phases = ['train','validation']
        for sequence, frames in self.split[phases[0]].items():
            for frame in frames:
                pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(frame):06d}.bin')
                label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(frame):06d}.label')
                self.train_pcd_path.append(pcd_path)
                self.train_label_path.append(label_path)
        for sequence, frames in self.split[phases[1]].items():
            for frame in frames:
                pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(frame):06d}.bin')
                label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(frame):06d}.label')
                self.val_pcd_path.append(pcd_path)
                self.val_label_path.append(label_path)
        print(f'[INFO]-->select {len(self.train_pcd_path)} pcd files for train, select {len(self.val_label_path)} pcd files for val')

        # pcd文件和label文件组合
        self.collation_pcd_path = self.train_pcd_path + self.val_pcd_path 
        self.collation_label_path = self.train_label_path + self.val_label_path

    def get_splits(self):
        if self.version == 'full':
            self.split_path = os.path.join(ABSOLUTE_PATH, '_splits/synlidar.pkl')
        elif self.version == 'mini':
            self.split_path = os.path.join(ABSOLUTE_PATH, '_splits/synlidar_mini.pkl')
        elif self.version == 'sequential':
            self.split_path = os.path.join(ABSOLUTE_PATH, '_splits/synlidar_sequential.pkl')
        else:
            raise NotImplementedError
        # 如果默认的split路径不存在
        if not os.path.isfile(self.split_path):
            self.split = {'train': {s: [] for s in self.sequences},
                          'validation': {s: [] for s in self.sequences}}
            if self.version != 'sequential':
                for sequence in self.sequences:
                    num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))
                    valid_frames = []

                    for v in np.arange(num_frames):
                        pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(v):06d}.bin')
                        label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(v):06d}.label')

                        if os.path.isfile(pcd_path) and os.path.isfile(label_path):
                            valid_frames.append(v)
                    if self.version == 'full':
                        train_selected = np.random.choice(valid_frames, int(num_frames/10), replace=False)
                    else:
                        train_selected = np.random.choice(valid_frames, int(num_frames/100), replace=False)

                    for t in train_selected:
                        valid_frames.remove(t)

                    validation_selected = np.random.choice(valid_frames, int(num_frames/100), replace=False)

                    self.split['train'][sequence].extend(train_selected)
                    self.split['validation'][sequence].extend(validation_selected)
                torch.save(self.split, self.split_path)
            else:
                for sequence in self.sequences:
                    num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))
                    total_for_sequence = int(num_frames/10)
                    print('--> TOTAL:', total_for_sequence)
                    train_selected = []
                    validation_selected = []

                    for v in np.arange(num_frames):
                        pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(v):06d}.bin')
                        label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(v):06d}.label')

                        if os.path.isfile(pcd_path) and os.path.isfile(label_path):
                            if len(train_selected) == 0:
                                train_selected.append(v)
                                last_added = v
                            elif len(train_selected) < total_for_sequence and (v-last_added) >= 5:
                                train_selected.append(v)
                                last_added = v
                                print(last_added)
                            else:
                                validation_selected.append(v)

                    validation_selected = np.random.choice(validation_selected, int(num_frames/100), replace=False)

                    self.split['train'][sequence].extend(train_selected)
                    self.split['validation'][sequence].extend(validation_selected)
                torch.save(self.split, self.split_path)
        # 如果默认的split路径存在
        else:
           # self.split = torch.load(split_path)
            self.split,self.count_train_val = get_sum_split(self.split_path,phase='train')
            self.split,self.count_validation_val = get_sum_split(self.split_path,phase='validation')
            print('[INFO] SEQUENCES', self.split.keys())
            print("[INFO] TRAIN SEQUENCES TOTAL", self.count_train_val)
            print("[INFO] VALIDATION SEQENCES TOTAL", self.count_validation_val)

    def get_num(self):
        return len(self.train_pcd_path),len(self.val_pcd_path)

    def __len__(self):
        return len(self.collation_pcd_path)
    
    def __getitem__(self, i: int):
        pcd_tmp = self.collation_pcd_path[i]
        label_tmp = self.collation_label_path[i]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = np.fromfile(label_tmp, dtype=np.uint32)
        label = label.reshape((-1))
        label = self.remap_lut_val[label].astype(np.int32)
        points = pcd[:, :3]
        if self.use_intensity:
            colors = pcd[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
        data = {'points': points, 'colors': colors, 'labels': label}

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        # print(f'[INFO] the shape of points {points.shape}, the shape of labels {labels.shape}')

        sampled_idx = np.arange(points.shape[0])
        if self.phase == 'train' and self.augment_data:
            sampled_idx = self.random_sample(points)
            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]

            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            points = homo_coords @ rigid_transformation.T[:, :3]
        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        # print(f'[INFO] the shape of points {points.shape}, the shape of labels {labels.shape}')

        quantized_coords, feats, labels, voxel_idx = ME.utils.sparse_quantize(
            coordinates=points,
            features=colors,
            labels=labels,
            ignore_label=vox_ign_label,
            quantization_size=self.voxel_size,
            return_index=True
        )

        # print(f'[INFO] the shape of points {quantized_coords.shape}, the shape of labels {labels.shape}')

        if isinstance(quantized_coords, np.ndarray):
            quantized_coords = torch.from_numpy(quantized_coords)

        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)

        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        if isinstance(voxel_idx, np.ndarray):
            voxel_idx = torch.from_numpy(voxel_idx)

        if sampled_idx is not None:
            sampled_idx = sampled_idx[voxel_idx]
            sampled_idx = torch.from_numpy(sampled_idx)
        else:
            sampled_idx = None
        
        return {"coordinates": quantized_coords,
                "features": feats,
                "labels": labels,
                "sampled_idx": sampled_idx,
                "idx": torch.tensor(i),
            }

    def get_data(self, i: int):
        pcd_tmp = self.collation_pcd_path[i]
        label_tmp = self.collation_label_path[i]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = np.fromfile(label_tmp, dtype=np.uint32)
        label = label.reshape((-1))
        label = self.remap_lut_val[label].astype(np.int32)
        points = pcd[:, :3]
        if self.use_intensity:
            colors = pcd[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
        data = {'points': points, 'colors': colors, 'labels': label}

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        return {"coordinates": points,
                "features": colors,
                "labels": labels,
                "idx": torch.tensor(i),
            }
        

