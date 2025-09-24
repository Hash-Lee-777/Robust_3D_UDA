# class adversarial synlidar dataset, using for training model
from utils.datasets.dataset import BaseDataset
import os
import torch
import yaml
import numpy as np
import pdb
import MinkowskiEngine as ME
import open3d as o3d
import concurrent.futures
import time
import glob
# time record
def timeit(func):
    """
        ":param func" 需要传入的函数
        ":return:"
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

# calculated adversarial coords path and labels path
@timeit
def calculate_index(adv_points_path: str='experiments/SAVEDIR/attackedLidar_Kitti100bck'):
    """
        access to the path of adversarial coords and lables
    """
    attacked_indices = np.array([])
    attacked_points_path = []
    attacked_labels_path = []
    adv_coords_path = adv_points_path + '/adv_preds'
    adv_labels_path = adv_points_path + '/adv_labels_pt'
    split_path =  os.path.join(ABSOLUTE_PATH, '_splits/synlidar.pkl')
    split = torch.load(split_path)
    sequences = ["00","01","02","03","04","05","06","07","08","09","10","11","12"]
    count_train_val=[293, 1719, 2694, 3681, 5350, 6841, 11348, 13306, 14452, 16194, 17292, 18717, 19834]
    for i,sequence in enumerate(sequences):
        adversarial_coords_path  = os.path.join(adv_coords_path,sequence) 
        sequence_arr = np.array(split['train'][sequence])
        if os.path.exists(adversarial_coords_path):
            for filename in os.listdir(adversarial_coords_path):
                frame_num = eval(filename.split('_')[0]) # 12365 293->0
                # score = eval(filename.split('_')[1].split('.')[0])
                indices = np.where(sequence_arr==frame_num) # 293-0,294-1,...293+23-23
                if i==0:
                    idx = indices[0]
                else:
                    idx = indices[0]+count_train_val[i-1]
                attacked_indices = np.append(attacked_indices,idx)
                attacked_points_path.append(adv_coords_path+'/'+sequence+'/'+filename)
                pattern = os.path.join(adv_labels_path+'/'+sequence+'/'+f'{frame_num}_*.pt')
                matching_files = glob.glob(pattern)
                attacked_labels_path.append(matching_files[0])

    # 计算对抗点云所在路径
    attacked_points_path_dict = dict(zip(attacked_indices.tolist(),attacked_points_path))
    attacked_labels_path_dict = dict(zip(attacked_indices.tolist(),attacked_labels_path))
    return attacked_indices.astype(np.int32),attacked_points_path_dict,attacked_labels_path_dict

# load adversarial coords 
def load_coords_cloud(adv_point_path):
    point_cloud = o3d.io.read_point_cloud(adv_point_path)
    coords = np.asarray(point_cloud.points)
    return torch.Tensor(coords).int()

# load adversrial labels
def load_labels_cloud(adv_label_path):
    return torch.load(adv_label_path).int()

# return attacked_indices, adv_coords_dict, adv_labels_dict of adversarial points
@timeit
def adv_all_dict(attacked_indices,attacked_coords_path,attacked_labels_path):
    # Test adversarial sample load
    adv_coords_dict = {}
    adv_labels_dict = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 加载点云数据
        coords_futures = {executor.submit(load_coords_cloud,attacked_coords_path[idx]): idx for idx in attacked_indices}
        labels_futures = {executor.submit(load_labels_cloud,attacked_labels_path[idx]): idx for idx in attacked_indices}
        for future in concurrent.futures.as_completed(coords_futures):
            idx = coords_futures[future]
            try:
                coords = future.result()
                adv_coords_dict[idx] = coords
            except Exception as exc:
                print(f"Failed to load point cloud for index {idx}:{exc}")
        for future in concurrent.futures.as_completed(labels_futures):
            idx = labels_futures[future]
            try:
                labels = future.result()
                adv_labels_dict[idx] = labels
            except Exception as exc:
                print(f'Failed to load point cloud for index {idx}:{exc}')
    return attacked_indices,adv_coords_dict,adv_labels_dict

# get adversarial coords dictionary and adversarial labels dictionary
@timeit
def get_adv_dict(path: str='experiments/SAVEDIR/attackedLidar_Poss100'):
    # 获取对抗样本坐标和标签的路径
    attacked_indices,attacked_points_path,attacked_labels_path= calculate_index(path)
    attacked_indices,adv_coords_dict,adv_labels_dict=adv_all_dict(
        attacked_indices=attacked_indices,
        attacked_coords_path=attacked_points_path,
        attacked_labels_path=attacked_labels_path
    )
    return attacked_indices,adv_coords_dict,adv_labels_dict

# randomly choose adversarial samples with probability (10%,30%,50%,100%)
@timeit
def get_adv_dict_random(prob:float=1.,path: str='experiments/SAVEDIR/attackedLidar_Poss100'):
    """
        prob: the probablity of choosing
        path: the path for adversarial sample
    """
    attacked_indices = None
    randomized_adv_coords_dict = {}
    randomized_adv_labels_dict = {}
    if prob<=1. and prob>=0.:
        attacked_indices,adv_coords_dict,adv_labels_dict = get_adv_dict(path)
        # 计算需要扰动的点的数量
        num_points = int(len(attacked_indices)*prob)
        random_indices = np.random.choice(attacked_indices,size=num_points,replace=False)
        for index in random_indices:
            randomized_adv_coords_dict[index] = adv_coords_dict[index]
            randomized_adv_labels_dict[index] = adv_labels_dict[index]
        # 生成采样的随机扰动
        return random_indices,randomized_adv_coords_dict,randomized_adv_labels_dict
    else:
        return random_indices,randomized_adv_coords_dict,randomized_adv_labels_dict

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

class AdvSynLiDARDataset(BaseDataset):
    def __init__(self,
                 version: str = 'full',
                 phase: str = 'train',
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
                 attacked_indices=None,
                 adv_coords_dict=None,
                 adv_labels_dict=None):

        if weights_path is not None:
            weights_path = os.path.join(ABSOLUTE_PATH, weights_path)
        # initialization
        self.attacked_indices = attacked_indices
        self.adv_coords_dict = adv_coords_dict
        self.adv_labels_dict = adv_labels_dict
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
        self.count_train_val=[293, 1719, 2694, 3681, 5350, 6841, 11348, 13306, 14452, 16194, 17292, 18717, 19834]

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

        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))

        self.pcd_path = []
        self.label_path = []
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
                self.pcd_path.append(pcd_path)
                self.label_path.append(label_path)
                # self.frames_record.append(frame)
                # self.sequences_record.append(sequence)
                

        print(f'--> Selected {len(self.pcd_path)} for {self.phase}')
      
    def get_splits(self):
        if self.version == 'full':
            split_path = os.path.join(ABSOLUTE_PATH, '_splits/synlidar.pkl')
        elif self.version == 'mini':
            split_path = os.path.join(ABSOLUTE_PATH, '_splits/synlidar_mini.pkl')
        elif self.version == 'sequential':
            split_path = os.path.join(ABSOLUTE_PATH, '_splits/synlidar_sequential.pkl')
        else:
            raise NotImplementedError

        if not os.path.isfile(split_path):
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

        else:
            self.split = torch.load(split_path)
            print('SEQUENCES', self.split.keys())
            print('TRAIN SEQUENCES', self.split['train'].keys())

    def __len__(self):
        return len(self.pcd_path)
    
    def __getitem__(self, i: int):
        # for i in attacked_indices, using the adversarial coordinates and adversrial labels
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

        quantized_coords, feats, labels, voxel_idx = ME.utils.sparse_quantize(points,
                                                                               colors,
                                                                               labels=labels,
                                                                               ignore_label=vox_ign_label,
                                                                               quantization_size=self.voxel_size,
                                                                               return_index=True)

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

        if i in self.attacked_indices.astype(np.int32):
            # if indices of adversarial sample exists
            adv_coords = self.adv_coords_dict[i]
            adv_labels = self.adv_labels_dict[i]
            adv_feats = torch.ones_like(adv_labels).unsqueeze(1).float()
            return{
                "coordinates": adv_coords,
                "features": adv_feats,
                "labels": adv_labels,
                "sampled_idx": sampled_idx,
                "idx": torch.tensor(i),
            }
        return {"coordinates": quantized_coords,
                "features": feats,
                "labels": labels,
                "sampled_idx": sampled_idx,
                "idx": torch.tensor(i),
            }

    def get_data(self, i: int):
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
    
