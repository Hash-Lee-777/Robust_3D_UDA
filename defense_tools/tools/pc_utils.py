# analysis feature propagation during training
import numpy as np
import open3d as o3d
import torch
import time
from torch.utils.data import DataLoader, TensorDataset
import math
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def vis(disp=False):
    """
        可视化装饰器
    """
    def _decorate(func):
        def calculate_vis_scope(pcd):
            # 获取点云数据的中心点
            if isinstance(pcd, tuple):
                pcd = pcd[0]
            pcd_legacy = pcd.to_legacy()
            center = pcd_legacy.get_center()
            # 计算点云数据的范围
            points = np.asarray(pcd_legacy.points)
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            extent = max(max_bound - min_bound) / 2
            # 设置视角为正前方
            lookat = center
            up = [0, 1, 1]
            front = [0, 0, 1]
            zoom = 0.3
            return zoom, front, lookat, up

        def visualization(pcd):
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            # 将点云添加到可视化窗口
            vis.add_geometry(pcd.to_legacy())
            ctr = vis.get_view_control()
            ctr.set_lookat(pcd.to_legacy().get_center())
            ctr.set_up([0, 1, 1])
            ctr.set_front([0, 0, 1])
            ctr.set_zoom(0.3)
            vis.run()
            vis.destroy_window()

        def _wrap(*args, **kwargs):
            pcd = func(*args, **kwargs)
            if disp:
                print(
                    f"[INFO] Visualization Information: The point cloud from the function: {func.__name__}"
                )
                visualization(pcd)
            return pcd

        return _wrap

    return _decorate

def visualization():
    return

def load_model(checkpoint_path, model):
    """ 
    加载模型
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

def timeit(func):
    def _wrap(*args,**kwargs):
        time1 = time.time()
        result = func(*args,**kwargs)
        time2 = time.time()
        print(f'[INFO] time recording for function {func.__name__} execution, the time cost is {time2-time1}')
        return result
    return _wrap 

# Batch and Nomalization tools

def batch_point_clouds(point_cloud, num_pts=4096):
    """
    将输入的点云数据分成多个批次，并返回每个批次的点云数据及其对应的原始索引。

    Args:
        point_cloud (torch.Tensor): 输入的点云数据，形状为 [N, 3]。
        num_pts (int): 每个批次的点数。

    Returns:
        batched_point_clouds (torch.Tensor): 分成批次的点云数据，形状为 [B, num_pts, 3]。
        original_indices (torch.Tensor): 每个批次对应的原始索引。
    """
    # 确保输入的点云数据是 [N, 3]
    assert point_cloud.ndimension() == 2 and point_cloud.shape[1] == 3, "点云数据必须是 [N, 3] 形状"

    num_points = len(point_cloud)
    num_batches = (num_points + num_pts - 1) // num_pts  # 计算批次数

    # 计算需要补充的点的数量，使每个批次的点云数量一致
    remainder = num_pts * num_batches - num_points
    if remainder > 0:
        additional_indices = torch.randint(0, num_points, size=(remainder,))
        additional_points = point_cloud[additional_indices]
        point_cloud = torch.cat([point_cloud, additional_points], dim=0)

    # 创建原始索引
    indices = torch.arange(len(point_cloud))

    # 创建数据集和数据加载器
    dataset = TensorDataset(point_cloud, indices)
    data_loader = DataLoader(dataset, batch_size=num_pts, shuffle=False)

    batch_point_clouds = []  # 存储每个批次的点云数据
    original_indices = []    # 存储每个批次对应的原始索引

    # 遍历数据加载器
    for i, (batch_data, indices) in enumerate(data_loader):
        batch_point_clouds.append(batch_data)  # 提取点云数据，第一个维度为批次维度
        original_indices.append(indices)       # 提取原始索引

    # 将每个批次的点云数据合并为一个张量
    batched_point_clouds = torch.stack(batch_point_clouds, dim=0)

    # 将每个批次对应的原始索引合并为一个张量
    original_indices = torch.stack(original_indices, dim=0)

    return batched_point_clouds, original_indices

def normalize_point_cloud(coords):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(coords)
    
    # 计算点云的均值
    mean = np.mean(coords, axis=0)
    
    # 将点云中心移动到原点
    centered_coords = coords - mean
    
    # 计算尺度因子，将点云进行归一化
    max_bound = np.max(centered_coords, axis=0)
    min_bound = np.min(centered_coords, axis=0)
    scale_factor = 1.0 / np.max(max_bound - min_bound)
    normalized_coords = centered_coords * scale_factor
    
    point_cloud.points = o3d.utility.Vector3dVector(normalized_coords)
    return point_cloud

def remove_duplicates(point_cloud):
    unique_points = torch.unique(point_cloud, dim=0)
    return unique_points

def normalize_pcd_tensor(coords:torch.Tensor):
    # 计算点云的均值
    coords = coords.float()
    mean = torch.mean(coords, dim=0)
    
    # 将点云中心移动到原点
    centered_coords = coords - mean.unsqueeze(0)
    
    # 计算尺度因子，将点云进行归一化
    max_bound, _ = torch.max(centered_coords, dim=0)
    min_bound, _ = torch.min(centered_coords, dim=0)
    scale_factor = 1.0 / torch.max(max_bound - min_bound)
    normalized_coords = centered_coords * scale_factor.unsqueeze(0)
    
    return normalized_coords

def load_former_params(checkpoint_path,model):
    """ 
    加载模型
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
    model.load_state_dict(ckpt, strict=False)
    return model

# calculate support radius
def calculate_support_radius(noise_patch_pts):
    device = noise_patch_pts.device
    noise_patch_pts = noise_patch_pts.detach().cpu().numpy()
    support_radius = np.linalg.norm(noise_patch_pts.max(0) - noise_patch_pts.min(0), 2) / noise_patch_pts.shape[0]
    support_radius = np.expand_dims(support_radius, axis=0)
    return torch.from_numpy(support_radius).to(device)

# PCA alignment tools, 估计点云的法线, 耗时比较长
def estimate_normals_gpu(points, k=5):
    """
    Estimate normals for a point cloud using PCA on GPU.

    Args:
        points (np.ndarray): A point cloud, size [N, 3].
        k (int): Number of nearest neighbors to use for normal estimation.

    Returns:
        np.ndarray: Normals for the point cloud, size [N, 3].
    """
    points_tensor = torch.from_numpy(points).float().cuda()
    # points_tensor = torch.from_numpy(points).float()
    N, _ = points.shape
    normals = torch.zeros((N, 3), dtype=points_tensor.dtype, device=points_tensor.device)

    for i in range(N):
        # Get the neighboring points
        neighbor_distances, neighbor_indices = torch.topk(torch.norm(points_tensor - points_tensor[i], dim=1), k, largest=False)
        neighbors = points_tensor[neighbor_indices]

        # Subtract mean
        neighbors -= torch.mean(neighbors, dim=0)

        # Compute covariance matrix
        cov_matrix = torch.matmul(neighbors.T, neighbors) / neighbors.shape[0]

        # Perform PCA (eigenvalue decomposition)
        eigvals, eigvecs = torch.linalg.eigh(cov_matrix)

        # The normal is the eigenvector corresponding to the smallest eigenvalue
        normal = eigvecs[:, 0]
        normals[i] = normal

    # Normalize normals
    normals /= torch.norm(normals, dim=1, keepdim=True)

    return normals.cpu().numpy()

def patch_sampling(patch_pts, sample_num):

    if patch_pts.shape[0] > sample_num:

        sample_index = np.random.choice(range(patch_pts.shape[0]), sample_num, replace=False)

    else:

        sample_index = np.random.choice(range(patch_pts.shape[0]), sample_num)

    return sample_index

def rotation_matrix(axis, theta):

    # Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.

    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.matrix(np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]]))

def compute_roatation_matrix(sour_vec, dest_vec, sour_vertical_vec=None):
    # http://immersivemath.com/forum/question/rotation-matrix-from-one-vector-to-another/
    if np.linalg.norm(np.cross(sour_vec, dest_vec), 2) == 0 or np.abs(np.dot(sour_vec, dest_vec)) >= 1.0:
        if np.dot(sour_vec, dest_vec) < 0:
            return rotation_matrix(sour_vertical_vec, np.pi)
        return np.identity(3)
    alpha = np.arccos(np.dot(sour_vec, dest_vec))
    a = np.cross(sour_vec, dest_vec) / np.linalg.norm(np.cross(sour_vec, dest_vec), 2)
    c = np.cos(alpha)
    s = np.sin(alpha)
    R1 = [a[0] * a[0] * (1.0 - c) + c,
          a[0] * a[1] * (1.0 - c) - s * a[2],
          a[0] * a[2] * (1.0 - c) + s * a[1]]

    R2 = [a[0] * a[1] * (1.0 - c) + s * a[2],
          a[1] * a[1] * (1.0 - c) + c,
          a[1] * a[2] * (1.0 - c) - s * a[0]]

    R3 = [a[0] * a[2] * (1.0 - c) - s * a[1],
          a[1] * a[2] * (1.0 - c) + s * a[0],
          a[2] * a[2] * (1.0 - c) + c]

    R = np.array([R1, R2, R3])

    return R

def get_principle_dirs(pts):

    pts_pca = PCA(n_components=3)
    pts_pca.fit(pts)
    principle_dirs = pts_pca.components_
    principle_dirs /= np.linalg.norm(principle_dirs, 2, axis=0)

    return principle_dirs

def pca_alignment(pts, random_flag=False):
    pca_dirs = get_principle_dirs(pts)

    if random_flag:
        pca_dirs *= np.random.choice([-1, 1], 1)

    rotate_1 = compute_roatation_matrix(pca_dirs[2], [0, 0, 1], pca_dirs[1])
    pca_dirs = np.array(rotate_1 @ pca_dirs.T).T  # 替换 * 为 @
    rotate_2 = compute_roatation_matrix(pca_dirs[1], [1, 0, 0], pca_dirs[2])
    pts = np.array(rotate_2 @ rotate_1 @ pts.T).T  # 替换 * 为 @

    inv_rotation = np.linalg.inv(rotate_2 @ rotate_1)  # 替换 * 为 @

    return pts, inv_rotation

# 调整点云大小，使其匹配
def resize_point_clouds(noise_coords,origin_coords):
    noise_len = len(noise_coords)
    origin_len = len(origin_coords)
    if noise_len == origin_len:
        return noise_coords,origin_coords
    
    target_len = min(noise_len,origin_len)
    # 如果噪声点云更长，则从噪声点云中随机选择与原始点云相同长度的点
    if noise_len > origin_len:
        indices = np.random.choice(noise_len, target_len, replace=False)
        resized_noise_coords = noise_coords[indices]
        resized_origin_coords = origin_coords
    # 如果原始点云更长，则从原始点云中随机选择与噪声点云相同长度的点
    else:
        indices = np.random.choice(origin_len, target_len, replace=False)
        resized_noise_coords = noise_coords
        resized_origin_coords = origin_coords[indices]
    
    return resized_noise_coords, resized_origin_coords

def batch_sampling(noise_coords,origin_coords,pts_num:float=4096*2):
        pts_len = min(pts_num,noise_coords.shape[0])
        if pts_len > pts_num:
            sample_index = np.random.choice(np.arange(noise_coords.shape[0]), pts_len, replace=False)
        else:
            sample_index = np.random.choice(np.arange(noise_coords.shape[0]), pts_len, replace=False)
        sample_index = np.sort(sample_index)
        return sample_index

# 处理点云得到处理后的噪声点云, 原始点云, 法线以及支持半径
def process_point_clouds(origin_coords, noise_coords, patch_radius:float = 750,pts_num=4096*2):
    device = origin_coords.device

    origin_coords = origin_coords.detach().cpu().numpy()
    noise_coords = noise_coords.detach().cpu().numpy()

    # 调整noise_coords和origin_coords的size大小,原始size 70000, 调整size pts_num
    sample_idx = batch_sampling(noise_coords,origin_coords,pts_num=pts_num)
    noise_coords = noise_coords[sample_idx]
    origin_coords = origin_coords[sample_idx]
    # noise_coords,origin_coords = resize_point_clouds(noise_coords,origin_coords)

    # 计算噪声点云的补丁
    noise_center = noise_coords.mean(axis=0)
    noise_patch_idx = np.linalg.norm(noise_coords - noise_center, axis=1) < patch_radius

    if len(noise_patch_idx) < 3:
        return None

    noise_patch_pts = noise_coords[noise_patch_idx] - noise_center

    # 对噪声点云进行 PCA 对齐
    noise_patch_pts, noise_patch_inv = pca_alignment(noise_patch_pts)
    noise_patch_pts /= patch_radius

    # 使用相同的对齐矩阵对原始点云进行对齐
    origin_patch_pts = origin_coords[noise_patch_idx] - noise_center
    origin_patch_pts /= patch_radius
    origin_patch_pts = np.array(np.linalg.inv(noise_patch_inv) @ origin_patch_pts.T).T

    # 估计法线
    normals = estimate_normals_gpu(noise_patch_pts)

    # 计算支持半径
    support_radius = np.linalg.norm(noise_patch_pts.max(0) - noise_patch_pts.min(0), 2) / noise_patch_pts.shape[0]
    support_radius = np.expand_dims(support_radius, axis=0)

    return {
        'nosie_patch_pts': torch.from_numpy(noise_patch_pts).float(),
        'origin_patch_pts': torch.from_numpy(origin_patch_pts).float(),
        'support_radius': torch.from_numpy(support_radius).float(),
        'normals': torch.from_numpy(normals)
    } 



utils_tools_mappings = {
    'load_model': load_model,
    'timeit': timeit,
    'batch_point_clouds':batch_point_clouds,
    'normalize_point_cloud': normalize_point_cloud,
    'remove_duplicates': remove_duplicates,
    'normalize_pcd_tensor': normalize_pcd_tensor,
    'load_former_params': load_former_params,
    # PCA Tools
    'get_principle_dirs': get_principle_dirs,
    'pca_alignment':pca_alignment,
    'patch_sampling':patch_sampling,
    'rotation_matrix':rotation_matrix,
    'compute_roatation_matrix':compute_roatation_matrix,
    'process_point_clouds':process_point_clouds


}