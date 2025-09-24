import torch
import torch.distributions
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import torch.linalg

def smooth_l1_loss(pred, target, beta=1.0):
    diff = torch.abs(pred - target)
    smooth_l1_loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return torch.mean(smooth_l1_loss)
# Custom function to compute covariance matrix
def compute_covariance_matrix(coords):
    """
    Compute the covariance matrix of point cloud data
    :param coords: Input coordinates, shape [N, D]
    :return: Covariance matrix, shape [D, D]
    """
    coords_mean = coords.mean(dim=0)
    coords_centered = coords - coords_mean
    covariance_matrix = torch.matmul(coords_centered.T, coords_centered) / (coords.shape[0] - 1)
    return covariance_matrix

# 找到最近的正定矩阵的函数，并在对角线上添加一个小值
def nearest_positive_definite(matrix):

    sym_matrix = (matrix + matrix.T) / 2
    sym_matrix += torch.eye(sym_matrix.size(0), device='cpu') * 1e-12 # 在对角线上添加一个小值
    
    if not torch.isfinite(sym_matrix).all():
        # 处理包含NaN或无穷大值的情况
        sym_matrix = torch.nan_to_num(sym_matrix, nan=0., posinf=1e6, neginf=-1e6)
    
    try:
        u, s, v = torch.svd(sym_matrix)
    except RuntimeError as e:
        print(f"SVD过程中出错: {e}")
        print(f"矩阵: {sym_matrix}")
        raise
    
    # 计算最近的正定矩阵
    H = torch.mm(v, torch.mm(torch.diag(s), v.T))
    nearest_pd = (sym_matrix + H) / 2
    
    # 在对角线上添加一个小值，确保矩阵为正定
    nearest_pd += torch.eye(nearest_pd.size(0), device='cpu') * 1e-14
    
    return nearest_pd

def kl_divergence_loss(output, target):
    # 确保所有张量在CPU上
    output = output.cpu()
    target = target.cpu()
    
    original_mean = torch.mean(output, dim=0)
    original_cov = compute_covariance_matrix(output)
    target_mean = torch.mean(target, dim=0)
    target_cov = compute_covariance_matrix(target)

    original_cov = nearest_positive_definite(original_cov)
    target_cov = nearest_positive_definite(target_cov)
    
    original_dist = MultivariateNormal(original_mean, covariance_matrix=original_cov)
    target_dist = MultivariateNormal(target_mean, covariance_matrix=target_cov)
    
    kl_div = torch.distributions.kl_divergence(original_dist, target_dist)
    return kl_div

# 内存消耗过大，采用下采样
def earth_movers_distance(x1, x2,num_pts=4096):
    """
    计算两个点云之间的 Earth Mover's Distance (EMD)

    参数:
        x1: 第一个点云, 形状为 (batch_size, num_points, num_dims)
        x2: 第二个点云, 形状为 (batch_size, num_points, num_dims)

    返回:
        emd: Earth Mover's Distance, 形状为 (batch_size,)
    """
    batch_size, num_points, num_dims = x1.shape

    # 确定采样比例
    sample_ratio = min(num_pts / num_points, 1.0)

    # 下采样
    sampled_points = int(num_points * sample_ratio)
    indices = torch.randperm(num_points)[:sampled_points]
    x1_sampled = x1[:, indices, :]
    x2_sampled = x2[:, indices, :]

    # 计算每个点与每个点之间的距离矩阵
    distances = torch.cdist(x1_sampled.view(batch_size * sampled_points, num_dims), 
                             x2_sampled.view(batch_size * sampled_points, num_dims))

    # 使用动态规划算法计算 Earth Mover's Distance
    emd = torch.zeros(batch_size, device=x1.device)
    for i in range(batch_size):
        c = distances[i].detach().cpu().numpy()  # 转换为 NumPy 数组
        n = c.shape[0]

        # 初始化动态规划矩阵
        dp = torch.zeros((n + 1, n + 1), device=x1.device)
        dp[0, 1:] = float('inf')
        dp[1:, 0] = float('inf')

        # 动态规划求解
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                dp[i, j] = c[i - 1, j - 1] + torch.min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

        emd[i] = dp[n, n]

    return emd.mean()  # 取平均值作为最终的 Earth Mover's Distance

# 最远点采样算法
def farthest_point_sample(points,num_samples=4096):
    B, N, C = points.shape
    centroids = torch.zeros(B, num_samples, dtype=torch.long).to(points.device)
    distances = torch.ones(B, N).to(points.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(points.device)

    batch_indices = torch.arange(B, dtype=torch.long).to(points.device)
    for i in range(num_samples):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest].view(B, 1, C)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = torch.max(distances, -1)[1]

    sampled_points = points[batch_indices[:, None], centroids]
    return sampled_points

# def chamfer_distance_batch(pred, target, num_pts=4096):
#     """
#     计算预测点云和目标点云之间的 Chamfer 距离

#     参数:
#         pred: 预测点云，形状为 (batch_size, num_points, num_dims)
#         target: 目标点云，形状为 (batch_size, num_points, num_dims)
#         num_pts: 采样点的数量 (默认值: 4096)

#     返回:
#         chamfer_dist: Chamfer 距离
#     """
#     chamfer_dists = []  # 存储每个批次的 Chamfer 距离

#     for i in range(len(pred)):
#         pred_sampled = pred[i]
#         target_sampled = target[i]

#         # 计算预测点到目标点的距离矩阵
#         pred_to_target_dist = torch.cdist(pred_sampled, target_sampled)
#         # 计算目标点到预测点的距离矩阵
#         target_to_pred_dist = torch.cdist(target_sampled, pred_sampled)

#         # 计算对称 Chamfer 距离
#         chamfer_dist = torch.mean(torch.min(pred_to_target_dist, dim=1)[0]) + torch.mean(torch.min(target_to_pred_dist, dim=1)[0])
#         chamfer_dists.append(chamfer_dist)  # 使用item()将张量转换为标量

#     # 返回每个批次的 Chamfer 距离的总和，保留计算图
#     return torch.stack(chamfer_dists).sum()
def chamfer_distance_batch(pred, target):
    """
    计算预测点云和目标点云之间的对称 Chamfer 距离, 点云数量过多,容易造成内存爆炸

    参数:
        pred: 预测点云，形状为 (batch_size, num_points, num_dims)
        target: 目标点云，形状为 (batch_size, num_points, num_dims)

    返回:
        chamfer_dist: Chamfer 距离
    """
    # 计算预测点到目标点的距离矩阵
    pred_to_target_dist = torch.cdist(pred, target)  # (batch_size, num_points_pred, num_points_target)
    # 计算目标点到预测点的距离矩阵
    target_to_pred_dist = torch.cdist(target, pred)  # (batch_size, num_points_target, num_points_pred)
    
    # 计算对称 Chamfer 距离
    chamfer_dist = torch.mean(torch.min(pred_to_target_dist, dim=2)[0], dim=1) + \
                   torch.mean(torch.min(target_to_pred_dist, dim=2)[0], dim=1)
    
    return torch.mean(chamfer_dist)

# compute_bilateral_loss_with_repulsion
# |Paper 'PointFilter'| pred_point_batch [B,4096,3] | gt_patch_pts [B,4096,3] |
def compute_bilateral_loss_with_repulsion(pred_point, gt_patch_pts, gt_patch_normals, support_radius, support_angle, alpha):
    """
    计算带有排斥项的双边损失
    :param pred_point: 预测点云, 形状为 [B, 4096, 3]
    :param gt_patch_pts: 真实点云块, 形状为 [B, 4096, 3]
    :param gt_patch_normals: 真实点云块的法向量, 形状为 [B, 4096, 3]
    :param support_radius: 支持半径
    :param support_angle: 支持角度
    :param alpha: 权重因子
    :return: 损失值
    """
    B, N, _ = pred_point.size()

    # 扩展pred_point以进行元素级操作
    pred_point_expanded = pred_point.unsqueeze(2)  # [B, 4096, 1, 3]
    gt_patch_pts_expanded = gt_patch_pts.unsqueeze(1)  # [B, 1, 4096, 3]

    # 计算欧氏距离的平方
    dist_square = ((pred_point_expanded - gt_patch_pts_expanded) ** 2).sum(3)  # [B, 4096, 4096]
    weight_theta = torch.exp(-1 * dist_square / (support_radius ** 2))

    # 找到最近的真实点的索引
    nearest_idx = torch.argmin(dist_square, dim=2)  # [B, 4096]

    # 根据最近点的索引，从gt_patch_normals中提取对应的法向量
    pred_point_normal = torch.stack([gt_patch_normals[b, nearest_idx[b]] for b in range(B)])  # [B, 4096, 3]

    # 扩展法向量以进行元素级操作
    pred_point_normal_expanded = pred_point_normal.unsqueeze(2)  # [B, 4096, 1, 3]
    gt_patch_normals_expanded = gt_patch_normals.unsqueeze(1)  # [B, 1, 4096, 3]

    # 计算法向量的投影距离
    normal_proj_dist = (pred_point_normal_expanded * gt_patch_normals_expanded).sum(3)  # [B, 4096, 4096]
    weight_phi = torch.exp(-1 * ((1 - normal_proj_dist) / (1 - torch.cos(torch.tensor(support_angle))))**2)

    # 避免除以零并归一化权重
    weight = weight_theta * weight_phi + 1e-12
    weight = weight / weight.sum(2, keepdim=True)

    # 计算投影距离
    project_dist = torch.abs(((pred_point_expanded - gt_patch_pts_expanded) * gt_patch_normals_expanded).sum(3))  # [B, 4096, 4096]
    imls_dist = (project_dist * weight).sum(2)  # [B, 4096]

    # 计算排斥损失
    max_dist = torch.max(dist_square, 2)[0]  # [B, 4096]

    # 计算最终损失
    dist = torch.mean(alpha * imls_dist + (1 - alpha) * max_dist)  # [B, 4096]

    return dist.mean()  # 返回所有批次的平均损失


loss_dict_mappings = {
    # loss tools
    'smooth_l1_loss': smooth_l1_loss,
    'kl_divergence_loss': kl_divergence_loss,
    'emd_loss': earth_movers_distance,
    'chamfer_distance_batch':chamfer_distance_batch,
    'compute_bilateral_loss_with_repulsion':compute_bilateral_loss_with_repulsion,
}