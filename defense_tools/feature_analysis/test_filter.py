import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置图像分辨率
# plt.figure(figsize=(12, 8), dpi=100)

synlidar2kitti = np.array(['car', 'bicycle', 'motorcycle',  'truck', 'other-vehicle', 'person',
                           'bicyclist', 'motorcyclist',
                           'road', 'parking', 'sidewalk', 'other-ground',
                           'building', 'fence', 'vegetation', 'trunk',
                           'terrain', 'pole', 'traffic-sign'])

synlidar2poss = np.array(['person', 'rider', 'car', 'trunk',
                          'plants', 'traffic-sign', 'pole', 'garbage-can',
                          'building', 'cone', 'fence', 'bike', 'ground'])

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
                                    (173, 23, 121)])/255.   # traffic-sign

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


# load adversrial labels
def load_labels_cloud(adv_label_path):
    return torch.load(adv_label_path).int()

# point cloud visualization, 写成一个装饰器，可视化与pcd相关的方法
def vis(func,
        # zoom=0.3,
        # front=[0,1,1],
        # lookat=[2.6172, 2.0475, 1.532],
        # up=[-0.0694, -0.9768, 0.2024]
    ):

    def calculate_vis_scope(pcd):
        # 获取点云数据的中心点
        if isinstance(pcd,tuple):
            pcd = pcd[0]
        pcd_legacy = pcd.to_legacy()
        center = pcd_legacy.get_center()
        # 计算点云数据的范围
        points = np.asarray(pcd_legacy.points)
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        extent = max(max_bound - min_bound) / 2
        # 设置视角为正前方，lookat为点云数据的中心点
        lookat = center
        up = [0, 1, 1]  # 或者 [0, 1, 0]，取决于希望的上方向
        front = [0, 0, 1]  # 正前方
        # zoom = 1.0/extent
        zoom = 0.3
        return zoom,front,lookat,up
    
    def visualization(pcd):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # 将点云添加到可视化窗口
        vis.add_geometry(pcd.to_legacy())
        ctr = vis.get_view_control()
        ctr.set_lookat(pcd.to_legacy().get_center())  # 设置视角中心
        ctr.set_up([0, 1, 1])  # 设置视角方向
        ctr.set_front([0, 0, 1])
        ctr.set_zoom(0.3)  # 设置缩放
        vis.run()
        vis.destroy_window()

    def _wrap(*args,**kwargs):
        pcd = func(*args,**kwargs)
        # zoom,front,lookat,up = calculate_vis_scope(pcd)
        # print("[INFO] Visualization Information:\n The point cloud from the function :{}\n The zoom setting: {}\n The front setting: {}\n The lookat setting: {}\n The up setting: {}".format(func.__name__,zoom,front,lookat,up))
        if isinstance(pcd,tuple):
            pcd = pcd[0]
        print("[INFO] Visualization Information:\n The point cloud from the function: {}".format(func.__name__))
        visualization(pcd)
        return pcd

    return _wrap

# load point cloud
@vis
def load_t_pcd(adv_point_path):
    return o3d.t.io.read_point_cloud(adv_point_path)

# DownSampling
@vis
def pcd_downsample(pcd=None,voxel_size=0.05):
    return pcd.voxel_down_sample(voxel_size=voxel_size)

# Uniform down sampling
@vis
def pcd_unisampling(pcd=None,every_k_point=5):
    return pcd.uniform_down_sample(every_k_points=every_k_point)

# SOR
@vis
def pcd_SOR(
        pcd=None,
        nb_neighbors=20,
        std_ratio=2.0
    ):
    cl, ind = pcd.remove_statistical_outliers(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return cl, ind


# test the use of SOR and the effection for point cloud
adv_path = 'attackedLidar_Kitti100/adv_preds/04/4989_18.ply'
ori_path = 'attackedLidar_Kitti100/ori_preds/04/4989_71.ply'
def test_SOR(pcd_path,nb_neighbors=5,std_radio=2):
    # get adversarial point cloud
    adv_pcd,idx = load_t_pcd(pcd_path)
    pcd_SOR(adv_pcd,nb_neighbors=nb_neighbors,std_ratio=std_radio)

# find the statistial distribution of the attacked point cloud and unattacked point cloud
# Kitti #
# adv_label_path = 'attackedLidar_Kitti100/adv_labels_pt/04/4989_71.pt'
# ori_label_path = 'attackedLidar_Kitti100/ori_labels_pt/04/4989_71.pt'

# adv_label_path = 'attackedLidar_Kitti100/adv_labels_pt/06/35166_76.pt'
# ori_label_path = 'attackedLidar_Kitti100/ori_labels_pt/06/35166_76.pt'

# adv_label_path = 'attackedLidar_Kitti100/adv_labels_pt/09/4876_70.pt'
# ori_label_path = 'attackedLidar_Kitti100/ori_labels_pt/09/4876_70.pt'

# Poss #
# adv_label_path = 'attckedLidar_Poss100/adv_labels_pt/04/4989_35.pt'
# ori_label_path = 'attckedLidar_Poss100/ori_labels_pt/04/4989_35.pt'

# adv_label_path = 'attckedLidar_Poss100/adv_labels_pt/06/35166_74.pt'
# ori_label_path = 'attckedLidar_Poss100/ori_labels_pt/06/35166_74.pt'

adv_label_path = 'attckedLidar_Poss100/adv_labels_pt/09/4876_42.pt'
ori_label_path = 'attckedLidar_Poss100/ori_labels_pt/09/4876_42.pt'

# attackedLidar_Kitti100
# 测试Kitti对抗样本和正常样本中的均值和方差特性
adv_label_path_set = []
ori_label_path_set = []


def test_class_distribution(label_path):
    labels = load_labels_cloud(label_path).flatten()
    valid_labels = labels[labels!=-1]
    # 统计labels中每个类别的数量
    print(f'[INFO] valid labels occupy the ratio of {valid_labels.size(0)/labels.size(0)}')
    label_counts = torch.bincount(valid_labels)
    # 绘制柱状图
    plt.figure(figsize=(10, 6),dpi=100)
    plt.bar(range(len(label_counts)), label_counts, tick_label=range(len(label_counts)))
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(f'Point Cloud Label Distribution for label {label_path}')
    plt.show()

if __name__ == '__main__':
    # Test

    # test_SOR(adv_path)
    # test_SOR(ori_path)
    test_class_distribution(ori_label_path)
    test_class_distribution(adv_label_path)


