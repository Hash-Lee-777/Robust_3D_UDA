# randomly sampling and label spreading
import open3d as o3d
import torch
import networkx as nx
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
# point cloud tools for subsequent operation

# point cloud visualization, Decoration
# 装饰器工厂：接受参数并返回装饰器
def vis(disp=False):
    def _decorate(func):
        def calculate_vis_scope(pcd):
            import numpy as np
            import open3d as o3d
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
            import open3d as o3d
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

# load point cloud
@vis(disp=False)
def load_pcd(adv_point_path):
    return o3d.t.io.read_point_cloud(adv_point_path)

# load point cloud as array
def load_pcd_array(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_coords = np.asarray(pcd.points) 
    pcd_colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    pcd_normals = np.asarray(pcd.normals) if pcd.has_normals() else None

    return (pcd_coords,pcd_colors,pcd_normals)

# load point cloud labels
def load_labels_cloud(adv_label_path):
    return torch.load(adv_label_path).int()

# Uniform down sampling (对于点云均匀随机采样)
@vis
def pcd_unisampling(pcd=None,every_k_point=5):
    return pcd.uniform_down_sample(every_k_points=every_k_point)

#<========== possible label diffusion way===================> #
# 处理源数据
# 创建输入输出Data
def point_cloud_to_graph(point_coords, point_labels, k_neighbors=5):
    """
    将点云数据转换为图结构。
    
    :param point_coords: 点云坐标的 numpy 数组，形状为 (N, 3)
    :param point_labels: 点云标签的 numpy 数组，形状为 (N,)
    :param k_neighbors: 每个点的 K 近邻数
    :return: PyTorch Geometric 图数据对象
    """
    # 创建 K-近邻结构
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(point_coords)
    distances, indices = nbrs.kneighbors(point_coords)

    # 创建 NetworkX 图
    G = nx.Graph()  # 初始化图
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:  # 避免自环
                G.add_edge(i, j)  # 添加边

    # 将 NetworkX 图转换为 PyTorch Geometric 图数据
    # edge_index = torch.tensor(list(G.edges)).t().contiguous().T  # 转置并连续化
    edge_index = torch.tensor(list(G.edges)).t().contiguous()  # 转置并连续化
    x = torch.tensor(point_coords, dtype=torch.float32)  # 节点特征
    y = torch.tensor(point_labels, dtype=torch.long)  # 标签

    # 创建 PyTorch Geometric 的 Data 对象
    graph_data = Data(x=x, edge_index=edge_index, y=y)

    return graph_data  # 返回图数据

# 加载点云数据
#1. get access to point cloud data (记载点云数据)
source_pcd_path = 'data/attackedLidar_Kitti100/adv_preds/04/4989_18.ply'
source_labels_path = 'data/attackedLidar_Kitti100/adv_labels_pt/04/4989_71.pt'

test_pcd_path = 'data/attackedLidar_Kitti100/ori_preds/06/35166_76.ply'
test_labels_path = 'data/attackedLidar_Kitti100/adv_labels_pt/06/35166_76.pt'

# source_pcd dataset
source_pcd_coords, source_pcd_colors, source_pcd_normals = load_pcd_array(pcd_path=source_pcd_path)
source_pcd_labels = load_labels_cloud(adv_label_path=source_labels_path)
source_pcd_labels = source_pcd_labels.numpy()

# target_pcd dataset
test_pcd_coords, test_pcd_colors, test_pcd_normals = load_pcd_array(pcd_path=test_pcd_path)
test_pcd_labels = load_labels_cloud(adv_label_path=test_labels_path)
test_pcd_labels = test_pcd_labels.numpy()

# 创建训练测试数据
Source_Data = point_cloud_to_graph(source_pcd_coords,source_pcd_labels,k_neighbors=5)
Test_Data = point_cloud_to_graph(test_pcd_coords,test_pcd_labels,k_neighbors=5)

print(f'Source Data points: {Source_Data.x}')
print(f'Test Data points: {Test_Data.x}')

# 创建图卷积模型
# 定义图卷积神经网络模型，用于标签扩散
class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x       

# 定义模型
input_dim = 3  # 输入维度：点云的三维坐标
hidden_dim = 3*24
output_dim = 19  # 两个类别
model = SimpleGCN(input_dim, hidden_dim, output_dim)
# 定义损失函数和优化器
criterion  = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
print(Source_Data.x.shape)
print(Test_Data.x.shape)

# 训练和评估
# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()  # 重置优化器梯度
    out = model(Source_Data)  # 前向传递

    # 仅使用已知标签进行损失计算
    labeled_indices = (Source_Data.y >= 0).nonzero().view(-1)
    loss = criterion(out[labeled_indices], Source_Data.y[labeled_indices])  # 计算损失

    loss.backward()  # 反向传播
    optimizer.step()  # 优化器更新参数

    if epoch % 10 == 0:
        # 计算训练准确率
        _, predicted = torch.max(out, 1)
        correct = (predicted == Source_Data.y).sum().item()
        accuracy = correct / len(Source_Data.y)
        print(f"Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy}")

# 在测试数据上进行评估
model.eval()  # 设置为评估模式
with torch.no_grad():  # 禁用梯度计算
    test_out = model(Test_Data)  # 测试数据的预测
    _, test_predicted = torch.max(test_out, 1)
    test_correct = (test_predicted == Test_Data.y).sum().item()
    test_accuracy = test_correct / len(Test_Data.y)
    print(f"Test Accuracy: {test_accuracy}")

# 保存训练好的模型
# torch.save(model.state_dict(), "simple_gcn_model.pth")  # 保存模型参数