import open3d as o3d
# 加载原始点云和对抗点云
import argparse 

# 创建一个互斥的参数组
parser = argparse.ArgumentParser(description="Point Cloud Visualization")
group = parser.add_mutually_exclusive_group()
group.add_argument('--ori_preds',type=str,default=None,help='origin preds by model')
group.add_argument('--adv_preds',type=str,default=None,help='adv preds by attack')
group.add_argument('--ori_labels',type=str,default=None,help='origin labels record')
args = parser.parse_args()

# 进行可视化
def visualization(path):
    points = o3d.io.read_point_cloud(path)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # 将点云添加到可视化窗口
    vis.add_geometry(points)
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])  # 设置视角中心
    ctr.set_up([0, 0, 1])  # 设置视角方向
    ctr.set_zoom(0.8)  # 设置缩放
    vis.run()
    vis.destroy_window()
if __name__ == "__main__":
    if args.ori_preds:
        visualization(args.ori_preds)
    elif args.adv_preds:
        visualization(args.adv_preds)
    else:
        visualization(args.ori_labels)