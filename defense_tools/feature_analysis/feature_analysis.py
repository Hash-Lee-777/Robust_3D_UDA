# analysis feature propagation during training
import torch
from torch.utils.data import DataLoader
import argparse
from configs import get_config
import os 
import numpy as np
import utils.models as models
import MinkowskiEngine as ME
from utils.collation import CollateFN
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger
from utils.losses import CELoss, DICELoss, SoftDICELoss
from utils.pipelines import PLTTrainer
from attack.datasets.SynlidarPhase import AdvSynLiDARDataset,get_dict_random
import collections
import pdb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.utils import resample
import time
from scipy.fftpack import dct
from scipy.signal import wavelets
from scipy.linalg import qr


parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/eval/synlidar2semantickitti.yaml",
                    type=str,
                    help="Path to config file")

# <==============tools===============>
def description(description:str):
    """ Description
    :type description:str:
    :param description:str:
    :raises:
    :rtype:
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"[INFO] Function {func.__name__} description: {description}\n")  
            result = func(*args, **kwargs)
            return result
        return wrapper  
    return decorator 

def load_model(checkpoint_path, model):
    """ Description 
    load pretrained models from .ckpt file
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


def capture_layer_outputs(func):
    def wrapper(model,*args,**kwrags):
        # 使用字典来保存每一层的输出
        outputs = collections.defaultdict(list)
        # 钩子函数
        def save_output(module,input,output):
            outputs[module].append(output)
        # 注册钩子
        for name,layer in model.named_modules():
            if isinstance(layer, (ME.MinkowskiConvolution, ME.MinkowskiBatchNorm)):
                layer.register_forward_hook(save_output)
        # 运行传入的函数
        result = func(model,*args,**kwrags)
        return *result, outputs
    return wrapper

def timeit(func):
    def _wrap(*args,**kwargs):
        time1 = time.time()
        result = func(*args,**kwargs)
        time2 = time.time()
        print(f'[INFO] time recording for function {func.__name__} execution, the time cost is {time2-time1}')
        return result
    return _wrap 

# <=============tools end=============>

@description('feature extraction')
@capture_layer_outputs
def feature_analysis(model,prob:float=1.,path:str='attacked_data/attackedLidar_Kitti100/',adv:bool=True):
    """ Description
    get the evaluate result and each feature of model layer output 
    :param model:
    :param prob:float: [0., 1.]
    :param path:str: point cloud saved dir
    :param adv:bool:
    :raises:
    :rtype: out,test_coords,test_colors,test_labels,output_features
    """
    random_indices,randomized_coords_dict,randomized_labels_dict,randomized_colors_dict = get_dict_random(
        prob=prob,
        path=path,
        adv=adv
    )
    test_coords = randomized_coords_dict[random_indices[0]]
    test_colors = randomized_colors_dict[random_indices[0]]
    test_labels = randomized_labels_dict[random_indices[0]]

    # 创建一列包含 0 的张量，形状为 N x 1, reshape point cloud as Nx4
    zeros_column = torch.zeros((test_coords.shape[0], 1), dtype=test_coords.dtype)
    test_coords_with_batch = torch.cat((zeros_column,test_coords),dim=1)
    test_features = torch.ones_like(test_labels).view(-1,1)
    criterion = SoftDICELoss(ignore_label=-1)
    # test forward propagation
    model.eval()
    stensor = ME.SparseTensor(coordinates=test_coords_with_batch.int(),features=test_features.float())
    stensor_out = model(stensor)
    out = stensor_out.F
    loss = criterion(out,test_labels.long())
    print(f'[INFO] recording loss {loss}')
    return out,test_coords,test_colors,test_labels

def test_feature_analysis(config,prob:float=1.,path:str='attacked_data/attackedLidar_Kitti100/',adv:bool=True):
    """ Description
    return feature analysis related dictionary
    :type config:
    :param config:
    :param prob:float:
    :param path:str:
    :param adv:bool:
    :rtype: result_dict={out=,test_coords,test_colors,test_labels,model_features}
    """
    # load pretrained model MinUnet34
    Model = getattr(models,config.model.name)
    model = Model(config.model.in_feat_size,config.model.out_classes)
    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    if config.model.resume_path is not None:
        print(f'[INFO] load trained model {config.model.resume_path} for training!')
        model=load_model(config.model.resume_path,model)
    # print(model)
    out,test_coords,test_colors,test_labels,outputs = feature_analysis(model,prob,path,adv)
    # test the output of the final layer
    final_layer_name = list(outputs)[-1]
    out_stensor = outputs[final_layer_name][0]
    out_feature = out_stensor.F # [66205,19]
    result_dict = dict(
        out=out,
        out_feature=out_feature,
        test_coords=test_coords,
        test_colors=test_colors,
        test_labels=test_labels,
        model_features=outputs
    )
    return result_dict

@description('analyse output feature with PCA, trans dimension to 2D')
def test_pca_2d(output_feature, desc: str = 'desc',c=None):
    # 使用 PCA 将数据降维至 2D
    pca = PCA(n_components=2)
    reduced_data_pca = pca.fit_transform(output_feature)

    # 创建2D图
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 2D图，不需要 'projection=3d'

    # 绘制2D散点图
    if c is None:
        ax.scatter(reduced_data_pca[:, 0], reduced_data_pca[:, 1], alpha=0.7, s=4)
    else:
        ax.scatter(reduced_data_pca[:, 0], reduced_data_pca[:, 1], alpha=0.7, s=4,c=c)

    # 设置轴标签和标题
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("2D PCA")

    # 保存图像到指定路径
    plt.savefig(f'images/PCA_2Dtest_{desc}.png')

    return

@description('analyse output feature with pca，trans dimension to 3D')
def test_pca_3d(output_feature,desc:str='desc',c=None):
    pca = PCA(n_components=3)
    reduced_data_pca = pca.fit_transform(output_feature)
    # 创建3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if c is None:
        ax.scatter(reduced_data_pca[:, 0], reduced_data_pca[:, 1], reduced_data_pca[:, 2], alpha=0.7,s=4)
    else:
        ax.scatter(reduced_data_pca[:, 0], reduced_data_pca[:, 1], reduced_data_pca[:, 2], alpha=0.7,s=4,c=c)
    # 设置标签和标题
    ax.set_xlabel("pca Component 1")
    ax.set_ylabel("pca Component 2")
    ax.set_zlabel("pca Component 3")
    ax.set_title("3D pca")
    plt.savefig(f'images/PCA_3Dtest_{desc}.png')
    return

@description('analyse with output feature with tsne, trans dimension to 3D')
def test_tsne_3d(output_feature,desc:str='desc',c=None):
    # tsne = TSNE(n_components=3, perplexity=10, n_iter=500, random_state=42)
    # learning_rates = [100, 500, 1000]
    learning_rates = [500]
    for lr in learning_rates:
        tsne = TSNE(n_components=3, perplexity=40, n_iter=600, random_state=1234,learning_rate=lr)
        reduced_data_tsne = tsne.fit_transform(output_feature)
        # 创建3D图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if c is None:
            ax.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], reduced_data_tsne[:, 2], alpha=0.7,s=4)
        else:
            ax.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], reduced_data_tsne[:, 2], alpha=0.7,s=4,c=c)
        # 设置标签和标题
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.set_zlabel("t-SNE Component 3")
        ax.set_title(f"3D t-SNE")
        plt.savefig(f'images/t-SNE_3Dtest_{desc}_lr:{lr}.png')
    return

@description('analyse with output feature with tsne, trans dimension to 2D')
def test_tsne_2d(output_feature, desc: str = 'desc',c=None):
    # learning_rates = [100, 500, 1000]
    learning_rates = [500]
    for lr in learning_rates:
        # 设定2D t-SNE参数
        tsne = TSNE(n_components=2, perplexity=40, n_iter=600, random_state=1234,learning_rate=lr)
        reduced_data_tsne = tsne.fit_transform(output_feature)

        # 创建2D散点图
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 不再是3D，因此没有'projection='3d'

        # 绘制2D散点图
        if c is None:
            ax.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], alpha=0.7, s=4)
        else:
            ax.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], alpha=0.7, s=4,c=c)

        # 设置轴标签和标题
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.set_title("2D t-SNE")

        # 保存图像到指定路径
        plt.savefig(f'images/t-SNE_2Dtest_{desc}_lr:{lr}.png')

    return

@timeit
@description('analyse sparsity under dct basis')
def is_sparse_dct(feature_space:torch.Tensor):
    """ Description
    analyse rank and recontruct basis
    :type feature_space:torch.Tensor:
    :param feature_space:torch.Tensor:
    :rtype: rank, basis 
    """
    basis,rank = None,None
    tol = 1e-5
    feature_space = feature_space.detach().numpy()
    # 进行DCT转换
    dct_features = dct(feature_space,norm='ortho',axis=1)
    # 计算非零系数的比例
    sparsity = np.sum(np.abs(dct_features)>tol) / np.prod(dct_features.shape)
    
    feature_space = feature_space.T
    # 使用QR分解找到线性无关的特征
    _, R, pivot_indices = qr(feature_space, mode='economic', pivoting=True)
    # 计算特征矩阵的秩
    rank = np.linalg.matrix_rank(feature_space,tol=tol)
    # 非零对角线对应的线性无关索引
    tol = 1e-5
    non_zero_indices = np.where(np.abs(np.diag(R))>tol)[0]
    # 提取线性无关的特征
    independent_feature = feature_space[:,pivot_indices[non_zero_indices]]
    # 输出线性无关特征及其索引
    return rank,pivot_indices[non_zero_indices],independent_feature.T

def disp3D(pcd):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # pcd: N*3
    x = pcd[:,0]
    y = pcd[:,1]
    z = pcd[:,2]
    # 创建 3D 图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 创建 3D 图

    # 绘制 3D 散点图
    ax.scatter(x, y, z, c='r', marker='o')  # 红色圆形散点

    # 设置轴标签
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # 保存 3D 图
    plt.savefig('3d_plot.png')  # 保存为图像文件

@timeit
@description('using for calculate the pseudo transformation matrix')
def test_pseudo_calculate(input_tensor:torch.Tensor,output_feature:torch.Tensor):
    """
    计算从 output_feature 到 input_tensor 的广义逆矩阵
    :param input_tensor: torch.Tensor, 输入矩阵 (N×3)
    :param output_feature: torch.Tensor, 输出矩阵 (N×19)
    :return: torch.Tensor, 广义逆矩阵
    """
    # 将 Torch Tensor 转换为 NumPy 数组
    input_np = input_tensor.detach().numpy()
    output_np = output_feature.detach().numpy()
    
    # 计算广义逆
    pseudo_inverse = np.linalg.pinv(output_np)  # 计算 output_feature 的广义逆
    
    # 计算从 output_feature 到 input_tensor 的转换矩阵
    transformation_matrix = np.dot(pseudo_inverse,input_np)  # 计算转换矩阵
    
    # 返回转换矩阵
    # return torch.from_numpy(transformation_matrix)  # 返回 Torch Tensor
    return transformation_matrix



if __name__ == "__main__":
    args = parser.parse_args()
    config = get_config(args.config_file)
    # fix random seed
    os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    np.random.seed(config.pipeline.seed)
    torch.manual_seed(config.pipeline.seed)
    torch.cuda.manual_seed(config.pipeline.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.max_split_size_mb = 512
    print('[INFO] start main:')
    test_feature_analysis()
    print('[INFO] end main:')

