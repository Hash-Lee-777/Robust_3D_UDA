"""
    训练编码器: 训练编码器用于恢复点云形状
    1、training dataset [features,labels]
    |  features |: MinkUnet34编码特征
    |  labels   |: 点云真实坐标
    2、training target:
    |   loss    |: 模型训练损失, training for descent geometry loss and distribution loss
    
"""
import argparse
from configs import get_config
import torch
import logging
import os
from attack.datasets.SynlidarPhase import get_sum_split,get_dict_random
import random
from attack.datasets.SynlidarPhase import AdvSynLiDARDataset
import numpy as np
import utils.models as models
import MinkowskiEngine as ME
from utils.collation import CollateFN
from torch.utils.data import DataLoader
import time
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.defense import DecoderTrainer
from defense_tools import utils_tools_mappings,loss_dict_mappings
import torch.nn as nn


normalize_pcd_tensor = utils_tools_mappings['normalize_pcd_tensor']
load_former_params = utils_tools_mappings['load_former_params']

smooth_l1_loss = loss_dict_mappings['smooth_l1_loss']
kl_loss = loss_dict_mappings['kl_divergence_loss']

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

def load_decoder(checkpoint_path,model,name:str='decoder'):
    """ 
    load pretrained models from .ckpt file
    """
    def clean_state_dict(state,name:str='decoder'):
        # 创建两个新的字典来保存分离后的 encoder 和 decoder 参数
        decoder_ckpt = {}
        encoder_ckpt = {}

        # 遍历原始字典并将键分类到相应的字典中
        for k in list(state.keys()):
            if "encoder" in k:
                # 分割键名并移除 'decoder'
                layer = k.split('.')
                layer.remove('encoder')
                # 重新组合键名
                layer_name = '.'.join(layer)
                encoder_ckpt[layer_name] = state[k]
        
        # 遍历原始字典并将键分类到相应的字典中
        for k in list(state.keys()):
            if "decoder" in k:
                # 分割键名并移除 'decoder'
                layer = k.split('.')
                layer.remove('decoder')
                # 重新组合键名
                layer_name = '.'.join(layer)
                decoder_ckpt[layer_name] = state[k]
        
        # 如果只需要 decoder 参数，可以直接返回 decoder_ckpt
        return decoder_ckpt if name=='decoder'else encoder_ckpt

    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
    ckpt = clean_state_dict(ckpt)
    model.load_state_dict(ckpt, strict=True)  # 设置 strict=False 以忽略未匹配的键
    return model

def setup_logger(name:str='defense.log',log_dir:str='log/defense'):
    """
        设置Logger
    """
    # 创建log所在目录 
    os.makedirs(log_dir,exist_ok=True)
   
    # 创建一个Logger对象
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 创建一个Handler，用于写入日志文件
    file_handler = logging.FileHandler(log_dir+'/'+name)
    file_handler.setLevel(logging.DEBUG)

    # 创建一个Handler，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建一个Formatter，并设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将Handler添加到Logger中
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('--config_file', type=str,
                    default='configs/defense/warmup/lidar2kitti.yaml',
                    help='Path to config file.')
args = parser.parse_args()

def save_train_validation(training_dataset,validation_dataset,selected_num=10):
    import open3d as o3d
    """
        save training and validation dataset
    """
    # the directory of saving dataset
    save_dir = "recovery/selected_samples"
    os.makedirs(save_dir, exist_ok=True)
    # 随机选择10个训练样本
    training_samples = random.sample(range(len(training_dataset)), selected_num)
    validation_samples = random.sample(range(len(validation_dataset)),selected_num)
    for idx, sample in enumerate(training_samples):
        # 获取点云数据
        points = training_dataset[sample]['coordinates'].numpy()
         # 创建一个Open3D点云对象
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        # 将点云数据保存为 Ply 文件
        o3d.io.write_point_cloud(os.path.join(save_dir, f"training_sample_{sample}.ply"), point_cloud)

    for idx, sample in enumerate(validation_samples):
        # 获取点云数据
        points = validation_dataset[sample]['coordinates'].numpy()
         # 创建一个Open3D点云对象
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        # 将点云数据保存为 Ply 文件
        o3d.io.write_point_cloud(os.path.join(save_dir, f"validation_sample_{sample}.ply"), point_cloud)

    print("Selected samples saved successfully.")

def load_train_validation(save_path:str='recovery/selected_samples'):
    """
        load test training and validation dataset
    """
    import open3d as o3d
    # 创建空列表存储所有点云的tensor
    all_points_cloud = []
    # 遍历指定目录中的所有 Ply 文件
    for file_name in os.listdir(save_path):
        if file_name.endswith('.ply'):
            # 加载 Ply 文件
            file_path = os.path.join(save_path, file_name)
            point_cloud = o3d.io.read_point_cloud(file_path)

            # 将点云数据转换为 Tensor，并添加到列表中
            points = torch.tensor(point_cloud.points, dtype=torch.float32)
            all_points_cloud.append(points)
    return all_points_cloud

def get_data_mappings():
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
    '''
    synlidar2poss_color = np.array([(255, 255, 255),  # unlabelled
                                    (250, 178, 50),  # person
                                    (255, 196, 0),  # rider
                                    (25, 25, 255),  # car
                                    (107, 98, 56),  # trunk
                                    (157, 234, 50),  # plants
                                    (233, 166, 250),  # building
                                    (255, 214, 251),  # fence
                                    (0, 0, 0),   # ground
                                    (173, 23, 121),  # traffic-sign
                                    (83, 93, 130)])/255.  # ground
    '''
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
    
    data_mappings = {
        'lidar2kitti': synlidar2kitti,
        'lidar2poss': synlidar2poss,
        'lidar2kitti_color': synlidar2kitti_color,
        'lidar2poss_color': synlidar2poss_color
    }
    return data_mappings

def get_dataloader(dataset, batch_size, collate_fn=CollateFN(), shuffle=False, pin_memory=True,config=None):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          shuffle=shuffle,
                          num_workers=config.pipeline.dataloader.num_workers,
                          pin_memory=pin_memory)

def train_decoder_pl(config,encoder=None,decoder=None,logger=None):
    # <==========   setting training and validation dataset   ==========>
    try:
        mapping_path = config.dataset.mapping_path
    except AttributeError('--> Setting default class mapping path!'):
        mapping_path = None

    data_mappings = get_data_mappings()

    # load clean traing dataset and label
    selected_split = select_dataset()
    
    logger.debug(
        "record training dataset size:{}".format(sum(len(selected_split['train'][key]) for key in selected_split['train']))
    )
    logger.debug(
        "record validation dataset size:{}".format(sum(len(selected_split['validation'][key]) for key in selected_split['validation']))
    )
    # training dataset
    training_dataset = AdvSynLiDARDataset(
        version=config.dataset.version,
        phase='train',
        dataset_path=config.dataset.dataset_path,
        mapping_path=mapping_path,
        voxel_size=config.dataset.voxel_size,
        augment_data=config.dataset.augment_data,
        sub_num=config.dataset.num_pts,
        num_classes=config.models.Encoder.out_classes,
        ignore_label=config.dataset.ignore_label,
        split=selected_split
    )
    # validation dataset
    validation_dataset = AdvSynLiDARDataset(
        version=config.dataset.version,
        phase='validation',
        dataset_path=config.dataset.dataset_path,
        mapping_path=mapping_path,
        voxel_size=config.dataset.voxel_size,
        augment_data=False,
        sub_num=config.dataset.num_pts,
        num_classes=config.models.Encoder.out_classes,
        ignore_label=config.dataset.ignore_label,
        attacked_phase='validation',
        split=selected_split

    )
    save_train_validation(training_dataset,validation_dataset)

    if config.dataset.target == 'SemanticKITTI':
        training_dataset.class2names = data_mappings['lidar2kitti']
        validation_dataset.class2names = data_mappings['lidar2kitti']
        training_dataset.color_map = data_mappings['lidar2kitti_color']
        validation_dataset.color_map = data_mappings['lidar2kitti_color']
    else:
        training_dataset.class2names = data_mappings['lidar2poss']
        validation_dataset.class2names = data_mappings['lidar2poss']
        training_dataset.color_map = data_mappings['lidar2poss_color']
        validation_dataset.color_map = data_mappings['lidar2poss_color']

    # 计算sample后的数据集大小, sample rate 5%
    logger.debug(f'clean dataset sampling rate: {config.defense.sampling_rate}')
    logger.debug(f'length of training dataset: {len(training_dataset)}')
    logger.debug(f'length of validation dataset: {len(validation_dataset)}')
    # setting dataloader
    collation = CollateFN()

    
    training_dataloader = get_dataloader(
        training_dataset,
        collate_fn=collation,
        batch_size=config.pipeline.dataloader.batch_size,
        shuffle=True,
        config=config
    )

    validation_dataloader = get_dataloader(
        validation_dataset,
        collate_fn=collation,
        batch_size=config.pipeline.dataloader.batch_size,
        shuffle=False,
        config=config
    )
    # setting decoder lightning
    pl_module = DecoderTrainer(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        encoder=encoder,
        decoder=decoder,
        criterion=config.pipeline.loss,
        optimizer_name=config.pipeline.optimizer.name,
        num_classes=config.models.Encoder.out_classes,
        batch_size=config.pipeline.dataloader.batch_size,
        val_batch_size=config.pipeline.dataloader.batch_size*4,
        lr=config.pipeline.optimizer.lr,
        train_num_workers=config.pipeline.dataloader.num_workers,
        val_num_workers=config.pipeline.dataloader.num_workers,
        clear_cache_int=config.pipeline.lightning.clear_cache_int,
        scheduler_name=config.pipeline.scheduler.name,
        debug_logger=logger
    )
    
    # 自定义数据集用于选择少量的干净数据集 5%
    logger.info(f'<========start training decoder=======>')
    run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())
    if config.pipeline.wandb.run_name is not None:
        run_name = run_time + '_' + config.pipeline.wandb.run_name
    else:
        run_name = run_time

    save_dir = os.path.join(config.pipeline.save_dir, run_name)

    wandb_logger = WandbLogger(project=config.pipeline.wandb.project_name,
                               entity=config.pipeline.wandb.entity_name,
                               name=run_name,
                               offline=config.pipeline.wandb.offline)

    loggers = [wandb_logger]

    checkpoint_callback = [ModelCheckpoint(dirpath=os.path.join(save_dir, 'checkpoints'), save_top_k=-1,every_n_epochs=5)]

    trainer = Trainer(max_epochs=config.pipeline.epochs,
                      gpus=config.pipeline.gpus,
                      accelerator="ddp",
                      default_root_dir=config.pipeline.save_dir,
                      weights_save_path=save_dir,
                      precision=config.pipeline.precision,
                      logger=loggers,
                      check_val_every_n_epoch=config.pipeline.lightning.check_val_every_n_epoch,
                      val_check_interval=1.0,
                      num_sanity_val_steps=2,
                      resume_from_checkpoint=config.pipeline.lightning.resume_checkpoint,
                      callbacks=checkpoint_callback)

    trainer.fit(pl_module,
                train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloader)

def select_dataset(percentage=0.05,split_path=None):
    """
        randomly select training dataset and validation dataset based on sample rate
    """
    os.makedirs(os.path.dirname(split_path),exist_ok=True)
    if not os.path.exists(split_path):
        split ,split_values_train = get_sum_split(phase='train')
        split ,split_values_validation = get_sum_split(phase='validation')
        selected_split = {'train': {}, 'validation': {}}
        
        for dataset in ['train', 'validation']:
            # 计算总的数据量
            total_data = sum(len(split[dataset][key]) for key in split[dataset])
            
            # 计算需要选取的数据量（percentage）
            num_samples_to_select = int(total_data * percentage)
            
            # 按比例从每个序列中选取数据
            for key in split[dataset]:
                # 计算当前序列中需要选取的数据量
                num_samples_from_current_sequence = int(len(split[dataset][key]) * percentage)
                
                # 如果当前序列的数据不足 percentage，则全选
                if num_samples_from_current_sequence < 1:
                    num_samples_from_current_sequence = len(split[dataset][key])
                
                # 从当前序列中随机抽取数据并保存在新的字典中
                selected_split[dataset][key] = random.sample(split[dataset][key], num_samples_from_current_sequence)
            
            # 如果选取的数据总量不足 percentage，可以从剩余数据中补充
            selected_data_flat = [item for sublist in selected_split[dataset].values() for item in sublist]
            if len(selected_data_flat) < num_samples_to_select:
                remaining_data = [item for key in split[dataset] for item in split[dataset][key] if item not in selected_data_flat]
                num_additional_samples_needed = num_samples_to_select - len(selected_data_flat)
                additional_samples = random.sample(remaining_data, num_additional_samples_needed)
                
                # 将补充的数据添加到各个序列中
                for sample in additional_samples:
                    for key in split[dataset]:
                        if sample in split[dataset][key]:
                            selected_split[dataset][key].append(sample)
                            break
        torch.save(selected_split,split_path)
    else:
        selected_split = torch.load(split_path)
    return selected_split

def train_decoder(config):
    # current time
    from datetime import datetime
    current_timestamp = datetime.now().timestamp()
    datetime_object = datetime.fromtimestamp(current_timestamp)
    formatted_time = datetime_object.strftime("%Y-%m-%d %H:%M:%S")
    # create logger
    save_dir = config.log.save_dir + formatted_time
    logger = setup_logger(name=config.log.name,log_dir=save_dir)

    logger.info('<==========load pretrained decoder========>')
    encoder, decoder = get_decoder(config)

    # start training defense module
    logger.info('start training defense module!')
    
    # logger.info('-->traning defense')
    train_decoder_pl(config=config,encoder=encoder,decoder=decoder,logger=logger)

    # end training defense module
    logger.info('<=========training defense module end!========>')

    return

def eval_decoder(config):
    # current time
    from datetime import datetime
    import open3d as o3d
    current_timestamp = datetime.now().timestamp()
    datetime_object = datetime.fromtimestamp(current_timestamp)
    formatted_time = datetime_object.strftime("%Y-%m-%d %H:%M:%S")
    
    # create logger
    save_dir = config.log.save_dir + formatted_time
    logger = setup_logger(name=config.log.name, log_dir=save_dir)

    logger.info('<==========load pretrained decoder and encoder========>')
    encoder, decoder = get_decoder(config)

    all_points_cloud = load_train_validation()
    test_coords = all_points_cloud[1]
    
    # 创建一列包含 0 的张量，形状为 N x 1, reshape point cloud as Nx4
    zeros_column = torch.zeros((test_coords.shape[0], 1), dtype=test_coords.dtype)
    test_coords = torch.cat((zeros_column, test_coords), dim=1)
    test_features = torch.ones_like(test_coords[:, 0]).view(-1, 1)
    
    stensor = ME.SparseTensor(coordinates=test_coords.int(), features=test_features.float())
    original_coords = test_coords[:, 1:].float()  # [N,3]

    # <============initialize encoder MinkUnet34 =============>
    encoder.eval()
    decoder.eval()
    # output_stensor
    stensor_out, _ = encoder(stensor, is_seg=False)
    encoded_feature = stensor_out.F
    
    # construct decoder input stensor
    decoder_stensor = ME.SparseTensor(coordinates=test_coords.int(), features=encoded_feature)
    recovery_coords = decoder(decoder_stensor).features  # [N,3]

    # 对输入输出点云归一化
    # Normalize and center original coords
    original_coords_normalized = normalize_pcd_tensor(original_coords)
    # Normalize and center recovery coords
    recovery_coords_normalized = normalize_pcd_tensor(recovery_coords)

    # 1、KL dist. loss
    kl_div_loss = kl_loss(original_coords_normalized, recovery_coords_normalized)
    # 3、reconstruct loss
    recovery_loss = smooth_l1_loss(original_coords_normalized, recovery_coords_normalized)

    cost = recovery_loss + kl_div_loss
    print(f'[INFO] the overall cost of original and recovery loss is {cost}')

    # 打印原始点云和恢复点云损失
    print(f'[INFO] Original Test coords: \n{original_coords_normalized}, size: \n{original_coords_normalized.size()}')
    print(f'[INFO] Normalized Original Test coords: \n{original_coords_normalized.detach().numpy()}')
    print(f'[INFO] Recovery Test coords: \n{recovery_coords_normalized}, size: \n{recovery_coords_normalized.size()}')
    print(f'[INFO] Normalized Recovery Test Coords: \n{recovery_coords_normalized.detach().numpy()}')

    # save data and visualization
    # 定义保存目录
    save_path = config.result.save_dir

    import re
    # 保留原始点云和恢复点云数据
    resume_path = config.models.Decoder.resume_path
    match = re.search(r'epoch=(\d+)-step=(\d+)', resume_path)

    if match:
        epoch = int(match.group(1))
        step = int(match.group(2))
        print("Epoch:", epoch)
        print("Step:", step)
    else:
        print("No epoch and step information found in the path.")

    save_path = os.path.join(save_path, f'epoch-{epoch}','cost-{:.2f}'.format(cost))
    # 创建保存目录（如果不存在）
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 创建Open3D点云对象并赋值
    original_point_cloud = o3d.geometry.PointCloud()
    original_point_cloud.points = o3d.utility.Vector3dVector(original_coords_normalized.detach().numpy())

    recovered_point_cloud = o3d.geometry.PointCloud()
    recovered_point_cloud.points = o3d.utility.Vector3dVector(recovery_coords_normalized.detach().numpy())

    o3d.io.write_point_cloud(os.path.join(save_path, "original_point_cloud.ply"), original_point_cloud)
    o3d.io.write_point_cloud(os.path.join(save_path, "recovered_point_cloud.ply"), recovered_point_cloud)

    # 打印保存路径信息
    print(f"Original point cloud saved to: {os.path.join(save_path, 'original_point_cloud.ply')}")
    print(f"Recovered point cloud saved to: {os.path.join(save_path, 'recovered_point_cloud.ply')}")

def get_decoder(config):
    """
        function output: (encoder,decoder)
    """
    print('<==========load pretrained decoder========>')
    Encoder = getattr(models,config.models.Encoder.name)
    encoder = Encoder(config.models.Encoder.in_feat_size,config.models.Encoder.out_classes)
    encoder = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(encoder)
 
    print('<==========load decoder for training======>')
    Decoder = getattr(models,config.models.Decoder.name)
    decoder = Decoder(config.models.Decoder.in_feat_size,config.models.Decoder.out_classes)
    decoder = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(decoder)
    print(f'[INFO] load decoder {decoder.name}')

    return encoder,decoder
    
decoder_tools={
    'get_decoder': get_decoder,
    'select_dataset': select_dataset,
    'get_data_mappings': get_data_mappings,
    'get_dataloader': get_dataloader,
    'setup_logger': setup_logger
}

if __name__ == "__main__":
    # create logger
    config = get_config(args.config_file)
    logger = setup_logger(name=config.log.name,log_dir=config.log.save_dir)

    # start training defense module
    logger.debug('training defense module start!')
    train_decoder(config)
    logger.debug('training defense module end!')
