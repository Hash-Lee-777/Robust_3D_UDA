import os
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import MinkowskiEngine as ME

import utils.models as models
from utils.datasets.initialization import get_dataset
from configs import get_config
from utils.collation import CollateFN
from utils.pipelines import PLTTrainer
from utils.losses import KPSLoss

parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/clean/warmup/synlidar2semantickitti.yaml",
                    type=str,
                    help="Path to config file")

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

def train(config):

    def get_dataloader(dataset, batch_size, collate_fn=CollateFN(), shuffle=False, pin_memory=True):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          shuffle=shuffle,
                          num_workers=config.pipeline.dataloader.num_workers,
                          pin_memory=pin_memory)
    try:
        mapping_path = config.dataset.mapping_path
    except AttributeError('--> Setting default class mapping path!'):
        mapping_path = None

    training_dataset, validation_dataset, target_dataset = get_dataset(dataset_name=config.dataset.name,
                                                                       dataset_path=config.dataset.dataset_path,
                                                                       target_name=config.dataset.target,
                                                                       voxel_size=config.dataset.voxel_size,
                                                                       augment_data=config.dataset.augment_data,
                                                                       version=config.dataset.version,
                                                                       sub_num=config.dataset.num_pts,
                                                                       num_classes=config.model.out_classes,
                                                                       ignore_label=config.dataset.ignore_label,
                                                                       mapping_path=mapping_path)

    collation = CollateFN()
    training_dataloader = get_dataloader(training_dataset,
                                         collate_fn=collation,
                                         batch_size=config.pipeline.dataloader.batch_size,
                                         shuffle=True)

    validation_dataloader = get_dataloader(validation_dataset,
                                           collate_fn=collation,
                                           batch_size=config.pipeline.dataloader.batch_size*4,
                                           shuffle=False)

    Model = getattr(models, config.model.name)
    model = Model(config.model.in_feat_size, config.model.out_classes)

    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    encoder = load_model('/home/chenjj/workspace/vscode/UDA/share/pretrained_models/synlidar/semantickitti/checkpoint/pretrained_model.ckpt',model)
    

    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    

    value_dict = training_dataset[1]
    initial_cls_num_list = training_dataset.get_cls_num_list(1)
    initial_cls_num_list = [x + 1 for x in initial_cls_num_list]


    # 将dict中-1的点删去
    # 找到 labels 中不等于 -1 的索引
    valid_indices = value_dict['labels'] != -1

    # 根据 valid_indices 筛选数据
    value_dict['coordinates'] = value_dict['coordinates'][valid_indices]
    value_dict['features'] = value_dict['features'][valid_indices]
    value_dict['labels'] = value_dict['labels'][valid_indices]
    value_dict['sampled_idx'] = value_dict['sampled_idx'][valid_indices]

    # for key in value_dict.keys():
    #     if value_dict[key].dim() > 0:  # 确保张量有至少一个维度
    #         original_length = value_dict[key].shape[0]
    #         half_length = original_length // 2
    #         value_dict[key] = value_dict[key][:half_length]

    coordinates = value_dict['coordinates']
    features = value_dict['features']
    # coordinates [N,3] 0列上添加一列0
    # features [[1.],[1.],...] torch.ones_like((N,1))
    # ME.SparseTensor(coordinates=coordinates,features=features)
    coordinates = torch.cat((torch.zeros(coordinates.size(0),1),coordinates),dim=1)
    features = torch.ones((coordinates.size(0),1))

    encoder = encoder.to(device)
    coordinates = coordinates.to(device)
    features = features.to(device)
    stensor = ME.SparseTensor(coordinates=coordinates,features=features)

    output = encoder(stensor).F

    criterion = KPSLoss(initial_cls_num_list)

    target = value_dict['labels']

    print(output.device)
    print(target.device)


    loss = criterion.forward(output,target)
    print(loss)

if __name__ == '__main__':
    args = parser.parse_args()

    config = get_config(args.config_file)

    # fix random seed
    # os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    # np.random.seed(config.pipeline.seed)
    # torch.manual_seed(config.pipeline.seed)
    # torch.cuda.manual_seed(config.pipeline.seed)
    # torch.backends.cudnn.benchmark = True

    train(config)
