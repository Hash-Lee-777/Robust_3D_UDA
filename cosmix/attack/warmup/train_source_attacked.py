import os
import time
import argparse
import numpy as np
import pdb
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
from attack.datasets.SynlidarPhase import AdvSynLiDARDataset,get_dict_random

# update train_source_attacked.py  ['attacked', 'clean'] ['training synlidar','validation synlidar']

parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                default="configs/attacked_warmup/lidar2kitti_advTrain_advVal.yaml",
                type=str,
                help="Path to config file")
args = parser.parse_args()

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

def load_model(checkpoint_path, model):
    # reloads model
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

def get_dataloader(dataset, batch_size, collate_fn=CollateFN(), shuffle=False, pin_memory=True, config=None):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          shuffle=shuffle,
                          num_workers=config.pipeline.dataloader.num_workers,
                          pin_memory=pin_memory)

def test_train_attacked_source():
    # <==========     ARGS INPUT         =========>
    config = get_config(args.config_file)

    # <==========   FIX RANDOM SEED     ==========>
    os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    np.random.seed(config.pipeline.seed)
    torch.manual_seed(config.pipeline.seed)
    torch.cuda.manual_seed(config.pipeline.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.max_split_size_mb = 512

    # <===========   GET DATA MAPPINGS  ==========>
    # ['lidar2kitti','lidar2poss','lidar2poss_color','lidar2poss_color']
    data_mappings = get_data_mappings() 

    # <==========   setting training and validation dataset   ==========>
    try:
        mapping_path = config.dataset.mapping_path
    except AttributeError('--> Setting default class mapping path!'):
        mapping_path = None
    prob = config.attacked.prob
    adv_train_path = config.attacked.adv_train_path
    adv_validation_path = config.attacked.adv_validation_path
    adv_attacked_train = config.attacked.attacked_train
    adv_attacked_validation = config.attacked.attacked_validation
    
    # adversarial training dataset or clean training dataset
    if adv_attacked_train:
        # load adversarial training dataset
        training_indices,training_coords_dict,training_labels_dict,training_colors_dict = get_dict_random(
            prob=prob,
            path = adv_train_path,
            adv=True
        )
        adv_training_dataset = AdvSynLiDARDataset(
            version=config.dataset.version,
            phase='train',
            dataset_path=config.dataset.dataset_path,
            mapping_path=mapping_path,
            voxel_size=config.dataset.voxel_size,
            augment_data=config.dataset.augment_data,
            sub_num=config.dataset.num_pts,
            num_classes=config.model.out_classes,
            ignore_label=config.dataset.ignore_label,
            attacked_indices=training_indices,
            adv_coords_dict=training_coords_dict,
            adv_labels_dict=training_labels_dict
        )
    else:
        adv_training_dataset = AdvSynLiDARDataset(
            version=config.dataset.version,
            phase='train',
            dataset_path=config.dataset.dataset_path,
            mapping_path=mapping_path,
            voxel_size=config.dataset.voxel_size,
            augment_data=config.dataset.augment_data,
            sub_num=config.dataset.num_pts,
            num_classes=config.model.out_classes,
            ignore_label=config.dataset.ignore_label
        )

    # adversarial validation dataset or clean validation dataset
    if adv_attacked_validation:
        # load adversarial validation dataset
        validation_indices,validation_coords_dict,validation_labels_dict,validation_colors_dict = get_dict_random(
            prob=prob,
            path = adv_validation_path,
            adv=True
        )
        adv_validation_dataset = AdvSynLiDARDataset(
            version=config.dataset.version,
            phase='validation',
            dataset_path=config.dataset.dataset_path,
            mapping_path=mapping_path,
            voxel_size=config.dataset.voxel_size,
            augment_data=config.dataset.augment_data,
            sub_num=config.dataset.num_pts,
            num_classes=config.model.out_classes,
            ignore_label=config.dataset.ignore_label,
            attacked_indices=validation_indices,
            adv_coords_dict=validation_coords_dict,
            adv_labels_dict=validation_labels_dict,
            attacked_phase='validation'
        )
    else:
        adv_validation_dataset = AdvSynLiDARDataset(
            version=config.dataset.version,
            phase='validation',
            dataset_path=config.dataset.dataset_path,
            mapping_path=mapping_path,
            voxel_size=config.dataset.voxel_size,
            augment_data=config.dataset.augment_data,
            sub_num=config.dataset.num_pts,
            num_classes=config.model.out_classes,
            ignore_label=config.dataset.ignore_label,
            attacked_phase='validation'
        )

    if config.dataset.target == 'SemanticKITTI':
        adv_training_dataset.class2names = data_mappings['lidar2kitti']
        adv_validation_dataset.class2names = data_mappings['lidar2kitti']
        adv_training_dataset.color_map = data_mappings['lidar2kitti_color']
        adv_validation_dataset.color_map = data_mappings['lidar2kitti_color']
        adv_training_dataset.ignore_label = -1
        adv_validation_dataset.ignore_label = -1
    else:
        adv_training_dataset.class2names = data_mappings['lidar2poss']
        adv_validation_dataset.class2names = data_mappings['lidar2poss']
        adv_training_dataset.color_map = data_mappings['lidar2poss_color']
        adv_validation_dataset.color_map = data_mappings['lidar2poss_color']
        adv_training_dataset.ignore_label = -1
        adv_validation_dataset.ignore_label = -1


    # <=========       start training         ======>
    collation = CollateFN()
    training_dataloader = get_dataloader(
        adv_training_dataset,
        collate_fn=collation,
        batch_size=config.pipeline.dataloader.batch_size,
        shuffle=True,
        config=config
    )

    validation_dataloader = get_dataloader(
        adv_validation_dataset,
        collate_fn=collation,
        batch_size=config.pipeline.dataloader.batch_size*4,
        shuffle=False,
        config=config
    )

    Model = getattr(models, config.model.name)
    model = Model(config.model.in_feat_size, config.model.out_classes)

    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    if config.model.resume_path is not None:
        print(f'[INFO] load trained model {config.model.resume_path} for training!')
        model=load_model(config.model.resume_path,model)

    # <=================  init PLTTrainer   ===================>
    pl_module = PLTTrainer(
        training_dataset=adv_training_dataset,
        validation_dataset=adv_validation_dataset,
        model=model,
        criterion=config.pipeline.loss,
        optimizer_name=config.pipeline.optimizer.name,
        batch_size=config.pipeline.dataloader.batch_size,
        val_batch_size=config.pipeline.dataloader.batch_size*4,
        lr=config.pipeline.optimizer.lr,
        num_classes=config.model.out_classes,
        train_num_workers=config.pipeline.dataloader.num_workers,
        val_num_workers=config.pipeline.dataloader.num_workers,
        clear_cache_int=config.pipeline.lightning.clear_cache_int,
        #scheduler_name=config.pipeline.scheduler.name
    )

    run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())
    if config.pipeline.wandb.run_name is not None:
        run_name = run_time + '_' + config.pipeline.wandb.run_name
    else:
        run_name = run_time

    save_dir = os.path.join(config.attacked.save_dir, run_name)

    wandb_logger = WandbLogger(
        project=config.pipeline.wandb.project_name,
        entity=config.pipeline.wandb.entity_name,
        name=run_name,
        offline=config.pipeline.wandb.offline
    )

    loggers = [wandb_logger]

    checkpoint_callback = [
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, 'checkpoints'),
            save_top_k=-1,
            every_n_epochs=20
        )
    ]

    trainer = Trainer(
        max_epochs=config.pipeline.epochs,
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
        callbacks=checkpoint_callback
    )
    print(f'[INFO] WARMUP STEP START:')
    print(f'[INFO] USING ADVERSARIAL TRAINING SAMPLE {adv_attacked_train}')
    print(f'[INFO] USING ADVERSARIAL VALIDATION SAMPLE {adv_attacked_validation}')
    
    try:
        pdb.set_trace()
        trainer.fit(pl_module,
                    train_dataloaders=training_dataloader,
                    val_dataloaders=validation_dataloader)
    except Exception as e:
        print(f'[ERROR] Error happens when traning model, error messages {e}!')
    
    print(f"[INFO] WARM UP STEP FINISHED!")
    print(f'[INFO] PRETRAINED MODEL SAVED WEIGHT IS AT {config.attacked.save_dir}')

 
if __name__ == '__main__':
    # <==========     ARGS INPUT         =========>
    config = get_config(args.config_file)

    # <==========   FIX RANDOM SEED     ==========>
    os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    np.random.seed(config.pipeline.seed)
    torch.manual_seed(config.pipeline.seed)
    torch.cuda.manual_seed(config.pipeline.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.max_split_size_mb = 512

    # <===========   GET DATA MAPPINGS  ==========>
    # ['lidar2kitti','lidar2poss','lidar2poss_color','lidar2poss_color']
    data_mappings = get_data_mappings() 

    # <==========   setting training and validation dataset   ==========>
    try:
        mapping_path = config.dataset.mapping_path
    except AttributeError('--> Setting default class mapping path!'):
        mapping_path = None
    prob = config.attacked.prob
    adv_train_path = config.attacked.adv_train_path
    adv_validation_path = config.attacked.adv_validation_path
    adv_attacked_train = config.attacked_train
    adv_attacked_validation = config.attacked_validation
    
    # adversarial training dataset or clean training dataset
    if adv_attacked_train:
        # load adversarial training dataset
        training_indices,training_coords_dict,training_labels_dict,training_colors_dict = get_dict_random(
            prob=prob,
            path = adv_train_path,
            adv=True
        )
        adv_training_dataset = AdvSynLiDARDataset(
            version=config.dataset.version,
            phase='train',
            dataset_path=config.dataset.dataset_path,
            mapping_path=mapping_path,
            voxel_size=config.dataset.voxel_size,
            augment_data=config.dataset.augment_data,
            sub_num=config.dataset.num_pts,
            num_classes=config.model.out_classes,
            ignore_label=config.dataset.ignore_label,
            attacked_indices=training_indices,
            adv_coords_dict=training_coords_dict,
            adv_labels_dict=training_labels_dict
        )
    else:
        adv_training_dataset = AdvSynLiDARDataset(
            version=config.dataset.version,
            phase='train',
            dataset_path=config.dataset.dataset_path,
            mapping_path=mapping_path,
            voxel_size=config.dataset.voxel_size,
            augment_data=config.dataset.augment_data,
            sub_num=config.dataset.num_pts,
            num_classes=config.model.out_classes,
            ignore_label=config.dataset.ignore_label
        )

    # adversarial validation dataset or clean validation dataset
    if adv_attacked_validation:
        # load adversarial validation dataset
        validation_indices,validation_coords_dict,validation_labels_dict,validation_colors_dict = get_dict_random(
            prob=prob,
            path = adv_validation_path,
            adv=True
        )
        adv_validation_dataset = AdvSynLiDARDataset(
            version=config.dataset.version,
            phase='validation',
            dataset_path=config.dataset.dataset_path,
            mapping_path=mapping_path,
            voxel_size=config.dataset.voxel_size,
            augment_data=config.dataset.augment_data,
            sub_num=config.dataset.num_pts,
            num_classes=config.model.out_classes,
            ignore_label=config.dataset.ignore_label,
            attacked_indices=validation_indices,
            adv_coords_dict=validation_coords_dict,
            adv_labels_dict=validation_labels_dict
        )
    else:
        adv_validation_dataset = AdvSynLiDARDataset(
            version=config.dataset.version,
            phase='validation',
            dataset_path=config.dataset.dataset_path,
            mapping_path=mapping_path,
            voxel_size=config.dataset.voxel_size,
            augment_data=config.dataset.augment_data,
            sub_num=config.dataset.num_pts,
            num_classes=config.model.out_classes,
            ignore_label=config.dataset.ignore_label,
        )

    # <=========       start training         ======>
    collation = CollateFN()
    training_dataloader = get_dataloader(
        adv_training_dataset,
        collate_fn=collation,
        batch_size=config.pipeline.dataloader.batch_size,
        shuffle=True
    )

    validation_dataloader = get_dataloader(
        adv_validation_dataset,
        collate_fn=collation,
        batch_size=config.pipeline.dataloader.batch_size*4,
        shuffle=False
    )

    Model = getattr(models, config.model.name)
    model = Model(config.model.in_feat_size, config.model.out_classes)

    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    if config.model.resume_path is not None:
        print(f'[INFO] load trained model {config.model.resume_path} for training!')
        model=load_model(config.model.resume_path,model)

    # <=================  init PLTTrainer   ===================>
    pl_module = PLTTrainer(
        training_dataset=adv_training_dataset,
        validation_dataset=adv_validation_dataset,
        model=model,
        criterion=config.pipeline.loss,
        optimizer_name=config.pipeline.optimizer.name,
        batch_size=config.pipeline.dataloader.batch_size,
        val_batch_size=config.pipeline.dataloader.batch_size*4,
        lr=config.pipeline.optimizer.lr,
        num_classes=config.model.out_classes,
        train_num_workers=config.pipeline.dataloader.num_workers,
        val_num_workers=config.pipeline.dataloader.num_workers,
        clear_cache_int=config.pipeline.lightning.clear_cache_int,
        #scheduler_name=config.pipeline.scheduler.name
    )

    run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())
    if config.pipeline.wandb.run_name is not None:
        run_name = run_time + '_' + config.pipeline.wandb.run_name
    else:
        run_name = run_time

    save_dir = os.path.join(config.attacked.save_dir, run_name)

    wandb_logger = WandbLogger(
        project=config.pipeline.wandb.project_name,
        entity=config.pipeline.wandb.entity_name,
        name=run_name,
        offline=config.pipeline.wandb.offline
    )

    loggers = [wandb_logger]

    checkpoint_callback = [
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, 'checkpoints'),
            save_top_k=-1,
            every_n_epochs=config.pipeline.lightning.check_val_every_n_epoch
        )
    ]

    trainer = Trainer(
        max_epochs=config.pipeline.epochs,
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
        callbacks=checkpoint_callback
    )
    print(f'[INFO] WARMUP STEP START:')
    print(f'[INFO] USING ADVERSARIAL TRAINING SAMPLE {adv_attacked_train}')
    print(f'[INFO] USING ADVERSARIAL VALIDATION SAMPLE {adv_attacked_validation}')
    trainer.fit(pl_module,
                train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloader)
    print(f"[INFO] WARM UP STEP FINISHED!")
    print(f'[INFO] PRETRAINED MODEL SAVED WEIGHT IS AT {config.attacked.save_dir}')


