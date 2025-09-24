from utils.datasets.dataset import BaseDataset
import os
import yaml
import numpy as np
import MinkowskiEngine as ME
import torch
from cosmix import decoder_tools
from configs import get_config
import pytorch_lightning as pl
from defense_tools import utils_tools_mappings,loss_dict_mappings
from utils.collation import CollateFN
import time
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from attack import NoiseSynlidar,CollateNoise,get_sum_split_known

# ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
ABSOLUTE_PATH = '/home/chenjj/workspace/vscode/UDA/share/'

# <================  load utils  ================>
normalize_pcd_tensor = utils_tools_mappings['normalize_pcd_tensor']
load_former_params = utils_tools_mappings['load_former_params']
batch_point_clouds = utils_tools_mappings['batch_point_clouds']

smooth_l1_loss = loss_dict_mappings['smooth_l1_loss']
chamfer_distance_batch = loss_dict_mappings['chamfer_distance_batch']
kl_loss = loss_dict_mappings['kl_divergence_loss']

get_decoder = decoder_tools['get_decoder']
select_dataset = decoder_tools['select_dataset']
get_data_mappings = decoder_tools['get_data_mappings']
get_dataloader = decoder_tools['get_dataloader']
setup_logger = decoder_tools['setup_logger']

class DecoderTester(pl.core.LightningModule):
    r"""
        Decoder Module for MinkowskiEngine for testing on the domain
    """
    def __init__(
            self,
            encoder,
            decoder,
            dataset,
            criterion,
            clear_cache_int=2,
            save_predictions=False,
            checkpoint_path=None,
            debug_logger=None,
            nn_size: int = 5,
            radius: float = 0.07,
            h: float = 0.03,
            eps: float = 1e-12,
            knn_batch = None,
            kl_beta:float = 1.0,
            l1_beta:float = 1e-2,
            save_path:str= 'perturb_back/points',
            split = None,
    ):
        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)
        self.count_validation_val = self.dataset.count_validation_val
        self.split = self.dataset.split
        self.sequences = ['00','01','02','03','04','05','06','07','08','09','10','11','12']

        self.ignore_label = self.dataset.ignore_label

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
            尝试使用L2距离训练解码器
        """
        clean_stensor = ME.SparseTensor(coordinates=batch["clean_coordinates"].int(), features=batch["clean_features"])
        noise_stensor = ME.SparseTensor(coordinates=batch["noise_coordinates"].int(), features=batch["noise_features"])
        
        origin_coords = batch['clean_coordinates']

        noise_coords = batch['noise_coordinates'][:,1:].float()  # [N,3]
        clean_coords = batch['clean_coordinates'][:,1:].float()  # [N,3]

        with torch.no_grad():
            stensor_out = self.encoder(noise_stensor)

        encoded_feature = stensor_out.F

        # construct decoder input stensor
        decoder_stesor = ME.SparseTensor(coordinates=batch['noise_coordinates'].int(), features=encoded_feature)

        # decoder_stesor.requires_grad_(True)

        recovery_eta = self.decoder(decoder_stesor).features  # [N,3]
        recovery_noise_coords = noise_coords+recovery_eta
        
        # 对输入输出点云归一化

        # Normalize clean coordinates
        # original_coords_normalized = normalize_pcd_tensor(clean_coords)
        original_coords_normalized = clean_coords

        # Normalize and center recovery coords
        # recovery_coords_normalized = normalize_pcd_tensor(recovery_noise_coords)
        recovery_coords_normalized = recovery_noise_coords
       # 1、KL dist. loss
        # kl_div_loss = kl_loss(original_coords_normalized,recovery_coords_normalized)
        # 3、reconstruct loss
        original_coords_batch = batch_point_clouds(original_coords_normalized,num_pts=4096)
        recovery_coords_batch = batch_point_clouds(recovery_coords_normalized,num_pts=4096)
        original_coords_batch = original_coords_batch[0]
        recovery_coords_batch = recovery_coords_batch[0]
        # original_coords_batch = original_coords_batch[0].transpose(0,1)
        # recovery_coords_batch = recovery_coords_batch[0].transpose(0,1)
        recovery_loss = chamfer_distance_batch(original_coords_batch,recovery_coords_batch)
        
        # cost = self.l1_beta*recovery_loss + self.kl_beta*kl_div_loss
        cost = self.l1_beta*recovery_loss

        self.debug_logger.info('Epoch {}: after validation,the recovery loss is {}'.format(self.current_epoch,cost))

        
        # return {'iou': iou, 'loss': loss.cpu()}
        result_dict = {
            'cost': cost,
            'recovery_loss':self.l1_beta*recovery_loss,
            'original_coords':original_coords_normalized,
            'recover_coords': recovery_coords_normalized,
            'noise_coords': noise_coords,
            'coords': origin_coords,
            'batch_idx':batch['idx']
        }
        self.save_prediction(result_dict)
        return result_dict

    def calculate_sequences(self,idx):
        count_val = self.count_validation_val
        i=0
        while i < len(count_val):
            if idx< count_val[0]:
                return 0,"00",'validation'
            elif idx< count_val[i+1]:
                return i,self.sequences[i+1],'validation'
            else:
                i+=1
    
    def save_prediction(self, outputs):
        import open3d as o3d
        cost = outputs['cost']
        coords = outputs['coords']
        noise_coords = outputs['noise_coords']
        recover_coords = outputs['recover_coords']
        original_coords = outputs['original_coords']
        idx = outputs['batch_idx']
        sequence_num,sequence_save,save_dir_phase = self.calculate_sequences(idx)
        
        if idx<self.count_validation_val[0]:
            frame_num = self.split['validation']['00'][idx]
        else:
            frame_num = self.split[save_dir_phase][sequence_save][idx-self.count_validation_val[sequence_num]]
      
        os.makedirs(os.path.join(self.save_path,save_dir_phase,'_original',sequence_save), exist_ok=True)
        os.makedirs(os.path.join(self.save_path,save_dir_phase,'_noise',sequence_save), exist_ok=True)
        os.makedirs(os.path.join(self.save_path,save_dir_phase,'_recovery',sequence_save), exist_ok=True)

        pcd_origin = o3d.geometry.PointCloud()
        pcd_origin.points = o3d.utility.Vector3dVector(original_coords.detach().cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(self.save_path,save_dir_phase,'_original',sequence_save, f"{frame_num}_{cost:.2f}.ply"), pcd_origin)

        pcd_noise = o3d.geometry.PointCloud()
        pcd_noise.points = o3d.utility.Vector3dVector(noise_coords.detach().cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(self.save_path,save_dir_phase,'_noise',sequence_save, f"{frame_num}_{cost:.2f}.ply"), pcd_noise)

        pcd_recovery = o3d.geometry.PointCloud()
        pcd_recovery.points = o3d.utility.Vector3dVector(recover_coords.detach().cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(self.save_path,save_dir_phase,'_recovery',sequence_save, f"{frame_num}_{cost:.2f}.ply"), pcd_recovery)
            
def test_noise_decoder(config,logger=None):

    # 设置logger
    logger = setup_logger(name='defense.log',log_dir='log/defense/validation') if logger == None else logger

    # 加载编码器和解码器
    encoder,decoder = get_decoder(config=config)
    print('[INFO] encoder state dict is as follows:\n')
    for key,value in encoder.state_dict().items():
        print('|kEY|:',key,'\n|VALUE|:',value)
    print('[INFO] decoder state dict is as follows:\n')
    for key,value in decoder.state_dict().items():
        print('|kEY|:',key,'\n|VALUE|:',value)
    

    data_mappings = get_data_mappings()

    # load clean traing dataset and label
    split_path = os.path.join(config.defense.split_dir,config.defense.split_name)
    selected_split = select_dataset(split_path = split_path)
    count_validation_val = get_sum_split_known(selected_split,'validation')
    
    try:
        mapping_path = ABSOLUTE_PATH + 'utils/datasets/_resources/synlidar_semantickitti.yaml'
    except AttributeError('--> Setting default class mapping path!'):
        mapping_path = None

    synlidar_noiseVal_dataset = NoiseSynlidar(
        version=config.dataset.version,
        phase='validation',
        dataset_path=config.dataset.dataset_path,
        mapping_path=mapping_path,
        voxel_size=config.dataset.voxel_size,
        augment_data=config.dataset.augment_data,
        sub_num=config.dataset.num_pts,
        num_classes=config.models.Encoder.out_classes,
        ignore_label=config.dataset.ignore_label,
        split=selected_split,
        clamp = config.defense.noise
    )
    synlidar_noiseVal_dataset.count_validation_val = count_validation_val

    if config.dataset.target == 'SemanticKITTI':
        synlidar_noiseVal_dataset.class2names = data_mappings['lidar2kitti']
        synlidar_noiseVal_dataset.color_map = data_mappings['lidar2kitti_color']
    else:
        synlidar_noiseVal_dataset.class2names = data_mappings['lidar2poss']
        synlidar_noiseVal_dataset.color_map = data_mappings['lidar2poss_color']

    # collation = CollateFN()
    collation = CollateNoise()

    validation_dataloader = get_dataloader(
        synlidar_noiseVal_dataset,
        collate_fn=collation,
        batch_size=config.pipeline.dataloader.batch_size,
        shuffle=False,
        config=config
    )

    # setting decoder lightning
    pl_module = DecoderTester(
        dataset=synlidar_noiseVal_dataset,
        encoder=encoder,
        decoder=decoder,
        criterion=config.pipeline.loss,
        clear_cache_int = 2,
        save_predictions = True,
        save_path = 'perturb_back/points',
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

    tester = Trainer(
        max_epochs=config.pipeline.epochs,
        gpus=[0],
        # logger=loggers,
        logger=None,
        default_root_dir=save_dir,
        weights_save_path=save_dir,
        val_check_interval=1.0,
        num_sanity_val_steps=0
    )

    tester.test(pl_module,dataloaders=validation_dataloader ,verbose=False)
