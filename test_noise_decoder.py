import unittest
from configs import get_config
from defense_tools import train_noise_decoder,test_noise_decoder
import parser
import os
import numpy as np 
import torch
import signal
from configs import get_config
from defense_tools import utils_tools_mappings,process_noise_decoder


class TestMyFunctions(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        random_seed = '1234'
        os.environ['PYTHONHASHSEED'] = random_seed
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        # 注册中断信号处理函数
        signal.signal(signal.SIGINT, cls.cleanup)

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        # 每个测试方法执行后运行的代码
        torch.cuda.empty_cache() # 清空GPU显存

    @classmethod
    def cleanup(cls,signum,frame):
        print("Clean up GPU memory")
        torch.cuda.empty_cache()
        raise SystemExit(1)
    
    # @unittest.skip('this test is temporarily skipped!')
    def test_train_noise_decoder(self):
        # config = 'configs/defense/warmup/train_encoder/lidar2kitti_chamfer_update.yaml'
        config = 'configs/defense/warmup/fix_encoder/lidar2kitti_chamfer_update.yaml'
        config = get_config(config)
        train_noise_decoder(config=config)
        print('[INFO] TRAIN NOISE DECODER END!')

    @unittest.skip('this test is temporarily skipped!')
    def test_eval_noise_decoder(self):
        config = 'configs/defense/warmup/fix_encoder/lidar2kitti_chamfer_update.yaml'
        config = get_config(config)
        test_noise_decoder(config=config)
        print('[INFO] TEST NOISE DECODER END!')
    #测试对点云数据的处理


if __name__ == '__main__':
    unittest.main()