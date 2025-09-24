import unittest
import signal
from configs import get_config
import argparse
import os
import torch
import numpy as np

# from cosmix.attack.adapt import adapt_cosmix_attacked

parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/clean/adaptation/synlidar2kitti_cosmix.yaml",
                    type=str,
                    help="Path to config file")
 


class TestMyFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize config and random seeds (run once)
        args = parser.parse_args()
        cls.config = get_config(args.config_file)
        os.environ['PYTHONHASHSEED'] = str(cls.config.pipeline.seed)
        np.random.seed(cls.config.pipeline.seed)
        torch.manual_seed(cls.config.pipeline.seed)
        torch.cuda.manual_seed(cls.config.pipeline.seed)
        # Register interrupt signal handler
        signal.signal(signal.SIGINT, cls.cleanup)

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        # Run after each test method
        torch.cuda.empty_cache() # Clear GPU memory

    @classmethod
    def cleanup(cls,signum,frame):
        print("Clean up GPU memory")
        torch.cuda.empty_cache()
        raise SystemExit(1)

    @unittest.skip('this test is skipped!')
    def testCosmix(self):
        import cosmix.clean.adaptation.adapt_cosmix as adapt_cosmix
        adapt = getattr(adapt_cosmix,"adapt")
        config = get_config("configs/attack/adaptation/model_only/lidar2kitti_cosmix.yaml")
        adapt(config)
    
    # @unittest.skip('this test is skipped!')
    def testDefenseCosmix(self):
        import cosmix.defense.adaptation.adapt_cosmix_defense as adapt_cosmix_defense
        adapt = getattr(adapt_cosmix_defense,"adapt")
        config = get_config('configs/defense/uda/lidar2kitti.yaml')
        adapt(config)

    
    
if __name__ == "__main__":
    args = parser.parse_args()
    config = get_config(args.config_file)
    # fix random seed
    os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    np.random.seed(config.pipeline.seed)
    torch.manual_seed(config.pipeline.seed)
    torch.cuda.manual_seed(config.pipeline.seed)
    unittest.main()