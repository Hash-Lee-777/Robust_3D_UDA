# 尝试根据点云几何特征，优化点云
# 噪声点云 N*3
# 原始点云 N*3
# 优化点云 N*3
from utils.models import ConvDecoderUpdate,MinkUnet34Decoder

if __name__== '__main__':
    import numpy as np
    coordinates = np.array(
        [[100,11,100],
         [101,23,344],
         [122,22,333]]
    )
    features = np.asarray([1.,1.,1.])
    encoder = MinkUnet34Decoder(1,19,3)
    decoder = ConvDecoderUpdate(latent_dim=96)
    print(encoder)
    print(decoder)
