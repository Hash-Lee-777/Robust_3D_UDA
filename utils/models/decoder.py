import MinkowskiEngine as ME
import torch.nn as nn
import torch
from .minkunet import MinkUNetBase,ResNetBase
from MinkowskiEngine.modules.resnet_block import BasicBlock

class MinkUnet34Decoder(ME.MinkowskiNetwork):
    dimension = (96, 64, 1024, 512, 64,3)

    def __init__(self, in_channels, out_channels, D=3,M=3):
        ME.MinkowskiNetwork.__init__(self, D)
        self.M = M
        self.shared_mlps = nn.ModuleList()
        # MinkowskiEngine construction
        self.conv1 = ME.MinkowskiConvolutionTranspose(
            in_channels,
            self.dimension[0],
            kernel_size=2,
            stride=1,
            dimension=D
        )
        self.bn1 = ME.MinkowskiBatchNorm(self.dimension[0])
        self.relu1 = ME.MinkowskiReLU(inplace=True)

        self.conv2 = ME.MinkowskiConvolutionTranspose(
            self.dimension[0],
            self.dimension[1] * self.M,
            kernel_size=2,
            stride=1,
            dimension=D
        )
        self.bn2 = ME.MinkowskiBatchNorm(self.dimension[1] * self.M)
        self.relu2 = ME.MinkowskiReLU(inplace=True)


        # 共享的线性层
        self.shared_mlp = nn.Sequential(
             ME.MinkowskiLinear(self.dimension[1], self.dimension[2], bias=False),
            ME.MinkowskiBatchNorm(self.dimension[2]),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(self.dimension[2], self.dimension[3], bias=False),
            ME.MinkowskiBatchNorm(self.dimension[3]),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(self.dimension[3], self.dimension[4], bias=False),
            ME.MinkowskiBatchNorm(self.dimension[4]),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(self.dimension[4], self.dimension[5], bias=False)
        )

        # 最后的下采样卷积层
        self.conv_final = ME.MinkowskiConvolution(
            self.dimension[5] * self.M , out_channels, kernel_size=1, stride=1, dimension=D
        )

    def forward(self, x:ME.SparseTensor):
        # 上采样
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # 将特征分为 M 份
        features = torch.chunk(out.F, self.M, dim=1)

        # 获取 coordinate_manager
        cm = out.coordinate_manager

        # 使用共享的卷积核处理每份特征
        new_features = []
        for i in range(self.M):
            stensor = ME.SparseTensor(features[i], coordinate_map_key=out.coordinate_map_key, coordinate_manager=cm)
            new_feature = self.shared_mlp(stensor)
            new_features.append(new_feature.F)

        # 将处理后的特征拼接起来
        out_features = torch.cat(new_features, dim=1)
        out = ME.SparseTensor(out_features, coordinate_map_key=out.coordinate_map_key, coordinate_manager=cm)

        # 最后的下采样卷积层
        out = self.conv_final(out)
        return out
    
class MinkUNet34Encoder(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)

    def forward(self, x, is_seg=False):
        if not is_seg:
            out,out_bottle = super().forward(x, is_seg)
            # output feature dimension 256
            return out
        else:
            out = super().forward(x,is_seg)
            return out


# An Easy Test

class ConvEncoder(ResNetBase):
    BLOCK = BasicBlock
    DILATIONS = (1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2)
    PLANES = (32, 64, 128, 256)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.dropout = ME.MinkowskiDropout(p=0.5)

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out_bottle = self.block4(out)
        
        return out_bottle
    
class ConvDecoder(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_feature, hidden_channels=[256, 128, 96], D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        self.hidden_channels = hidden_channels

        # Define the convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()

        prev_channels = in_channels
        for out_channels in self.hidden_channels:
            conv_layer = ME.MinkowskiConvolutionTranspose(
                prev_channels,
                out_channels,
                kernel_size=2,
                stride=1,  # Ensure stride is compatible
                dimension=D
            )
            self.conv_layers.append(conv_layer)

            bn_layer = ME.MinkowskiBatchNorm(out_channels)
            self.bn_layers.append(bn_layer)

            relu_layer = ME.MinkowskiReLU(inplace=True)
            self.relu_layers.append(relu_layer)

            prev_channels = out_channels
        

        # Final layer to match the required output channels
        self.conv_final = ME.MinkowskiConvolution(
            prev_channels,
            out_feature,
            kernel_size=1,
            stride=1,
            dimension=D
        )

    def forward(self, x: ME.SparseTensor):
        out = x
        for conv, bn, relu in zip(self.conv_layers, self.bn_layers, self.relu_layers):
            out = conv(out)
            out = bn(out)
            out = relu(out)

        out = self.conv_final(out)
        # out = ME.MinkowskiFunctional.tanh(out)
        return out

# 添加dropout    
class ConvDecoderUpdate(ME.MinkowskiNetwork):
    def __init__(self, latent_dim, out_channels=3, hidden_channels=[512, 256, 128, 96], D=3, dropout_p=0.3):
        super().__init__(D)
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim  # 潜在空间的维度
        self.out_channels = out_channels
        self.dropout_p = dropout_p

        # 定义全连接层从潜在向量映射到第一个卷积层的输入
        self.fc = nn.Linear(self.latent_dim, self.hidden_channels[0])
        
        # 定义转置卷积层
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        prev_channels = self.hidden_channels[0]
        for out_channels in self.hidden_channels:
            conv_layer = ME.MinkowskiConvolutionTranspose(
                prev_channels,
                out_channels,
                kernel_size=2,
                stride=1,  # Ensure stride is compatible
                dimension=D
            )
            self.conv_layers.append(conv_layer)

            bn_layer = ME.MinkowskiBatchNorm(out_channels)
            self.bn_layers.append(bn_layer)

            relu_layer = ME.MinkowskiReLU(inplace=True)
            self.relu_layers.append(relu_layer)
            
            dropout_layer = nn.Dropout(self.dropout_p)
            self.dropout_layers.append(dropout_layer)

            prev_channels = out_channels
        
        # 最后一层卷积用于从特征映射到输出空间
        self.conv_final = ME.MinkowskiConvolution(
            prev_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            dimension=D
        )

    def forward(self, x: ME.SparseTensor):
        # 假设x是编码器输出的潜在特征的稀疏张量
        # 首先通过一个线性层将潜在特征映射到解码器的第一个卷积层的输入大小
        out = self.fc(x.F)  # x.F是稀疏张量的特征矩阵

        # 将线性层的输出转换为稀疏张量以供解码器使用
        out = ME.SparseTensor(coordinates=x.coordinates, features=out)  # 假设x.C包含正确的坐标信息

        # 通过转置卷积层逐步上采样
        for conv, bn, relu, dropout in zip(self.conv_layers, self.bn_layers, self.relu_layers, self.dropout_layers):
            out = conv(out)
            out = bn(out)
            out = relu(out)
            out = ME.SparseTensor(coordinates=out.coordinates, features=dropout(out.F))  # 应用Dropout到特征矩阵

        # 通过最后的卷积层将特征映射到输出空间
        out = self.conv_final(out)
        return out

class EnhancedConvDecoder(ME.MinkowskiNetwork):
    def __init__(self, latent_dim, out_channels=3, hidden_channels=[1024, 512, 512, 96], D=3, dropout_p=0.2):
        self.name = 'EnhancedConvDecoder'
        super().__init__(D)
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.dropout_p = dropout_p

        self.fc = nn.Linear(self.latent_dim, self.hidden_channels[0])

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        prev_channels = self.hidden_channels[0]
        for out_channels in self.hidden_channels:
            conv_layer = ME.MinkowskiConvolutionTranspose(
                prev_channels,
                out_channels,
                kernel_size=2,
                stride=1,  # 使用stride=1进行逐步上采样
                dimension=D
            )
            self.conv_layers.append(conv_layer)

            bn_layer = ME.MinkowskiBatchNorm(out_channels)
            self.bn_layers.append(bn_layer)

            relu_layer = ME.MinkowskiReLU(inplace=True)
            self.relu_layers.append(relu_layer)

            dropout_layer = nn.Dropout(self.dropout_p)
            self.dropout_layers.append(dropout_layer)

            skip_connection = ME.MinkowskiConvolution(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                dimension=D
            )
            self.skip_connections.append(skip_connection)

            prev_channels = out_channels

        self.conv_final = ME.MinkowskiConvolution(
            prev_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            dimension=D
        )

        # 调用初始化函数
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, ME.MinkowskiConvolution, ME.MinkowskiConvolutionTranspose)):
                # Kaiming初始化
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, ME.MinkowskiBatchNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: ME.SparseTensor, is_seg=False):
        out = self.fc(x.F)
        out = ME.SparseTensor(coordinates=x.coordinates, features=out)
        for conv, bn, relu, dropout, skip in zip(self.conv_layers, self.bn_layers, self.relu_layers, self.dropout_layers, self.skip_connections):
            out = conv(out)
            out = bn(out)
            out = relu(out)
            out = ME.SparseTensor(coordinates=out.coordinates, features=dropout(out.F))
            out = skip(out) + out
        out_recovery = self.conv_final(out)
        return out if is_seg else out_recovery
    
# 拆分解码器 [特征空间恢复Module,点云几何空间恢复Module]
class FeatureUpsampleModule(nn.Module):
    def __init__(self, latent_dim=96, hidden_channels=[256,512,256,96], dropout_p=0.2, D=3):
        super(FeatureUpsampleModule, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_channels[0])
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        prev_channels = hidden_channels[0]
        for out_channels in hidden_channels:
            conv_layer = ME.MinkowskiConvolutionTranspose(
                prev_channels,
                out_channels,
                kernel_size=2,
                stride=1,
                dimension=D
            )
            self.conv_layers.append(conv_layer)

            bn_layer = ME.MinkowskiBatchNorm(out_channels)
            self.bn_layers.append(bn_layer)

            relu_layer = ME.MinkowskiReLU(inplace=True)
            self.relu_layers.append(relu_layer)
            
            dropout_layer = nn.Dropout(dropout_p)
            self.dropout_layers.append(dropout_layer)
            
            skip_connection = ME.MinkowskiConvolution(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                dimension=D
            )
            self.skip_connections.append(skip_connection)

            prev_channels = out_channels

    def forward(self, x):
        out = self.fc(x.F)
        out = ME.SparseTensor(coordinates=x.coordinates, features=out)
        for conv, bn, relu, dropout, skip in zip(self.conv_layers, self.bn_layers, self.relu_layers, self.dropout_layers, self.skip_connections):
            out = conv(out)
            out = bn(out)
            out = relu(out)
            out = ME.SparseTensor(coordinates=out.coordinates, features=dropout(out.F))
            out = skip(out) + out
        return out

class PCRecoveryModule(nn.Module):
    def __init__(self, latent_dim, hidden_channels, out_channels, dropout_p, D):
        super(PCRecoveryModule, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_channels[-1])
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        prev_channels = hidden_channels[-1]
        for out_channels in hidden_channels[::-1]:
            conv_layer = ME.MinkowskiConvolutionTranspose(
                prev_channels,
                out_channels,
                kernel_size=2,
                stride=1,
                dimension=D
            )
            self.conv_layers.append(conv_layer)

            bn_layer = ME.MinkowskiBatchNorm(out_channels)
            self.bn_layers.append(bn_layer)

            relu_layer = ME.MinkowskiReLU(inplace=True)
            self.relu_layers.append(relu_layer)
            
            dropout_layer = nn.Dropout(dropout_p)
            self.dropout_layers.append(dropout_layer)
            
            skip_connection = ME.MinkowskiConvolution(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                dimension=D
            )
            self.skip_connections.append(skip_connection)

            prev_channels = out_channels
        
        out_channels = 3
        self.conv_final = ME.MinkowskiConvolution(
            prev_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dimension=D
        )

    def forward(self, x):
        out = self.fc(x.F)
        out = ME.SparseTensor(coordinates=x.coordinates, features=out)
        for conv, bn, relu, dropout, skip in zip(self.conv_layers, self.bn_layers, self.relu_layers, self.dropout_layers, self.skip_connections):
            out = conv(out)
            out = bn(out)
            out = relu(out)
            out = ME.SparseTensor(coordinates=out.coordinates, features=dropout(out.F))
            out = skip(out) + out
        out_recovery = self.conv_final(out)
        return out_recovery
    
class CombineDecoder(ME.MinkowskiNetwork):
    def __init__(self, latent_dim, out_channels=3, hidden_channels=[[256,512,256,96],[96,512,512,1024]], D=3, dropout_p=0.2):
        super().__init__(D)
        self.name = 'CombineDecoder'
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.dropout_p = dropout_p

        self.feature_upsample_module = FeatureUpsampleModule(latent_dim, hidden_channels[0], dropout_p=0.2, D=D)
        self.point_cloud_recovery_module = PCRecoveryModule(latent_dim, hidden_channels[1], out_channels, dropout_p=0.3, D=D)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights in feature_upsample_module
        nn.init.xavier_uniform_(self.feature_upsample_module.fc.weight)
        if self.feature_upsample_module.fc.bias is not None:
            nn.init.constant_(self.feature_upsample_module.fc.bias, 0)

        for conv in self.feature_upsample_module.conv_layers:
            if isinstance(conv, ME.MinkowskiConvolutionTranspose):
                nn.init.xavier_uniform_(conv.kernel)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
                
        for conv in self.feature_upsample_module.skip_connections:
            if isinstance(conv, ME.MinkowskiConvolution):
                nn.init.xavier_uniform_(conv.kernel)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)

        # Initialize weights in point_cloud_recovery_module
        for conv in self.point_cloud_recovery_module.conv_layers:
            if isinstance(conv, ME.MinkowskiConvolutionTranspose):
                nn.init.xavier_uniform_(conv.kernel)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
                
        nn.init.xavier_uniform_(self.point_cloud_recovery_module.conv_final.kernel)
        if self.point_cloud_recovery_module.conv_final.bias is not None:
            nn.init.constant_(self.point_cloud_recovery_module.conv_final.bias, 0)

    def forward(self, x: ME.SparseTensor, is_seg=False):
        features = self.feature_upsample_module(x)
        point_cloud = self.point_cloud_recovery_module(features)
        return features if is_seg else point_cloud