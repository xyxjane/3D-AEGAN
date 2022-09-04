from distutils.command.upload import upload
from doctest import OutputChecker
from math import fabs
from re import S, X
import re
from sys import api_version
from turtle import forward, up
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import torch
import numpy as np
from unet3d.buildingblocks import DoubleConv, ExtResNetBlock, create_encoders, SingleConv, create_decoders, AR_Encoder_block, AR_Decoder_block, Encoder, Decoder_nocat
from unet3d.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from unet3d.utils import get_class,number_of_features_per_level

class Abstract3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, **kwargs):
        super(Abstract3DUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        upsample=True)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, input):
        # encoder part
        x = input
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        # x = self.final_conv(x+input)
        # x = self.final_relu(x)
        # x = torch.clamp(x, min=0.0)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     **kwargs)


class ResidualUNet3D(Abstract3DUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ExtResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             **kwargs)


class UNet2D(Abstract3DUNet):
    """
    Just a standard 2D Unet. Arises naturally by specifying conv_kernel_size=(1, 3, 3), pool_kernel_size=(1, 2, 2).
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        if conv_padding == 1:
            conv_padding = (0, 1, 1)
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_kernel_size=(1, 3, 3),
                                     pool_kernel_size=(1, 2, 2),
                                     conv_padding=conv_padding,
                                     **kwargs)


class AbstractCasDUNet(nn.Module):
    def __init__(self, in_channels, out_channels, D, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, **kwargs):
        super(AbstractCasDUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        upsample=True)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None
        self.D = D
        # self.final_relu = nn.ReLU(inplace=True)

    def forward(self, input):
        # encoder part
        # x = torch.cat((input, staged_input),1)
        for _ in range(self.D):
            x = input
            encoders_features = []
            for encoder in self.encoders:
                x = encoder(x)
                # reverse the encoder outputs to be aligned with the decoder
                encoders_features.insert(0, x)

            # remove the last encoder's output from the list
            # !!remember: it's the 1st in the list
            encoders_features = encoders_features[1:]

            # decoder part
            for decoder, encoder_features in zip(self.decoders, encoders_features):
                # pass the output from the corresponding encoder and the output
                # of the previous decoder
                x = decoder(encoder_features, x)

            x = self.final_conv(x)
            # x = self.final_conv(x+input)
            # x = self.final_relu(x)
            # x = torch.clamp(x, min=0.0)

            # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs logits
            if not self.training and self.final_activation is not None:
                x = self.final_activation(x)
            input = x

        return x


class CasUNet3D(AbstractCasDUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(CasUNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     **kwargs)


class AENet(nn.Module):
    # def __init__(self, in_channels, out_channels, UNet3D_maps,layer_order,f_maps,num_groups,is_segmentation, **kwargs):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(AENet, self).__init__()
        # self.pre_net = UNet3D(in_channels, out_channels,f_maps=UNet3D_maps,layer_order=layer_order,num_groups=num_groups,
        # is_segmentation=is_segmentation)
        self.encoder_1 = AR_Encoder_block(in_channels,16, apply_pooling=True)
        self.encoder_2 = AR_Encoder_block(16,32, apply_pooling=True)
        self.encoder_3 = AR_Encoder_block(32,64, apply_pooling=False)
        self.encoder_4 = AR_Encoder_block(64,128, apply_pooling=False)
        self.decoder_1 = AR_Decoder_block(128,64,apply_transconv=True)
        self.decoder_2 = AR_Decoder_block(64,32,apply_transconv=True)
        self.decoder_3 = AR_Decoder_block(32,16,apply_transconv=False)
        # self.decoder_4 = AR_Decoder_block(64,32,apply_transconv=False)
        self.last_conv = nn.Conv3d(16,out_channels,kernel_size=3,padding=1)
        self.final_activation = nn.Tanh()

    def forward(self,x):
        # x_pre = self.pre_net(x)
        x_1 = self.encoder_1(x)
        x_2 = self.encoder_2(x_1)
        x_3 = self.encoder_3(x_2)
        x_4 = self.encoder_4(x_3)
        x_5 = self.decoder_1(x_4)
        x_6 = self.decoder_2(x_5)
        x_7 = self.decoder_3(x_6)
        # x_8 = self.decoder_4(x_7)
        x_final = self.last_conv(x_7)
        logit = self.final_activation(x_final)
        # print(logit.shape)
        rectified_para = torch.mul(x,logit)
        return rectified_para
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AENet_V2(nn.Module):
    def __init__(self, in_channels, out_channels,layer_order,f_maps,num_groups,is_segmentation, **kwargs):
    # def __init__(self, in_channels, out_channels, **kwargs):
        super(AENet_V2, self).__init__()
        self.pre_net = UNet3D(in_channels, out_channels,f_maps=f_maps,layer_order=layer_order,num_groups=num_groups,
        is_segmentation=is_segmentation)
        self.encoder_1 = AR_Encoder_block(in_channels,16, apply_pooling=True)
        self.encoder_2 = AR_Encoder_block(16,32, apply_pooling=True)
        self.encoder_3 = AR_Encoder_block(32,64, apply_pooling=False)
        self.encoder_4 = AR_Encoder_block(64,128, apply_pooling=False)
        self.decoder_1 = AR_Decoder_block(128,64,apply_transconv=True)
        self.decoder_2 = AR_Decoder_block(64,32,apply_transconv=True)
        self.decoder_3 = AR_Decoder_block(32,16,apply_transconv=False)
        # self.decoder_4 = AR_Decoder_block(64,32,apply_transconv=False)
        self.last_conv = nn.Conv3d(16,out_channels,kernel_size=3,padding=1)
        self.final_activation = nn.Tanh()

    def forward(self,x):
        x_pre = self.pre_net(x)
        residual = x_pre-x
        x_1 = self.encoder_1(residual)
        x_2 = self.encoder_2(x_1)
        x_3 = self.encoder_3(x_2)
        x_4 = self.encoder_4(x_3)
        x_5 = self.decoder_1(x_4)
        x_6 = self.decoder_2(x_5)
        x_7 = self.decoder_3(x_6)
        # x_8 = self.decoder_4(x_7)
        x_final = self.last_conv(x_7)
        logit = self.final_activation(x_final)
        # print(logit.shape)
        rectified_para = torch.mul(residual,logit)
        return x_pre, rectified_para
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = [nn.Conv3d(input_nc, ngf, kernel_size=7, padding=3),
                 norm_layer(ngf, affine=True),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2, affine=True),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2), affine=True),
                      nn.ReLU(True)]

        model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=3)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert(padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out



class DilatedNet(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=32,
                 norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[]):
        super(DilatedNet, self).__init__()
        self.gpu_ids = gpu_ids

        DilatedCount = 1
        inputCount = 0

        # construct unet structure
        model = [nn.Conv3d(input_nc, ngf, kernel_size=3, padding=DilatedCount, 
                       dilation = DilatedCount),
                       norm_layer(ngf, affine=True),
                       nn.LeakyReLU(0.2, True)]
        DilatedCount *= 2
        inputCount += ngf
        for i in range(num_downs - 2):
            model += [DilatedBlock(inputCount, ngf, DilatedCount, norm_layer, use_dropout)]
            DilatedCount *= 2
            inputCount += ngf

        model += [nn.Conv3d(inputCount, ngf, kernel_size=3, padding=1),
                       norm_layer(ngf, affine=True),
                       nn.LeakyReLU(0.2, True)]
    
        # model += [nn.Conv3d(ngf, output_nc, kernel_size=3, padding=1),
        #           nn.Tanh()]
        model += [nn.Conv3d(ngf, output_nc, kernel_size=3, padding=1)]

        # unet_block += [nn.Softmax(dim=1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if  self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class DilatedBlock(nn.Module):
    def __init__(self, input_nc, output_nc, dilatedRate, norm_layer, use_dropout):
        super(DilatedBlock, self).__init__()
        self.conv_block = self.build_Dilatedconv_block(input_nc, output_nc, dilatedRate, norm_layer, use_dropout)

    def build_Dilatedconv_block(self, input_nc, output_nc, dilatedRate, norm_layer, use_dropout):
        conv_block = []
        # TODO: InstanceNorm
        conv_block += [nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=dilatedRate, 
                       dilation = dilatedRate),
                       norm_layer(output_nc, affine=True),
                       nn.LeakyReLU(0.2, True)]
        # if use_dropout:
        #     conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = torch.cat([self.conv_block(x), x], 1)
        return out



# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids


        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        # unet_block += [nn.Softmax(dim=1)]

        self.model = unet_block

    def forward(self, input):
        # print(input.size(), self.model(input).size())
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1)
            # upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            down = [downconv]
            # up = [uprelu, upconv, nn.Tanh()]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up


        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            # print(self.model(x).size())
            return self.model(x)
        else:
            # return torch.cat([self.model(x), x], 1)
            output = self.model(x)
            size = x.size()[2:]
            middle_output = F.interpolate(output, size=size, mode='trilinear')
            return torch.cat([middle_output, x], 1)

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=True,**kwargs):
        super(Discriminator, self).__init__()

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        input_conv = nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
        sequence = [
            input_conv,
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            intermediate_conv = nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                                kernel_size=kw, stride=2, padding=padw)
            sequence += [
                intermediate_conv,
                # TODO: use InstanceNorm
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        intermediate_conv2 = nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw)
        sequence += [
            intermediate_conv2,
            # TODO: useInstanceNorm
            norm_layer(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        last_conv = nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)

        sequence += [last_conv]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids)  and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        #     return self.model(input)
        return self.model(input)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def init_weights(net, init_type='normal'):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=0.02)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=1)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm3d') != -1:
            init.normal(m.weight.data, 1.0, 0.02)
            init.constant(m.bias.data, 0.0)
    return init_func

def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm3d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm3d
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type=init_type)
    net.apply(init_weights(init_type))
    return net

# class Generator:
#     def __init__(self, in_channels, out_channels, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], **kwargs):
#         super(Generator, self).__init__()
def define_G(in_channels, out_channels, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], **kwargs):
        netG = None
        use_gpu = len(gpu_ids) > 0
        norm_layer = get_norm_layer(norm_type=norm)

        if use_gpu:
            assert(torch.cuda.is_available())

        input_nc = in_channels
        output_nc = out_channels

        if which_model_netG == 'resnet_9blocks':
            netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
        elif which_model_netG == 'resnet_6blocks':
            netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
        elif which_model_netG == 'resnet_5blocks':
            netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=5, gpu_ids=gpu_ids)
        elif which_model_netG == 'unet_32':
            netG = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
        elif which_model_netG == 'unet_64':
            netG = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
        elif which_model_netG == 'unet_128':
            netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
        elif which_model_netG == 'unet_256':
            netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
        elif which_model_netG == 'dilated_32':
            netG = DilatedNet(input_nc, output_nc, 5, ngf=32, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
        elif which_model_netG == 'dilated_64':
            netG = DilatedNet(input_nc, output_nc, 6, ngf=32, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
        elif which_model_netG == 'dilated_128':
            netG = DilatedNet(input_nc, output_nc, 7, ngf=32, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
        else:
            print('Generator model name [%s] is not recognized' % which_model_netG)

        init_type='kaiming'

        ###### new version #############
        return init_net(netG, init_type, gpu_ids)
        # return netG

def get_model(model_config):
    model_class = get_class(model_config['name'], modules=['unet3d.model'])
    return model_class(**model_config)
