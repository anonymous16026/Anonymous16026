import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
from timm.layers.helpers import to_2tuple
import collections.abc
from .net_modules import *


class UFormer(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        dims=[64, 128, 320],
        depths=[2, 2, 3],
        token_mixers_encoder=nn.Identity,
        token_mixers_decoder=None,
        skip_connection_layers=nn.Identity,
        fuse_layers=FuseAdd,
        downsample_layers=[convStemDown] + [convDown] * 2,
        upsample_layers=[transposeStemUp] + [transposeUp] * 2,
        mlps=Mlp,
        norm_layers=nn.Identity,
        drop_path_rate=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=None,
        act_layer=nn.Tanh,
    ):
        super(UFormer, self).__init__()

        if not isinstance(depths, (list, tuple)):
            depths = [depths]  # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_channels] + dims
        self.downsample_layers = nn.ModuleList(
            [
                downsample_layers[i](down_dims[i], down_dims[i + 1])
                for i in range(num_stage)
            ]
        )

        if not isinstance(upsample_layers, (list, tuple)):
            upsample_layers = [upsample_layers] * num_stage
        up_dims = [out_channels] + dims
        self.upsample_layers = nn.ModuleList(
            [upsample_layers[i](up_dims[i + 1], up_dims[i]) for i in range(num_stage)]
        )

        if not isinstance(token_mixers_encoder, (list, tuple)):
            token_mixers_encoder = [token_mixers_encoder] * num_stage
        if token_mixers_decoder == None:
            token_mixers_decoder = token_mixers_encoder
        if not isinstance(token_mixers_decoder, (list, tuple)):
            token_mixers_decoder = [token_mixers_decoder] * num_stage

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage

        if not isinstance(skip_connection_layers, (list, tuple)):
            skip_connection_layers = [skip_connection_layers] * (num_stage - 1)

        if not isinstance(fuse_layers, (list, tuple)):
            fuse_layers = [fuse_layers] * (num_stage - 1)

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        # ENCODER
        self.stages_encoder = (
            nn.ModuleList()
        )  # each stage consists of multiple metaformer blocks
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[
                    MetaFormerBlock(
                        dim=dims[i],
                        token_mixer=token_mixers_encoder[i],
                        mlp=mlps[i],
                        norm_layer=norm_layers[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_values[i],
                        res_scale_init_value=res_scale_init_values[i],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages_encoder.append(stage)
            cur += depths[i]

        # DECODER
        self.stages_decoder = (
            nn.ModuleList()
        )  # each stage consists of multiple metaformer blocks
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[
                    MetaFormerBlock(
                        dim=dims[i],
                        token_mixer=token_mixers_decoder[i],
                        mlp=mlps[i],
                        norm_layer=norm_layers[i],
                        drop_path=dp_rates[-cur - j - 1],
                        layer_scale_init_value=layer_scale_init_values[i],
                        res_scale_init_value=res_scale_init_values[i],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages_decoder.append(stage)
            cur += depths[i]

        # SKIP CONNECTIONS
        self.skip_connections = nn.ModuleList()
        for i in range(num_stage - 1):
            skip_connection = skip_connection_layers[i](
                decoder_in_channels=dims[i + 1],
                skip_in_channels=dims[i],
            )
            self.skip_connections.append(skip_connection)

        # FUSION
        self.fuse_layers = nn.ModuleList()
        for i in range(num_stage - 1):
            fuse_layer = fuse_layers[i](channels=dims[i])
            self.fuse_layers.append(fuse_layer)

        self.act_layer = act_layer()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        feats = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages_encoder[i](x)
            feats.append(x)

        return feats  # [enc1, enc2, enc3]

    def forward_decoder(self, skips):
        x = skips[-1]
        feats = []
        for i in range(self.num_stage):
            x = self.stages_decoder[-i - 1](x)
            feats.append(x)
            if i != self.num_stage - 1:
                skip_connection = self.skip_connections[-i - 1](
                    decoder=x, skip=skips[-i - 2]
                )

            x = self.upsample_layers[-i - 1](x)

            if i != self.num_stage - 1:
                x = self.fuse_layers[-i - 1](x, skip_connection)

        feats.append(x)
        return feats

    def forward(self, x):
        x = self.forward_encoder(x)
        x = self.forward_decoder(x)
        return self.act_layer(x[-1]), x[:-1]


#from .blocks import FFCResNetGenerator
# class Lama(nn.Module):
#     def __init__(self, input_nc=4, output_nc=3):
#         super(Lama, self).__init__()
#         self.net = FFCResNetGenerator(
#             input_nc=input_nc,
#             output_nc=3,
#             init_conv_kwargs={
#                 "ratio_gin": 0,
#                 "ratio_gout": 0,
#                 "enable_lfu": False,
#             },
#             downsample_conv_kwargs={
#                 "ratio_gin": 0,
#                 "ratio_gout": 0,
#                 "enable_lfu": False,
#             },
#             resnet_conv_kwargs={
#                 "ratio_gin": 0.75,
#                 "ratio_gout": 0.75,
#                 "enable_lfu": False,
#             },
#         )
#     def forward(self, x):
#         rgb = self.net(x)
#         return rgb, 0
# def lama():
#     """ """
#     model = Lama(input_nc=4, output_nc=3)
#     return model

def uformer_ffc_ffc():
    """ """
    model = UFormer(
        in_channels=4,
        out_channels=3,
        dims=[64, 128, 256, 512],
        depths=[2, 2, 2, 2],
        token_mixers_encoder=convFFC_at_75_global_ratio,
        token_mixers_decoder=convFFC_at_75_global_ratio,
        skip_connection_layers=SkipConnectionIdentity,
        fuse_layers=fuseconv1x1,
        downsample_layers=[convStemDown] + [convDown] * 3,
        upsample_layers=[transposeStemUp] + [transposeUp] * 3,
        mlps= Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=None,
        act_layer=nn.Tanh,
    )
    return model

def uformer_GatedConv():
    """ """
    model = UFormer(
        in_channels=4,
        out_channels=3,
        dims=[64, 128, 256, 512],
        depths=[2, 2, 2, 2],
        token_mixers_encoder=GatedConv,
        token_mixers_decoder=GatedConv,
        skip_connection_layers=SkipConnectionIdentity,
        fuse_layers=fuseconv1x1,
        downsample_layers=[convStemDown] + [convDown] * 3,
        upsample_layers=[transposeStemUp] + [transposeUp] * 3,
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=None,
        act_layer=nn.Tanh,
    )
    return model

def uformer_fourierWindow():
    """ """
    model = UFormer(
        in_channels=4,
        out_channels=3,
        dims=[64, 128, 256, 512],
        depths=[2, 2, 2, 2],
        token_mixers_encoder=FourierWindow,
        token_mixers_decoder=FourierWindow,
        skip_connection_layers=SkipConnectionIdentity,
        fuse_layers=fuseconv1x1,
        downsample_layers=[convStemDown] + [convDown] * 3,
        upsample_layers=[transposeStemUp] + [transposeUp] * 3,
        mlps= Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=None,
        act_layer=nn.Tanh,
    )
    return model


def uformer_fourierMix():
    """ """
    model = UFormer(
        in_channels=4,
        out_channels=3,
        dims=[64, 128, 256, 512],
        depths=[2, 2, 2, 2],
        token_mixers_encoder=SpectralMixGated,
        token_mixers_decoder=SpectralMixGated,
        skip_connection_layers=SkipConnectionIdentity,
        fuse_layers=fuseconv1x1,
        downsample_layers=[convStemDown] + [convDown] * 3,
        upsample_layers=[transposeStemUp] + [transposeUp] * 3,
        mlps= Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=None,
        act_layer=nn.Tanh,
    )
    return model

def uformer_fourierMixWindow():
    """ """
    model = UFormer(
        in_channels=4,
        out_channels=3,
        dims=[64, 128, 256, 512],
        depths=[2, 2, 2, 2],
        token_mixers_encoder=SpectralMixGatedWindow,
        token_mixers_decoder=SpectralMixGatedWindow,
        skip_connection_layers=SkipConnectionIdentity,
        fuse_layers=fuseconv1x1,
        downsample_layers=[convStemDown] + [convDown] * 3,
        upsample_layers=[transposeStemUp] + [transposeUp] * 3,
        mlps= Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=None,
        act_layer=nn.Tanh,
    )
    return model



