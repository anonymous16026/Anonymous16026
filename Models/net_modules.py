import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
#from einops import rearrange as rearrange

from timm.models.layers import trunc_normal_, DropPath
from timm.layers.helpers import to_2tuple

from itertools import repeat


###############################################################################################
###############################################################################################
class Pad(nn.Module):
    """Pad input tensor B,C,H, W in a circular fashion along longitudinal axis and reflect along the poles.
    """

    def __init__(
        self,
        pad=3,
    ):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, (self.pad, self.pad, 0, 0), "circular")
        if self.pad <= H:
            x = F.pad(x, (0, 0, self.pad, self.pad), "reflect")
        else:
            x = F.pad(x, (0, 0, self.pad, self.pad), "constant", value=0)
        return x


class ConvPano(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        padding=1,
        kernel_size=3,
        stride=1,
        bias=False,
        groups=1,
        dilation=1,
    ):
        super(ConvPano, self).__init__()
        self.pad = Pad(padding)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=stride,
            bias=bias,
            groups=groups,
            dilation=dilation,
        )

    def forward(self, x):
        return self.conv(self.pad(x))


###############################################################################################
###############################################################################################

"""
adapted from 
https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py
"""
class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x * self.scale
        x = x.permute(0, 3, 1, 2)
        return x


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(
        self,
        scale_value=1.0,
        bias_value=0.0,
        scale_learnable=True,
        bias_learnable=True,
        mode=None,
        inplace=False,
    ):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(
            scale_value * torch.ones(1), requires_grad=scale_learnable
        )
        self.bias = nn.Parameter(
            bias_value * torch.ones(1), requires_grad=bias_learnable
        )

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class LayerNormGeneral(nn.Module):
    r"""General LayerNorm for different situations.

    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default.
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance.
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.

        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.

        For the several metaformer baslines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """

    def __init__(
        self, affine_shape=None, normalized_dim=(-1,), scale=True, bias=True, eps=1e-5
    ):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)

        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        x = x.permute(0, 3, 1, 2)
        return x


class LayerNormWithoutBias(nn.Module):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster,
    because it directly utilizes otpimized F.layer_norm
    """

    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        return F.layer_norm(
            x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps
        ).permute(0, 3, 1, 2)

class FourierUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groups=1,
        fft_norm="ortho",
        norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
        act_layer=partial(StarReLU),
        spectral_pos_encoding=False,
    ):
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )

        self.act = act_layer()
        self.norm = norm_layer(out_channels * 2)
        self.spectral_pos_encoding = spectral_pos_encoding

        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view(
            (
                batch,
                -1,
            )
            + ffted.size()[3:]
        )

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = (
                torch.linspace(0, 1, height)[None, None, :, None]
                .expand(batch, 1, height, width)
                .to(ffted)
            )
            coords_hor = (
                torch.linspace(0, 1, width)[None, None, None, :]
                .expand(batch, 1, height, width)
                .to(ffted)
            )
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        # ffted = ffted + self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act(self.norm(ffted))

        ffted = (
            ffted.view(
                (
                    batch,
                    -1,
                    2,
                )
                + ffted.size()[2:]
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(
            ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm
        )
        return output


class SpectralTransform(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groups=1,
        enable_lfu=False,
        act_layer=StarReLU,
        norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
        **fu_kwargs,
    ):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False
            ),
            norm_layer(out_channels // 2),
            act_layer(),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)

        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)

        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False
        )
        # self.scale = Scale(dim=out_channels // 2, init_value=0)

    def forward(self, x):
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(
                torch.split(x[:, : c // 4], split_s_h, dim=-2), dim=1
            ).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        # output = self.conv2(x + self.scale(output) + xs)
        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):
    def __init__(
        self,
        dim,
        ratio_gin=0.5,
        ratio_gout=0.5,
        spatial_block=nn.Identity,
        enable_lfu=True,
        gated=False,
        groups=1,
        act_layer=nn.Identity,
        norm_Layer=nn.Identity,
        **kwargs,
    ):
        super(FFC, self).__init__()

        in_cg = int(dim * ratio_gin)
        in_cl = dim - in_cg
        out_cg = int(dim * ratio_gout)
        out_cl = dim - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        # Local to local
        module = nn.Identity if in_cl == 0 or out_cl == 0 else spatial_block
        self.convl2l = module(in_channels=in_cl, out_channels=out_cl)

        # Local to global
        module = nn.Identity if in_cl == 0 or out_cg == 0 else spatial_block
        self.convl2g = module(in_channels=in_cl, out_channels=out_cg)

        # Global to local
        module = nn.Identity if in_cg == 0 or out_cl == 0 else spatial_block
        self.convg2l = module(in_channels=in_cg, out_channels=out_cl)

        # Global to global
        module = (
            nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        )  # SpectralTransform
        self.convg2g = module(
            in_channels=in_cg,
            out_channels=out_cg,
            groups=1 if groups == 1 else groups // 2,
            enable_lfu=enable_lfu,
        )

        self.gated = gated
        module = (
            nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        )
        self.gate = module(dim, 2, 1)

        self.actl = act_layer() if out_cl > 0 else nn.Identity()
        self.actg = act_layer() if out_cg > 0 else nn.Identity()
        self.norml = norm_Layer(out_cl) if out_cl > 0 else nn.Identity()
        self.normg = norm_Layer(out_cg) if out_cg > 0 else nn.Identity()

    def forward(self, x):
        if self.global_in_num != 0:
            x_l, x_g = (
                x[:, : -self.global_in_num],
                x[:, -self.global_in_num :],
            )
        else:
            x_l, x_g = x, 0
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            i1 = self.convl2l(x_l)
            i2 = self.convg2l(x_g)
            out_xl = i1 + i2 * g2l_gate
        if self.ratio_gout != 0:
            if self.ratio_gin == 1:
                return self.convg2g(x_g)

            else:
                i1 = self.convl2g(x_l) * l2g_gate
                i2 = self.convg2g(x_g)
                out_xg = i1 + i2

        out_xl = self.actl(self.norml(out_xl))
        out_xg = self.actg(self.normg(out_xg))
        out = torch.cat([out_xl, out_xg], dim=1)
        return out


class Mlp(nn.Module):
    """MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4,
        out_features=None,
        act_layer=StarReLU,
        drop=0.0,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """

    def __init__(
        self,
        dim,
        token_mixer=nn.Identity,
        mlp=Mlp,
        norm_layer=nn.LayerNorm,
        drop=0.0,
        drop_path=0.0,
        layer_scale_init_value=None,
        res_scale_init_value=None,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim)  # token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale1 = (
            Scale(dim=dim, init_value=layer_scale_init_value)
            if layer_scale_init_value
            else nn.Identity()
        )
        self.res_scale1 = (
            Scale(dim=dim, init_value=res_scale_init_value)
            if res_scale_init_value
            else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale2 = (
            Scale(dim=dim, init_value=layer_scale_init_value)
            if layer_scale_init_value
            else nn.Identity()
        )
        self.res_scale2 = (
            Scale(dim=dim, init_value=res_scale_init_value)
            if res_scale_init_value
            else nn.Identity()
        )

    def forward(self, x):
        x = self.res_scale1(x) + self.layer_scale1(
            self.drop_path1(self.token_mixer(self.norm1(x)))
        )
        x = self.res_scale2(x) + self.layer_scale2(
            self.drop_path2(self.mlp(self.norm2(x)))
        )
        return x


class Downsample_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        downsample_op=partial(
            nn.Conv2d,
            kernel_size=3,
            padding=1,
            stride=2,
        ),
        norm_layer=nn.Identity,
        prenorm=True,
        postnorm=False,
    ):
        super(Downsample_block, self).__init__()

        self.prenorm = norm_layer(in_channels) if prenorm else nn.Identity()
        self.postnorm = norm_layer(out_channels) if postnorm else nn.Identity()
        self.downsample = downsample_op(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, x):
        x = self.prenorm(x)
        x = self.downsample(x)
        x = self.postnorm(x)
        return x


class Upsample_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        upsample_op=partial(
            nn.ConvTranspose2d,
            kernel_size=4,
            padding=1,
            stride=2,
        ),
        norm_layer=nn.Identity,
        prenorm=True,
        postnorm=False,
    ):
        super(Upsample_block, self).__init__()

        self.prenorm = norm_layer(in_channels) if prenorm else nn.Identity()
        self.postnorm = norm_layer(out_channels) if postnorm else nn.Identity()

        self.downsample = upsample_op(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, x):
        x = self.prenorm(x)
        x = self.downsample(x)
        x = self.postnorm(x)
        return x


class FuseConv(nn.Module):
    def __init__(
        self, channels, kernel_size=1, padding=0, stride=1, padding_mode="reflect"
    ):
        super(FuseConv, self).__init__()

        self.fuse_op = nn.Conv2d(
            in_channels=channels * 2,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            padding_mode=padding_mode,
        )
        self.norm_layer = LayerNormWithoutBias(channels * 2)

    def forward(self, dec, skip):
        x = torch.cat([dec, skip], dim=1)
        x = self.fuse_op(self.norm_layer(x))
        # x = self.fuse_op(x)
        return x


class FuseAdd(nn.Module):
    def __init__(self, **conv_kwargs):
        super(FuseAdd, self).__init__()

    def forward(self, dec, skip):
        x = dec + skip
        return x


class SkipConnectionIdentity(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, decoder, skip):
        return skip

class GatedConv(torch.nn.Module):
    def __init__(
        self,
        dim,
        out_channels=None,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        norm_layer=nn.Identity,  # partial(LayerNormWithoutBias, eps=1e-6),
        act_layer=nn.Identity,  # StarReLU,
    ):
        super(GatedConv, self).__init__()
        out_channels = out_channels or dim
        self.norm_layer = norm_layer(out_channels)
        self.act_layer = act_layer()
        self.conv2d = ConvPano(
            dim,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.mask_conv2d = ConvPano(
            dim,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = x * self.sigmoid(mask)
        return self.act_layer(self.norm_layer(x))


convFFC_at_75_global_ratio = partial(
    FFC,
    ratio_gin=0.75,
    ratio_gout=0.75,
    spatial_block=partial(
        ConvPano,
        kernel_size=3,
        padding=1,
        stride=1,
    ),
    enable_lfu=False,
    gated=False,
    groups=1,
)

fuseconv1x1 = partial(
    FuseConv, kernel_size=1, padding=0, stride=1, padding_mode="reflect"
)

fuseconv3x3 = partial(
    FuseConv, kernel_size=3, padding=1, stride=1, padding_mode="zeros"
)

transposeStemUp = partial(
    Upsample_block,
    upsample_op=partial(
        nn.ConvTranspose2d,
        kernel_size=7,
        stride=4,
        padding=3,
        output_padding=3,
    ),
    norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
    prenorm=True,
    postnorm=False,
)
transposeUp = partial(
    Upsample_block,
    upsample_op=partial(
        nn.ConvTranspose2d,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
    ),
    norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
    prenorm=True,
    postnorm=False,
)

convStemDown = partial(
    Downsample_block,
    downsample_op=partial(
        ConvPano,
        kernel_size=7,
        stride=4,
        padding=3,
    ),
    norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
    prenorm=False,
    postnorm=True,
)


convDown = partial(
    Downsample_block,
    downsample_op=partial(
        ConvPano,
        kernel_size=3,
        stride=2,
        padding=1,
    ),
    norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
    prenorm=True,
    postnorm=False,
)


class NoSkip(nn.Module):
    def __init__(self, **kwargs):
        super(NoSkip, self).__init__()

    def forward(self, dec, skip):
        return dec


class FourierUnitColumnWise(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groups=1,
        fft_norm="ortho",
        norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
        act_layer=partial(StarReLU),
        spectral_pos_encoding=False,
    ):
        super(FourierUnitColumnWise, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )

        self.act = act_layer()
        self.norm = norm_layer(in_channels * 2)
        self.spectral_pos_encoding = spectral_pos_encoding

        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        # (batch, c, h, w/2+1, 2)
        fft_dim = -2
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view(
            (
                batch,
                -1,
            )
            + ffted.size()[3:]
        )

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = (
                torch.linspace(0, 1, height)[None, None, :, None]
                .expand(batch, 1, height, width)
                .to(ffted)
            )
            coords_hor = (
                torch.linspace(0, 1, width)[None, None, None, :]
                .expand(batch, 1, height, width)
                .to(ffted)
            )
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        # ffted = ffted + self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act(self.norm(ffted))

        ffted = (
            ffted.view(
                (
                    batch,
                    -1,
                    2,
                )
                + ffted.size()[2:]
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2]
        output = torch.fft.irfftn(
            ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm
        )
        return output


class FourierUnitRowWise(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groups=1,
        fft_norm="ortho",
        norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
        act_layer=partial(StarReLU),
        spectral_pos_encoding=False,
    ):
        super(FourierUnitRowWise, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )

        self.act = act_layer()
        self.norm = norm_layer(in_channels * 2)
        self.spectral_pos_encoding = spectral_pos_encoding

        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        # (batch, c, h, w/2+1, 2)
        fft_dim = -1
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view(
            (
                batch,
                -1,
            )
            + ffted.size()[3:]
        )

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = (
                torch.linspace(0, 1, height)[None, None, :, None]
                .expand(batch, 1, height, width)
                .to(ffted)
            )
            coords_hor = (
                torch.linspace(0, 1, width)[None, None, None, :]
                .expand(batch, 1, height, width)
                .to(ffted)
            )
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        # ffted = ffted + self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act(self.norm(ffted))

        ffted = (
            ffted.view(
                (
                    batch,
                    -1,
                    2,
                )
                + ffted.size()[2:]
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-1]
        output = torch.fft.irfftn(
            ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm
        )
        return output


def window_partition(x, window_size_h, window_size_w):
    B, C, H, W = x.shape
    x = x.view(
        B, C, H // window_size_h, window_size_h, W // window_size_w, window_size_w
    )
    windows = (
        x.permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .view(-1, C, window_size_h, window_size_w)
    )
    return windows


def window_reverse(windows, window_size_h, window_size_w, H, W):
    B = int(windows.shape[0] / (H * W / window_size_h / window_size_w))
    x = windows.view(
        B, H // window_size_h, W // window_size_w, -1, window_size_h, window_size_w
    )
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return x

class SpectralMixGated(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super(SpectralMixGated, self).__init__()
        self.spectral_row = FourierUnitRowWise(dim, dim)
        self.spectral_column = FourierUnitColumnWise(dim, dim)
        self.spatial_mix = GatedConv(dim * 3, dim, padding=1, kernel_size=3)

    def forward(self, x):
        x_row = self.spectral_row(x)
        x_column = self.spectral_column(x)
        x = self.spatial_mix(torch.cat([x, x_row, x_column], dim=1))
        return x


class SpectralMixGatedWindow(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super(SpectralMixGatedWindow, self).__init__()
        self.conv1 = GatedConv(
            dim,
            dim // 2,
            norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
            act_layer=partial(StarReLU),
        )
        self.spectral_row = FourierUnitRowWise(dim // 2, dim // 2)
        self.spectral_column = FourierUnitColumnWise(dim // 2, dim // 2)
        self.spatial_mix = GatedConv(dim * 3, dim, padding=1, kernel_size=3)

    def forward(self, x):
        x_init = x
        B, C, H, W = x.shape
        w = W // 2
        x = self.conv1(x)
        x_row = self.spectral_row(x)
        x_row_window = window_reverse(
            self.spectral_row(window_partition(x, H, w)), H, w, H, W
        )
        x_column = self.spectral_column(x)
        x_column_window = window_reverse(
            self.spectral_column(window_partition(x, H, w)), H, w, H, W
        )
        x = self.spatial_mix(
            torch.cat([x_init, x_row, x_column, x_row_window, x_column_window], dim=1)
        )
        return x

class FourierWindow(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super(FourierWindow, self).__init__()
        self.conv1 = GatedConv(
            dim,
            dim // 2,
            norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
            act_layer=partial(StarReLU),
        )
        self.spectral_mix = FourierUnit(dim // 2, dim // 2)
        self.spatial_mix = GatedConv(dim * 2, dim, padding=1, kernel_size=3)

    def forward(self, x):
        x_init = x
        B, C, H, W = x.shape
        w = W // 2
        x = self.conv1(x)
        x_spectral = self.spectral_mix(x)
        x_spectral_window = window_reverse(
            self.spectral_mix(window_partition(x, H, w)), H, w, H, W
        )
        x = self.spatial_mix(torch.cat([x_init, x_spectral, x_spectral_window], dim=1))
        return x