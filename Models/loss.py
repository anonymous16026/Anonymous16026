import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from typing import Tuple, Dict, Optional, List

from Models.ade20k import ModelBuilder

"""
original from 
https://github.com/advimman/lama/blob/main/saicinpainting/training/losses/adversarial.py
"""


class BaseAdversarialLoss:
    def pre_generator_step(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        """
        Prepare for generator step
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param generator:
        :param discriminator:
        :return: None
        """

    def pre_discriminator_step(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        """
        Prepare for discriminator step
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param generator:
        :param discriminator:
        :return: None
        """

    def generator_loss(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        discr_real_pred: torch.Tensor,
        discr_fake_pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate generator loss
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param discr_real_pred: Tensor, discriminator output for real_batch
        :param discr_fake_pred: Tensor, discriminator output for fake_batch
        :param mask: Tensor, actual mask, which was at input of generator when making fake_batch
        :return: total generator loss along with some values that might be interesting to log
        """
        raise NotImplemented()

    def discriminator_loss(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        discr_real_pred: torch.Tensor,
        discr_fake_pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate discriminator loss and call .backward() on it
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param discr_real_pred: Tensor, discriminator output for real_batch
        :param discr_fake_pred: Tensor, discriminator output for fake_batch
        :param mask: Tensor, actual mask, which was at input of generator when making fake_batch
        :return: total discriminator loss along with some values that might be interesting to log
        """
        raise NotImplemented()

    def interpolate_mask(self, mask, shape):
        assert mask is not None
        assert self.allow_scale_mask or shape == mask.shape[-2:]
        if shape != mask.shape[-2:] and self.allow_scale_mask:
            if self.mask_scale_mode == "maxpool":
                mask = F.adaptive_max_pool2d(mask, shape)
            else:
                mask = F.interpolate(mask, size=shape, mode=self.mask_scale_mode)
        return mask


def make_r1_gp(discr_real_pred, real_batch):
    if torch.is_grad_enabled():
        grad_real = torch.autograd.grad(
            outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True
        )[0]
        grad_penalty = (
            grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2
        ).mean()
    else:
        grad_penalty = 0
    real_batch.requires_grad = False

    return grad_penalty


class NonSaturatingWithR1(BaseAdversarialLoss):
    def __init__(
        self,
        gp_coef=5,
        weight=1,
        mask_as_fake_target=True,
        allow_scale_mask=True,
        mask_scale_mode="nearest",
        extra_mask_weight_for_gen=0,
        use_unmasked_for_gen=True,
        use_unmasked_for_discr=True,
    ):
        self.gp_coef = gp_coef
        self.weight = weight
        # use for discr => use for gen;
        # otherwise we teach only the discr to pay attention to very small difference
        assert use_unmasked_for_gen or (not use_unmasked_for_discr)
        # mask as target => use unmasked for discr:
        # if we don't care about unmasked regions at all
        # then it doesn't matter if the value of mask_as_fake_target is true or false
        assert use_unmasked_for_discr or (not mask_as_fake_target)
        self.use_unmasked_for_gen = use_unmasked_for_gen
        self.use_unmasked_for_discr = use_unmasked_for_discr
        self.mask_as_fake_target = mask_as_fake_target
        self.allow_scale_mask = allow_scale_mask
        self.mask_scale_mode = mask_scale_mode
        self.extra_mask_weight_for_gen = extra_mask_weight_for_gen

    def generator_loss(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        discr_real_pred: torch.Tensor,
        discr_fake_pred: torch.Tensor,
        mask=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fake_loss = F.softplus(-discr_fake_pred)
        if (
            self.mask_as_fake_target and self.extra_mask_weight_for_gen > 0
        ) or not self.use_unmasked_for_gen:  # == if masked region should be treated differently
            mask = self.interpolate_mask(mask, discr_fake_pred.shape[-2:])
            if not self.use_unmasked_for_gen:
                fake_loss = fake_loss * mask
            else:
                pixel_weights = 1 + mask * self.extra_mask_weight_for_gen
                fake_loss = fake_loss * pixel_weights

        return fake_loss.mean() * self.weight, dict()

    def pre_discriminator_step(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        real_batch.requires_grad = True

    def discriminator_loss(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        discr_real_pred: torch.Tensor,
        discr_fake_pred: torch.Tensor,
        mask=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        real_loss = F.softplus(-discr_real_pred)
        grad_penalty = make_r1_gp(discr_real_pred, real_batch) * self.gp_coef
        fake_loss = F.softplus(discr_fake_pred)

        if not self.use_unmasked_for_discr or self.mask_as_fake_target:
            # == if masked region should be treated differently
            mask = self.interpolate_mask(mask, discr_fake_pred.shape[-2:])
            # use_unmasked_for_discr=False only makes sense for fakes;
            # for reals there is no difference beetween two regions
            fake_loss = fake_loss * mask
            if self.mask_as_fake_target:
                fake_loss = fake_loss + (1 - mask) * F.softplus(-discr_fake_pred)

        sum_discr_loss = real_loss + grad_penalty + fake_loss
        metrics = dict(
            discr_real_out=discr_real_pred.mean(),
            discr_fake_out=discr_fake_pred.mean(),
            discr_real_gp=grad_penalty,
        )
        return sum_discr_loss.mean(), metrics


def masked_l1_loss(pred, target, mask, weight_known, weight_missing):
    per_pixel_l1 = F.l1_loss(pred, target, reduction="none")
    # per_pixel_l1 = F.mse_loss(pred, target, reduction="none")
    pixel_weights = mask * weight_missing + (1 - mask) * weight_known
    return (pixel_weights * per_pixel_l1).mean()


def feature_matching_loss(
    fake_features: List[torch.Tensor], target_features: List[torch.Tensor], mask=None
):
    if mask is None:
        res = torch.stack(
            [
                F.mse_loss(fake_feat, target_feat)
                for fake_feat, target_feat in zip(fake_features, target_features)
            ]
        ).mean()
    else:
        res = 0
        norm = 0
        for fake_feat, target_feat in zip(fake_features, target_features):
            cur_mask = F.interpolate(
                mask, size=fake_feat.shape[-2:], mode="bilinear", align_corners=False
            )
            error_weights = 1 - cur_mask
            cur_val = ((fake_feat - target_feat).pow(2) * error_weights).mean()
            res = res + cur_val
            norm += 1
        res = res / norm
    return res


IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]


class ResNetPL(nn.Module):
    def __init__(
        self,
        weight=1,
        weights_path="weights",
        arch_encoder="resnet50dilated",
        segmentation=True,
    ):
        super().__init__()
        self.impl = ModelBuilder.get_encoder(
            weights_path=weights_path,
            arch_encoder=arch_encoder,
            arch_decoder="ppm_deepsup",
            fc_dim=2048,
            segmentation=segmentation,
        )
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)

        self.weight = weight

    def forward(self, pred, target):
        pred = (pred - IMAGENET_MEAN.to(pred)) / IMAGENET_STD.to(pred)
        target = (target - IMAGENET_MEAN.to(target)) / IMAGENET_STD.to(target)

        pred_feats = self.impl(pred, return_feature_maps=True)
        target_feats = self.impl(target, return_feature_maps=True)

        result = (
            torch.stack(
                [
                    F.mse_loss(cur_pred, cur_target)
                    for cur_pred, cur_target in zip(pred_feats, target_feats)
                ]
            ).sum()
            * self.weight
        )
        return result


class BCELoss(BaseAdversarialLoss):
    def __init__(self, weight):
        self.weight = weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def generator_loss(
        self, discr_fake_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        real_mask_gt = torch.zeros(discr_fake_pred.shape).to(discr_fake_pred.device)
        fake_loss = self.bce_loss(discr_fake_pred, real_mask_gt) * self.weight
        return fake_loss, dict()

    def pre_discriminator_step(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        real_batch.requires_grad = True

    def discriminator_loss(
        self,
        mask: torch.Tensor,
        discr_real_pred: torch.Tensor,
        discr_fake_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        real_mask_gt = torch.zeros(discr_real_pred.shape).to(discr_real_pred.device)
        sum_discr_loss = (
            self.bce_loss(discr_real_pred, real_mask_gt)
            + self.bce_loss(discr_fake_pred, mask)
        ) / 2
        metrics = dict(
            discr_real_out=discr_real_pred.mean(),
            discr_fake_out=discr_fake_pred.mean(),
            discr_real_gp=0,
        )
        return sum_discr_loss, metrics


#################################################################################################################
#################################################################################################################
class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module("vgg", VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(
            self.compute_gram(x_vgg["relu2_2"]), self.compute_gram(y_vgg["relu2_2"])
        )
        style_loss += self.criterion(
            self.compute_gram(x_vgg["relu3_4"]), self.compute_gram(y_vgg["relu3_4"])
        )
        style_loss += self.criterion(
            self.compute_gram(x_vgg["relu4_4"]), self.compute_gram(y_vgg["relu4_4"])
        )
        style_loss += self.criterion(
            self.compute_gram(x_vgg["relu5_2"]), self.compute_gram(y_vgg["relu5_2"])
        )

        return style_loss


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module("vgg", VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(
            x_vgg["relu1_1"], y_vgg["relu1_1"]
        )
        content_loss += self.weights[1] * self.criterion(
            x_vgg["relu2_1"], y_vgg["relu2_1"]
        )
        content_loss += self.weights[2] * self.criterion(
            x_vgg["relu3_1"], y_vgg["relu3_1"]
        )
        content_loss += self.weights[3] * self.criterion(
            x_vgg["relu4_1"], y_vgg["relu4_1"]
        )
        content_loss += self.weights[4] * self.criterion(
            x_vgg["relu5_1"], y_vgg["relu5_1"]
        )

        return content_loss


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            "relu1_1": relu1_1,
            "relu1_2": relu1_2,
            "relu2_1": relu2_1,
            "relu2_2": relu2_2,
            "relu3_1": relu3_1,
            "relu3_2": relu3_2,
            "relu3_3": relu3_3,
            "relu3_4": relu3_4,
            "relu4_1": relu4_1,
            "relu4_2": relu4_2,
            "relu4_3": relu4_3,
            "relu4_4": relu4_4,
            "relu5_1": relu5_1,
            "relu5_2": relu5_2,
            "relu5_3": relu5_3,
            "relu5_4": relu5_4,
        }
        return out
