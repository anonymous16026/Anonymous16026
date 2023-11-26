import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from easydict import EasyDict


from tqdm import tqdm

from .discriminator import NLayerDiscriminator
from .loss import (
    NonSaturatingWithR1,
    ResNetPL,
    feature_matching_loss,
    masked_l1_loss,
    PerceptualLoss,
)
from Datasets.datasetPano import (
    Structured3D,
    Phase,
)

import pprint

from .utils import (
    Logger,
    denormalize,
    unpack_data,
    load_weights,
    save_checkpoints_with_optimizer,
)


class RoomCleaner(nn.Module):
    def __init__(self, model):
        super(RoomCleaner, self).__init__()
        self.net = model()

    def forward(self, x):
        rgb, feats = self.net(x)
        return rgb


def create_discriminator_sample(network_input, empty_room_rgb, mask):
    return empty_room_rgb
    # return torch.cat([empty_room_rgb, mask], dim=1)


def create_losses(config):
    pixel_loss = masked_l1_loss
    # pixel_loss = nn.L1Loss(reduction="mean").cuda()
    perceptual_loss = ResNetPL().cuda()  # PerceptualLoss().cuda()  #
    adversarial_loss = NonSaturatingWithR1(
        gp_coef=config.adversarial_loss_gp_coef, weight=1
    )
    feature_matching = feature_matching_loss
    return (
        pixel_loss,
        perceptual_loss,
        adversarial_loss,
        feature_matching,
    )


def generator_loss(
    config,
    input_rgb,
    fake_rgb,
    real_rgb,
    discriminator,
    pixel_loss,
    adversarial_loss,
    perceptual_loss,
    feature_matching_loss,
    mask,
):
    # PIXEL
    pixel = pixel_loss(
        fake_rgb, real_rgb, mask, 1, 0
    )  # + pixel_loss(fake_rgb_lr, real_rgb_lr, mask_lr, 1, 0)

    # ADVERSARIAL
    fake_sample = create_discriminator_sample(input_rgb, fake_rgb, mask)
    real_sample = create_discriminator_sample(input_rgb, real_rgb, mask)
    discr_real_pred, discr_real_features = discriminator(real_sample)
    discr_fake_pred, discr_fake_features = discriminator(fake_sample)
    adversarial, m = adversarial_loss.generator_loss(
        real_batch=real_sample,
        fake_batch=fake_sample,
        discr_real_pred=discr_real_pred,
        discr_fake_pred=discr_fake_pred,
        mask=mask,
    )

    # PERCEPTUAL
    perceptual = perceptual_loss(denormalize(fake_rgb), denormalize(real_rgb))

    # feature
    fm_loss = feature_matching_loss(
        discr_fake_features, discr_real_features
    )  # , mask=mask

    generator_loss = (
        (pixel * config.l1_pixel_loss_weight)
        + (adversarial * config.adversarial_loss_weight)
        + (perceptual * config.resnet_perceptual_loss_weight)
        + (fm_loss * config.feature_matching_loss_weight)
    )
    metrics_dict = {
        "Pixel": pixel,
        "Adversarial": adversarial,
        "Perceptual": perceptual,
        "Feature_matching": fm_loss,
        "Total": generator_loss,
    }

    return generator_loss, metrics_dict


def discriminator_loss(
    config,
    input_rgb,
    fake_rgb,
    real_rgb,
    discriminator,
    adversarial_loss,
    mask,
):
    # ADVERSARIAL
    fake_sample = create_discriminator_sample(
        input_rgb.detach(), fake_rgb.detach(), mask.detach()
    )
    real_sample = create_discriminator_sample(
        input_rgb.detach(), real_rgb.detach(), mask.detach()
    )
    real_sample.requires_grad = True
    discr_real_pred, discr_real_features = discriminator(real_sample)
    discr_fake_pred, discr_fake_features = discriminator(fake_sample)
    adversarial, adv_metrics = adversarial_loss.discriminator_loss(
        real_batch=real_sample,
        fake_batch=fake_sample,
        discr_real_pred=discr_real_pred,
        discr_fake_pred=discr_fake_pred,
        mask=mask,
    )

    discriminator_loss = adversarial
    metrics_dict = {**adv_metrics}
    metrics_dict["Total"] = discriminator_loss

    return discriminator_loss, metrics_dict


def create_optimizers(config, generator, discriminator):
    optimizer_Generator = torch.optim.AdamW(
        generator.parameters(),
        lr=config.generator_optimizer_lr,
        betas=config.generator_optimizer_betas,
        weight_decay=0.05,
    )
    optimizer_Discriminator = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config.discriminator_optimizer_lr,
        betas=config.discriminator_optimizer_betas,
        weight_decay=0.05,
    )
    discriminator.zero_grad()
    generator.zero_grad()

    return optimizer_Generator, optimizer_Discriminator


def create_dataloaders(config):
    train_dataloader = DataLoader(
        Structured3D(config.config_train_dataloader, Phase.TRAIN),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers_dataloader,
        drop_last=True,
    )
    validation_dataloader = DataLoader(
        Structured3D(config.config_validation_dataloader, Phase.VALIDATION),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers_dataloader,
        drop_last=True,
    )

    return train_dataloader, validation_dataloader


def make_scheduler(
    config,
    optimizer_Generator,
    optimizer_Discriminator,
    grad_accumulation_steps,
    train_dataloader,
):
    # scheduler_generator = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer_Generator, 10, 1
    # )
    # scheduler_discriminator = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer_Discriminator, 10, 1
    # )

    scheduler_generator = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_Generator,
        max_lr=config.generator_optimizer_lr,
        steps_per_epoch=grad_accumulation_steps * len(train_dataloader),
        epochs=config.num_epochs,
        pct_start=0.05,
        final_div_factor=1e6,
    )

    scheduler_discriminator = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_Discriminator,
        max_lr=config.discriminator_optimizer_lr,
        steps_per_epoch=grad_accumulation_steps * len(train_dataloader),
        epochs=config.num_epochs,
        pct_start=0.05,
        final_div_factor=1e6,
    )

    return scheduler_generator, scheduler_discriminator


def update_log_bools(config, images_seen_total):
    b_should_save_images = (
        images_seen_total % (config.save_images_every_kiter * 1e3)
    ) == 0
    b_should_log = (images_seen_total % (config.log_loss_every_kiter * 1e3)) == 0
    b_should_save_checkpoint = (
        images_seen_total % (config.save_checkpoint_every_kiter * 1e3)
    ) == 0
    return b_should_save_images, b_should_log, b_should_save_checkpoint


def train(config):
    torch.manual_seed(42)

    pprint.pprint(config)
    logger = Logger(config)

    train_dataloader, validation_dataloader = create_dataloaders(config)

    print("--------------------------------------------------")
    print(f"Train set has {len(train_dataloader.dataset)} samples")
    print(f"Validation set has {len(validation_dataloader.dataset)} samples")
    print("--------------------------------------------------")

    # Loss functions
    (
        pixel_loss,
        perceptual_loss,
        adversarial_loss,
        feature_matching,
    ) = create_losses(config)

    # create generator and discriminator
    generator = RoomCleaner(config.model).cuda()
    discriminator = NLayerDiscriminator(input_nc=3).cuda()

    # create optimizers
    optimizer_Generator, optimizer_Discriminator = create_optimizers(
        config, generator, discriminator
    )

    # load weights
    load_weights(
        config, generator, discriminator, optimizer_Generator, optimizer_Discriminator
    )

    grad_accumulation_steps = config.gradient_accumulation_steps

    # scheduler_generator, scheduler_discriminator = make_scheduler(
    #     config,
    #     optimizer_Generator,
    #     optimizer_Discriminator,
    #     grad_accumulation_steps,
    #     train_dataloader,
    # )

    # train loop
    images_seen_total = config.current_iter

    idx = 0
    for epoch in tqdm(range(config.current_epoch, config.num_epochs)):
        generator.train()
        discriminator.train()

        print("\n--TRAIN--\n")
        loss_gen = 0
        loss_disc = 0
        i = 0
        for data in tqdm(train_dataloader):
            (
                b_should_save_images,
                b_should_log,
                b_should_save_checkpoint,
            ) = update_log_bools(config, images_seen_total)

            network_input, empty_room, mask = unpack_data(data)

            batch_size = network_input.size(0)

            fake_rgb = generator(network_input.cuda())

            ###############
            ##### GENERATOR
            ###############
            metrics_dict = {}
            generator.zero_grad()
            loss_generator = 0
            loss_generator, metrics_dict = generator_loss(
                config=config,
                input_rgb=network_input.cuda(),
                fake_rgb=fake_rgb,
                real_rgb=empty_room.cuda(),
                discriminator=discriminator,
                pixel_loss=pixel_loss,
                perceptual_loss=perceptual_loss,
                adversarial_loss=adversarial_loss,
                feature_matching_loss=feature_matching,
                mask=mask.cuda(),
            )

            loss_generator = loss_generator / grad_accumulation_steps
            loss_gen += loss_generator.detach()
            loss_generator.backward()

            if ((idx + 1) % grad_accumulation_steps == 0) or (
                idx + 1 == len(train_dataloader)
            ):
                optimizer_Generator.step()
                optimizer_Generator.zero_grad()
                generator.zero_grad()
                # scheduler_generator.step()

            # Log generator losses
            if b_should_log:
                logger.log_metrics_dict(
                    metrics_dict, images_seen_total, "Training", "Generator"
                )

            ###################
            ##### DISCRIMINATOR
            ###################
            metrics_dict = {}
            discriminator.zero_grad()

            # fake_rgb = generator(network_input.cuda())

            loss_discriminator, metrics_dict = discriminator_loss(
                config=config,
                input_rgb=network_input.cuda(),
                fake_rgb=fake_rgb,
                real_rgb=empty_room.cuda(),
                discriminator=discriminator,
                adversarial_loss=adversarial_loss,
                mask=mask.cuda(),
            )
            loss_discriminator = loss_discriminator / grad_accumulation_steps
            loss_disc += loss_discriminator.detach()

            # Log discriminator losses
            if b_should_log:
                logger.log_metrics_dict(
                    metrics_dict, images_seen_total, "Training", "Discriminator"
                )
            if ((idx + 1) % grad_accumulation_steps == 0) or (
                idx + 1 == len(train_dataloader)
            ):
                loss_discriminator.backward()
                optimizer_Discriminator.step()
                optimizer_Discriminator.zero_grad()
                # scheduler_discriminator.step()

            if b_should_save_images:
                logger.save_images(
                    config=config,
                    data=data,
                    prediction_rgb=fake_rgb,
                    phase="Training",
                    epoch=epoch,
                    images_seen=images_seen_total,
                )

            if b_should_save_checkpoint:
                save_checkpoints_with_optimizer(
                    generator,
                    discriminator,
                    optimizer_Generator,
                    optimizer_Discriminator,
                    config,
                    epoch,
                    images_seen_total,
                )

            images_seen_total += 1
            idx += 1

        loss_disc = loss_disc / len(train_dataloader)
        loss_gen = loss_gen / len(train_dataloader)
        logger.log_metrics_dict(
            {"/Epoch_loss": loss_gen}, images_seen_total, "Training", "Generator"
        )
        logger.log_metrics_dict(
            {"/Epoch_loss": loss_disc}, images_seen_total, "Training", "Discriminator"
        )

        ##---------------------------------------------------------------------------------
        ##---------------------------------------------------------------------------------
        ##---------------------------------------------------------------------------------

        print("\n--VALIDATION--\n")
        # For each batch in the validation dataloader
        i = 0
        validation_loss_generator = 0
        validation_pixel_loss_generator = 0
        validation_adversarial_loss_generator = 0
        validation_perceptual_loss_generator = 0

        validation_loss_discriminator = 0
        with torch.no_grad():
            generator.eval()
            for data in tqdm(validation_dataloader):
                network_input, empty_room, mask = unpack_data(data)
                mask = mask.float()

                ###################
                ######### GENERATOR
                ###################
                generator.zero_grad()
                fake_rgb = generator(network_input.cuda())

                loss_generator, metrics_dict = generator_loss(
                    config=config,
                    input_rgb=network_input.cuda(),
                    fake_rgb=fake_rgb,
                    real_rgb=empty_room.cuda(),
                    discriminator=discriminator,
                    pixel_loss=pixel_loss,
                    perceptual_loss=perceptual_loss,
                    adversarial_loss=adversarial_loss,
                    feature_matching_loss=feature_matching,
                    mask=mask.cuda(),
                )
                validation_pixel_loss_generator += metrics_dict["Pixel"]
                validation_perceptual_loss_generator += metrics_dict["Perceptual"]
                validation_adversarial_loss_generator += metrics_dict["Adversarial"]
                validation_adversarial_loss_generator += metrics_dict[
                    "Feature_matching"
                ]
                validation_loss_generator += loss_generator

                ###################
                ##### DISCRIMINATOR
                ###################
                discriminator.zero_grad()
                loss_discriminator, metrics_dict = discriminator_loss(
                    config=config,
                    input_rgb=network_input.cuda(),
                    fake_rgb=fake_rgb,
                    real_rgb=empty_room.cuda(),
                    discriminator=discriminator,
                    adversarial_loss=adversarial_loss,
                    mask=mask.cuda(),
                )

                validation_loss_discriminator += loss_discriminator.item()

                # Save images
                if i <= 5:
                    logger.save_images(
                        config,
                        data,
                        fake_rgb,
                        "Validation",
                        epoch=epoch,
                        images_seen=images_seen_total + i,
                    )
                    i += 1

            metrics_dict = {}
            # Log generator losses
            metrics_dict["Pixel"] = validation_pixel_loss_generator / len(
                validation_dataloader
            )

            metrics_dict["Perceptual"] = validation_perceptual_loss_generator / len(
                validation_dataloader
            )

            metrics_dict["Adversarial"] = validation_adversarial_loss_generator / len(
                validation_dataloader
            )

            metrics_dict["Total"] = validation_loss_generator / (
                len(validation_dataloader)
            )

            logger.log_metrics_dict(
                metrics_dict, images_seen_total, "Validation", "Generator"
            )

            metrics_dict = {}
            metrics_dict["Total"] = validation_loss_discriminator / (
                len(validation_dataloader)
            )
            logger.log_metrics_dict(
                metrics_dict, images_seen_total, "Validation", "Discriminator"
            )

        save_checkpoints_with_optimizer(
            generator,
            discriminator,
            optimizer_Generator,
            optimizer_Discriminator,
            config,
            epoch,
            images_seen_total,
        )

    return