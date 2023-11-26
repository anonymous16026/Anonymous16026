import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision


im2tensor = transforms.ToTensor()
sem2tensor = transforms.PILToTensor()
tensor2pil = transforms.ToPILImage()


def denormalize(img):
    return (img * 0.5) + 0.5


def createdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Logger:
    def __init__(self, config):
        # create log dirs
        createdir(f"{config.log_dir}/{config.name}/tensorboard")
        createdir(f"{config.log_dir}/{config.name}/images/rgb/Training")
        createdir(f"{config.log_dir}/{config.name}/images/rgb/Validation")

        self.writer = SummaryWriter(f"{config.log_dir}/{config.name}/tensorboard")

    def log_metrics_dict(self, metrics_dict, time, phase, agent):
        for key, value in metrics_dict.items():
            self.writer.add_scalar(
                f"{phase}/Loss/{agent}/{key}_loss",
                value,
                time,
            )

    def log_norm(self, name, value, time):
        self.writer.add_scalar("Weights_Norm/" + name, value, time)

    def save_images(
        self,
        config,
        data,
        prediction_rgb,
        phase,
        epoch,
        images_seen,
    ):
        network_input = data.network_input[:, :-1].cpu()
        network_input = denormalize(network_input)
        label_rgb = data.empty_room.cpu()
        label_rgb = denormalize(label_rgb)
        output_rgb = prediction_rgb.cpu()
        output_rgb = denormalize(output_rgb)
        images_rgb = torch.cat([network_input, label_rgb, output_rgb], dim=0)

        grid_rgb = torchvision.utils.make_grid(images_rgb)

        tensor2pil(grid_rgb).save(
            f"{config.log_dir}/{config.name}/images/rgb/{phase}/{epoch}_{images_seen}.jpg"
        )


def unpack_data(data):
    network_input = data.network_input
    empty_room = data.empty_room
    mask = data.mask

    return (network_input, empty_room, mask)


def load_weights(
    config, generator, discriminator, generator_optimizer, discriminator_optimizer
):
    createdir(f"{config.save_checkpoint_dir}/{config.name}/{config.dataset_name}")
    if config.current_epoch != 0 or config.current_iter != 0:
        checkpoint_generator = torch.load(
            f"{config.save_checkpoint_dir}/{config.name}/{config.dataset_name}/Generator_{config.current_epoch}_{config.current_iter}.tar"
        )
        generator.load_state_dict(checkpoint_generator["model_state_dict"])
        generator_optimizer.load_state_dict(
            checkpoint_generator["optimizer_state_dict"]
        )

        checkpoint_discriminator = torch.load(
            f"{config.save_checkpoint_dir}/{config.name}/{config.dataset_name}/Discriminator_{config.current_epoch}_{config.current_iter}.tar"
        )
        discriminator.load_state_dict(checkpoint_discriminator["model_state_dict"])
        discriminator_optimizer.load_state_dict(
            checkpoint_discriminator["optimizer_state_dict"]
        )


def save_checkpoints_with_optimizer(
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    config,
    epoch,
    images_seen_total,
):
    save_model(
        model=generator,
        optimizer=generator_optimizer,
        config=config,
        epoch=epoch,
        images_seen_total=images_seen_total,
        name="Generator",
    )
    save_model(
        model=discriminator,
        optimizer=discriminator_optimizer,
        config=config,
        epoch=epoch,
        images_seen_total=images_seen_total,
        name="Discriminator",
    )


def save_model(model, optimizer, epoch, config, images_seen_total, name):
    createdir(f"{config.save_checkpoint_dir}/{config.name}/{config.dataset_name}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            #'loss': loss,
        },
        f"{config.save_checkpoint_dir}/{config.name}/{config.dataset_name}/{name}_{epoch}_{images_seen_total}.tar",
    )


def apply_blur(images_seen_total, blur_fade_kimg, blur_init_value, empty_room_rgb):
    blur_sigma = (
        max(
            1 - images_seen_total / (blur_fade_kimg * 1e3),
            0,
        )
        * blur_init_value
        if blur_fade_kimg > 0
        else 0
    )
    if blur_sigma > 0:
        blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=blur_sigma)
        empty_room_rgb = blur(empty_room_rgb)

    return empty_room_rgb