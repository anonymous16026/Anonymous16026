from easydict import EasyDict
from Models.net_archis import (
    uformer_ffc_ffc,
    uformer_GatedConv,
    uformer_fourierWindow,
    uformer_fourierMix,
    uformer_fourierMixWindow,
)

##############
## Log
##############
log_params = {
    "log_dir": "logs",
    "save_images_every_kiter": 2,
    "log_loss_every_kiter": 0.1,
}

##############
## Checkpoints
##############
checkpoint_params = {
    "save_checkpoint_dir": "checkpoints",
    "save_checkpoint_every_kiter": 5,
    "current_epoch": 0,
    "current_iter": 0,
}

##############
## Loss
##############
loss_params = {
    "adversarial_loss_weight": 10,
    "adversarial_loss_gp_coef": 0.001,
    "feature_matching_loss_weight": 30,
    "resnet_perceptual_loss_weight": 100,
    "l1_pixel_loss_weight": 10,
}

##########
## Train
##########
train_params = {
    "batch_size": 6,
    "gradient_accumulation_steps": 1,
    "num_workers_dataloader": 10,
    "num_epochs": 40,
    "discriminator_optimizer_lr": 1e-4,
    "generator_optimizer_lr": 1e-3,
    "discriminator_optimizer_betas": (0.9, 0.999),
    "generator_optimizer_betas": (0.9, 0.999),
}
train_params_low_res = {
    "batch_size": 24,
    "gradient_accumulation_steps": 1,
    "num_workers_dataloader": 10,
    "num_epochs": 20,
    "discriminator_optimizer_lr": 1e-4,
    "generator_optimizer_lr": 1e-3,
    "discriminator_optimizer_betas": (0.9, 0.999),
    "generator_optimizer_betas": (0.9, 0.999),
}

###
### DATASET DATALOADERs
###
Structure3d_pano = {
    "dataset_name": "Structured3D",
    "config_train_dataloader": {
        "input_height": 256,
        "input_width": 512,
        "dataset_path": "E:/Structured3D_Pano_Dataset",
        "flist_samples_path": "E:/PanoFlist/train.flist",
        "maskGenerator_type": 1,
        "maskGenerator_kwargs": {},
        "furnished": False,
    },
    "config_test_dataloader": {
        "input_height": 256,
        "input_width": 512,
        "dataset_path": "E:/Structured3D_Pano_Dataset",
        "flist_samples_path": "E:/PanoFlist/test.txt",
        "maskGenerator_type": 1,
        "maskGenerator_kwargs": {},
        "furnished": False,
        "masks_path": "E:/Structured3D_Pano_Dataset_Test_Masks_0.01_0.1",
    },
    "config_validation_dataloader": {
        "input_height": 256,
        "input_width": 512,
        "dataset_path": "E:/Structured3D_Pano_Dataset",
        "flist_samples_path": "E:/PanoFlist/val.flist",
        "maskGenerator_type": 1,
        "maskGenerator_kwargs": {},
        "furnished": False,
        "masks_path": "E:/Structured3D_Pano_Dataset_Test_Masks_0.01_0.5",
    },
}
Structure3d_pano_low_res = {
    "dataset_name": "Structured3D",
    "config_train_dataloader": {
        "input_height": 128,
        "input_width": 256,
        "dataset_path": "E:/Structured3D_Pano_Dataset",
        "flist_samples_path": "E:/PanoFlist/train.flist",
        "maskGenerator_type": 1,
        "maskGenerator_kwargs": {},
        "furnished": False,
    },
    "config_test_dataloader": {
        "input_height": 128,
        "input_width": 256,
        "dataset_path": "E:/Structured3D_Pano_Dataset",
        "flist_samples_path": "E:/PanoFlist/test.txt",
        "maskGenerator_type": 1,
        "maskGenerator_kwargs": {},
        "furnished": False,
        "masks_path": "E:/Structured3D_Pano_Dataset_Test_Masks_0.01_0.1",
    },
    "config_validation_dataloader": {
        "input_height": 128,
        "input_width": 256,
        "dataset_path": "E:/Structured3D_Pano_Dataset",
        "flist_samples_path": "E:/PanoFlist/val.flist",
        "maskGenerator_type": 1,
        "maskGenerator_kwargs": {},
        "furnished": False,
        "masks_path": "E:/Structured3D_Pano_Dataset_Test_Masks_0.01_0.5",
    },
}


config_RoomCleanerPano_Default = {
    **log_params,
    **checkpoint_params,
    **loss_params,
    **train_params,
    **Structure3d_pano,
}
config_RoomCleanerPano_Low_res = {
    **log_params,
    **checkpoint_params,
    **loss_params,
    **train_params_low_res,
    **Structure3d_pano_low_res,
}


config_uformer_ffc_ffc = {
    **{
        "name": "uformer_ffc_ffc",
        "model": uformer_ffc_ffc,
    },
    **config_RoomCleanerPano_Low_res,
}

config_uformer_GatedConv = {
    **{
        "name": "uformer_GatedConv",
        "model": uformer_GatedConv,
    },
    **config_RoomCleanerPano_Low_res,
}

config_uformer_FourierWindow = {
    **{
        "name": "uformer_FourierWindow",
        "model": uformer_fourierWindow,
    },
    **config_RoomCleanerPano_Low_res,
}

config_uformer_fourierMix = {
    **{
        "name": "uformer_fourierMix",
        "model": uformer_fourierMix,
    },
    **config_RoomCleanerPano_Low_res,
}

config_uformer_fourierMixWindow_low_res = {
    **{
        "name": "uformer_fourierMixWindow",
        "model": uformer_fourierMixWindow,
    },
    **config_RoomCleanerPano_Low_res,
}

config_uformer_fourierMixWindow = {
    **{
        "name": "uformer_fourierMixWindow",
        "model": uformer_fourierMixWindow,
    },
    **config_RoomCleanerPano_Default,
}


config_dic = {
    "uformer_ffc_ffc": EasyDict(config_uformer_ffc_ffc),
    "uformer_fourierMixWindow": EasyDict(config_uformer_fourierMixWindow),
    "uformer_fourierMix": EasyDict(config_uformer_fourierMix),
    "uformer_GatedConv": EasyDict(config_uformer_GatedConv),
    "uformer_FourierWindow": EasyDict(config_uformer_FourierWindow),
}