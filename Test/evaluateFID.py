import os
import glob
from collections import namedtuple
from enum import Enum
from tqdm import tqdm
import torch
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from models.utils import im2tensor, sem2tensor
from torchmetrics.image.fid import FrechetInceptionDistance

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Structured3DGroundTruth(Dataset):
    def __init__(self, config):
        """

        """
        self.dataset_path = config.dataset_path

        self.flist_samples_path = config.flist_samples_path
        self.samples = self.load_flist(self.flist_samples_path)

        self.input_height = config.input_height
        self.input_width = config.input_width

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # load empty room panorama
        empty_image_path = os.path.join(
            self.dataset_path, self.samples[index], "empty/rgb_rawlight.png"
        )
        empty_image = (
            Image.open(empty_image_path)
            .convert("RGB")
            .resize((self.input_width, self.input_height))
        )

        (
            empty_image,
        ) = self.im2tensor(
            empty_image,
        )

        return empty_image

    def im2tensor(
        self,
        empty_image
    ):
        #norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        #empty_image = norm(im2tensor(empty_image))
        empty_image = im2tensor(empty_image)

        return empty_image

    def load_flist(self, flist):
        """
        loads flist = refs to all samples
        """
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + "/*.jpg")) + list(
                    glob.glob(flist + "/*.png")
                )
                flist.sort()
                return flist
            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=str, encoding="utf-8")
                except:
                    return [flist]

        return []


class Structured3DPrediction(Dataset):
    def __init__(self, config):
        """

        """
        self.predictions_path = config.predictions_path

        self.flist_samples_path = config.flist_samples_path
        self.samples = self.load_flist(self.flist_samples_path)

        self.input_height = config.input_height
        self.input_width = config.input_width

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # load prediction room panorama

        prediction_image_path = os.path.join(
            self.prediction_path, self.samples[index], "empty/diminished.png"
        )
        prediction_image = (
            Image.open(prediction_image_path)
            .convert("RGB")
            .resize((self.input_width, self.input_height))
        )

        (
            prediction_image,
        ) = self.im2tensor(
            prediction_image,
        )

        return prediction_image

    def im2tensor(
        self,
        prediction_image,
    ):
        #norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        #empty_image = norm(im2tensor(empty_image))
        prediction_image = im2tensor(prediction_image)

        return prediction_image

    def load_flist(self, flist):
        """
        loads flist = refs to all samples
        """
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + "/*.jpg")) + list(
                    glob.glob(flist + "/*.png")
                )
                flist.sort()
                return flist
            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=str, encoding="utf-8")
                except:
                    return [flist]

        return []


def start_fid_stats_ground_truth(ground_truth, fid):
    with torch.no_grad():
        for data in tqdm(ground_truth):
            fid.update(data, real=True)


def evaluate_FID(prediction, fid, type, interval):
    fid.reset()
    with torch.no_grad():
        for data in tqdm(prediction):
            fid.update(data, real=False)
    fid_score = fid.compute()
    print("===============================")
    print("===============================")
    print(f"FID for {type}_{interval}:  {fid_score}")
    print("===============================")
    print("===============================")


def main():
    predictions_path = ""
    batch_size = 12
    num_workers_dataloader = 10

    conf_ground_truth = {
        "dataset_path": "F:/Structured3D_Pano_Dataset",
        "flist_samples_path": "F:/PanoFlist/test.txt",
        "input_height": 256,
        "input_width": 512,
    }
    conf_ground_truth = EasyDict(conf_ground_truth)

    masks = [
        "Segmentation",
        "Quadrants",
        "Outpainting",
        "Irregular",
        "Rectangular",
        # "EveryOtherLine",
        # "EveryOtherColumn",
        # "SR2X",
    ]

    intervals = [
        "0.01_0.1",
        "0.1_0.2",
        "0.2_0.3",
        "0.3_0.4",
        "0.4_0.5",
    ]

    # create dataloaders
    gr_dataloader = DataLoader(
        Structured3DGroundTruth(conf_ground_truth),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_dataloader,
        drop_last=False,
    )

    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True)
    # compute gt stats
    start_fid_stats_ground_truth(gr_dataloader, fid)

    for mask_type in masks:
        mask_intervals = intervals
        mask_type_path = os.path.join(predictions_path, mask_type)
        for interval in mask_intervals:
            mask_type_interval_path = os.path.join(
                mask_type_path, interval)
            conf_predictions = {
                "predictions_path": mask_type_interval_path,
                "flist_samples_path": "F:/PanoFlist/test.txt",
                "input_height": 256,
                "input_width": 512,
            }
            conf_predictions = EasyDict(conf_predictions)
            pred_dataloader = DataLoader(
                Structured3DPrediction(conf_predictions),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers_dataloader,
                drop_last=False,
            )
            evaluate_FID(pred_dataloader, fid, mask_type, interval)


if __name__ == "__main__":
    main()
