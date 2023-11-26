
import os
from json import load
import torch
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter


from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.regression import MeanAbsoluteError

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Evaluator:
    def __init__(self):
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)  # data_range=1.0
        self.mean_absolute_error = MeanAbsoluteError()
        # self.fid = FrechetInceptionDistance(feature=64)

    def calculate(self, preds, target, masks):
        preds = (preds * masks) + (target * (1 - masks))
        # PNSR
        psnr_value = self.psnr(preds, target)

        # SSIM
        ssim_value = self.ssim(preds, target)

        # MAE
        mae_value = self.mean_absolute_error(preds, target)

        # FID
        # fid_value = self.fid(preds, target)

        return psnr_value, ssim_value, mae_value

    def calculate_from_disk(
        self, flist, ground_truth_directory_path, predictions_directory_path
    ):
        im2tensor = transforms.ToTensor()
        # load file list
        samples_paths = np.genfromtxt(flist, dtype=str, encoding="utf-8")

        PSNR, SSIM, MAE = 0, 0, 0

        for sample_path in tqdm(samples_paths):
            gt_image_path = os.path.join(
                ground_truth_directory_path, sample_path, "empty/rgb_rawlight.png"
            )
            ground_truth_image = (
                Image.open(gt_image_path).convert("RGB").resize((512, 256))
            )

            diminishied_image_path = os.path.join(
                predictions_directory_path, sample_path, "diminished.png"
            )
            diminished_image = Image.open(diminishied_image_path).convert("RGB")

            ground_truth_image = im2tensor(ground_truth_image).unsqueeze(0)
            diminished_image = im2tensor(diminished_image).unsqueeze(0)

            # PNSR
            psnr_value = self.psnr(diminished_image, ground_truth_image)

            # SSIM
            ssim_value = self.ssim(diminished_image, ground_truth_image)

            # MAE
            mae_value = self.mean_absolute_error(diminished_image, ground_truth_image)

            PSNR += psnr_value.item()
            SSIM += ssim_value.item()
            MAE += mae_value.item()

        # FID
        # fid_value = self.fid(preds, target)

        PSNR = PSNR / len(samples_paths)
        SSIM = SSIM / len(samples_paths)
        MAE = MAE / len(samples_paths)

        return PSNR, SSIM, MAE
