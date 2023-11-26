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


from .maskGenerators import GeneratorFactory_get

from Models.utils import im2tensor, sem2tensor

FLOOR_ID = 2
WALL_ID = 1
CEILING_ID = 22

"""
adapted from
https://github.com/ericsujw/LGPN-net/blob/main/src/dataset.py#L629
"""

# UTIL TUPLES
Sample = namedtuple("Sample", ["network_input", "empty_room", "mask"])

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Phase(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


class Structured3D(Dataset):
    def __init__(self, config, phase):
        """

        Parameters
        ----------
        config: {
            'input_height':int,
            'input_width':int,
            'flist_samples_path':str,
            'flist_masks_path': str,
            'dataset_path':str,
            'furnished' : bool,
            'maskGeneration_seed' : int,
            'maskGenerator_type': 1:Mixed, 2:SemanticSegmentation, 3: RandomIrregular, 4:RandomRectangular, 5: Quadrant, 6: Outpainting
            'maskGenerator_kwargs': _ ,
            'masks_path': str,
        }

        phase: Phase.TRAINING|Phase.VALIDATION|Phase.TEST

        """
        self.dataset_path = config.dataset_path

        self.flist_samples_path = config.flist_samples_path
        self.samples = self.load_flist(self.flist_samples_path)

        if phase == Phase.VALIDATION or phase == Phase.TEST:
            self.masks_path = config.masks_path

        self.phase = phase

        self.furnished = config.furnished

        self.input_height = config.input_height
        self.input_width = config.input_width

        if phase == Phase.TRAIN:
            self.mask_generator = GeneratorFactory_get(
                config.maskGenerator_type, config.maskGenerator_kwargs
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ### load empty room panorama
        empty_image_path = os.path.join(
            self.dataset_path, self.samples[index], "empty/rgb_rawlight.png"
        )
        empty_image = (
            Image.open(empty_image_path)
            .convert("RGB")
            .resize((self.input_width, self.input_height))
        )

        ### load full room panorama

        full_image_path = os.path.join(
            self.dataset_path, self.samples[index], "full/rgb_rawlight.png"
        )
        full_image = (
            Image.open(full_image_path)
            .convert("RGB")
            .resize((self.input_width, self.input_height))
        )

        ### load empty semantic map
        empty_semantic_image_path = os.path.join(
            self.dataset_path, self.samples[index], "empty/semantic.png"
        )
        empty_semantic_image = (
            Image.open(empty_semantic_image_path)
            .convert("P")
            .resize((self.input_width, self.input_height))
        )

        ### load full semantic map
        full_semantic_image_path = os.path.join(
            self.dataset_path, self.samples[index], "full/semantic.png"
        )
        full_semantic_image = (
            Image.open(full_semantic_image_path)
            .convert("P")
            .resize((self.input_width, self.input_height))
        )

        (
            empty_image,
            full_image,
            empty_semantic_image,
            full_semantic_image,
        ) = self.im2tensor(
            empty_image,
            full_image,
            empty_semantic_image,
            full_semantic_image,
        )

        if self.furnished:
            foreground = 255 * (
                (full_semantic_image.numpy() != CEILING_ID).astype(np.uint8)
                * (full_semantic_image.numpy() != FLOOR_ID).astype(np.uint8)
                * (full_semantic_image.numpy() != WALL_ID).astype(np.uint8)
            )
            network_input = np.where(
                foreground.astype(bool),
                full_image.numpy(),
                empty_image.numpy(),
            )
            network_input = torch.Tensor(network_input)

        else:
            network_input = empty_image

        ### load mask
        mask = self.load_mask(
            network_input.numpy(),
            index,
            empty_semantic_map=empty_semantic_image.numpy(),
            full_semantic_map=full_semantic_image.numpy(),
        )

        network_input = network_input * (1 - mask)
        network_input = torch.cat([network_input, mask], dim=0)

        sample = Sample(network_input, empty_image, mask)

        return sample

    def im2tensor(
        self,
        empty_image,
        full_image,
        empty_semantic_image,
        full_semantic_image,
    ):
        norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        empty_semantic_image = sem2tensor(empty_semantic_image)
        full_semantic_image = sem2tensor(full_semantic_image)

        empty_image = norm(im2tensor(empty_image))
        full_image = norm(im2tensor(full_image))

        return empty_image, full_image, empty_semantic_image, full_semantic_image

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

    def load_mask(self, img, index, empty_semantic_map=None, full_semantic_map=None):
        if self.phase == Phase.TRAIN:
            mask = self.mask_generator(
                img=img,
                empty_semantic_map=empty_semantic_map,
                full_semantic_map=full_semantic_map,
            )
            mask = torch.Tensor(mask).float()
        else:
            mask_path = os.path.join(
                self.masks_path, self.samples[index][:-9], "mask.png"
            )
            mask = Image.open(mask_path).resize((self.input_width, self.input_height))
            mask = sem2tensor(mask) // 255

        return mask


    def create_test_ratio(self, min_ratio, max_ratio, outputdir):
        tensor2pil = transforms.ToPILImage()
        for name in tqdm(self.samples):
            empty_image_path = os.path.join(
                self.dataset_path, name, "empty/rgb_rawlight.png"
            )
            empty_image = (
                Image.open(empty_image_path)
                .convert("RGB")
                .resize((self.input_width, self.input_height))
            )

            ### load full room panorama
            full_image_path = os.path.join(
                self.dataset_path, name, "full/rgb_rawlight.png"
            )
            full_image = (
                Image.open(full_image_path)
                .convert("RGB")
                .resize((self.input_width, self.input_height))
            )

            ### load empty semantic map
            empty_semantic_image_path = os.path.join(
                self.dataset_path, name, "empty/semantic.png"
            )
            empty_semantic_image = (
                Image.open(empty_semantic_image_path)
                .convert("P")
                .resize((self.input_width, self.input_height))
            )
            full_semantic_image_path = os.path.join(
                self.dataset_path, name, "full/semantic.png"
            )
            full_semantic_image = (
                Image.open(full_semantic_image_path)
                .convert("P")
                .resize((self.input_width, self.input_height))
            )

            (
                empty_image,
                full_image,
                empty_semantic_image,
                full_semantic_image,
            ) = self.im2tensor(
                empty_image,
                full_image,
                empty_semantic_image,
                full_semantic_image,
            )

            if self.furnished:
                foreground = 255 * (
                    (full_semantic_image.numpy() != CEILING_ID).astype(np.uint8)
                    * (full_semantic_image.numpy() != FLOOR_ID).astype(np.uint8)
                    * (full_semantic_image.numpy() != WALL_ID).astype(np.uint8)
                )
                network_input = np.where(
                    foreground.astype(bool),
                    full_image.numpy(),
                    empty_image.numpy(),
                )
                network_input = torch.Tensor(network_input)

            else:
                network_input = empty_image

            ### load mask
            b_is_mask_found = False
            while not b_is_mask_found:
                mask = self.load_mask(
                    network_input.numpy(),
                    0,
                    empty_semantic_map=empty_semantic_image.numpy(),
                    full_semantic_map=full_semantic_image.numpy(),
                )
                mask_np = mask.numpy()
                image_area = mask_np.size
                area = np.sum(mask_np)
                mask_ratio = area / image_area
                b_is_mask_found = (mask_ratio > min_ratio) and (mask_ratio <= max_ratio)

            mask_img = tensor2pil(mask)
            maskpath = f"{outputdir}/{min_ratio}_{max_ratio}/{name[:-9]}"
            if not os.path.exists(maskpath):
                os.makedirs(maskpath)
            mask_img.save(f"{maskpath}/mask.png")
        return