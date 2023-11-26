import argparse
import os
import torch
import numpy as np
from models.model import RoomCleaner
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from models.utils import im2tensor, sem2tensor, tensor2pil
from tqdm import tqdm
from Models.net_archis import (
    uformer_ffc_ffc,
    uformer_fourierMixWindow,
    uformer_fourierMix,
    uformer_GatedConv,
    uformer_fourierWindow,
)
from torch.utils.data import DataLoader
from Datasets.datasetPano import (
    Structured3D,
    Phase,
)
from easydict import EasyDict
import glob
from collections import namedtuple
from enum import Enum
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image, ImageFile


from Models.utils import im2tensor, sem2tensor

ImageFile.LOAD_TRUNCATED_IMAGES = True
FLOOR_ID = 2
WALL_ID = 1
CEILING_ID = 22

# UTIL TUPLES
Sample = namedtuple("Sample", ["network_input", "empty_room", "mask", "sample_path"])

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

        self.input_height = config.input_height
        self.input_width = config.input_width

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

        empty_image = self.im2tensor(
            empty_image,
        )

        network_input = empty_image

        ### load mask
        mask = self.load_mask(
            network_input.numpy(),
            index,
            None,
            None,
        )

        network_input = network_input * (1 - mask)
        network_input = torch.cat([network_input, mask], dim=0)
        sample_path = self.samples[index]

        sample = Sample(network_input, empty_image, mask, sample_path)

        return sample

    def im2tensor(
        self,
        empty_image,
    ):
        norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        empty_image = norm(im2tensor(empty_image))

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


def inference(generator, dataloader, outputdir):
    with torch.no_grad():
        for data in tqdm(dataloader):
            network_input, empty_image, mask, sample_path = data

            raw_output = generator(network_input.cuda()) * 0.5 + 0.5
            output = (empty_image * 0.5 + 0.5) * (1 - mask) + raw_output.cpu() * mask
            network_input = network_input[:, :3] * 0.5 + 0.5

            # save results
            b = output.shape[0]
            for i in range(b):
                os.makedirs(
                    os.path.join(outputdir, sample_path[i] + "/"), exist_ok=True
                )
                input_path = os.path.join(outputdir, sample_path[i] + "/", "input.png")
                pred_path = os.path.join(
                    outputdir, sample_path[i] + "/", "diminished.png"
                )
                raw_pred_path = os.path.join(
                    outputdir, sample_path[i] + "/", "raw_pred.png"
                )
                tensor2pil(network_input[i]).save(input_path)
                tensor2pil(output[i]).save(pred_path)
                tensor2pil(raw_output[i]).save(raw_pred_path)


def main():
    flist = "F:/PanoFlist/test.txt"
    samples_paths = np.genfromtxt(flist, dtype=str, encoding="utf-8")
    outputdir = "D:/RLE/RoomInpainting/Results"
    input_images_path = "F:/Structured3D_Pano_Dataset"
    masks_path = "F:/Structured3D_Pano_Masks"
    batch_size = 12
    num_workers_dataloader = 10

    models = [
        uformer_ffc_ffc,
        uformer_fourierMixWindow,
        uformer_fourierMix,
        uformer_GatedConv,
        uformer_fourierWindow,
    ]

    checkpoints = [
        # TODO
    ]  

    masks = [
        "Segmentation",
        "Quadrants",
        "Outpainting",
        "Irregular",
        "Rectangular",
    ]

    intervals = [
        "0.01_0.1",
        "0.1_0.2",
        "0.2_0.3",
        "0.3_0.4",
        "0.4_0.5",
    ]

    for i, model in enumerate(models):
        generator = RoomCleaner(model).cuda()
        checkpoint_generator = torch.load(checkpoints[i])
        generator.load_state_dict(checkpoint_generator["model_state_dict"])
        generator.eval()

        for mask_type in masks:
            mask_intervals = intervals
            mask_type_path = os.path.join(masks_path, mask_type)

            outs = os.path.join(outputdir, mask_type)
            for interval in mask_intervals:
                mask_type_interval_path = os.path.join(mask_type_path, interval)

                conf_dataloader = {
                    "input_height": 256,
                    "input_width": 512,
                    "dataset_path": "F:/Structured3D_Pano_Dataset",
                    "flist_samples_path": "F:/PanoFlist/test.txt",
                    "maskGenerator_type": 1,
                    "maskGenerator_kwargs": {},
                    "furnished": False,
                    "masks_path": mask_type_interval_path,
                }
                conf_dataloader = EasyDict(conf_dataloader)

                test_dataloader = DataLoader(
                    Structured3D(conf_dataloader, Phase.TEST),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers_dataloader,
                    drop_last=False,
                )

                outs_interval = os.path.join(outs, interval)
                inference(
                    generator,
                    test_dataloader,
                    outs_interval,
                )


if __name__ == "__main__":
    main()