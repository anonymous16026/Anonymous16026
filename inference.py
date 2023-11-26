import argparse
import os
import glob
import torch
import numpy as np
from Models.model import RoomCleaner
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from Models.utils import im2tensor, sem2tensor, tensor2pil
from tqdm import tqdm
from Models.net_archis import (
    uformer_fourierMixWindow,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--input_rgb_path",
        required=True,
        type=str,
        help="directory path to input images xxx.png and masks xxx_mask.png",
        metavar="DIR",
    )
    parser.add_argument(
        "--input_mask_path",
        required=True,
        type=str,
        help="directory path to input masks xxx_mask.png",
        metavar="DIR",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="model checkpoint",
        metavar="DIR",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="output directory",
        metavar="DIR",
    )
    return parser.parse_args()

def load_files_path(input_path):
    pattern = f"{input_path}/**.png"

    input_files = glob.glob(pattern, recursive=True)

    return input_files

def load_rgb(rgb_path):
    norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    rgb_image = (
            Image.open(rgb_path)
            .convert("RGB")
            .resize((512, 256))
        )
    rgb_image = norm(im2tensor(rgb_image))
    return rgb_image.unsqueeze(0)

def load_mask(mask_path):
    mask = Image.open(mask_path).resize((512, 256))
    mask = sem2tensor(mask) // 255
    return mask

def load_mask_for_rgb(rgb_path, masks_path):
    rgb_dir_fileName = os.path.split(rgb_path)
    rgbName = rgb_dir_fileName[1][:-4]
    mask_path = os.path.join(masks_path, rgbName+"_mask.png")
    mask = load_mask(mask_path)
    return mask.unsqueeze(0)

def prepare_net_input(rgb, mask):
    network_input = torch.cat([rgb*(1-mask), mask], dim=1)
    return network_input


def main():
    args = parse_args()
    if args.input_rgb_path == args.input_mask_path:
        raise()

    generator = RoomCleaner(uformer_fourierMixWindow).cuda()
    checkpoint_generator = torch.load(args.checkpoint)
    generator.load_state_dict(checkpoint_generator["model_state_dict"])

    generator.eval()
    import time
    rgb_paths = load_files_path(args.input_rgb_path)
    with torch.no_grad():
        for rgb_path in tqdm(rgb_paths):
            rgb = load_rgb(rgb_path)
            mask = load_mask_for_rgb(rgb_path, args.input_mask_path)
            network_input = prepare_net_input(rgb, mask).cuda()
            #inference
            t_s = time.time()
            raw_output = generator(network_input) * 0.5 + 0.5
            print(t_s-time.time())
            output = (rgb * 0.5 + 0.5) * (1 - mask) + raw_output.cpu() * mask
            #save_image
            os.makedirs(args.output_path, exist_ok=True)
            rgb_dir_fileName = os.path.split(rgb_path)
            rgbName = rgb_dir_fileName[1][:-4]
            input_path = os.path.join(args.output_path, rgbName+"_input.png")
            pred_path = os.path.join(
                args.output_path, rgbName+"_diminished.png"
            )
            raw_pred_path = os.path.join(
                args.output_path, rgbName+"_raw_pred.png"
            )
            tensor2pil((network_input[:,:3] * 0.5 + 0.5).squeeze(0)).save(input_path)
            tensor2pil(output.squeeze(0)).save(pred_path)
            tensor2pil(raw_output.squeeze(0)).save(raw_pred_path)

if __name__ == "__main__":
    main()