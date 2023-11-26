import argparse
import os
from easydict import EasyDict
from Datasets.datasetPano import Phase, Structured3D
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create test/validation masks between x_min and x_max ratio and of type t"
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        type=str,
        help="directory path to input images",
        metavar="DIR",
    )
    parser.add_argument(
        "--flist",
        required=True,
        type=str,
        help="filelist",
        metavar="DIR",
    )
    parser.add_argument(
        "--outputdir",
        required=True,
        type=str,
        help="output directory",
        metavar="DIR",
    )
    parser.add_argument(
        "--mask_type",
        required=True,
        type=int,
        help="mask type: 1_Mixed, 2_, 3_ ...",
    )
    parser.add_argument(
        "--mask_min_ratio",
        required=True,
        type=float,
        help="mask min ratio [0-1]",
    )
    parser.add_argument(
        "--mask_max_ratio",
        required=True,
        type=float,
        help="mask max ratio [0-1]",
    )
    return parser.parse_args()


def create_test_masks():
    tensor2pil = transforms.ToPILImage()
    config = {
        "input_height": 256,
        "input_width": 512,
        "dataset_path": "F:/Structured3D_Pano_Dataset",
        "flist_samples_path": "F:/PanoFlist/val.flist",  # "F:/PanoFlist/test.txt",  # "F:/PanoFlist/train.flist",
        "flist_masks_path": None,
        "maskGenerator_type": 1,
        "maskGenerator_kwargs": {
            "semantic_kwargs": {"min_mask_area": 0.01, "max_mask_area": 0.5}
        },
        "furnished": False,
    }

    args = parse_args()
    config["maskGenerator_type"] = args.mask_type
    config["dataset_path"] = args.dataset_path
    config["flist_samples_path"] = args.flist
    if args.mask_type == 1:
        config["maskGenerator_kwargs"] = {
            "semantic_kwargs": {
                "min_mask_area": args.mask_min_ratio,
                "max_mask_area": args.mask_max_ratio,
            }
        }
    elif args.mask_type == 2:
        config["maskGenerator_kwargs"] = {
            "min_mask_area": args.mask_min_ratio,
            "max_mask_area": args.mask_max_ratio,
        }
    else:
        config["maskGenerator_kwargs"] = {}

    config = EasyDict(config)

    dt = Structured3D(config, Phase.TRAIN)
    dt.create_test_ratio(
        min_ratio=args.mask_min_ratio,
        max_ratio=args.mask_max_ratio,
        outputdir=args.outputdir,
    )


if __name__ == "__main__":
    create_test_masks()


# python create_test_masks.py --dataset_path F:\Structured3D_Pano_Dataset --flist "F:\PanoFlist\test.txt" --outputdir F:\Structured3D_Pano_Masks\Segmentation --mask_type 2 --mask_min_ratio 0.3 --mask_max_ratio 0.4