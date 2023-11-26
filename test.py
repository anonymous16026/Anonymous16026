import argparse
from Test.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument(
        "--ground_truth_path",
        required=True,
        type=str,
        help="directory path to ground truth images",
        metavar="DIR",
    )
    parser.add_argument(
        "--predictions_path",
        required=True,
        type=str,
        help="directory path to prediction (diminished reality) images",
        metavar="DIR",
    )
    parser.add_argument(
        "--flist",
        required=True,
        type=str,
        help="path to flist file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    evaluator = Evaluator()
    psnr, ssim, mae = evaluator.calculate_from_disk(
        flist=args.flist,
        ground_truth_directory_path=args.ground_truth_path,
        predictions_directory_path=args.predictions_path,
    )

    print(f"PSNR: {psnr} \n SSIM: {ssim} \n MAE: {mae}")


if __name__ == "__main__":
    main()
