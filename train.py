from Models.model import train

from Configs import configs
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="training_config must be specified in config/configs.py",
        metavar="DIR",
    )
    parser.add_argument(
        "--current_epoch",
        required=False,
        default=0,
        type=int,
        help="current epoch: loads model if different from 0",
    )
    parser.add_argument(
        "--current_iter", required=False, default=0, type=int, help="current iter"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = configs.config_dic[args.config]
    config.current_epoch = args.current_epoch
    config.current_iter = args.current_iter

    train(config)


if __name__ == "__main__":
    main()