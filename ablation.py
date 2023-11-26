from Models.model import train
from Configs import configs
import argparse


def main():
    configs_list = [
        "uformer_FourierWindow",
        "uformer_ffc_ffc",
        "uformer_GatedConv",
        "uformer_fourierMix",
        "uformer_fourierMixWindow",
    ]
    for cfg in configs_list:
        config = configs.config_dic[cfg]
        train(config)


if __name__ == "__main__":
    main()