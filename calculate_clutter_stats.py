import argparse
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

clutter_labels = [
    #1,  # wall
    #2,  # floor
    3,  # cabinet
    4,  # bed
    5,  # chair
    6,  # sofa
    7,  # table
    #8,  # door
    #9,  # window
    10,  # bookshelf
    11,  # picture
    12,  # counter
    #13,  # blinds
    14,  # desk
    15,  # shelves
    #16,  # curtain
    17,  # dresser
    18,  # pillow
    19,  # mirror
    20,  # floor mat
    21,  # clothes
    #22,  # ceiling
    23,  # books
    24,  # refrigerator
    25,  # television
    26,  # paper
    27,  # towel
    #28,  # shower curtain
    29,  # box
    30,  # whiteboard
    31,  # person
    32,  # nightstand
    33,  # toilet
    34,  # sink
    35,  # lamp
    36,  # bathtub
    37,  # bag
    #38,  # otherstructure
    39,  # otherfurniture
    40,  # otherprop
]


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate clutter stats")
    parser.add_argument(
        "--dataset_path",
        required=True,
        type=str,
        help="directory path to Structured3D dataset",
        metavar="DIR",
    )
    return parser.parse_args()


def calculate_clutter_percentage(semantic_map, clutter_labels):
    total_pixels = semantic_map.size
    clutter_pixels = np.sum(np.isin(semantic_map, clutter_labels))
    clutter_percentage = (clutter_pixels / total_pixels) * 100
    return clutter_percentage


def process_images(images, clutter_labels):
    clutter_percentages = []

    for image in images:
        clutter_percentage = calculate_clutter_percentage(image, clutter_labels)
        clutter_percentages.append(clutter_percentage)

    return clutter_percentages


def plot_histogram(clutter_percentages):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot histogram
    ax1.hist(clutter_percentages, bins=np.arange(0, 101, 5), edgecolor="black")
    ax1.set_xlabel("Clutter Percentage")
    ax1.set_ylabel("Number of Images")
    ax1.set_title("Clutter Percentage Histogram")
    ax1.grid(True)

    # Plot statistics
    ax2.axis("off")
    ax2.text(0.1, 0.9, f"Mean: {np.mean(clutter_percentages):.2f}%", fontsize=12)
    ax2.text(0.1, 0.8, f"Variance: {np.var(clutter_percentages):.2f}", fontsize=12)
    quantiles = np.percentile(clutter_percentages, [25, 50, 75])
    ax2.text(0.1, 0.7, f"25th Percentile: {quantiles[0]:.2f}%", fontsize=12)
    ax2.text(0.1, 0.6, f"50th Percentile (Median): {quantiles[1]:.2f}%", fontsize=12)
    ax2.text(0.1, 0.5, f"75th Percentile: {quantiles[2]:.2f}%", fontsize=12)

    plt.show()


def load_files_path(dataset_path):
    pattern = f"{dataset_path}/**/full/semantic.png"

    semantic_files = glob.glob(pattern, recursive=True)

    return semantic_files


def calculate_clutter_stats(dataset_path):
    """
    calculates percentage of clutter per room
    saves histogram pr interval
    mean
    var
    quantiles
    """
    clutter_percentages = []
    semantic_maps_paths = load_files_path(dataset_path)
    for semantic_map_path in tqdm(semantic_maps_paths):  # [10:11][11:12]
        try:
            semantic_map = np.array(
                Image.open(semantic_map_path).convert("P").resize((512, 256))
            )
            clutter_percentages.append(
                calculate_clutter_percentage(semantic_map, clutter_labels)
            )
        except:
            continue

    plot_histogram(clutter_percentages)


if __name__ == "__main__":
    args = parse_args()
    calculate_clutter_stats(args.dataset_path)
