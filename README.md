# Windowed-FourierMixer: Enhancing Clutter-Free Room Modeling with Fourier Transform

## Environment setup
1. Clone the repo.

2. Setup environment using Conda
```bash
    cd <path_to_repo>
    conda env create -f conda_env.yml
    conda activate WFourierMixer
```
## Inference
1. Download pre-trained model [here](https://1drv.ms/u/s!Ak8Yx9uuLGLmajVDANSmBz8I9C0?e=J9Hvnm) (anonymous link)

2. Run:
```bash
    cd <path_to_repo>
    python inference.py --input_rgb_path <directory_path_to_rgbs> --input_mask_path <directory_path_to_masks> --checkpoint <path_to_pretrained_model> --output_path <designated_output_directory>
```
input_rgb_path is expected to be a folder containing png files corresponding to indoor scenes (image_name.png), and input_mask_path the correponding masks (image_name_mask.png).

We additionnaly provide a simple tkinter app allowing the user to create their own masks, just run:
```bash
    python draw_mask.py
```
then under File > Open Image, draw a mask and then again, under File > Save Mask


## Dataset & Masks
We use [Structured3D](https://structured3d-dataset.org/) dataset. Please download the dataset from the official website. We follow the official training, validation, and testing splits as defined by the authors.

Masks for testing and validation can be created by:
```bash
    python create_test_masks.py --dataset_path <path_to_structured3d_dataset> --flist <path_to_flist_in_Datasets_folder> --outputdir <designated_output_directory> --mask_type [1|2|3|4|5|6|7]--mask_min_ratio [0.0-1.0] --mask_max_ratio [0.0-1.0]
```
with mask types being:

1. Mixed
1. Semantic
1. Iregular
1. Rectangular
1. Outpainting
1. Quadrants

## Training 
In order to train the model, first specify the required parameters in Configs/configs.py file:
- "dataset_path": <path_to_structured3d_dataset>
- "flist_samples_path": <path_to_flist_in_Datasets_folder>

You can additionnaly change parameters in configs.py such as batch_size, image resolution, etc.

Download models for perceptual loss [here](http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth) and put them in weights\ade20k\ade20k-resnet50dilated-ppm_deepsup folder

Training can be done by:
```bash
    python train.py
```
And the ablation by:
```bash
    python ablation.py
```

## Testing
Assuming you have generated masks for testing, and saved the associated model predictions in a folder.
You can run: 
```bash
    python test.py --ground_truth_path <path_to_structured3d_dataset> --predictions_path <path_to_predictions> --flist <path_to_test_flist>
```

## Acknowledgments
Our code borrows from:

- [Metaformer](https://github.com/sail-sg/metaformer) : Metaformer like architecture
- [LaMa](https://github.com/advimman/lama/tree/main) : irregular, rectangular masks and losses
- [LGPN](https://github.com/ericsujw/LGPN-net) : flist files, dataset
