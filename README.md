# NOISe: Nuclei-Aware Osteoclast Instance Segmentation for Mouse-to-Human Domain Transfer
<div align="center">
  <p>
    <img width="100%" src="images/banner.png" alt="NOISe banner">
  </p>
</div>

NOISe builds on top of [YOLOv8](https://github.com/ultralytics/ultralytics) for the task of osteoclast instance segmentation. This repository contains training, validation, and inference scripts to train an instance segmentation model from scratch and finetune on top of the provided checkpoints trained for osteoclast and nuclei detection as proposed in our [paper](https://arxiv.org/pdf/2404.10130.pdf):

Manne, S.K.R., Martin, B., Roy, T., Neilson, R., Peters, R., Chillara, M., Lary, C.W., Motyl, K.J., Wan, M., "NOISe: Nuclei-Aware Osteoclast Instance Segmentation for Mouse-to-Human Domain Transfer.” IEEE/CVF Conference on Computer Vision and Pattern Recognition 2024 (CVPR 2024), Workshop on Computer Vision for Microscopy Image Analysis (CVMI).

<details open>
<summary>Install</summary>

Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```bash
pip install ultralytics
```

For alternative installation methods including [Conda](https://anaconda.org/conda-forge/ultralytics), [Docker](https://hub.docker.com/r/ultralytics/ultralytics), and Git, please refer to the [Quickstart Guide](https://docs.ultralytics.com/quickstart).

</details>

## Dataset
Our dataset consists of full slide images and corresponding instance segmentation annotations, along with patches used for training and validation in our experiments. Please download the dataset from [here](https://drive.google.com/drive/folders/1hwGVKH4pN1Ftcl9bDKUykTIU8mcZfmiu?usp=drive_link), unzip the data and place it in the dataset folder with the following folder structure:

```
noise
    |...
    |-- dataset
        |-- m1
            |-- images
            |-- labels
        |-- m2
            |-- images
            |-- labels
        |...
        |-- m5
            |-- images
            |-- labels
    |...
```

## Whole Slide Inference
Instance segmentation prediction can be done for a whole slide image by creating overlapping patches of ```832x832``` resolution, that are then merged to generate a full-scale output. Different models trained on various configurations of data are available [here](https://drive.google.com/drive/folders/1pHpwhwJSKN47Dbtcy92F2XHydpURMb4O?usp=drive_link). Please download the checkpoints and place them in the checkpoints folder.

| Model Name                                | Info                                                                                           | Recommendation                         
| ----------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------
| `yolo_mouse_ins.pt` (YOLOv8 M→M)          | YOLOv8 model trained on entire mouse data for osteoclast instance segmentation                 | Recommended model to test on mouse data             |
| `yolo_mouse_det_pretrain.pt`              | NOISE pretrain - YOLOv8 model trained on entire mouse data for osteoclast and nuclei detection | Recommended model to finetune on new data           |
| `noise_h1_ins_finetune.pt` (NOISE H→H)    | NOISe model finetuned on H1 dataset for osteoclast instance segmentation                       | Recommended model to test new human osteoclast data |
| `noise_h2_ins_finetune.pt` (NOISE H→H)    | NOISe model finetuned on H2 dataset for osteoclast instance segmentation                       | Recommended model to test new human osteoclast data |

```
python wsi_inference.py path/to/checkpoint.pt path/to/images
```

Outputs will be stored in ```path/to/output```.

## Training
Training from scratch or finetuning a specific checkpoint can be done using the following command:

```
python train.py --ckpt path/to/checkpoint.pt
```

Sample command to train YOLOv8 M→H model:
```
python train.py --ckpt checkpoints/yolo_mouse_det_pretrain.pt --data ./osteo.yaml --name YOLOv8MH
```

## Validation
Running evaluation on test data can be done using the following command:

```
python val.py --ckpt path/to/checkpoint.pt
```

Sample command to validate the YOLOv8 M→M model:
```
python train.py --ckpt checkpoints/yolo_mouse_ins.pt --data ./osteo.yaml
```

## Training on your data
NOISe can be further improved by training on custom data following these steps:

### Preparing dataset
We provide a helper script to convert whole slide images into overlapping patches of a fixed size for training and validation of models.

```
python create_patches.py --img_foldername path/to/wsi_imagefolder --roi_foldername path/to/imageJ_rois --out_foldername path/to/patch_folder
```

### Configs
Update the config.yaml according to the dataset created:
```
path: path/to/patch_data
train: [train_folder1, train_folder2, ...]
val: [test_folder1, test_folder2, ...]
```

### Train
Once the dataset is generated, and the `config.yaml` file is adjusted accordingly, training can be done using:

```
python train.py --ckpt path/to/checkpoint.pt --data config.yaml
```

## Acknowledgement
NOISe is largely based on [YOLOv8](https://github.com/ultralytics/ultralytics).

## Citation
If you find the dataset, model weights, or the code useful for your work, please consider citing us:
```
@InProceedings{ManneMartin_2023_CVPR,
    author    = {Manne, S.K.R. and Martin, B. and Roy, T. and Neilson, R. and Peters, R. and Chillara, M. and Lary, C.W. and Motyl, K.J. and Wan, M.},
    title     = {NOISe: Nuclei-Aware Osteoclast Instance Segmentation for Mouse-to-Human Domain Transfer},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024}
}
```
Note that the first two authors contributed equally.
