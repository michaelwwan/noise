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

Below verified setup uses Pytorch 2.3 with CUDA 12.1 support:
```bash
conda create --name noise python=3.8
conda activate noise
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install ultralytics
```

For alternative installation methods including [Conda](https://anaconda.org/conda-forge/ultralytics), [Docker](https://hub.docker.com/r/ultralytics/ultralytics), and Git, please refer to the [Quickstart Guide](https://docs.ultralytics.com/quickstart).

</details>

## Quickstart Guide: Whole Slide Inference 
This section will walk you through applying one of our osteoclast instange segmentation models on your own whole slide or well images, without needing to do any machine learning training, and with minimal setup and computing requirements. Internally, our script will break your image down into overlapping ```832x832``` resolution patches, apply the specific instance segmentation model on those patches, and then intelligently merge the results to generate results for your original image. 

You do need to download the model "checkpoint" corresponding to the model you want to run, and place it in a `/checkpoint` directory. You can follow our recommendations below for which model to use, or take a look at our [paper](https://arxiv.org/pdf/2404.10130.pdf) for more information about each model.

| Model Name                                | Info                                                                                           | Recommendation                         
| ----------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------
| `yolo_mouse_ins.pt` <br/> (YOLOv8 M→M)          | YOLOv8 model trained on entire mouse data for osteoclast instance segmentation                 | Ideal for detecting osteoclasts in new mouse well images            |
| `yolo_mouse_det_pretrain.pt`              | NOISE pretrain - YOLOv8 model trained on entire mouse data for osteoclast and nuclei detection | Ideal starting point if you want to train on your own annotated data  |
| `noise_h1_ins_finetune.pt`<br/> (NOISE H→H)    | NOISe model finetuned on H1 dataset for osteoclast instance segmentation                       | Ideal for detecting osteoclasts in new human well images |
| `noise_h2_ins_finetune.pt` <br/> (NOISE H→H)    | NOISe model finetuned on H2 dataset for osteoclast instance segmentation                       | Ideal for detecting osteoclasts in new human well images (alternative) |

"New" data means images not drawn from our own mouse and human datasets, as described in our paper. (If you want to test these models on our datasets, then you should take that you are not using a model that was trained on that very same data. But for new data, this is not a problem.)

Inference can be performed with the following command.

```
python wholeslide_inference.py --model_path path/to/checkpoint.pt --img_foldername path/to/images --out_foldername path/to/output --ratio r --device dev
```

Set ```--ratio``` to the appropriate $\mu m / pixel$ ratio for your dataset; the image will be patched accordingly, and the patches will then be scaled to  832x832 pixel resolution. The training dataset of our images used a ratio of 0.7784.

You can select a cuda device (default is the cpu) to use for inference with ```--device```.

Outputs will be stored in ```path/to/output```. The output for each image consists of a text file containing all predicted bounding boxes, objectness scores, and segmentation masks as well as an image representing these same results.

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

Update the dataset folder absolute path in ```osteo.yaml``` accordingly.

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
