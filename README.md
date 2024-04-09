# NOISe: Nuclei-Aware Osteoclast Instance Segmentation for Mouse-to-Human Domain Transfer
<div align="center">
  <p>
    <img width="100%" src="images/banner.png" alt="NOISe banner">
  </p>
</div>

NOISe builds on top of [YOLOv8](https://github.com/ultralytics/ultralytics) for the task of osteoclast instance segmentation. This repository contains training, validation, and inference scripts to train an instance segmentation model from scratch and finetune on top of the provided checkpoints trained for osteoclast and nuclei detection as proposed in our [paper]().

<details open>
<summary>Install</summary>

Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

[![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

```bash
pip install ultralytics
```

For alternative installation methods including [Conda](https://anaconda.org/conda-forge/ultralytics), [Docker](https://hub.docker.com/r/ultralytics/ultralytics), and Git, please refer to the [Quickstart Guide](https://docs.ultralytics.com/quickstart).

</details>

## Dataset
Our dataset consists of full slide images and corresponding instance segmentation annotations, along with patches used for training and validation in our experiments. Please download the dataset from [here]() and and place it in the dataset folder with the following folder structure:

```
noise
  |- ...
  |- ...
  |- dataset
    |- images
    |- labels
  |- ultralytics
  |- ...
  |- ...
```

## Whole Slide Inference
Instance segmentation prediction can be done for a whole slide image by creating overlapping patches of ```832x832``` resolution, that are then merged to generate a full-scale output. Different models trained on various configurations of data are available [here]().

```
python wsi_inference.py path/to/checkpoint.pt path/to/images
```

Outputs will be stored in ```path/to/output```.

## Training
Training from scratch or finetuning a specific checkpoint can be done using the following command:

```
python train.py --ckpt path/to/checkpoint.pt
```

## Validation
Running evaluation on test data can be done using the following command:

```
python val.py --ckpt path/to/checkpoint.pt
```

## Acknowledgement
NOISe is largely based on [YOLOv8](https://github.com/ultralytics/ultralytics).

## Citation
If you find the dataset, model weights, or the code useful for your work, please consider citing us:
```
```