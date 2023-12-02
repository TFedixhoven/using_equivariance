# Using and Abusing Equivariance
[[arXiv](https://arxiv.org/abs/2308.11316)] - ICCVw 2023 paper, by [Tom Edixhoven](https://github.com/TFedixhoven), [Attila Lengyel](https://attila94.github.io) and [Jan van Gemert](http://jvgemert.github.io/index.html).

This repository contains the PyTorch implementation the experiments described in the paper.

## Abstract
In this paper we show how Group Equivariant Convolutional Neural Networks use subsampling to learn to break equivariance to their symmetries. We focus on 2D rotations and reflections and investigate the impact of broken equivariance on network performance. We show that a change in the input dimension of a network as small as a single pixel can be enough for commonly used architectures to become approximately equivariant, rather than exactly. We investigate the impact of networks not being exactly equivariant and find that approximately equivariant networks generalise significantly worse to unseen symmetries compared to their exactly equivariant counterparts. However, when the symmetries in the training data are not identical to the symmetries of the network, we find that approximately equivariant networks are able to relax their own equivariant constraints, causing them to match or outperform exactly equivariant networks on common benchmark datasets.

## Getting started

Create a local clone of this repository:
```bash
git clone https://github.com/TFedixhoven/using_equivariance
```

See requirements.txt for the required Python packages. You can install them using:
```bash
pip install -r requirements.txt
```

Set the following environment variables:
```bash
export WANDB_DIR=/path/to/wandb/dir
export DATA_DIR=/path/to/data/dir
export OUTPUT_DIR=/path/to/output/dir
```

## Experiments


### Classification
To reproduce the classification experiments, use the following command:

```bash
python -m experiments.classification.train --dataset "cifar10"  # CIFAR-10
python -m experiments.classification.train --dataset "cifar100"  # CIFAR-100
python -m experiments.classification.train --dataset "flowers102"  # Flowers-102
```
This will train a ResNet model on the specified dataset and evaluate it on both the upright and rotated test set. The results will be logged to Weights & Biases.

#### PatchCamelyon
After downloading the PCAM dataset, it can be converted to PyTorch data using the following command:
```bash
python -m experiments.classification.pcam.preprocess_data
```
To then train a model, one can use:
```bash
python -m experiments.classification.pcam.main
```

### ImageNet
We use TFRecords to load the ImageNet dataset - see [more details here](https://www.kaggle.com/code/ipythonx/tfrecord-imagenet-basic-starter-on-tpu-vm). To reproduce the ImageNet experiments, use the following command:

```bash
python -m experiments.imagenet.main "/path/to/tfrecords"
```

### Learning Unequivariance
To train a single P4 convolution to become unequivariant to rotations, one can use `/experiments/learning_unequivariance/learned_unequi_horses.ipynb`


### Rotated MNIST Experiments
```bash
python -m experiments.rotmnist.trainer
```
This will train a single model on the MNIST dataset. To run the training, one has to download the MNIST dataset and extract it into folders first.
This will be rewritten to use the torchvision dataset version of MNIST.

Outputs of our training can be found under `/experiments/rotmnist/training_output`, and they can be visualised using `/experiments/rotmnist/total_comparison_visualisation.ipynb`.

## Citation

If you find this repository useful for your work, please cite as follows:

```
@InProceedings{edixhoven2023using,
    author    = {Edixhoven, Tom and Lengyel, Attila and van Gemert, Jan C.},
    title     = {Using and Abusing Equivariance},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {10},
    year      = {2023},
    pages     = {119-128}
}
```
