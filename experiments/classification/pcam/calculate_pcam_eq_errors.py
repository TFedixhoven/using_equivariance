import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchinfo import summary

from torch.utils.data import TensorDataset, DataLoader
from resnet44 import resnet44
from resnet44_p4 import p4resnet44

import torchvision
from torchvision.transforms.functional import resize
import torch.nn.functional as F

from sepgroupy.gconv.make_gconv_indices import *
import os

from datetime import datetime

# --------------------------------
# FEATURE MAP EVALUATION FUNCTIONS
# --------------------------------

def rotate_featuremap(fm, rotations):
    # This can likely be written faster
    results = torch.zeros_like(fm)
    for b in range(fm.shape[0]):
        for c in range(fm.shape[1]):
            for i in range(fm.shape[2]):
                results[b, c, (i + rotations) % fm.shape[2], :, :] =\
                    torchvision.transforms.RandomRotation((rotations*90, rotations*90))(
                        torch.unsqueeze(fm[b, c, i, :, :], 0))
    return results


def calc_eq_error_single(fm1, fm2, rotations):
    r_fm1 = rotate_featuremap(fm1, rotations)
    err = nn.functional.l1_loss(r_fm1, fm2)
    del r_fm1
    return err


def calc_eq_error(inp, model, depth):
    err = 0
    fm_0 = model.get_fm(inp, depth)
    rot_inp = inp.clone()

    for i in range(1, 4):
        rot_inp = torchvision.transforms.RandomRotation((90, 90))(rot_inp)
        fm = model.get_fm(rot_inp, depth)
        err += calc_eq_error_single(fm_0, fm, i)
    del fm_0; del fm; del rot_inp;
    return err
# --------------------------------

def get_data(args):
    class CircleCrop(torch.nn.Module):

        def __init__(self, inplace=False):
            super().__init__()
            self.inplace = inplace

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            if self.inplace:
                res = tensor
            else:
                res = tensor.clone()

            m = (tensor.shape[-1] - 1) / 2
            for i in range(tensor.shape[-2]):
                for j in range(tensor.shape[-1]):
                    if ((i - m) ** 2 + (j - m) ** 2) > m * m:
                        if len(tensor.shape) == 4:
                            res[:, :, i, j] = 0
                        else:
                            res[:, i, j] = 0
            return res

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}(inplace={self.inplace})"
    
    normalize = torchvision.transforms.Normalize(mean=[0.3949, 0.0696, 0.3760], std=[0.4721, 0.5571, 0.4291], inplace=True)
    
    print('Loading data...')
    test_data = torch.load('./data/PCAM_test_data.pt')
    test_labels = torch.load('./data/PCAM_test_labels.pt').squeeze().long()

    normalize(test_data)

    if args.circle_crop:
        c = CircleCrop(inplace=True)
        c(test_data)

    test_loader = DataLoader(TensorDataset(test_data, test_labels),
                             batch_size=args.test_batch_size, shuffle=False, pin_memory=False, num_workers=2)
    print('Data loaded')
    return test_loader

def main(args):
    # create model
    model = p4resnet44(num_classes=2)
    
    # push model to gpu
    if args.cuda:
        model = model.cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
        else:
            print('invalid file path... {:}'.format(args.resume))

    # Data loading code
    test_loader = get_data(args)
    avg_eq_err = [0, 0, 0, 0]
    for i, data in enumerate(test_loader):
        if i >= args.eq_batches:
            break
        if args.cuda:
            inp = data[0].cuda()
        else:
            inp = data[0]

        inp = resize(inp, (args.input_size, args.input_size))
        for d in range(4):
            err = calc_eq_error(inp, model, d) / inp.shape[0]
            avg_eq_err[d] += err.item()
        del inp
        torch.cuda.empty_cache()
    for i in range(4):
        avg_eq_err[i] = avg_eq_err[i] / args.eq_batches
    print(f'Total Equivariance error: {avg_eq_err}')
    return


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PatchCamelyon')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--test-batch-size', type=int, default=2, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--circle-crop', action='store_true', default=False)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--eq-batches', type=int, default=10,
                        help='amount of batches used to evaluate equivariance error')
    parser.add_argument('--input-size', '-i', type=int, default=96)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)

    main(args)
