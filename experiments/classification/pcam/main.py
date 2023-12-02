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

import PIL.Image

import copy
from sepgroupy.gconv.make_gconv_indices import *

from datetime import datetime

model_dict = {'resnet44': resnet44,
              'resnet44_p4': p4resnet44}


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
    fm1 = rotate_featuremap(fm1, rotations)
    return nn.functional.l1_loss(fm1, fm2)


def calc_eq_error(inp, model, depth):
    err = 0
    fm_0 = model.get_fm(inp, depth)
    rot_inp = inp.clone()

    for i in range(1, 4):
        rot_inp = torchvision.transforms.RandomRotation((90, 90))(rot_inp)
        fm = model.get_fm(rot_inp, depth)
        err += calc_eq_error_single(fm_0, fm, i)
    del fm_0; del fm
    return err
# --------------------------------

def train(epoch, model, criterion, optimizer, train_loader, test_loader, args):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        # Temporarily done during training due to RAM consumption
        data = resize(data, (args.input_size, args.input_size))

        target = target.long()
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx} / {len(train_loader)} Done (Acc: {correct / ((batch_idx + 1) * len(data))})')

    return 100. * correct / len(train_loader.dataset)


def test(model, criterion, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    eq_error = [0, 0, 0, 0]
    eq_batch = 0
    EQ_BATCHES = 10

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # Temporary due to RAM consumption
        data = resize(data, (args.input_size, args.input_size))

        target = target.long()
        with torch.no_grad():
            output = model(data)
        test_loss += criterion(output, target).data  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if args.test_eq:
            if eq_batch >= EQ_BATCHES:
                continue
            for depth in range(4):
                eq_error[depth] += calc_eq_error(data, model, depth)
            eq_batch += 1

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if args.test_eq:
        print('Equivariance errors: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
            eq_error[0] / EQ_BATCHES,
            eq_error[1] / EQ_BATCHES,
            eq_error[2] / EQ_BATCHES,
            eq_error[3] / EQ_BATCHES))

    return 100. * correct / len(test_loader.dataset)


def test360(model, test_loader, args, step_size=10):
        model.eval()
        accs = {}
        for angle in range(0, 359, step_size):
            print('-= Testing Rotation {0} =-'.format(angle))
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                # Temporary due to RAM consumption
                data = resize(data, (args.input_size, args.input_size))
                target = target.long()

                data = torchvision.transforms.RandomRotation((angle, angle), resample=PIL.Image.BILINEAR)(data)

                with torch.no_grad():
                    output = model(data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').data
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(test_loader.dataset)
            acc = 100. * correct / len(test_loader.dataset)
            print('Test set: Average loss: {:.4f}, \
                  Accuracy: {}/{} ({:.2f}%)'.format(
                test_loss, correct, len(test_loader.dataset),
                acc))
            accs[angle] = acc
        return accs


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
                    if ((i - m)**2 + (j - m)**2) > m*m:
                        if len(tensor.shape) == 4:
                            res[:, :, i, j] = 0
                        else:
                            res[:, i, j] = 0
            return res

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}(inplace={self.inplace})"

    normalize = torchvision.transforms.Normalize(mean=[0.3949, 0.0696, 0.3760], std=[0.4721, 0.5571, 0.4291], inplace=True)
    
    print('Loading data...')
    train_data = torch.load('./data/PCAM_train_data.pt')
    train_labels = torch.load('./data/PCAM_train_labels.pt').squeeze().long()

    valid_data = torch.load('./data/PCAM_valid_data.pt')
    valid_labels = torch.load('./data/PCAM_valid_labels.pt').squeeze().long()

    test_data = torch.load('./data/PCAM_test_data.pt')
    test_labels = torch.load('./data/PCAM_test_labels.pt').squeeze().long()
    
    normalize(train_data)
    normalize(valid_data)
    normalize(test_data)
    
    if args.crop:
        print('Cropping data...')
        train_data = train_data[:, :, 32:64, 32:64]
        test_data = test_data[:, :, 32:64, 32:64]

    if args.circle_crop:
        c = CircleCrop(inplace=True)
        c(train_data)
        c(valid_data)
        c(test_data)

    # Temporarily moved due to memory restrictions
    #  print('Resizing data...')
    #  train_data = resize(train_data, (args.input_size, args.input_size))
    #  test_data = resize(test_data, (args.input_size, args.input_size))

    train_loader = DataLoader(TensorDataset(train_data, train_labels),
                              batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_loader = DataLoader(TensorDataset(valid_data, valid_labels),
                              batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    test_loader = DataLoader(TensorDataset(test_data, test_labels),
                             batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=2)
    print(f'Data loaded, train {train_data.shape}, test {test_data.shape}, train labels {train_labels.shape}')
    return train_loader, valid_loader, test_loader


def main(args):
    # Fix seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print('.: USING CUDA :.')

    # Build model
    model = model_dict[args.model](num_classes=2)
    summary(model, (64,3,args.input_size,args.input_size), col_names=("input_size", "output_size", "num_params"))
    if args.cuda: model = model.cuda()

    # Get data
    train_loader, valid_loader, test_loader = get_data(args)

    # Optimization settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 16, 20, 24, 26, 28], gamma=0.75)

    acc_best = 0.0
    model_best = copy.deepcopy(model)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print("Train start: Current Time =", datetime.now().strftime("%H:%M:%S"))
        acc = train(epoch, model, criterion, optimizer, train_loader, test_loader, args)
        scheduler.step()
        print("Valid start: Current Time =", datetime.now().strftime("%H:%M:%S"))
        valid_acc = test(model, criterion, valid_loader, args)
        print("Valid done: Current Time =", datetime.now().strftime("%H:%M:%S"))

        augstr = '_cropped' if args.crop else ''
        #torch.save(model.state_dict(), '{}{}_seed_{:d}_epoch{}.pth.tar'.format(args.model, augstr,
                                                                                        #args.seed, epoch))

        if valid_acc > acc_best:
            acc_best = valid_acc
            model_best = copy.deepcopy(model)

        if epoch % args.log_interval == 0:
            print('Learning rate: {:.1e}'.format(optimizer.param_groups[0]['lr']))
            print('Epoch {:d} train accuracy: {:.2f}%, valid accuracy {:.2f}%'.format(epoch, acc, valid_acc))
            test(model, criterion, test_loader, args)

    print('Testing best model on test data')
    test(model_best, criterion, test_loader, args)
    print(test360(model_best, test_loader, args))
    
    # Save model
    augstr = '_cropped' if args.crop else ''
    torch.save(model_best.state_dict(), '{}{}_seed_{:d}.pth.tar'.format(args.model, augstr, args.seed))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PatchCamelyon')
    parser.add_argument('--dataset_path', metavar='path/to/dataset/root', default='./data/',
                        type=str, help='location of cifar10 files')
    parser.add_argument('--model', metavar='resnet44', default='resnet44_p4',
                        choices=['resnet44', 'resnet44_p4'],
                        type=str, help='Select model to use for experiment.')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-5, metavar='WD',
                        help='weight decay (default: 0.0)')
    parser.add_argument('--crop', action='store_true', default=False,
                        help='crops the input to the center 32x32 patch')
    parser.add_argument('--circle-crop', action='store_true', default=False)
    parser.add_argument('--hflip', action='store_true', default=False,
                        help='perform random horizontal flipping')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--freeze', action='store_true', default=False,
                        help='freezes the convolution part of the network')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--input-size', '-i', type=int, default=97)
    parser.add_argument('--test-eq', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)

    main(args)
