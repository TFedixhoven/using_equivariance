import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torchinfo import summary

from loader import load
from models_groupy import SepCNN
from models_nopool import NoPoolCNN
from models_4xcnn import SharedCNN
from models_dropout import GCNN

from torch.utils.data.sampler import SubsetRandomSampler

import copy
import torchvision
import PIL.Image


def main(args):

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    # Fix seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

    def train(epoch):
        model.train()
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            if args.biased_training:
                if not args.uniform:
                    a = random.gauss(args.biased_mean, args.biased_std)
                    data = torchvision.transforms.RandomRotation((a, a), resample=PIL.Image.BILINEAR)(data)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        scheduler.step()
        return 100. * correct / (len(train_loader.dataset) * 0.8)

    def test(loaders):
        model.eval()
        accs = {}

        for name, test_loader in loaders:
            print('-= Testing {0} =-'.format(name))
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                with torch.no_grad():
                    output = model(data)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').data
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            size = len(test_loader.dataset)
            if name == 'valid':
                size = len(test_loader.dataset) * 0.2
            test_loss /= size
            normal_acc = 100. * correct / size
            print('Test set: Average loss: {:.4f}, \
                  Accuracy: {}/{} ({:.2f}%)'.format(
                test_loss, correct, size,
                normal_acc))

            accs[name] = normal_acc.item()
        print('\n\n')
        return accs

    def test360(m, step_size=10):
        m.eval()
        accs = {}
        for angle in range(0, 359, step_size):
            print('-= Testing Rotation {0} =-'.format(angle))
            test_loader = test_loaders[1][1]
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data = torchvision.transforms.RandomRotation((angle, angle), resample=PIL.Image.BILINEAR)(data)

                with torch.no_grad():
                    output = m(data)
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
       
    model = GCNN(args.planes, equivariance=args.equivariance, pad=args.pad)

    model_best = copy.deepcopy(model)
    acc_best = 0.0

    summary(model, (256, 3, args.input_size, args.input_size))

    if args.cuda:
        model.cuda()

    data_dict = {False: 'MNIST_nonrot_validation',
                 True: 'MNIST_rot_validation'}

    data = load(data_dict[args.rotated], inp_size=args.input_size,
                fraction=args.train_fraction, shift=args.shift_augment)

    if args.uniform:
        data.randomRotate()

    num_train = len(data)

    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))

    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                               sampler=train_sampler, **kwargs)
    valid_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                               sampler=valid_sampler, **kwargs)

    test_loaders = [('rotated', torch.utils.data.DataLoader(load('MNIST_rot', inp_size=args.input_size), batch_size=args.batch_size, shuffle=False, **kwargs)),
                    ('nonrotated', torch.utils.data.DataLoader(load('MNIST_nonrot', inp_size=args.input_size), batch_size=args.batch_size, shuffle=False, **kwargs))]

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step, gamma=0.5)

    accs = []
    for epoch in range(1, args.epochs + 1):
        acc = train(epoch)

        test_acc = test([('valid', valid_loader)])
        if test_acc['valid'] > acc_best:
            acc_best = test_acc['valid']
            model_best = copy.deepcopy(model)

        if epoch % args.log_interval == 0:
            print('Epoch {:d} train accuracy: {:.2f}%, valid accuracy {:.2f}%'.format(epoch, acc, test_acc['valid']))
            print('Epoch {:d} train accuracy: {:.2f}%'.format(epoch, acc))
            test_acc = test(test_loaders)
            accs.append(test_acc)

    print(f'Accuracies:\n{accs}')

    if args.savestr:
        torch.save(model.state_dict(),'./{}_trained_{}_{}.pth.tar'.format(args.savestr,args.planes, args.equivariance))

    test_accs = test360(model)
    print(test_accs)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='RotatedMNIST Experiment')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=8, metavar='N',
                        help='number of epochs to train (default: 8)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--wd', type=float, default=1e-5, metavar='WD',
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--step', type=int, default=10, metavar='LRS',
                        help='step size for lr scheduler (default: 10)')
    parser.add_argument('--planes', type=int, default=20, metavar='planes',
                        help='number of channels in CNN (default: 20)')
    parser.add_argument('--samples', type=int, default=None, metavar='samples',
                        help='amount of training samples to use (default: None (all))')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--savestr', type=str, default='', metavar='STR',
                        help='weights file name addendum')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--train-fraction', type=float, default=1.0, metavar='F',
                        help='the fraction of the training data to use (default: 1.0)')
    parser.add_argument('-i', '--input-size', type=int, default=48)
    parser.add_argument('-r', '--rotated', action='store_true')

    parser.add_argument('--biased-training', action='store_true')
    parser.add_argument('--biased-mean', type=float, default=45.0)
    parser.add_argument('--biased-std', type=float, default=5.0)
    parser.add_argument('--uniform', action='store_true')

    parser.add_argument('--equivariance', type=str, default='P4')
    parser.add_argument('--shift-augment', action='store_true')
    parser.add_argument('--pad', type=int, default=1)
    args = parser.parse_args()

    main(args)
