import argparse
import math
import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from experiments.imagenet.imagenet_tfrecord import ImageNet_TFRecord
from models.resnet_p4 import P4ResNet18
from torch.distributed import init_process_group
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, "item"):
        return t.item()
    else:
        return t[0]


def main_process(args):
    # set address for master process to localhost since we use a single node
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # use all gpus pytorch can find
    args.world_size = torch.cuda.device_count()
    print("Found {} GPUs:".format(args.world_size))
    for i in range(args.world_size):
        print("{} : {}".format(i, torch.cuda.get_device_name(i)))

    # total batch size = batch size per gpu * ngpus
    args.total_batch_size = args.world_size * args.batch_size

    # TODO: find out what this stuff does
    print("\nCUDNN VERSION: {}\n".format(cudnn.version()))
    cudnn.benchmark = True
    assert cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if not len(args.data):
        raise Exception("error: No data set provided")

    # start processes for all gpus
    spawn(gpu_process, nprocs=args.world_size, args=(args,))


def gpu_process(gpu, args):
    # each gpu runs in a separate proces
    torch.cuda.set_device(gpu)
    init_process_group(
        backend="nccl", init_method="env://", rank=gpu, world_size=args.world_size
    )

    # create model
    model = P4ResNet18()
    if gpu == 0:
        summary(model, (2, 3, 224, 224), device="cpu")

    # Set cudnn to deterministic setting
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(gpu)
        torch.set_printoptions(precision=10)

    # push model to gpu
    model = model.cuda(gpu)

    # Scale learning rate based on global batch size
    args.lr = args.lr * float(args.batch_size * args.world_size) / 256.0
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Use DistributedDataParallel for distributed training
    model = DDP(model, device_ids=[gpu], output_device=gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    best_prec1 = 0

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(
                    args.resume, map_location=lambda storage, loc: storage.cuda(gpu)
                )
                args.start_epoch = checkpoint["epoch"]
                best_prec1 = checkpoint["best_prec1"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                print(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, checkpoint["epoch"]
                    )
                )
                return best_prec1
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
                return 0

        best_prec1 = resume()

    # Data loading code
    train_loader = ImageNet_TFRecord(
        args.data,
        "train",
        args.batch_size,
        args.workers,
        gpu,
        args.world_size,
        augment=True,
    )
    val_loader = ImageNet_TFRecord(
        args.data,
        "val",
        args.batch_size,
        args.workers,
        gpu,
        args.world_size,
        augment=False,
    )

    # only evaluate model, no training
    if args.evaluate:
        validate(val_loader, model, criterion, gpu, args)
        return

    total_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        avg_train_time = train(
            train_loader, model, criterion, optimizer, epoch, gpu, args
        )
        total_time.update(avg_train_time)

        # if in test mode quit after 1st epoch
        if args.test:
            break

        # evaluate on validation set
        [prec1, prec5] = validate(val_loader, model, criterion, gpu, args)

        # remember best prec@1 and save checkpoint
        if gpu == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
            )
            if epoch == args.epochs - 1:
                print(
                    "##Top-1 {0}\n"
                    "##Top-5 {1}\n"
                    "##Perf  {2}".format(
                        prec1, prec5, args.total_batch_size / total_time.avg
                    )
                )

        train_loader.reset()
        val_loader.reset()


def train(train_loader, model, criterion, optimizer, epoch, gpu, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda(gpu).long()
        train_loader_len = int(math.ceil(train_loader._size / args.batch_size))

        # lr schedule
        adjust_learning_rate(args.lr, optimizer, epoch, i, train_loader_len)

        # if in test mode, quit after 100 iterations
        if args.test and i > 100:
            break

        # compute output
        output, _ = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            prec1 = reduce_tensor(prec1, args.world_size)
            prec5 = reduce_tensor(prec5, args.world_size)

            # # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if gpu == 0:  # only print for main process
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    # 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    "Speed {3:.3f} ({4:.3f})\t"
                    "Loss {loss.val:.10f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        epoch,
                        i,
                        train_loader_len,
                        args.world_size * args.batch_size / batch_time.val,
                        args.world_size * args.batch_size / batch_time.avg,
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )

    return batch_time.avg


def validate(val_loader, model, criterion, gpu, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda(gpu).long()
        val_loader_len = int(val_loader._size / args.batch_size)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        reduced_loss = reduce_tensor(loss.data, args.world_size)
        prec1 = reduce_tensor(prec1, args.world_size)
        prec5 = reduce_tensor(prec5, args.world_size)
        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if gpu == 0 and i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Speed {2:.3f} ({3:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    i,
                    val_loader_len,
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )

    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(top1=top1, top5=top5))

    return [top1.avg, top5.avg]


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(lr, optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = lr * (0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5.0 * len_epoch)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument(
        "data",
        metavar="DIR",
        nargs="*",
        help="path(s) to dataset",
        default="/tudelft.net/staff-bulk/ewi/insy/CV-DataSets/imagenet/tfrecords",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers per GPU (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=64,
        type=int,
        metavar="N",
        help="mini-batch size per process (default: 64)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="Initial learning rate.  Will be scaled by <global batch size>/64: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model",
    )
    parser.add_argument(
        "--dali_cpu",
        action="store_true",
        help="Runs CPU based version of DALI pipeline.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "-t", "--test", action="store_true", help="Run short training script."
    )
    args = parser.parse_args()

    print(args)

    main_process(args)
