import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from experiments.imagenet.imagenet_tfrecord import ImageNet_TFRecord
from models.resnet_p4 import P4ResNet18
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary
from torchvision.models import resnet18
from torch.distributed import init_process_group
from torch.multiprocessing.spawn import spawn
import torch.backends.cudnn as cudnn


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, "item"):
        return t.item()
    else:
        return t[0]


def main_process(args):
    # set address for master process to localhost since we use a single node
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(args.port)

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
    if not args.standard:
        model = P4ResNet18()
    else:
        model = resnet18(pretrained=True)

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
    if args.resume and not args.standard:
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
    val_loader = ImageNet_TFRecord(
        args.data,
        "val",
        args.batch_size,
        args.workers,
        gpu,
        args.world_size,
        augment=False,
        crop=args.crop,
    )

    # only evaluate model, no training
    validate_rotations(val_loader, model, gpu, args)


def validate_rotations(val_loader, model, gpu, args):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    rotations = {}
    for rotation in range(0, 360, 90):
        rotations[rotation] = AverageMeter()

    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda(gpu).long()

        for rotation in range(0, 360, 90):
            inp = torchvision.transforms.RandomRotation((rotation, rotation))(input)
            # compute output
            with torch.no_grad():
                output = model(inp)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            prec1 = reduce_tensor(prec1, args.world_size)

            rotations[rotation].update(to_python_float(prec1), inp.size(0))

        val_loader_len = int(val_loader._size / args.batch_size)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if gpu == 0 and i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Speed {2:.3f} ({3:.3f})".format(
                    i,
                    val_loader_len,
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time,
                )
            )

    for rotation in rotations.keys():
        print(f"Rotation {rotation} -> Average Top1: {rotations[rotation].avg}")

    return rotations


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

    parser.add_argument("--port", default=12365, type=int)
    parser.add_argument("--crop", default=224, type=int)
    parser.add_argument("--standard", action="store_true")
    args = parser.parse_args()

    print(args)

    main_process(args)
