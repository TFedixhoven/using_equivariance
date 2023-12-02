"""Image classification experiments for Exact Equivariance."""

import argparse
import datetime
import math
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from models.resnet import ResNet18, ResNet44
from models.resnet_p4 import P4ResNet18, P4ResNet44, gcP4ResNet18, gcP4ResNet44
from models.resnet_p4m import P4MResNet18, P4MResNet44, gcP4MResNet18, gcP4MResNet44
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchinfo import summary

from experiments.classification.datasets import getDataset


class PL_model(pl.LightningModule):
    def __init__(self, args) -> None:
        super(PL_model, self).__init__()

        # Logging.
        self.save_hyperparameters()
        self.train_acc = torchmetrics.Accuracy()
        self.test_acc_seen = torchmetrics.Accuracy()
        self.test_acc_unseen = torchmetrics.Accuracy()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Model definition.
        architectures = {
            "resnet18": ResNet18,
            "resnet44": ResNet44,
            "p4resnet18": P4ResNet18,
            "p4resnet44": P4ResNet44,
            "p4mresnet18": P4MResNet18,
            "p4mresnet44": P4MResNet44,
            "gcp4resnet18": gcP4ResNet18,
            "gcp4resnet44": gcP4ResNet44,
            "gcp4mresnet18": gcP4MResNet18,
            "gcp4mresnet44": gcP4MResNet44,
        }
        assert args.architecture in architectures.keys(), "Model not supported."
        kwargs = {}
        # Use default network width if not provided as argument.
        if args.width is not None:
            kwargs["width"] = args.width
        # Only GConv ResNets have groupcosetmaxpool argument.
        if "p4" in args.architecture:
            kwargs["groupcosetmaxpool"] = args.groupcosetmaxpool
        self.model = architectures[args.architecture](
            num_classes=len(args.classes),
            **kwargs,
        )

        # Print model summary.
        summary(self.model, (2, 3, args.resolution, args.resolution), device="cpu")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
        parser.add_argument("--groupcosetmaxpool", action="store_true")
        parser.add_argument(
            "--width",
            type=int,
            default=None,
            help="override base width for network, default if None",
        )
        parser.add_argument(
            "--architecture", default="resnet44", type=str, help="network architecture"
        )
        return parent_parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.wd
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
                optimizer,
                max_lr=args.lr,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
            ),
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        x, y = batch

        # Forward pass and compute loss.
        y_pred, _ = self.model(x)
        loss = self.criterion(y_pred, y)

        # Logging.
        batch_acc = self.train_acc(y_pred, y)
        self.log("train_acc_step", batch_acc)
        self.log("train_loss_step", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        self.log("train_acc_epoch", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        x, y = batch
        bs = x.shape[0]

        # Rotate input by 90, 180, 270 degrees.
        x = torch.cat(
            [
                x,
                torch.rot90(x, 1, [2, 3]),
                torch.rot90(x, 2, [2, 3]),
                torch.rot90(x, 3, [2, 3]),
            ],
            dim=0,
        )
        y = torch.cat([y, y, y, y], dim=0)

        # Forward pass and compute loss.
        with torch.no_grad():
            y_pred, _ = self.model(x)
        loss = self.criterion(y_pred[:bs, ...], y[:bs])

        # Logging.
        self.test_acc_seen.update(y_pred[:bs, ...], y[:bs])
        self.test_acc_unseen.update(y_pred, y)

        return {"loss": loss}

    def validation_epoch_end(self, outputs) -> None:
        self.log("test_acc_seen_epoch", self.test_acc_seen.compute())
        self.log("test_acc_unseen_epoch", self.test_acc_unseen.compute())
        self.test_acc_seen.reset()
        self.test_acc_unseen.reset()


def main(args) -> None:
    # Create temp dir for wandb.
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    # Use fixed seed.
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    # Get data loaders.
    trainloader, testloader = getDataset(args)
    args.steps_per_epoch = len(trainloader)
    args.epochs = math.ceil(args.epochs / args.split)

    # Initialize model.
    model = PL_model(args)

    # Callbacks and loggers.
    run_name = "{}-{}-{}-{}-split_{}-seed_{}".format(
        args.dataset,
        args.architecture,
        args.resolution,
        str(args.groupcosetmaxpool).lower(),
        str(args.split).replace(".", "_"),
        args.seed,
    )
    run_name += "-rot" if args.rot else ""
    mylogger = pl_loggers.WandbLogger(  # type: ignore
        project="exact-equivariance",
        entity="tudcv",
        config=vars(args),
        name=run_name,
        save_dir=os.environ["WANDB_DIR"],
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Define callback to store model weights.
    weights_dir = os.path.join(
        os.environ["OUT_DIR"], "color_equivariance/classification/"
    )
    os.makedirs(weights_dir, exist_ok=True)
    weights_name = "{}-{}.pth.tar".format(
        datetime.datetime.now().strftime("%Y%m%dT%H%M%S"), run_name
    )

    print("Saving model weights to: {}".format(weights_name))
    checkpoint_callback = ModelCheckpoint(dirpath=weights_dir, filename=weights_name)

    # Train model.
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=mylogger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=args.epochs,
        log_every_n_steps=10,
        deterministic=(args.seed is not None),
        check_val_every_n_epoch=5 * int(1 / args.split),
        # resume_from_checkpoint=args.resume,
    )
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=[testloader],
        ckpt_path=args.resume,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset settings.
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument(
        "--split", default=1.0, type=float, help="Fraction of training set to use."
    )
    parser.add_argument("--resolution", type=int, default=32, help="image resolution")
    parser.add_argument(
        "--rot", action="store_true", help="train with 90-deg rotations"
    )
    parser.add_argument("--padding", type=int, default=4, help="image padding")

    # Training settings.
    parser.add_argument(
        "--bs", type=int, default=256, help="training batch size (default: 256)"
    )
    parser.add_argument(
        "--test-bs", type=int, default=256, help="test batch size (default: 256)"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="number of epochs (default: 200)"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )

    parser.add_argument(
        "--resume", type=str, default=None, help="path to checkpoint to resume from"
    )

    parser = PL_model.add_model_specific_args(parser)

    args = parser.parse_args()
    main(args)
