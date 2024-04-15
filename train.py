import os
import time
import torch
import random
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import PolynomialLR

from models.bisenet import BiSeNet
from utils.dataset import CelebAMaskHQ
from utils.loss import OhemLossWrapper
from utils.transform import TrainTransform


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Argument Parser for Training Configuration")

    # Dataset
    parser.add_argument('--num-classes', type=int, default=19, help='Number of classes in the dataset')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=12, help='Number of workers for data loading')
    parser.add_argument('--image-size', type=int, nargs=2, default=[448, 448], help='Size of input images')
    parser.add_argument('--data-root', type=str, default='/mnt/d/Datasets/CelebAMask-HQ/',
                        help='Root directory of the dataset')

    # Optimizer
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--lr-start', type=float, default=1e-2, help='Initial learning rate')
    parser.add_argument('--max-iter', type=int, default=80000, help='Maximum number of iterations')
    parser.add_argument('--power', type=float, default=0.9, help='Power for learning rate policy')
    parser.add_argument('--lr-warmup-epochs', type=int, default=1, help='Number of warmup epochs')
    parser.add_argument('--warmup-start-lr', type=float, default=1e-5, help='Warmup starting learning rate')
    parser.add_argument('--score-thres', type=float, default=0.7, help='Score threshold')

    # Training loop
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone architecture')

    # Train loop
    parser.add_argument('--print-freq', type=int, default=50, help='Print frequency during training')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')

    args = parser.parse_args()
    return args


def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_weight_decay(model, weight_decay=1e-5):
    """Applying weight decay to only weights, not biases"""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or isinstance(param, nn.BatchNorm2d) or "bn" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.},
            {"params": decay, "weight_decay": weight_decay}]


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, scaler=None):
    model.train()
    batch_loss = []
    for batch_idx, (image, target) in enumerate(data_loader):
        start_time = time.time()
        image = image.to(device)
        target = target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        batch_loss.append(loss.item())

        if (batch_idx + 1) % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f'Train: [{epoch:>3d}][{batch_idx + 1:>4d}/{len(data_loader)}] '
                f'Loss: {loss.item():.4f}  '
                f'Time: {(time.time() - start_time):.3f}s '
                f'LR: {lr:.7f} '
            )
    print(f"Avg batch loss: {np.mean(batch_loss):.7f}")


def main(params):
    random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_dir = os.path.join(params.data_root, 'CelebA-HQ-img')
    labels_dir = os.path.join(params.data_root, 'mask')

    dataset = CelebAMaskHQ(images_dir, labels_dir, transform=TrainTransform(image_size=params.image_size))
    data_loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # model
    model = BiSeNet(num_classes=params.num_classes, backbone_name=params.backbone)
    model.to(device)

    n_min = params.batch_size * params.image_size[0] * params.image_size[1] // 16
    criterion = OhemLossWrapper(thresh=params.score_thres, min_kept=n_min)

    # optimizer
    parameters = add_weight_decay(model, params.weight_decay)
    optimizer = torch.optim.SGD(parameters, lr=params.lr_start, momentum=params.momentum,
                                weight_decay=params.weight_decay)

    iters_per_epoch = len(data_loader)
    lr_scheduler = PolynomialLR(
        optimizer, total_iters=iters_per_epoch * (params.epochs - params.lr_warmup_epochs), power=params.power
    )
    start_epoch = 0
    if params.resume:
        checkpoint = torch.load(f"./weights/{params.backbone}.ckpt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, params.epochs):
        train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            lr_scheduler,
            device,
            epoch,
            params.print_freq,
            scaler=None
        )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
        }
        torch.save(ckpt, f'./weights/{params.backbone}.ckpt')

    #  save final model
    state = model.state_dict()
    torch.save(state, f'./weights/{params.backbone}.pt')


if __name__ == "__main__":
    args = parse_args()
    main(args)
