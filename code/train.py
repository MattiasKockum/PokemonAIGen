import os
import subprocess
subprocess.call(["pip", "install", "wandb==0.15.11"])
subprocess.call(["wandb", "login", os.environ["WANDB_API_KEY"]])

import argparse
import json
import logging
import sys
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Net
from dataset import PokemonSprites
from utils import add_variable_gaussian_noise

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(args):
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda > 0 else "cpu")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_set = PokemonSprites(args.train)
    test_set = PokemonSprites(args.test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)

    net = Net(train_set.images_size, 1, args.noise).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(
        net.parameters(), betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay
    )

    wandb.init(
        project="PokemonSprites",
        config={
            "learning_rate": args.learning_rate,
            "architecture": "ConvAutoEncoder",
            "dataset": "PokemonSprites",
            "epochs": args.epochs,
            "noise": args.noise
            }
    )

    logger.info("Start training ...")
    for epoch in range(1, args.epochs + 1):
        net.train()
        for batch_idx, images in enumerate(train_loader, 1):
            images = images.to(device)
            noisy_images = add_variable_gaussian_noise(images, args.noise)
            output = net(noisy_images)
            loss = loss_fn(output, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(batch_idx, args.log_interval, batch_idx % args.log_interval)
            if batch_idx % args.log_interval == 0:
                log(loss, epoch, batch_idx, images, train_loader)

        # test the model
        test(net, test_loader, device, loss_fn, args)

    # save model checkpoint
    save_model(net, args.model_dir)
    wandb.finish()
    return


def log(loss, epoch, batch_idx, imgs, train_loader):
    wandb.log({"loss": loss.item(), "epoch": epoch})
    print(
        "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
            epoch,
            batch_idx * len(imgs),
            len(train_loader.sampler),
            100.0 * batch_idx / len(train_loader),
            loss.item(),
        )
    )


def test(model, test_loader, device, loss_fn, args):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            noisy_images = add_variable_gaussian_noise(images, args.noise)
            output = model(noisy_images)
            test_loss += loss_fn(output, images).item()


    test_loss /= len(test_loader.dataset)
    wandb.log({"test": test_loss})
    logger.info(
        "Test set: Average loss: {:.4f}\n".format(test_loss)
    )
    return


def save_model(model, model_dir):
    logger.info("Saving the model")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    return


def parse_args():
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, metavar="n", help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.2,
        metavar="no",
        help="How much noise the model is trained to remove",
    )
    parser.add_argument(
        "--beta_1", type=float, default=0.9, metavar="BETA1", help="beta1 (default: 0.9)"
    )
    parser.add_argument(
        "--beta_2", type=float, default=0.999, metavar="BETA2", help="beta2 (default: 0.999)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        metavar="WD",
        help="L2 weight decay (default: 1e-4)",
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
