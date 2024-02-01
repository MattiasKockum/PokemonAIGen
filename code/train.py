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

from model import Net, save_model
from dataset import PokemonSprites
from utils import add_variable_gaussian_noise, augment_data

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

    net = Net(train_set.images_size, 1, args.denoising_steps).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(
        net.parameters(), betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay
    )

    config={
        "learning_rate": args.learning_rate,
        "architecture": "DiffusionVisionTransformerModel",
        "dataset": "PokemonSprites",
        "epochs": args.epochs,
        "denoising-steps": args.denoising_steps,
        "distortion-factor": args.distortion_factor
        }

    wandb.init(
        project="PokemonSprites",
        config=config
    )

    logger.info("Start training ...")
    for epoch in range(1, args.epochs + 1):
        config["current-epoch"] = epoch
        config["current-learning-rate"] = args.learning_rate
        net.train()
        for batch_idx, images in enumerate(train_loader, 1):
            images = images.to(device)
            images = augment_data(images, args)
            noisy_images = add_variable_gaussian_noise(images, 1 / args.denoising_steps)
            output = net(noisy_images)
            loss = loss_fn(output, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                train_log(loss, epoch, batch_idx, images, train_loader)

        # test the model
        test(net, test_loader, device, loss_fn, args)
        if epoch % args.checkpoint_interval == 0:
            save_model(net, config, f"{args.model_dir}/checkpoint_model_{epoch}.pth")

    # save model checkpoint
    save_model(net, config, f"{args.model_dir}/final_model_{epoch}.pth")
    wandb.finish()
    return


def train_log(loss, epoch, batch_idx, imgs, train_loader):
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
    return


def diffusion(model, batch_size=4):
    noise = torch.rand((batch_size, model.channels, *model.image_size))
    L = [noise]
    with torch.no_grad():
        for _ in range(model.denoising_steps):
            L.append(model(L[-1]))
    return L


def test_log(model, test_loss):
    exemple_images = diffusion(model)
    exemple_images = [batch.unsqueeze(0) for batch in exemple_images]
    exemple_images = torch.cat(exemple_images)
    exemple_images = exemple_images.squeeze(2)  # Removing channels
    exemple_images = exemple_images.permute(1, 0, 2, 3)
    exemple_images = [torch.cat(list(l), dim=1) for l in list(exemple_images)]
    exemple_images = torch.cat(exemple_images)
    exemple_images = wandb.Image(exemple_images, caption="Diffusion trail")
    wandb.log({
        "test": test_loss,
        "inference-exemples": exemple_images
    })
    return


def test(model, test_loader, device, loss_fn, args):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            noisy_images = add_variable_gaussian_noise(images, 1 / args.denoising_steps)
            output = model(noisy_images)
            test_loss += loss_fn(output, images).item()


    test_loss /= len(test_loader.dataset)
    test_log(model, test_loss)
    logger.info("Test set: Average loss: {:.4f}\n".format(test_loss))
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
        "--distortion-factor",
        type=int,
        default=4,
        metavar="daf",
        help="How much the images are stretch during data augmentation",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=20,
        metavar="no",
        help="How much denoising steps the model is trained to do",
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
        "--checkpoint-interval",
        type=int,
        default=100,
        metavar="CI",
        help="how many epochs to wait before saving the model",
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
