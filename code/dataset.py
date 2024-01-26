import os
import numpy as np

import gzip
import tarfile

from io import BytesIO
from PIL import Image

import torch
from torch.utils.data import Dataset

def normalize(x, axis):
    eps = np.finfo(float).eps
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True) + eps
    return (x - mean) / std


def read_tar_gz(tar_gz_path):
    images = []
    with gzip.open(tar_gz_path, 'rb') as gz_file:
        with tarfile.TarFile(fileobj=gz_file, mode='r') as tar:
            for member in tar.getmembers():
                image_bytes = tar.extractfile(member).read()
                image_array = np.frombuffer(image_bytes, np.uint8).reshape(56, 56).astype(np.float32)
                images.append(image_array)
    return images


def convert_to_tensor(archive_path):
    images = read_tar_gz(archive_path)
    images = normalize(images, axis=(1, 2))
    images = torch.tensor(images, dtype=torch.float32)
    return images


class PokemonSprites(Dataset):
    def __init__(self, archive_path):
        self.images = convert_to_tensor(archive_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
