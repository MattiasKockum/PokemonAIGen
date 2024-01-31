import os
import boto3

import numpy as np

import gzip
import tarfile

from io import BytesIO
from PIL import Image


import subprocess

from dotenv import load_dotenv

load_dotenv()

import sagemaker

sess = sagemaker.Session()

bucket = sess.default_bucket()

prefix = os.getenv("prefix")

np.random.seed(151)
test_proportion = 0.2

if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/training"):
    os.makedirs("data/training")
if not os.path.exists("data/testing"):
    os.makedirs("data/testing")

subprocess.call(["wget", "https://veekun.com/static/pokedex/downloads/generation-1.tar.gz", "-O", "data/generation-1.tar.gz"])


def read_images_from_tar_gz(archive_path, images_file_pattern):
    with gzip.open(archive_path, 'rb') as gz_file:
        with tarfile.TarFile(fileobj=gz_file, mode='r') as tar:
            images_list = []
            for member in tar.getmembers():
                if member.isfile() and images_file_pattern in member.name:
                    file_content = tar.extractfile(member).read()
                    image = Image.open(BytesIO(file_content))
                    images_list.append(image)
    return images_list


def resize_images(images):
    size = max([i.size[0] for i in images])
    images = [
        np.pad(image, int((size - image.size[0])/2), constant_values=255)
        for image in images
    ]
    return images


def create_tar_gz(images, output_path):
    with BytesIO() as tar_buffer:
        with tarfile.TarFile(fileobj=tar_buffer, mode='w') as tar:
            for i, image in enumerate(images):
                image_bytes = image.tobytes()
                image_io = BytesIO(image_bytes)
                tar_info = tarfile.TarInfo(f'image_{i}.png')
                tar_info.size = len(image_bytes)
                tar.addfile(tar_info, image_io)
        tar_buffer.seek(0)
        with gzip.open(output_path, 'wb') as gz_file:
            gz_file.write(tar_buffer.read())


archive_path = "data/generation-1.tar.gz"

images_file_patterns = [
    'pokemon/main-sprites/red-blue/gray',
    'pokemon/main-sprites/yellow/gray'
]

images = []
for images_file_pattern in images_file_patterns:
    images += read_images_from_tar_gz(archive_path, images_file_pattern)

images = resize_images(images)
np.random.shuffle(images)

test_size = int(len(images) * test_proportion)
test_sample = images[:test_size]
train_sample = images[test_size:]

create_tar_gz(test_sample, "data/testing/images.tar.gz")
create_tar_gz(train_sample, "data/training/images.tar.gz")


s3 = boto3.resource("s3")
s3.Bucket(bucket).upload_file("data/testing/images.tar.gz", f"{prefix}/data/testing/images.tar.gz")
s3.Bucket(bucket).upload_file("data/training/images.tar.gz", f"{prefix}/data/training/images.tar.gz")
