import os
from dotenv import load_dotenv
import yaml

import sagemaker
from sagemaker.pytorch import PyTorch

import boto3

load_dotenv()

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


sess = sagemaker.Session()

role = os.getenv("role")

prefix = config["prefix"]

bucket = sess.default_bucket()

output_path = f"s3://{bucket}/{prefix}"

channels = {
        "training": f"{output_path}/data/training",
        "testing": f"{output_path}/data/testing"
}

estimator = PyTorch(
    entry_point="train.py",
    source_dir="code",
    role=role,
    framework_version="1.5.0",
    py_version="py3",
    instance_type=config["learning-instance"],
    instance_count=1,
    volume_size=250,
    output_path=f"{output_path}/models",
    hyperparameters={
        "batch-size": config["batch-size"],
        "epochs": config["epochs"],
        "learning-rate": config["learning-rate"],
        "noise": config["noise"],
        "log-interval": config["log-interval"]},
    environment={"WANDB_API_KEY": os.getenv("wandb_api_key")}
    )
estimator.fit(inputs=channels)

pt_mnist_model_data = estimator.model_data

with open('config/save_last_model.yaml', 'w') as file:
    yaml.dump({"last-trained-model": pt_mnist_model_data}, file)

download_trained_model = False

if download_trained_model:
    if not os.path.exists("models"):
        os.makedirs("models")
    s3 = boto3.client("s3")
    l = pt_mnist_model_data.split('/')
    key = '/'.join(l[3:])
    dest = "models/" + l[-1]
    s3.download_file(bucket, key, dest)
