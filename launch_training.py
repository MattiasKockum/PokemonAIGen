import os
from dotenv import load_dotenv

import sagemaker
from sagemaker.pytorch import PyTorch

import boto3

load_dotenv()

sess = sagemaker.Session()

role = os.getenv("role")

prefix = os.getenv("prefix")

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
    instance_type="ml.c5.xlarge",
    instance_count=1,
    volume_size=250,
    output_path=f"{output_path}/models",
    hyperparameters={
        "batch-size": 128,
        "epochs": 3,
        "learning-rate": 1e-3,
        "log-interval": 10},
    environment={"WANDB_API_KEY": os.getenv("wandb_api_key")}
)
estimator.fit(inputs=channels)

pt_mnist_model_data = estimator.model_data
print("Model artifact saved at:\n", pt_mnist_model_data)
print("Please copy this into your .env file.")


download_trained_model = False

if download_trained_model:
    if not os.path.exists("models"):
        os.makedirs("models")
    s3 = boto3.client("s3")
    l = pt_mnist_model_data.split('/')
    key = '/'.join(l[3:])
    dest = "models/" + l[-1]
    s3.download_file(bucket, key, dest)
