import os
import time
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

wait = config["wait"]

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
        "denoising-steps": config["denoising-steps"],
        "distortion-factor": config["distortion-factor"],
        "log-interval": config["log-interval"],
        "checkpoint-interval": config["checkpoint-interval"]
        },
    environment={"WANDB_API_KEY": os.getenv("wandb_api_key")}
    )
estimator.fit(inputs=channels, wait=wait)

job_name = estimator.latest_training_job.name

with open('config/save_last_model.yaml', 'w') as file:
    yaml.dump({"last-job-name": job_name}, file)

print("\nFrom now on the local machine can be disconnected\n")
print("\nKeep up with the training on Weights&Biases\n")

follow = False
while follow:
    logs = sess.logs_for_job(job_name, wait=True)
    print(logs)
    if 'Training job completed' in logs:
        break
    time.sleep(10)


download_trained_model = True

if wait:
    pt_mnist_model_data = estimator.model_data
    if download_trained_model:
        if not os.path.exists("models"):
            os.makedirs("models")
        s3 = boto3.client("s3")
        l = pt_mnist_model_data.split('/')
        key = '/'.join(l[3:])
        dest = "models/" + l[-1]
        s3.download_file(bucket, key, dest)
