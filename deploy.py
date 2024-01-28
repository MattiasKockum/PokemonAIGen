import os
from dotenv import load_dotenv
import yaml

import gzip
import numpy as np

from sagemaker.pytorch import PyTorchModel
from sagemaker import Session
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

import matplotlib.pyplot as plt

load_dotenv()

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

with open('config/save_last_model.yaml', 'r') as file:
    saveconfig = yaml.safe_load(file)


# Deploying

sess = Session()

role = os.getenv("role")

pt_mnist_model_data = saveconfig["last-trained-model"]

model = PyTorchModel(
    entry_point="inference.py",
    source_dir="code",
    role=role,
    model_data=pt_mnist_model_data,
    framework_version="1.5.0",
    py_version="py3",
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type=config["deploy-instance"],
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)


# Calling



data = {"x": 56, "y": 56}

res = predictor.predict(data)


# Shutting down

predictor.delete_endpoint()


# Showing results

for img in res:
    print(res)
    #plt.imshow(img)
    #plt.show()
