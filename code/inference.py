import os
import json
import torch

from model import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_fn(model_dir):
    model = Net((56, 56), 1, 0.05)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    return model

def input_fn(request_body, request_content_type):
    assert request_content_type=='application/json'
    x = json.loads(request_body)['x']
    y = json.loads(request_body)['y']
    data = torch.rand((1, x, y))
    return data

def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = input_object
        for i in range(int(1 / model.noise)):
            prediction = model(prediction)
    return prediction

def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)

