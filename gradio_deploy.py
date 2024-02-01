import torch
import numpy as np
import gradio as gr
import yaml

from code.model import load_model

with open('config/gradio.yaml', 'r') as file:
    config = yaml.safe_load(file)

model = load_model(config["model-path"])[0]

def preprocess_image(input_image):
    input_image = np.pad(input_image, int((model.image_size[0] - input_image.shape[0])/2), constant_values=255)
    input_tensor = torch.tensor(input_image, dtype=torch.float32)
    input_tensor = input_tensor.sum(dim=2)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def inference_function(input_image):
    output_tensor = preprocess_image(input_image)
    with torch.no_grad():
        for _ in range(model.denoising_steps):
            output_tensor = model(output_tensor)
    output_tensor = output_tensor.squeeze(0)
    output_tensor = output_tensor.squeeze(0)
    output_image = output_tensor.numpy()
    return output_image


iface = gr.Interface(
    fn=inference_function,
    inputs=gr.Image(),
    outputs=gr.Image(),
)

iface.launch(share=False)
