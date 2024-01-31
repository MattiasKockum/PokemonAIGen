import os
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, image_size, channels, denoising_steps, num_heads=32, num_layers=24, embedding_dim=512):
        super(Net, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.denoising_steps = denoising_steps
        self.embedding = nn.Linear(image_size[0] * image_size[1] * channels, embedding_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, image_size[0] * image_size[1] * channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)
        x = self.fc(x)
        x = x.view(x.size(0), -1, self.image_size[0], self.image_size[1])
        return x


def save_model(model, info, path):
    model_card = {
        "image_size": model.image_size,
        "channels":model.channels,
        "denoising_steps": model.denoising_steps,
        "model_dict": model.state_dict()
    }
    model_card = {**model_card, **info}
    torch.save(model_card, path)
    return model_card


def load_model(path):

    with open(path, "rb") as f:
        info_dict = torch.load(f)

    image_size = info_dict["image_size"]
    channels = info_dict["channels"]
    denoising_steps = info_dict["denoising_steps"]
    model_dict = info_dict["model_dict"]

    model = Net(image_size, channels, denoising_steps)
    model.load_state_dict(model_dict)

    return model, model_dict
