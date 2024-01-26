import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, image_size, channels, num_heads=8, num_layers=6, embedding_dim=512):
        super(Net, self).__init__()
        self.image_size = image_size
        self.embedding = nn.Linear(image_size[0] * image_size[1] * channels, embedding_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, image_size[0] * image_size[1] * channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1, self.image_size[0], self.image_size[1])
        return x

