import torch

def add_variable_gaussian_noise(tensor, noise_proportion):
    std = torch.std(tensor)
    noise = torch.randn_like(tensor) * std
    noisy_tensor = tensor * (1 - noise_proportion) + noise * noise_proportion
    return noisy_tensor
