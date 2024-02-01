import torch
import numpy as np

def add_variable_gaussian_noise(tensor, noise_proportion):
    std = torch.std(tensor)
    noise = torch.randn_like(tensor) * std
    noisy_tensor = tensor * (1 - noise_proportion) + noise * noise_proportion
    return noisy_tensor


def augment_data(batch, args):
    l_max = args.distortion_factor
    denoising_steps = args.denoising_steps
    h, w = batch.shape[-2], batch.shape[-1]
    ih = np.random.randint(h)
    iw = np.random.randint(w)
    l = np.random.randint(1, l_max)
    flip = np.random.choice([True, False])
    orientation = np.random.choice([1, 2, 3, 4])

    # Flip
    if flip:
        batch = batch.flip(dims=[3])

    # Distortion
    if orientation == 1:
        col = batch[:, :, :, iw:iw+1]
        batch = torch.cat([batch[:, :, :, :iw]] + l * [col] +[batch[:, :, :, iw:]], dim=3)
        batch = batch[:, :, :, :-l]

    elif orientation == 2:
        col = batch[:, :, :, iw:iw+1]
        batch = torch.cat([batch[:, :, :, :iw+1]] + l * [col] + [batch[:, :, :, iw+1:]], dim=3)
        batch = batch[:, :, :, l:]

    elif orientation == 3:
        row = batch[:, :, ih:ih+1, :]
        batch = torch.cat([batch[:, :, :ih, :]] + l * [row] +[batch[:, :, ih:, :]], dim=2)
        batch = batch[:, :, :-l, :]

    elif orientation == 4:
        row = batch[:, :, ih:ih+1, :]
        batch = torch.cat([batch[:, :, :ih+1, :]] + l * [row] + [batch[:, :, ih+1:, :]], dim=2)
        batch = batch[:, :, l:, :]

    # Sliding

    # Adding noise
    noise_step = np.random.randint(denoising_steps - 1)
    batch = add_variable_gaussian_noise(batch, noise_step / denoising_steps)

    return batch
