import pytest

import torch

from code.model import Net


@pytest.mark.parametrize("image_size, channels", [((64, 64), 3), ((64, 64), 2), ((64, 64), 1), ((64, 52), 1), ((52, 52), 1), ((52, 52), 1)])
def test_output_size(image_size, channels):
    # Create a random input image tensor
    batch_size = 32
    random_image = torch.randn((batch_size, channels, image_size[0], image_size[1]))

    # Instantiate the Net model
    model = Net(image_size, channels, 0.2)

    # Forward pass
    output = model(random_image)

    # Check if the output size matches the input size
    assert output.size() == random_image.size(), "Output size does not match input size."

