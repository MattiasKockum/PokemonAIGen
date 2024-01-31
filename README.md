# WIP Project

# Pokemon Visual Transformer Diffusion

This is a transformer based model trained for denoising task on pokémon sprites dataset from first generation.
The goal is to produce new and original sprites while keeping a coherent style.
This is still a work in progress project.

# Best results for now

![Image 1](/images/results/1.png?raw=true "Image 1")
![Image 2](/images/results/2.png?raw=true "Image 2")


## How tu use it

``` bash
git clone https://github.com/MattiasKockum/PokemonAIGen.git
cd PokemonAIGen
python -m venv venv
source venv/bin/activate
#sudo mount -o remount,size=16G /tmp # This might be needed
pip install -r requirements.txt
```

Fill a .env file with your own data
``` python
role = "..." # Get it from AWS
pt_mnist_model_data = "..." # You get it by running launch_training.py
wandb_api_key = "..." # Get it from Weights And Biases
```

``` bash
python prepare_data.py
python launch_training.py
```

``` bash
python deploy.py
```

Look into outputs directory

# TODO

Early stopping

Regularization

Add color (multiple channels)

# Data augmentation

Here are exemples of data augmentation done to ensure better robustness of the model.

![Image 3](/images/data_augmentation/data_augmentation_example_1.png?raw=true "Image 3")
![Image 4](/images/data_augmentation/data_augmentation_example_2.png?raw=true "Image 4")
![Image 5](/images/data_augmentation/data_augmentation_example_3.png?raw=true "Image 5")
![Image 6](/images/data_augmentation/data_augmentation_example_4.png?raw=true "Image 6")
