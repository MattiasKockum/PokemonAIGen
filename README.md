# WIP Project

# Pokemon Visual Transformer Diffusion

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
prefix = "DEMO-Pokemon-Generator"
role = "..." # Get it from AWS
pt_mnist_model_data = "..." # You get it by running launch_training.py
wandb_api_key = "..." # Get it from Weights And Biases
```

``` bash
python prepare_data.py
python launch_training.py
```

Fill the .env file with your pt_mnist_model_data

``` bash
python deploy.py
```

Look into outputs directory
