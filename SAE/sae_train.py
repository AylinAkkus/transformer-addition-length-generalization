from model import load_model_from_file
from tokenizer import Tokenizer
from config import Config
import torch
from .dataset import AdditionDataset
from torch.utils.data import DataLoader
from .sae_model import SAE, SAEConfig

if __name__ == "__main__":
    # Importing the model
    path = "saved_runs/fixed_digit_add_30/final.pth"
    model = load_model_from_file(path)
    model.eval()

    # Initializing the SAE model
    sae_config = SAEConfig(d_in=model.config.d_mlp, d_sae=128)
    sae = SAE(sae_config, model)
    sae.optimize(resample_method="advanced")





