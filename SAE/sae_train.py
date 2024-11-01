from model import load_model_from_file
from tokenizer import Tokenizer
from config import Config
import torch
from .dataset import AdditionDataset
from torch.utils.data import DataLoader

# Importing the model
path = "saved_runs/fixed_digit_add_30/final.pth"
model = load_model_from_file(path)
model.eval()
tokenizer = Tokenizer(Config())

def cache_mlp(model, cache):
    # Caches mlp activations for = 
    def save_hook(tensor, name):
        cache.append(tensor.detach().squeeze())
    model.blocks[0].hook_mlp_out.add_hook(save_hook, 'fwd')

if __name__ == "__main__":

    # Caching a batch of activations
    data_path = "saved_runs/fixed_digit_add_30/data.csv" # TODO make sure model and data are from same dir
    dataset = AdditionDataset(data_path)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    cache = [] # A list which will store the MLP activations for all tokens
    cache_mlp(model, cache)

    for i, batch in enumerate(dataloader):
        # Convert batch to a list of lists for forward pass
        stacked = torch.stack(batch, dim=1)
        model(stacked)

    # Reshaping the activations to: (num_tokens x samples)  hidden_size
    cache = torch.cat(cache, dim=0)
    cache = cache.reshape(-1, 128)
    print(cache.shape)

    # Now we can train the SAE model on the cached activations




