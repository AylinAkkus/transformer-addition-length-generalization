# AUTOGENERATED! DO NOT EDIT! File to edit: ../transformer.ipynb.

# %% auto 0
__all__ = ['Config', 'HookPoint', 'Embed', 'Unembed', 'PosEmbed', 'LayerNorm', 'Attention', 'MLP', 'TransformerBlock',
           'Transformer', 'make_fourier_basis', 'calculate_key_freqs', 'get_components_of_trig_loss',
           'calculate_excluded_loss', 'calculate_trig_loss', 'calculate_coefficients', 'Tokenizer', 'gen_train_test', 'loss',
           'Trainer', 'train_model']

# %% ../transformer.ipynb 3
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import einops
import random 
import helpers
from helpers import *
from dataclasses import dataclass
import os
import wandb
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import re
import pandas as pd
import json

# %% ../transformer.ipynb 4
# TODO does dataclass really require type annotations lol

class TokenizedDataset(Dataset):
    """
    Converts the entire data (pandas data frame) into a PyTorch Dataset
    """
    def __init__(self, data, train = True):
        # Select train/test
        filtered_data = data[data["is_train"] == train]
        list_of_lists = filtered_data["tokenized"].values.tolist()
        self.tokenized_data = t.tensor(list_of_lists, dtype=t.long)

    def __len__(self):
        return self.tokenized_data.shape[0]

    def __getitem__(self, idx):
        return self.tokenized_data[idx]
    

@dataclass(frozen = True)
class Config():
    lr: float = 3e-3 #@param
    weight_decay: float = 1.0 #@param
    batch_size: int = 64 #@param
    p: int = 113 #@param
    d_model: int = 128 #@param
    fn_name: str = 'add' #@param ['add', 'subtract', 'x2xyy2','rand']
    frac_train: float = 0.1 #@param
    num_epochs: int = 2000 #@param
    save_models: bool = True #@param
    save_every: int = 50 #@param

    # TODO for the first 1000 steps, save every 10 because 'interesting stuff happens at the start'
    # TODO add a helper function to generate indices here

    # Stop training when test loss is <stopping_thresh
    stopping_thresh: int = -1 #@param
    seed: int = 0 #@param

    token_to_tokenid = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        '+': 10,
        '=': 11,
        'EOS': 12,
        'PAD': 13
    }

    tokenid_to_token = {v: k for k, v in token_to_tokenid.items()}

    num_layers: int = 1
    batch_style: str = 'full'
    d_vocab: int = len(token_to_tokenid)
    num_digits: int = 3
    n_ctx: int = 12
    d_mlp: int = 4*d_model
    num_heads: int = 4

    act_type: str = 'ReLU' #@param ['ReLU', 'GeLU']


    device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    # TODO ankify the privileged basis concept- a priori vs etc. ; consider writing up an explanation of privileged basis

    use_ln: bool = False

    take_metrics_every_n_epochs: int = 10 #@param

    @property
    def d_head(self):
        return self.d_model // self.num_heads

    @property
    def random_answers(self):
        return np.random.randint(low=0, high=self.p, size=(self.p, self.p))

    @property 
    def fns_dict(self):
        return {
            'add': lambda x,y:(x+y) % self.p,
            'subtract': lambda x,y:(x-y) % self.p,
            'x2xyy2': lambda x,y:(x**2+x*y+y**2) % self.p,
            'rand': lambda x,y:self.random_answers[x][y]
            }

    @property
    def fn(self):
        return self.fns_dict[self.fn_name]
    
    def serialize(self):
        """Serialize the config to JSON so that I can be stored in the saved_runs folder"""
        config_as_dict = dataclasses.asdict(self)
        # Ignore device as it can not be serialized
        del config_as_dict['device']
        return json.dumps(config_as_dict)

    def is_train_is_test(self, train):
        '''Creates an array of Boolean indices according to whether each data point is in train or test.
        Used to index into the big batch of all possible data'''
        pass

    def is_it_time_to_save(self, epoch):
        """Save every 10 epochs for the first 1000 epochs, then less frequently"""
        if epoch < 1000:
            return (epoch % 5 == 0)
        else:
            return (epoch % self.save_every == 0)

    def is_it_time_to_take_metrics(self, epoch):
        return epoch % self.take_metrics_every_n_epochs == 0

# TODO make this an assert inside the consturctor
assert Config.d_model % Config.num_heads == 0

# %% ../transformer.ipynb 5
class HookPoint(nn.Module):
    '''A helper class to get access to intermediate activations (inspired by Garcon)
    It's a dummy module that is the identity function by default
    I can wrap any intermediate activation in a HookPoint and get a convenient way to add PyTorch hooks
    '''
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
    
    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name
    
    def add_hook(self, hook, dir='fwd'):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output, 
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")
    
    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")
    
    def forward(self, x):
        return x

# %% ../transformer.ipynb 6
class Embed(nn.Module):
    '''Define network architecture
    I defined my own transformer from scratch so I'd fully understand each component 
    - I expect this wasn't necessary or particularly important, and a bunch of this replicates existing Pyt functionality
    '''
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(t.randn(d_model, d_vocab)/np.sqrt(d_model))
    
    def forward(self, x):
        return t.einsum('dbp -> bpd', self.W_E[:, x])
    

#| export
class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(t.randn(d_model, d_vocab)/np.sqrt(d_vocab))
    
    def forward(self, x):
        return (x @ self.W_U)

#| export
class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(t.randn(max_ctx, d_model)/np.sqrt(d_model))
    
    def forward(self, x):
        return x+self.W_pos[:x.shape[-2]]

#| export
class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon = 1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(t.ones(d_model))
        self.b_ln = nn.Parameter(t.zeros(d_model))
        self.epsilon = epsilon
    
    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x

#| export
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(t.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.register_buffer('mask', t.tril(t.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        k = self.hook_k(t.einsum('ihd,bpd->biph', self.W_K, x))
        q = self.hook_q(t.einsum('ihd,bpd->biph', self.W_Q, x))
        v = self.hook_v(t.einsum('ihd,bpd->biph', self.W_V, x))
        attn_scores_pre = t.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = t.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = self.hook_attn(F.softmax(self.hook_attn_pre(attn_scores_masked/np.sqrt(self.d_head)), dim=-1))
        z = self.hook_z(t.einsum('biph,biqp->biqh', v, attn_matrix))
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = t.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out

#| export
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(t.randn(d_mlp, d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.W_out = nn.Parameter(t.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))
        self.act_type = act_type
        # self.ln = LayerNorm(d_mlp, model=self.model)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ['ReLU', 'GeLU']
        
    def forward(self, x):
        x = self.hook_pre(t.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
        if self.act_type=='ReLU':
            x = F.relu(x)
        elif self.act_type=='GeLU':
            x = F.gelu(x)
        x = self.hook_post(x)
        x = t.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x

# export
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        # self.ln1 = LayerNorm(d_model, model=self.model)
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        # self.ln2 = LayerNorm(d_model, model=self.model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()
    
    def forward(self, x):
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn((self.hook_resid_pre(x)))))
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp((x))))
        return x

#| export
class Transformer(nn.Module):
    def __init__(self, config: Config, use_cache=False, use_ln=True):
        '''this function could be augmented to contain more options for creating different architectures'''
        super().__init__()
        self.cache = {}
        self.config = config
        self.use_cache = use_cache
        self.embed = Embed(d_vocab = config.d_vocab, d_model = config.d_model)
        self.pos_embed = PosEmbed(max_ctx = config.n_ctx, d_model = config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model = config.d_model,
            d_mlp = config.d_mlp,
            d_head = config.d_head,
            num_heads = config.num_heads,
            n_ctx = config.n_ctx,
            act_type = config.act_type,
            model=[self]) for i in range(config.num_layers)])
        self.unembed = Unembed(d_vocab = config.d_vocab, d_model = config.d_model)
        self.use_ln = use_ln

        for name, module in self.named_modules():
            if type(module)==HookPoint:
                module.give_name(name)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        # x = self.ln(x)
        x = self.unembed(x)
        return x
    
    @t.no_grad()
    def generate_greedy(self, x):
        # Greedy generation for a sequence (non-batched)

        # make a copy of x
        x = x.copy()

        while len(x) <= self.config.n_ctx:
            logits = self([x])[0, -1]
            next_token = t.argmax(logits).item()
            x.append(next_token)

            if next_token == self.config.token_to_tokenid['EOS']:
                return x

        return x

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache
    
    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')
    
    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')
                

# %% ../transformer.ipynb 7
def make_fourier_basis(config: Config):
    fourier_basis = []
    fourier_basis.append(t.ones(config.p)/np.sqrt(config.p))
    fourier_basis_names = ['Const']
    # Note that if p is even, we need to explicitly add a term for cos(kpi), ie 
    # alternating +1 and -1
    for i in range(1, config.p//2 +1):
        fourier_basis.append(t.cos(2*t.pi*t.arange(config.p)*i/config.p))
        fourier_basis.append(t.sin(2*t.pi*t.arange(config.p)*i/config.p))
        fourier_basis[-2]/=fourier_basis[-2].norm()
        fourier_basis[-1]/=fourier_basis[-1].norm()
        fourier_basis_names.append(f'cos {i}')
        fourier_basis_names.append(f'sin {i}')
    return t.stack(fourier_basis, dim=0).to(config.device)


def calculate_key_freqs(config: Config, model: Transformer, all_data):
    # TODO this was moved from the app code; probably move it around
    labels = t.tensor([config.fn(i, j) for i, j, _ in all_data]).to(config.device)
    cache = {}
    model.remove_all_hooks() # TODO is this line fucky??
    model.cache_all(cache)
    model(all_data)
    neuron_acts = cache['blocks.0.mlp.hook_post'][:, -1]
    # Center the neurons to remove the constant term
    neuron_acts_centered = neuron_acts - einops.reduce(neuron_acts, 'batch neuron -> 1 neuron', 'mean')
    # Note that fourier_neuron_acts[(0, 0), i]==0 for all i, because we centered the activations
    fourier_basis = make_fourier_basis(config = config)
    fourier_neuron_acts = helpers.fft2d(neuron_acts_centered, p = config.p, fourier_basis=fourier_basis)

    fourier_neuron_acts_square = fourier_neuron_acts.reshape(config.p, config.p, config.d_mlp)
    neuron_freqs = []
    neuron_frac_explained = []
    for ni in range(config.d_mlp):
        best_frac_explained = -1e6
        best_freq = -1
        for freq in range(1, config.p//2):
            # We extract the linear and quadratic fourier terms of frequency freq, 
            # and look at how much of the variance of the full vector this explains
            # If neurons specialise into specific frequencies, one frequency should 
            # have a large value
            numerator = helpers.extract_freq_2d(fourier_neuron_acts_square[:, :, ni], freq, p = config.p).pow(2).sum()
            denominator = fourier_neuron_acts_square[:, :, ni].pow(2).sum().item()
            frac_explained = numerator / denominator
            if frac_explained > best_frac_explained:
                best_freq = freq
                best_frac_explained = frac_explained
        neuron_freqs.append(best_freq)
        neuron_frac_explained.append(best_frac_explained)
    neuron_freqs = np.array(neuron_freqs)
    neuron_frac_explained = helpers.to_numpy(neuron_frac_explained)
    key_freqs, neuron_freq_counts = np.unique(neuron_freqs, return_counts=True)
    return key_freqs

def get_components_of_trig_loss(logits, freq, fourier_basis):
    cos = helpers.get_component_cos_xpy(logits, freq, fourier_basis=fourier_basis)
    sin = helpers.get_component_sin_xpy(logits, freq, fourier_basis=fourier_basis)
    return cos + sin


def calculate_excluded_loss(config: Config, fourier_basis, key_freqs, is_train, is_test, labels, logits):
    row = []
    for freq in key_freqs:
        cos = helpers.get_component_cos_xpy(logits, freq, fourier_basis=fourier_basis) 
        sin = helpers.get_component_sin_xpy(logits, freq, fourier_basis=fourier_basis) 
        value = helpers.test_logits(logits - cos - sin, bias_correction=False, mode='train', p = config.p,
           is_train = is_train, is_test = is_test, labels = labels)
        row.append(value.item())
    return row

def calculate_trig_loss(config: Config, model, train, logits, key_freqs, fourier_basis, all_data, is_train, is_test, labels, mode='all'):
    trig_logits = sum([get_components_of_trig_loss(logits, freq, fourier_basis) for freq in key_freqs])
    return helpers.test_logits(trig_logits, 
                        p = config.p,
                        is_train = is_train, 
                        is_test = is_test,
                        labels = labels,
                        bias_correction=True, 
                        original_logits=logits, 
                        mode=mode)


def calculate_coefficients(logits, fourier_basis, key_freqs, p, device):
    '''updated version from https://colab.research.google.com/drive/1ScVRL8OCtTFpOHpgfz0PLTFvX4g_YbuN?usp=sharing#scrollTo=WY4nPUDwl9UN
    '''
    x = t.arange(p)[None, :, None, None]
    y = t.arange(p)[None, None, :, None]
    z = t.arange(p)[None, None, None, :]
    w = t.arange(1, (p//2+1))[:, None, None, None]
    coses = t.cos(w*t.pi*2/p * (x + y - z)).to(device)
    coses = coses.reshape(p//2, p*p, p)
    coses/= coses.pow(2).sum([-2, -1], keepdim=True).sqrt()
    cos_coefficients = (coses * logits).sum([-2, -1])
    return cos_coefficients

# %% ../transformer.ipynb 8
# TODO move this into the config?
import dataclasses
from collections import defaultdict

def gen_train_test(config: Config, fixed_digit = False):
    tokenizer = Tokenizer(config)
    '''Generate a dataframe with:
    - operand_1, operand_2, result
    - input string of form: "operator_1+operator_2=resultEOSPADPAD"
    - start index of the result
    - a bool indicating whether the sequence is in the training set
    '''
    num_to_generate = config.p

    # Upsample the pairs with both operands being single digits
    upsample_frac = 0.8
    pairs_1 = [(i,j,(i+j)%num_to_generate) for i in range(10) for j in range(10)]
    df_1 = pd.DataFrame(pairs_1, columns=['operand_1', 'operand_2', 'result'])
    random.seed(config.seed)
    num_train = int(upsample_frac * len(pairs_1))
    num_test = len(pairs_1) - num_train
    train_idx = np.array([True]*num_train + [False]*num_test)
    random.shuffle(train_idx)
    df_1['is_train'] = train_idx

    pairs_2 = [(i,j,(i+j)%num_to_generate) for i in range(10) for j in range(10, 100)]
    pairs_2 += [(i,j,(i+j)%num_to_generate) for i in range(10, 100) for j in range(10)]
    pairs_2 += [(i,j,(i+j)%num_to_generate) for i in range(10, 100) for j in range(10, 100)]
    df_2 = pd.DataFrame(pairs_2, columns=['operand_1', 'operand_2', 'result'])
    random.seed(config.seed)
    num_train = int(config.frac_train * len(pairs_2))
    num_test = len(pairs_2) - num_train
    train_idx = np.array([True]*num_train + [False]*num_test)
    random.shuffle(train_idx)
    df_2['is_train'] = train_idx

    df = pd.concat([df_1, df_2])

    # Parse the data into input strings and store in the dataframe
    if fixed_digit:
        df["input_str"] = df.apply(lambda row: to_fixed_digit_format(row), axis=1)
    else:
        df["input_str"] = df.apply(lambda row: to_variable_digit_format(row), axis=1)

    # Add the tokenized version which is fed into the LLM
    df["tokenized"] = df["input_str"].apply(lambda x: tokenizer.tokenize(x))

    # Calculate the start and end index of the result in the input string
    df["start_idx"] = df.apply(lambda row: start_index(row), axis=1)
    df["end_idx"] = df.apply(lambda row: end_index(row), axis=1)

    # Write to file
    # with open('data.csv', 'w') as f:
    #    df.to_csv(f)

    return df

def to_fixed_digit_format(row):
    """
    A function that parses the input string into the fixed digit format:
    "0 0 1 + 0 8 9 = 0 9 0 EOS PAD" 
    """
    tokenizer = Tokenizer(config)
    num_digits = config.num_digits
    # Pad the operands and result with zeros
    operand_1 = str(row['operand_1']).zfill(num_digits)
    operand_2 = row['operand_2'] = str(row['operand_2']).zfill(num_digits)
    result = row['result'] = str(row['result']).zfill(num_digits)

    # Create the tokenized string
    input_str = f"{operand_1}+{operand_2}={result}EOS"

    # Pad the tokenized string
    if len(tokenizer.tokenize(input_str)) < (config.n_ctx +1):
        input_str += 'PAD'*((config.n_ctx+1) - len(tokenizer.tokenize(input_str)))

    return input_str

def to_variable_digit_format(row):
    """
    A function that parses the input string into the variable digit format:
    "1 + 1 2 = 13 EOS PAD PAD PAD PAD PAD"
    """
    tokenizer = Tokenizer(config)
    operand_1 = str(row['operand_1'])
    operand_2 = str(row['operand_2'])
    result = row['result'] = str(row['result'])

    # Create the tokenized string
    input_str = f"{operand_1}+{operand_2}={result}EOS"

    # Pad the tokenized string
    if len(tokenizer.tokenize(input_str)) < (config.n_ctx +1):
        input_str += 'PAD'*((config.n_ctx+1) - len(tokenizer.tokenize(input_str)))
    return input_str

def start_index(row):
    """
    A function that calculates the start index of the result in the input string.
    """
    tokenizer = Tokenizer(config)
    idx = row["tokenized"].index(tokenizer.tokenize("=")[0])
    return idx

def end_index(row):
    """
    A function that calculates the end index of the result in the input string.
    """
    tokenizer = Tokenizer(config)
    idx = row["tokenized"].index(tokenizer.tokenize("EOS")[0])
    return idx


def filter_data(data, train, operand_1_len=None, operand_2_len=None, res_len=None):
    """
    Selects data from the data frame based on the length of the operands and the result.
    Distingushes between training and testing data.
    """
    if operand_1_len:
        data = data[data['operand_1'].apply(lambda x: len(str(x))==operand_1_len)]

    if operand_2_len:
        data = data[data['operand_2'].apply(lambda x: len(str(x))==operand_2_len)]
    
    if res_len:
        data = data[data['result'].apply(lambda x: len(str(x))==res_len)]
    
    assert train in [True, False]

    if train:
        return data[data['is train']]
    else:
        return data[~data["is train"]]

# TODO what type for model?
def loss(config : Config, model: Transformer, data):
    # We need this to extract the position of the result in the input string
    tokenizer = Tokenizer(config)
    start_tokenid = tokenizer.tokenize("=")[0]
    end_tokenid = tokenizer.tokenize("EOS")[0]
    '''Takes the cross entropy loss of the model on the data'''
    fwd_data = data[:,:-1]
    logits = model(fwd_data)
    cross_entropy_per_seq_losses = []

    for i, row in enumerate(data):
        start_idx = t.nonzero(row == start_tokenid).item()
        end_idx = t.nonzero(row == end_tokenid).item()
        relevant_logits = logits[i][start_idx:end_idx+1]
        targets = row[start_idx+1:end_idx+2].to(config.device)
        if not all(0 <= label < config.d_vocab for label in targets):
            raise ValueError(f"Invalid label found in sequence {i}: {targets}")

        cross_entropy_per_seq = F.cross_entropy(relevant_logits, targets)
        cross_entropy_per_seq_losses.append(cross_entropy_per_seq)

    loss = t.stack(cross_entropy_per_seq_losses).mean()

    return loss

class Trainer:
    '''TODO
    ways this stinks:
    - callbacks every k epochs 
    - training on infinite data
    - general abstract class w/o assumption and subclasses w/ more assumptions
    - check out hugging face trainer
    - disentangle optimization step and taking gradients
    - forward compatibility, e.g. batches per step
    '''

    def __init__(self, config : Config, model = None) -> None:
        wandb.init(project = "grokking", config = dataclasses.asdict(config))
        self.model = model if model is not None else Transformer(config, use_cache=False)
        self.model.to(config.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr = config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))

        def lr_lambda(step, num_epochs=config.num_epochs):
            n_warmup = 20
            if step <= n_warmup:
                return min(step / n_warmup, 1)  # Linear warm-up
            else:
                # Linear decay from the end of the warm-up to 1/10 of the original LR
                decay_factor = 0.03  # Final LR will be 1/10 of the original LR
                total_decay_steps = num_epochs - n_warmup
                step_after_warmup = step - n_warmup
                decay = 1 - (1 - decay_factor) * (step_after_warmup / total_decay_steps)
                return max(decay, decay_factor)  # Ensures LR never goes below 1/10

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda) # TODO make this a config option
        self.run_name = f"mod_digit_add_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        self.data = gen_train_test(config = config)
        self.train = TokenizedDataset(self.data, train=True)
        print("Success")
        self.test = TokenizedDataset(self.data, train=False)
        print("Success")
        self.metrics_dictionary = defaultdict(dict) # so we can safely call 'update' on keys
        print('training length = ', len(self.train))
        print('testing length = ', len(self.test))
        self.train_losses = []
        self.test_losses = []
        self.config = config

    def save_epoch(self, epoch, save_to_wandb = True):
        ''' precondition! train loss and test losses have been appended to '''
        save_dict = {
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'epoch': epoch,
        }
        if save_to_wandb:
            wandb.log(save_dict)
            print("Saved epoch to wandb")
        if self.config.save_models: 
            t.save(save_dict, root/self.run_name/f"{epoch}.pth")
            print(f"Saved model to {root/self.run_name/f'{epoch}.pth'}")
        self.metrics_dictionary[epoch].update(save_dict)

    def do_a_training_step(self, epoch: int):
        '''returns train_loss, test_loss'''
        dataloader_train = DataLoader(self.train, batch_size = self.config.batch_size, shuffle = True)
        self.model.train()

        # Train the model for one epoch using mini-batches
        epoch_loss = []
        for batch in dataloader_train:
            self.optimizer.zero_grad()
            train_loss = loss(config = self.config, model = self.model, data = batch)
            train_loss.backward()
            self.optimizer.step()
            epoch_loss.append(train_loss.item())
        self.scheduler.step()

        # Test the model on entire test set
        # We need to use this weird construction because want the test data to be in the same format as the training data
        dataloader_test = DataLoader(self.test, batch_size = len(self.test), shuffle = False)
        self.model.eval()
        for batch in dataloader_test:
            test_loss = loss(config = self.config, model = self.model, data = self.test)

        # Log the train and test losses
        train_loss = t.tensor(epoch_loss).mean() # we mean the train loss over the epoch
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss.item())

        # Print the train and test losses
        if epoch % 1 == 0:
            # TODO is this ok? this was np.log, and it was barking at me ; i think np.log was being interpreted as a logging module
            print(f'Epoch {epoch}, train loss {t.log(train_loss).item():.4f}, test loss {t.log(test_loss).item():.4f}')

        return train_loss, test_loss

    def initial_save_if_appropriate(self):
        """
        Save the model, config and entire data at the start of training
        """
        if self.config.save_models:
            os.mkdir(root/self.run_name)

            # Save model
            save_dict = {'model': self.model.state_dict()}
            t.save(save_dict, root/self.run_name/'init.pth')
            
            # Save the config
            config_json = self.config.serialize()
            with open(root/self.run_name/'config.json', 'w') as f:
                f.write(config_json)

            # Save entire data as csv
            self.data.to_csv(root/self.run_name/'data.csv')



    def post_training_save(self, save_optimizer_and_scheduler = True, log_to_wandb = True):
        if not self.config.save_models:
            os.makedirs(root/self.run_name, exist_ok=True)
        save_dict = {
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'epoch': self.config.num_epochs,
        }
        if save_optimizer_and_scheduler:
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['scheduler'] = self.scheduler.state_dict()
        if log_to_wandb:
            wandb.log(save_dict)
        t.save(save_dict, root/self.run_name/f"final.pth")
        print(f"Saved model to {root/self.run_name/f'final.pth'}")
        self.metrics_dictionary[save_dict['epoch']].update(save_dict)


    def take_metrics(self, train, epoch):
        with t.inference_mode():
            def sum_sq_weights():
                # TODO refactor- taken from app code
                row = []
                for name, param in self.model.named_parameters():
                    row.append(param.pow(2).sum().item())
                return row

            print('taking metrics')

            all_data = t.tensor([(i, j, self.config.p) for i in range(self.config.p) for j in range(self.config.p)]).to(self.config.device)
            # TODO calculate key freqs is the most expensive part of this
            key_freqs = calculate_key_freqs(config = self.config, model = self.model, all_data = all_data) 
            logits = self.model(all_data)[:, -1, :-1] # TODO i think this is equivalent to what's in the new paper?
            fourier_basis = make_fourier_basis(config = self.config)
            is_train, is_test = self.config.is_train_is_test(train = train)
            labels = t.tensor([self.config.fn(i, j) for i, j, _ in all_data]).to(self.config.device)

            metrics = {
                'epoch': epoch, 
                'trig_loss': calculate_trig_loss(config = self.config,
                    model = self.model,
                    train = train,
                    key_freqs = key_freqs,
                    is_test=is_test,
                    is_train=is_train,
                    labels=labels,
                    logits = logits,
                    fourier_basis=fourier_basis, 
                    all_data=all_data),
                'sum_of_squared_weights': sum_sq_weights(),
                'excluded_loss': calculate_excluded_loss(
                    logits = logits,
                    key_freqs = key_freqs,
                    fourier_basis = fourier_basis,
                    is_train=is_train,
                    config = self.config,
                    is_test = is_test,
                    labels=labels),
                'coefficients': calculate_coefficients(p = self.config.p, logits = logits, fourier_basis = fourier_basis, key_freqs = key_freqs, device = self.config.device),
            }
            wandb.log(metrics)
            print("Logged metrics to wandb")
            self.metrics_dictionary[epoch].update(metrics)

def train_model(config: Config):
    world = Trainer(config = config)
    print(f'Run name {world.run_name}')
    world.initial_save_if_appropriate()

    for epoch in range(config.num_epochs):
        t0 = time.time()
        train_loss, test_loss = world.do_a_training_step(epoch)
        print(f"Epoch {epoch} took {time.time() - t0:.2f} seconds")
        if test_loss.item() < config.stopping_thresh:
            break
        if config.is_it_time_to_save(epoch = epoch):
            # TODO this also used to do a check about test loss- pretty sure not necessary
            world.save_epoch(epoch = epoch)
        if config.is_it_time_to_take_metrics(epoch = epoch):
            pass
            #world.take_metrics(epoch = epoch, train = world.train)

    world.post_training_save(save_optimizer_and_scheduler=True)
    helpers.lines([world.train_losses, world.test_losses], labels=['train', 'test'], log_y=True)
    return world # to export the dictionary with the training metrics

class Tokenizer():
    def __init__(self, config: Config):
        self.config = config
        
    def tokenize(self, sequence):
        sorted_vocab = sorted(self.config.token_to_tokenid.keys(), key=lambda x: len(x), reverse=True)
        pattern = '|'.join(re.escape(token) for token in sorted_vocab)
        # TODO: Check whether there are tokens not in the vocab
        return [self.config.token_to_tokenid[token] for token in re.findall(pattern, sequence)]
    
    def detokenize(self, tokenized):
        return ''.join([self.config.tokenid_to_token[token] for token in tokenized])
        

if __name__ == '__main__':
    config = Config()
    train_model(config)
    
