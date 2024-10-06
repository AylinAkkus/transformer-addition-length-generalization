__all__ = ['use_drive', 'root', 'imshow', 'imshow_div', 'run_shell_command_as_python', 'is_inside_colab', 'install_dependencies',
           'download_model_from_drive', 'cuda_memory', 'cross_entropy_high_precision', 'test_logits', 'fft1d',
           'fourier_2d_basis_term', 'fft2d', 'analyse_fourier_2d', 'get_2d_fourier_component', 'get_component_cos_xpy',
           'get_component_sin_xpy', 'to_numpy', 'unflatten_first', 'inputs_heatmap', 'scatter', 'line', 'lines',
           'line_marker', 'animate_lines', 'imshow_fourier', 'animate_multi_lines', 'animate_scatter', 'cos', 'mod_div',
           'normalize', 'extract_freq_2d', 'get_cov', 'is_close', 'cpu_aware_load_at_root',
           'load_mod_addition_frac_train_sweep', 'load_5_digit_addition_infinite', 'load_5_digit_addition_finite',
           'load_induction_head_finite', 'load_induction_head_infinite', 'load_infinite_data_losses',
           'load_finite_data_losses', 'load_no_wd_width_scan', 'take_metrics', 'extract_answer_from_prediction']

# %% ../ipynb 2
def run_shell_command_as_python(shell):
    '''helpful for python functions; thanks https://stackoverflow.com/questions/70068720/jupyter-shell-commands-in-a-function'''
    from IPython import get_ipython
    ipython = get_ipython()
    code = ipython.transform_cell(f'!{shell}')
    print(f'Executing {code}')
    exec(code)

# %% ../helpers.ipynb 5
def install_dependencies():
    # TODO how to make this run at the right times?
    run_shell_command_as_python("nvidia-smi") # TODO what if this isn't available? maybe this is just for the main notebook?
    run_shell_command_as_python("pip install einops matplotlib pandas plotly")

use_drive = False #@param

# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
import tqdm.notebook as tqdm

# TODO from google.colab import drive
from pathlib import Path

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.io as pio
'''if is_inside_colab():
    pio.renderers.default = "colab"
else:
    pio.renderers.default = "vscode"'''
import plotly.graph_objects as go

from torch.utils.data import DataLoader

from functools import *
import pandas as pd

# %% ../helpers.ipynb 7
root = Path('./saved_runs')

# %% ../helpers.ipynb 9
def cuda_memory():
    print(torch.cuda.memory_allocated()/1e9)

#| export
def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly 
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes 
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss


#| export
def test_logits(logits, p, is_train, is_test, labels, bias_correction=False, original_logits=None, mode='all'):
    # Calculates cross entropy loss of logits representing a batch of all p^2 
    # possible inputs
    # Batch dimension is assumed to be first
    if logits.shape[1]==p*p:
        logits = logits.T
    if logits.shape==torch.Size([p*p, p+1]):
        logits = logits[:, :-1]
    logits = logits.reshape(p*p, p)
    if bias_correction:
        # Applies bias correction - we correct for any missing bias terms, 
        # independent of the input, by centering the new logits along the batch 
        # dimension, and then adding the average original logits across all inputs
        logits = einops.reduce(original_logits - logits, 'batch ... -> ...', 'mean') + logits
    if mode=='train':
        return cross_entropy_high_precision(logits[is_train], labels[is_train])
    elif mode=='test':
        return cross_entropy_high_precision(logits[is_test], labels[is_test])
    elif mode=='all':
        return cross_entropy_high_precision(logits, labels)

# %% ../helpers.ipynb 10
# Fourier transform stuff


def fft1d(tensor, fourier_basis):
    # Converts a tensor with dimension p into the Fourier basis
    return tensor @ fourier_basis.T

def fourier_2d_basis_term(x_index, y_index, fourier_basis):
    # Returns the 2D Fourier basis term corresponding to the outer product of 
    # the x_index th component in the x direction and y_index th component in the 
    # y direction
    # Returns a 1D vector of length p^2
    return (fourier_basis[x_index][:, None] * fourier_basis[y_index][None, :]).flatten()

def fft2d(mat, p, fourier_basis):
    # Converts a pxpx... or batch x ... tensor into the 2D Fourier basis.
    # Output has the same shape as the original
    shape = mat.shape
    mat = einops.rearrange(mat, '(x y) ... -> x y (...)', x=p, y=p)
    fourier_mat = torch.einsum('xyz,fx,Fy->fFz', mat, fourier_basis, fourier_basis)
    return fourier_mat.reshape(shape)

"""
def analyse_fourier_2d(p, tensor, top_k=10):
    # Processes a (p,p) or (p*p) tensor in the 2D Fourier Basis, showing the 
    # top_k terms and how large a fraction of the variance they explain
    values, indices = tensor.flatten().pow(2).sort(descending=True)
    rows = []
    total = values.sum().item()
    for i in range(top_k):
        rows.append([tensor.flatten()[indices[i]].item(),
                     values[i].item()/total, 
                     values[:i+1].sum().item()/total, 
                     fourier_basis_names[indices[i].item()//p], 
                     fourier_basis_names[indices[i]%p]])
    display(pd.DataFrame(rows, columns=['Coefficient', 'Frac explained', 'Cumulative frac explained', 'x', 'y']))
"""

def get_2d_fourier_component(tensor, x, y, fourier_basis):
    # Takes in a batch x ... tensor and projects it onto the 2D Fourier Component 
    # (x, y)
    vec = fourier_2d_basis_term(x, y, fourier_basis).flatten()
    return vec[:, None] @ (vec[None, :] @ tensor)

def get_component_cos_xpy(tensor, freq, fourier_basis, collapse_dim=False):
    # Gets the component corresponding to cos(freq*(x+y)) in the 2D Fourier basis
    # This is equivalent to the matrix cos((x+y)*freq*2pi/p)
    cosx_cosy_direction = fourier_2d_basis_term(2*freq-1, 2*freq-1, fourier_basis=fourier_basis).flatten()
    sinx_siny_direction = fourier_2d_basis_term(2*freq, 2*freq, fourier_basis=fourier_basis).flatten()
    # Divide by sqrt(2) to ensure it remains normalised
    cos_xpy_direction = (cosx_cosy_direction - sinx_siny_direction)/np.sqrt(2)
    # Collapse_dim says whether to project back into R^(p*p) space or not
    if collapse_dim:
        return (cos_xpy_direction @ tensor)
    else:
        return cos_xpy_direction[:, None] @ (cos_xpy_direction[None, :] @ tensor)

def get_component_sin_xpy(tensor, freq, fourier_basis, collapse_dim=False):
    # Gets the component corresponding to sin((x+y)*freq*2pi/p) in the 2D Fourier basis
    sinx_cosy_direction = fourier_2d_basis_term(2*freq, 2*freq-1, fourier_basis=fourier_basis).flatten()
    cosx_siny_direction = fourier_2d_basis_term(2*freq-1, 2*freq, fourier_basis=fourier_basis).flatten()
    sin_xpy_direction = (sinx_cosy_direction + cosx_siny_direction)/np.sqrt(2)
    if collapse_dim:
        return (sin_xpy_direction @ tensor)
    else:
        return sin_xpy_direction[:, None] @ (sin_xpy_direction[None, :] @ tensor)

# %% ../helpers.ipynb 13
def to_numpy(tensor, flat=False):
    if type(tensor)!=torch.Tensor:
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()

def unflatten_first(tensor, p):
    if tensor.shape[0]==p*p:
        return einops.rearrange(tensor, '(x y) ... -> x y ...', x=p, y=p)
    else: 
        return tensor

def imshow(tensor, p, xaxis=None, yaxis=None, animation_name='Snapshot', **kwargs):
    if tensor.shape[0]==p*p:
        tensor = unflatten_first(tensor, p)
    tensor = torch.squeeze(tensor)
    px.imshow(to_numpy(tensor, flat=False), 
              labels={'x':xaxis, 'y':yaxis, 'animation_name':animation_name}, 
              **kwargs).show()

# Set default colour scheme
#| export
imshow = partial(imshow, color_continuous_scale='Blues')

# Creates good defaults for showing divergent colour scales (ie with both 
# positive and negative values, where 0 is white)
imshow_div = partial(imshow, color_continuous_scale='RdBu', color_continuous_midpoint=0.0)

# Presets a bunch of defaults to imshow to make it suitable for showing heatmaps 
# of activations with x axis being input 1 and y axis being input 2.
#| export
def inputs_heatmap(*args, **kwargs):
    return imshow(*args, **kwargs, xaxis='Input 1', yaxis='Input 2', color_continuous_scale='RdBu', color_continuous_midpoint=0.0)

def scatter(x, y, **kwargs):
    px.scatter(x=to_numpy(x, flat=True), y=to_numpy(y, flat=True), **kwargs).show()

# %% ../helpers.ipynb 14
def line(x, y=None, hover=None, xaxis='', yaxis='', **kwargs):
    if type(y)==torch.Tensor:
        y = to_numpy(y, flat=True)
    if type(x)==torch.Tensor:
        x=to_numpy(x, flat=True)
    fig = px.line(x, y=y, hover_name=hover, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    fig.show()

# %% ../helpers.ipynb 15
def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    '''Helper function to plot multiple lines'''
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()

def line_marker(x, **kwargs):
    lines([x], mode='lines+markers', **kwargs)

def animate_lines(lines_list, snapshot_index = None, snapshot='snapshot', hover=None, xaxis='x', yaxis='y', **kwargs):
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[1]):
            rows.append([lines_list[i][j], snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=[yaxis, snapshot, xaxis])
    px.line(df, x=xaxis, y=yaxis, animation_frame=snapshot, range_y=[lines_list.min(), lines_list.max()], hover_name=hover,**kwargs).show()


def imshow_fourier(tensor, p, fourier_basis_names, title='', animation_name='snapshot', facet_labels=[], **kwargs):
    # Set nice defaults for plotting functions in the 2D fourier basis
    # tensor is assumed to already be in the Fourier Basis
    if tensor.shape[0]==p*p:
        tensor = unflatten_first(tensor, p)
    tensor = torch.squeeze(tensor)
    fig=px.imshow(to_numpy(tensor),
            x=fourier_basis_names, 
            y=fourier_basis_names, 
            labels={'x':'x Component', 
                    'y':'y Component', 
                    'animation_frame':animation_name},
            title=title,
            color_continuous_midpoint=0., 
            color_continuous_scale='RdBu', 
            **kwargs)
    fig.update(data=[{'hovertemplate':"%{x}x * %{y}y<br>Value:%{z:.4f}"}])
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    fig.show()

def animate_multi_lines(lines_list, y_index=None, snapshot_index = None, snapshot='snapshot', hover=None, swap_y_animate=False, **kwargs):
    # Can plot an animation of lines with multiple lines on the plot.
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if swap_y_animate:
        lines_list = lines_list.transpose(1, 0, 2)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if y_index is None:
        y_index = [str(i) for i in range(lines_list.shape[1])]
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append(list(lines_list[i, :, j])+[snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=y_index+[snapshot, 'x'])
    px.line(df, x='x', y=y_index, animation_frame=snapshot, range_y=[lines_list.min(), lines_list.max()], hover_name=hover, **kwargs).show()

def animate_scatter(lines_list, snapshot_index = None, snapshot='snapshot', hover=None, yaxis='y', xaxis='x', color=None, color_name = 'color', **kwargs):
    # Can plot an animated scatter plot
    # lines_list has shape snapshot x 2 x line
    if type(lines_list)==list:
        lines_list = torch.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list, flat=False)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    if color is None:
        color = np.ones(lines_list.shape[-1])
    if type(color)==torch.Tensor:
        color = to_numpy(color)
    if len(color.shape)==1:
        color = einops.repeat(color, 'x -> snapshot x', snapshot=lines_list.shape[0])
    print(lines_list.shape)
    rows=[]
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append([lines_list[i, 0, j].item(), lines_list[i, 1, j].item(), snapshot_index[i], color[i, j]])
    print([lines_list[:, 0].min(), lines_list[:, 0].max()])
    print([lines_list[:, 1].min(), lines_list[:, 1].max()])
    df = pd.DataFrame(rows, columns=[xaxis, yaxis, snapshot, color_name])
    px.scatter(df, x=xaxis, y=yaxis, animation_frame=snapshot, range_x=[lines_list[:, 0].min(), lines_list[:, 0].max()], range_y=[lines_list[:, 1].min(), lines_list[:, 1].max()], hover_name=hover, color=color_name, **kwargs).show()

# %% ../helpers.ipynb 16
def cos(x, y):
    return (x.dot(y))/x.norm()/y.norm()
def mod_div(a, b, p):
    return (a*pow(b, p-2, p))%p
def normalize(tensor, axis=0):
    return tensor/(tensor).pow(2).sum(keepdim=True, axis=axis).sqrt()
def extract_freq_2d(tensor, freq, p):
    # Takes in a pxpx... or batch x ... tensor, returns a 3x3x... tensor of the 
    # Linear and quadratic terms of frequency freq
    tensor = unflatten_first(tensor, p)
    # Extracts the linear and quadratic terms corresponding to frequency freq
    index_1d = [0, 2*freq-1, 2*freq]
    # Some dumb manipulation to use fancy array indexing rules
    # Gets the rows and columns in index_1d
    return tensor[[[i]*3 for i in index_1d], [index_1d]*3]
def get_cov(tensor, norm=True):
    # Calculate covariance matrix
    if norm:
        tensor = normalize(tensor, axis=1)
    return tensor @ tensor.T
def is_close(a, b):
    return ((a-b).pow(2).sum()/(a.pow(2).sum().sqrt())/(b.pow(2).sum().sqrt())).item()

# %% ../helpers.ipynb 18
def cpu_aware_load_at_root(path):
    path = root / path
    if torch.cuda.is_available():
        return torch.load(path)
    else:
        return torch.load(path, map_location=torch.device('cpu'))

#| export
def load_mod_addition_frac_train_sweep():
    return cpu_aware_load_at_root('mod_addition_frac_train_sweep.pth')

#| export
def load_5_digit_addition_infinite():
    return cpu_aware_load_at_root('5_digit_addition_infinite.pth')

#| export
def load_5_digit_addition_finite():
    return cpu_aware_load_at_root('5_digit_addition_finite.pth')

#| export
def load_induction_head_finite():
    return cpu_aware_load_at_root('induction_head_finite.pth')

#| export
def load_induction_head_infinite():
    return cpu_aware_load_at_root('induction_head_infinite.pth')

#| export
def load_infinite_data_losses():
    return cpu_aware_load_at_root('skip_trigram_infinite.pth')

#| export
def load_finite_data_losses():
    return cpu_aware_load_at_root('skip_trigram_finite.pth')

#| export
def load_no_wd_width_scan():
    return cpu_aware_load_at_root('no_wd_width_scan.pth')

# AUTOGENERATED! DO NOT EDIT! File to edit: ../transformer.ipynb.

# %% auto 0
__all__ = ['make_fourier_basis', 'calculate_key_freqs', 'get_components_of_trig_loss',
           'calculate_excluded_loss', 'calculate_trig_loss', 'calculate_coefficients', 'loss']

# %% ../transformer.ipynb 3
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import einops
import random 
from dataclasses import dataclass
import os
import wandb
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import re
import pandas as pd
import json
from config import *
from dataset import *
from model import *
from tokenizer import *

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
    fourier_neuron_acts = fft2d(neuron_acts_centered, p = config.p, fourier_basis=fourier_basis)

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
            numerator = extract_freq_2d(fourier_neuron_acts_square[:, :, ni], freq, p = config.p).pow(2).sum()
            denominator = fourier_neuron_acts_square[:, :, ni].pow(2).sum().item()
            frac_explained = numerator / denominator
            if frac_explained > best_frac_explained:
                best_freq = freq
                best_frac_explained = frac_explained
        neuron_freqs.append(best_freq)
        neuron_frac_explained.append(best_frac_explained)
    neuron_freqs = np.array(neuron_freqs)
    neuron_frac_explained = to_numpy(neuron_frac_explained)
    key_freqs, neuron_freq_counts = np.unique(neuron_freqs, return_counts=True)
    return key_freqs

def get_components_of_trig_loss(logits, freq, fourier_basis):
    cos = get_component_cos_xpy(logits, freq, fourier_basis=fourier_basis)
    sin = get_component_sin_xpy(logits, freq, fourier_basis=fourier_basis)
    return cos + sin


def calculate_excluded_loss(config: Config, fourier_basis, key_freqs, is_train, is_test, labels, logits):
    row = []
    for freq in key_freqs:
        cos = get_component_cos_xpy(logits, freq, fourier_basis=fourier_basis) 
        sin = get_component_sin_xpy(logits, freq, fourier_basis=fourier_basis) 
        value = test_logits(logits - cos - sin, bias_correction=False, mode='train', p = config.p,
           is_train = is_train, is_test = is_test, labels = labels)
        row.append(value.item())
    return row

def calculate_trig_loss(config: Config, model, train, logits, key_freqs, fourier_basis, all_data, is_train, is_test, labels, mode='all'):
    trig_logits = sum([get_components_of_trig_loss(logits, freq, fourier_basis) for freq in key_freqs])
    return test_logits(trig_logits, 
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

# TODO what type for model?
def loss(config : Config, model: Transformer, data):
    # We need this to extract the position of the result in the input string
    tokenizer = Tokenizer(config)
    start_tokenid = tokenizer.tokenize("=")[0]
    end_tokenid = tokenizer.tokenize("EOS")[0]
    '''Takes the cross entropy loss of the model on the data'''
    fwd_data = data[:,:-1].to(config.device)
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
    
def extract_answer_from_prediction(pred, tokenizer):
    """
    Takes the prediction and extracts the answer from it
    pred: list of token ids like [0, 10, 0, 11, 0, 12, 13]
    return: answer which is an integer like 0
    """
    equal_tokenid = tokenizer.tokenize("=")[0]
    eos_tokenid = tokenizer.tokenize("EOS")[0]
    answer_start_idx = pred.index(equal_tokenid) + 1
    answer_end_idx = pred.index(eos_tokenid)
    answer = pred[answer_start_idx:answer_end_idx]
    try:
        answer = int(tokenizer.detokenize(answer))
    except:
        print("Could not convert answer to integer")
    return answer

def select_results_n_digits(data, n):
    """
    Selects the results with n or more digits
    """
    return data[data["result"].apply(lambda x: len(str(x)) >= n)]

def get_counts(data):
    """
    Initializes the counts for the metrics
    """
    counts = {}
    train = data[data["is_train"]]
    test = data[~data["is_train"]]

    # Counts for overall accuracy
    counts["train_total"] = len(train)
    counts["test_total"] = len(test)

    # Counts for individual digits
    max_len = len(str(train["result"].max()))

    for i in range(max_len):
        train_n_digits = select_results_n_digits(train, i+1)
        test_n_digits = select_results_n_digits(test, i+1)
        counts[f"train_digit_{i}"] = len(train_n_digits)
        counts[f"test_digit_{i}"] = len(test_n_digits)
    return counts

def check_individual_digits(pred, ground_truth, istrain, frequencies):
        """
        Updates the frequency for the individual digits metric
        """
        # Fill shorter number with zeros
        max_len = max(len(str(pred)), len(str(ground_truth)))
        pred_str = f"{pred:0{max_len}d}"
        ground_truth_str = f"{ground_truth:0{max_len}d}"
        
        assert len(pred_str) == len(ground_truth_str)

        for i in range(len(pred_str)):
            # Initialize the dictionary entry
            if istrain and f"train_digit_{i}" not in frequencies.keys():
                frequencies[f'train_digit_{i}'] = 0
            elif not istrain and f"test_digit_{i}" not in frequencies.keys():
                frequencies[f'test_digit_{i}'] = 0
            
            # Now update the dictionary entry if the prediction is correct
            if pred_str[-(i+1)] == ground_truth_str[-(i+1)]:
                #print("correct digit prediction for digit ", i)
                #print("ground_truth:", ground_truth_str)
                #print("pred:", pred_str)
                if istrain:
                    frequencies[f'train_digit_{i}'] += 1
                else:
                    frequencies[f'test_digit_{i}'] += 1
            '''
            else:
                print("incorrect digit prediction for digit ", i)
                print("ground_truth:", ground_truth_str)
                print("pred:", pred_str)
            '''

def get_frequencies(data, model, tokenizer):
    """
    Calculates the frequencies for the metrics
    """
    frequencies = {
        "train_total": 0,
        "test_total": 0
    }
    # Calculate the accuracy (digits and overall) for train and test
    for _, row in data.iterrows():

        is_train = row["is_train"]
        # Get the ground truth and prediction
        ground_truth = int(row["result"])
        # Convert the tokenized input to a list
        input = tokenizer.tokenize(f"{row['operand_1']}+{row['operand_2']}=")
        pred = model.generate_greedy(input)
        answer = extract_answer_from_prediction(pred, tokenizer)
        
        # Check prediction and ground truth overall
        if answer == ground_truth:
            if is_train:
                frequencies["train_total"] += 1
            else:
                frequencies["test_total"] += 1
        
        # Check prediction and ground truth for individual digits

        check_individual_digits(answer, ground_truth, is_train, frequencies)

    return frequencies

def take_metrics(model_path):

    # Model and tokenizer
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    config = Config()
    model = Transformer(config)
    model.load_state_dict(t.load(model_path, map_location = device)["model"])
    model.to(device)
    model.eval()
    tokenizer = Tokenizer(config)

    # We evaluate directly on the data df saved in the same directory as the model
    data_path = "saved_runs/variable_digit_add_50/data.csv"
    data = pd.read_csv(data_path)

    # Get counts
    counts = get_counts(data)
    # print("counts", counts)

    # Get the frequencies
    frequencies = get_frequencies(data, model, tokenizer)
    # print("frequencies", frequencies)

    # Divide the frequencies by the counts to get the metrics
    metrics = {}
    for key in frequencies.keys():
        metrics[key + "_accuracy"] = frequencies[key] / counts[key]

    # print("metrics", metrics)

    # Save the metrics to a json file
    save_metrics(metrics, model_path)

def save_metrics(metrics_dict, model_path):
    """
    Function which saves the metrics of a model to a json file
    called "metrics.json" in the same directory as the model
    """
    # Extract the directory and model name
    dir_path = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)

    # Save the metrics to metric.json
    metrics_data = {model_name: metrics_dict}
    metrics_file_path = os.path.join(dir_path, "metrics.json")

    # Append or create the metrics file
    if os.path.exists(metrics_file_path):
        with open(metrics_file_path, "r+") as f:
            existing_data = json.load(f)
            existing_data.update(metrics_data)
            f.seek(0)
            json.dump(existing_data, f, indent=4)
    else:
        # Create and write the file if it doesn't exist
        with open(metrics_file_path, "w") as f:
            json.dump(metrics_data, f, indent=4)

    
