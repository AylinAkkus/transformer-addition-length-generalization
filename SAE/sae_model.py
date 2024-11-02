"""Copied from ARENA tutorial 1.3.1 by Callum McDougall:
https://arena3-chapter1-transformer-interp.streamlit.app/[1.3.1]_Superposition_&_SAEs
"""

from types import NoneType
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import einops
import numpy as np
import torch as t
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
from model import Transformer
from torch.utils.data import DataLoader
from .dataset import AdditionDataset

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

from re import X
def linear_lr(step, steps):
    return (1 - (step / steps))

def constant_lr(*_):
    return 1.0

def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))

def cache_mlp(model, cache):
    # Hook func to cache mlp activations for =
    # TODO is this the right place?
    def save_hook(tensor, name):
        cache.append(tensor.detach().squeeze())
    model.blocks[0].mlp.hook_post.add_hook(save_hook, 'fwd')

@dataclass
class SAEConfig:
    d_in: int
    d_sae: int
    n_inst: int = 1 # TODO: probably remove this
    l1_coeff: float = 0.2
    weight_normalize_eps: float = 1e-8
    tied_weights: bool = False
    architecture: Literal["standard", "gated"] = "standard"
    batch_size: int = 32


class SAE(nn.Module):
    W_enc: Float[Tensor, "inst d_in d_sae"]
    _W_dec: Float[Tensor, "inst d_sae d_in"] | None
    b_enc: Float[Tensor, "inst d_sae"]
    b_dec: Float[Tensor, "inst d_in"]

    def __init__(self, config: SAEConfig, model: Transformer) -> None:
        super(SAE, self).__init__()

        assert config.d_in == model.config.d_mlp, "Model's hidden dim doesn't match SAE input dim"
        self.config = config
        self.model = model.requires_grad_(False)

        # Initialize weights
        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(t.empty((config.n_inst, config.d_in,config.d_sae)))
            )
        if config.tied_weights:
          self._W_dec = None
        else:
          self._W_dec = nn.Parameter(nn.init.kaiming_uniform_(t.empty((config.n_inst, config.d_sae, config.d_in))))
        self.b_enc = nn.Parameter(t.zeros(config.n_inst, config.d_sae))
        self.b_dec = nn.Parameter(t.zeros(config.n_inst, config.d_in))
        self.to(device)

        # Initialize dataloader
        data_path = "saved_runs/fixed_digit_add_30/data.csv" # TODO make sure model and data are from same dir, maybe add this as model attribute
        dataset = AdditionDataset(data_path)
        self.dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
    
    @classmethod
    def from_file(
            cls,
            path: str,
        ) -> None:
        """
        Loads the SAE model including config and transformer model from a file.
        Args:
            path:   path to load the model
        """
        save_dict = t.load(path)
        # keys = ["state_dict", "config", "transformer_state_dict", "transformer_config"]
        config = save_dict["config"]
        transformer_model = Transformer(save_dict["transformer_config"])
        transformer_model.load_state_dict(save_dict["transformer_state_dict"])
        sae = cls(config, transformer_model)
        return sae

    @property
    def W_dec(self) -> Float[Tensor, "inst d_sae d_in"]:
        return self._W_dec if self._W_dec is not None else self.W_enc.transpose(-1, -2)

    @property
    def W_dec_normalized(self) -> Float[Tensor, "inst d_sae d_in"]:
        """Returns decoder weights, normalized over the autoencoder input dimension."""
        return self.W_dec / (self.W_dec.norm(dim=-1, keepdim=True) + self.config.weight_normalize_eps)

    def generate_batch(self) -> Float[Tensor, "batch inst d_in"]:
        """
        Generates a batch of hidden activations from our model.
        """
        #return einops.einsum(features, W, "batch inst features, inst hidden features -> batch inst hidden")
        self.model.remove_all_hooks()
        cache = [] # A list which will store the MLP activations for all tokens
        cache_mlp(self.model, cache)
        batch = next(iter(self.dataloader))
        batch_stacked = t.stack(batch, dim=1)
        
        # Forward pass
        self.model(batch_stacked)

        # Reshaping the activations to: (num_tokens x samples) inst hidden_size
        batch = t.cat(cache, dim=0)
        batch = batch.reshape(-1, self.model.config.d_mlp).unsqueeze(dim=1)
        return batch

    def forward(
        self,
        h: Float[Tensor, "batch inst d_in"],
    ) -> tuple[
        dict[str, Float[Tensor, "batch inst"]],
        Float[Tensor, ""],
        Float[Tensor, "batch inst d_sae"],
        Float[Tensor, "batch inst d_in"],
    ]:
        """
        Forward pass on the autoencoder.

        Args:
            h: hidden layer activations of model

        Returns:
            loss_dict: dict of different loss function term values, for every (batch elem, instance)
            loss: scalar total loss (summed over instances & averaged over batch dim)
            acts: autoencoder feature activations
            h_reconstructed: reconstructed autoencoder input
        """
        # forward
        acts = einops.einsum(self.W_enc, h - self.b_dec, "inst hidden features, batch inst hidden -> batch inst features")
        acts = nn.functional.relu(acts + self.b_enc)
        h_reconstructed = einops.einsum(self.W_dec, acts, "inst features hidden, batch inst features -> batch inst hidden") + self.b_dec
        # calculating logging values
        reconstruction_loss = (h - h_reconstructed).pow(2).mean(dim=-1)
        sparsity = acts.norm(p=1,dim=-1)
        loss_dict = {
            "L_reconstruction": reconstruction_loss,
            "L_sparsity": sparsity
            }
        loss = reconstruction_loss.mean(dim=0).sum() + self.config.l1_coeff * sparsity.mean(dim=0).sum()
        return loss_dict, loss, acts, h_reconstructed

    def optimize(
        self,
        steps: int = 10_000,
        log_freq: int = 50,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
        resample_method: Literal["simple", "advanced", None] = None,
        resample_freq: int = 2500,
        resample_window: int = 500,
        resample_scale: float = 0.5,
    ) -> dict[str, list]:
        """
        Optimizes the autoencoder using the given hyperparameters.

        Args:
            model:              we reconstruct features from model's hidden activations
            batch_size:         size of batches we pass through model & train autoencoder on
            steps:              number of optimization steps
            log_freq:           number of optimization steps between logging
            lr:                 learning rate
            lr_scale:           learning rate scaling function
            resample_method:    method for resampling dead latents
            resample_freq:      number of optimization steps between resampling dead latents
            resample_window:    number of steps needed for us to classify a neuron as dead
            resample_scale:     scale factor for resampled neurons

        Returns:
            data_log:               dictionary containing data we'll use for visualization
        """
        assert resample_window <= resample_freq

        optimizer = t.optim.Adam(list(self.parameters()), lr=lr, betas=(0.0, 0.999))
        frac_active_list = []
        progress_bar = tqdm(range(steps))

        # Create lists to store data we'll eventually be plotting
        data_log = {"steps": [], "W_enc": [], "W_dec": [], "frac_active": []}

        for step in progress_bar:
            # Resample dead latents
            if (resample_method is not None) and ((step + 1) % resample_freq == 0):
                frac_active_in_window = t.stack(frac_active_list[-resample_window:], dim=0)
                if resample_method == "simple":
                    self.resample_simple(frac_active_in_window, resample_scale)
                elif resample_method == "advanced":
                    self.resample_advanced(frac_active_in_window, resample_scale)

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Get a batch of hidden activations from the model
            with t.inference_mode():
                h = self.generate_batch()

            # Optimize
            loss_dict, loss, acts, _ = self.forward(h)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Normalize decoder weights by modifying them inplace (if not using tied weights)
            if not self.config.tied_weights:
                self.W_dec.data = self.W_dec_normalized

            # Calculate the mean sparsities over batch dim for each feature
            frac_active = (acts.abs() > 1e-8).float().mean(0)
            frac_active_list.append(frac_active)

            # Display progress bar, and append new values for plotting
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(
                    lr=step_lr,
                    frac_active=frac_active.mean().item(),
                    **{k: v.mean(0).sum().item() for k, v in loss_dict.items()},  # type: ignore
                )
                data_log["W_enc"].append(self.W_enc.detach().cpu().clone())
                data_log["W_dec"].append(self.W_dec.detach().cpu().clone())
                data_log["frac_active"].append(frac_active.detach().cpu().clone())
                data_log["steps"].append(step)

        return data_log


    @t.no_grad()
    def resample_advanced(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
    ) -> None:
        """
        Resamples latents that have been dead for 'dead_feature_window' steps, according to `frac_active`.

        Resampling method is:
            - Compute the L2 reconstruction loss produced from the hidden state vectors `h`
            - Randomly choose values of `h` with probability proportional to their reconstruction loss
            - Set new values of W_dec and W_enc to be these (centered and normalized) vectors, at each dead neuron
            - Set b_enc to be zero, at each dead neuron
        """
        h = self.generate_batch()
        l2_loss = self.forward(h)[0]["L_reconstruction"]

        for instance in range(self.config.n_inst):
            # Find the dead latents in this instance. If all latents are alive, continue
            is_dead = (frac_active_in_window[:, instance] < 1e-8).all(dim=0)
            dead_latents = t.nonzero(is_dead).squeeze(-1)
            n_dead = dead_latents.numel()
            if n_dead == 0:
                continue  # If we have no dead features, then we don't need to resample

            # Compute L2 loss for each element in the batch
            l2_loss_instance = l2_loss[:, instance]  # [batch_size]
            if l2_loss_instance.max() < 1e-6:
                continue  # If we have zero reconstruction loss, we don't need to resample

            # Draw `d_sae` samples from [0, 1, ..., batch_size-1], with probabilities proportional to l2_loss
            distn = Categorical(probs=l2_loss_instance.pow(2) / l2_loss_instance.pow(2).sum())
            replacement_indices = distn.sample((n_dead,))  # type: ignore

            # Index into the batch of hidden activations to get our replacement values
            replacement_values = (h - self.b_dec)[replacement_indices, instance]  # [n_dead d_in]
            replacement_values_normalized = replacement_values / (
                replacement_values.norm(dim=-1, keepdim=True) + self.config.weight_normalize_eps
            )

            # Get the norm of alive neurons (or 1.0 if there are no alive neurons)
            W_enc_norm_alive_mean = (
                self.W_enc[instance, :, ~is_dead].norm(dim=0).mean().item()
                if (~is_dead).any()
                else 1.0
            )

            # Lastly, set the new weights & biases (W_dec is normalized, W_enc needs specific scaling, b_enc is zero)
            self.W_dec.data[instance, dead_latents, :] = replacement_values_normalized
            self.W_enc.data[instance, :, dead_latents] = (
                replacement_values_normalized.T * W_enc_norm_alive_mean * resample_scale
            )
            self.b_enc.data[instance, dead_latents] = 0.0

    @t.no_grad()
    def save_model(
            self,
            path: str,
    ) -> None:
        """
        Saves the SAE model including config and transformer model to a file.
        Args:
            path:   path to save the model to
        """
        save_dict = {
            "state_dict": self.state_dict(),
            "config": self.config,
            "transformer_state_dict": self.model.state_dict(),
            "transformer_config": self.model.config,
        }
        t.save(save_dict, path)

