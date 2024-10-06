__all__ = ['Config']

import numpy as np
import torch as t
import torch.nn.functional as F
import random 
import dataclasses
import pandas as pd
import json

@dataclasses.dataclass
class Config():
    lr: float = 1e-3 #@param
    weight_decay: float = 1.0 #@param
    batch_size: int = 256 #@param
    p: int = 100 #@param
    d_model: int = 128 #@param
    fn_name: str = 'add' #@param ['add', 'subtract', 'x2xyy2','rand']
    frac_train: float = 0.5 #@param
    num_epochs: int = 3000 #@param
    save_models: bool = True #@param
    save_every: int = 5 #@param
    fixed_digit: bool = False #@param
    n_warmup: int = 10 #@param

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
        return config_as_dict

    def is_train_is_test(self, train):
        '''Creates an array of Boolean indices according to whether each data point is in train or test.
        Used to index into the big batch of all possible data'''
        pass

    def is_it_time_to_save(self, epoch):
        """Save every 10 epochs for the first 1000 epochs, then less frequently"""
        if epoch < 10:
            return (epoch % 1 == 0)
        else:
            return (epoch % self.save_every == 0)

    def is_it_time_to_take_metrics(self, epoch):
        # for now take metrics whenever we save
        return self.is_it_time_to_save(epoch)

# TODO make this an assert inside the consturctor
assert Config.d_model % Config.num_heads == 0
