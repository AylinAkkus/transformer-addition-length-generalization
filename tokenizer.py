__all__ = ['Tokenizer']
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import re
import pandas as pd
from config import *


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