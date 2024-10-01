__all__ = ['TokenizedDataset', 'gen_train_test', 'to_fixed_digit_format', 'to_variable_digit_format', 'filter_data']

from tokenizer import *
from config import *
import numpy as np
import torch as t
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import random

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
    pairs_1 = [(i,j,(i+j)) for i in range(10) for j in range(10)]
    df_1 = pd.DataFrame(pairs_1, columns=['operand_1', 'operand_2', 'result'])
    random.seed(config.seed)
    num_train = int(upsample_frac * len(pairs_1))
    num_test = len(pairs_1) - num_train
    train_idx = np.array([True]*num_train + [False]*num_test)
    random.shuffle(train_idx)
    df_1['is_train'] = train_idx

    pairs_2 = [(i,j,(i+j)) for i in range(10) for j in range(10, 100)]
    pairs_2 += [(i,j,(i+j)) for i in range(10, 100) for j in range(10)]
    pairs_2 += [(i,j,(i+j)) for i in range(10, 100) for j in range(10, 100)]
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
        df["input_str"] = df.apply(lambda row: to_fixed_digit_format(row, config), axis=1)
    else:
        df["input_str"] = df.apply(lambda row: to_variable_digit_format(row, config), axis=1)

    # Add the tokenized version which is fed into the LLM
    df["tokenized"] = df["input_str"].apply(lambda x: tokenizer.tokenize(x))

    # Write to file
    # with open('data.csv', 'w') as f:
    #    df.to_csv(f)

    return df

def to_fixed_digit_format(row, config):
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

def to_variable_digit_format(row, config):
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