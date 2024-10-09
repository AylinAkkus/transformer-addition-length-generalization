import torch as t
from config import Config
from model import Transformer 
from tokenizer import Tokenizer
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def carry_digits(row, max_len):
    """
    Checks whether there is a carry digit at position n in the result of the addition
    """
    # Read the operands
    operand_1 = str(row["operand_1"])
    operand_2 = str(row["operand_2"])

    # Pad the operands with 0s
    paddded_op_1 = operand_1.zfill(max_len)
    padded_op_2 = operand_2.zfill(max_len)
    # Check for all carry digits
    carries = [0] * (max_len-1)
    for i in range(max_len):
        if int(paddded_op_1[-(i+1)]) + int(padded_op_2[-(i+1)]) + carries[-i] >= 10:
            carries[-(i+1)] = 1

    assert len(carries) == max_len-1
    return carries
   
def add_carry_columns_to_df(data_path, new_data_path=None):
    df = pd.read_csv(data_path)
    max_len = df["result"].apply(lambda x: len(str(x))).max()
    carry_col_names = [f"carry_{i}" for i in range(1, max_len)]
    res = df.apply(lambda x: carry_digits(x, max_len), axis=1)
    df[carry_col_names] = pd.DataFrame(res.tolist(), index=df.index)
    if new_data_path:
        df.to_csv(new_data_path, index=False)
    else:
        df.to_csv(data_path, index=False)

# Load the model
config = Config()
model_name = "variable_digit_add_30"
model_path = f"saved_runs/{model_name}/final.pth"
data_path = f"saved_runs/{model_name}/data.csv"
repr_type = "resid_post"
new_data_path = f"saved_runs/{model_name}/{repr_type}_carry_log_reg_data.csv"
device = config.device
tokenizer = Tokenizer(Config())
model = Transformer(Config())
model.load_state_dict(t.load(model_path, map_location = device)["model"])
model.eval()

hidden_dict = {}

def get_hidden_equals(idx, activation, **kwargs):
    name = kwargs.get('name', 'Unknown')
    activation = activation.squeeze() # Remove the batch dimension
    hidden_equals = activation[-1]
    hidden_dict[idx] = hidden_equals

def get_hidden_equals_with_idx(idx):
    return lambda activation, **kwargs: get_hidden_equals(idx, activation, **kwargs)

data = pd.read_csv(data_path)
for idx, row in data.iterrows():
    # Adding the hook to the model
    print(f"idx: {idx}")
    if repr_type == "resid_mid":
        model.blocks[-1].hook_resid_mid.add_hook(get_hidden_equals_with_idx(idx))
    else:
        model.blocks[-1].hook_resid_post.add_hook(get_hidden_equals_with_idx(idx))

    # Tokenizing the sentence
    op_1 = row["operand_1"]
    op_2 = row["operand_2"]
    input_sentence = f"{op_1}+{op_2}="
    input_sentence_tokenized = [tokenizer.tokenize(input_sentence)]
    input_sentence_tokenized = t.tensor(input_sentence_tokenized).to(device)

    # Forward pass
    with t.no_grad():
        output = model(input_sentence_tokenized)

    model.remove_all_hooks()

    # Merge the dict into the dataframe
d_model = config.d_model
col_list = [f"hidden_{i}" for i in range(d_model)]

for idx, hidden_state in hidden_dict.items():
    hidden_state = hidden_state.cpu().tolist()
    data.loc[idx, col_list] = hidden_state

# Write the dataframe to a csv file

# Drop the columns that are not needed
data_2 = data.drop(["Unnamed: 0", "input_str", "tokenized"], axis=1)
data_2.to_csv(new_data_path, index=False)
# Add the carry columns
add_carry_columns_to_df(new_data_path)