from torch.utils.data import Dataset
import pandas as pd

class AdditionDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)
        # TODO: This is a work around because the data.csv is suboptimal
        self.data["tokenized"] = self.data["tokenized"].apply(lambda x: [int(i) for i in x.strip("[]").split(",")])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]["tokenized"][:-1]
    