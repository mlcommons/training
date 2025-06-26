import pandas as pd
from torch.utils.data import Dataset, DataLoader


class TsvDataset(Dataset):
    def __init__(self, annotations_file, keys):
        self.df = pd.read_csv(annotations_file, sep='\t', header=0)
        self.keys = keys

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {}
        for key in self.keys:
            sample[key] = self.df[key].iloc[idx]
        return sample


def build_dataloader(annotations_file,
                     keys,
                     batch_size,
                     shuffle=False,
                     num_workers=1,
                     pin_memory=True):
    dataset = TsvDataset(annotations_file, keys=keys)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
