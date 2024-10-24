import torch 

class UnsupervisedDS(torch.utils.data.Dataset):
    def __init__(self, x):
        self.data = torch.tensor(x).float()
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]
    