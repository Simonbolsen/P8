import torch
from torch.utils.data import Dataset

class SupportDataset(torch.utils.data.Dataset):
    def __init__(self, support_data, support_data_targets):
        self.data = support_data
        self.targets = support_data_targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# removes in-place
def remove_classes_from_data(data : Dataset, classes : list[int]) -> tuple[Dataset, list[int]]:
    original_length = len(data)

    idx_to_remove = [i for i, x in enumerate(data.targets) if x in classes]
    idx_to_keep = [i for i, x in enumerate(data.targets) if x not in classes]
    
    support_data = data.data[idx_to_remove]
    support_targets = data.data[idx_to_remove]
    
    support_dataset = SupportDataset(support_data, support_targets)
    
    data.data = data.data[idx_to_keep]
    data.targets = data.targets[idx_to_keep]

    assert len(data.data) + len(support_data.data) == original_length
    assert len(data.targets) + len(support_data.targets) == original_length

    return data, support_dataset