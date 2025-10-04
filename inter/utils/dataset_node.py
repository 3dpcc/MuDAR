import glob

import numpy as np
from torch.utils.data import Dataset


class NodeDataset(Dataset):
    def __init__(self, point_path):
        self.file_list = sorted(glob.glob(point_path + "/**/*.npz", recursive=True))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        prev = np.load(self.file_list[idx])["octree_nodes_prev"]
        curr = np.load(self.file_list[idx])["octree_nodes_curr"]

        prev[:, :, 0] = prev[:, :, 0] - 1
        curr[:, :, 0] = curr[:, :, 0] - 1

        return prev, curr


if __name__ == "__main__":
    import torch.utils.data.dataloader as DataLoader

    train_data_root = "/zhu/data/SemanticKITTI/train/process_lidar_1024_dynamic"

    train_data = NodeDataset(point_path=train_data_root)
    train_loader = DataLoader.DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

    for data in train_loader:
        print(data.shape)
