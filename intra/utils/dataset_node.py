import glob

import numpy as np

from torch.utils.data import Dataset


class NodeDataset(Dataset):
    def __init__(self, point_path):
        self.file_list = sorted(glob.glob(point_path + "/**/*.npz", recursive=True))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        nodes = np.load(self.file_list[idx])["octree_nodes"]

        nodes[:, :, 0] = nodes[:, :, 0] - 1

        return nodes


if __name__ == "__main__":
    import torch.utils.data.dataloader as DataLoader

    train_data_root = "/zhu/data/Ford/Train/Ford_01_q_1mm_octree_lidar_fixed_1024_static"

    train_data = NodeDataset(point_path=train_data_root)
    train_loader = DataLoader.DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

    for idx, data in enumerate(train_loader):
        print(idx, data.shape)
