from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional as F

class NPZDataset(Dataset):
    def __init__(self, path, indices=None, resize=256, device="cpu"):
        self.data = np.load(path, allow_pickle=True, mmap_mode="r")
        self.images = self.data["images"]
        self.labels = self.data["labels"]

        if indices is None:
            self.indices = np.arange(len(self.labels))
        else:
            self.indices = indices

        self.resize = resize
        self.device = device

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        img = torch.tensor(self.images[i]).float()
        lbl = torch.tensor(self.labels[i]).float()

        img = img.permute(2,0,1)
        if img.max() > 1:
            img = img / 255.

        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.resize, self.resize),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return img.to(self.device), lbl.to(self.device)

class SyntheticDataset(Dataset):
    def __init__(self, path, device):
        self.data = np.load(path, mmap_mode="r")
        self.images = self.data["images"]
        self.labels = self.data["labels"]
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).float().permute(2,0,1)
        lbl = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img.to(self.device), lbl.to(self.device)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def step(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def reset(self):
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False