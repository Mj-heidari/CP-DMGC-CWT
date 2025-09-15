import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.utils import shuffle
from collections import defaultdict
from typing import Callable, List

class CHBMITDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        group_ids,
        online_transforms: List[Callable] = None,
        offline_transforms: List[Callable] = None,
    ):
        """
        Args:
            X (ndarray): EEG data, shape (N, C, T)
            y (ndarray): labels, shape (N,)
            group_ids (ndarray): group labels (e.g. 'preictal_1', 'interictal_2')
        """
        self.online_transform = online_transforms or []

        # Encode labels to 0/1
        self.y_full = np.array([1 if label == "preictal" else 0 for label in y])
        self.X_full = X
        self.group_ids = group_ids

        # Apply offline transforms once
        for transform in offline_transforms or []:
            self.X_full = transform(self.X_full)

        # Preictal groups
        self.preictal_groups = sorted(
            set(g for g in self.group_ids if g.startswith("preictal"))
        )
        self.N = len(self.preictal_groups)

        # Interictal chunks (precomputed)
        interictal_idx = [i for i, g in enumerate(self.group_ids) if g.startswith("interictal")]
        interictal_idx = shuffle(interictal_idx, random_state=42)
        self.interictal_chunks = np.array_split(interictal_idx, self.N)

        # Storage for current split
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.X, self.y = None, None

    def set_leave_out(self, leave_out_idx: int):
        """Set which seizure group is left out for testing."""
        test_preictal_group = self.preictal_groups[leave_out_idx]
        test_interictal_idx = self.interictal_chunks[leave_out_idx]

        test_mask = np.array(
            [(g == test_preictal_group) or (i in test_interictal_idx)
             for i, g in enumerate(self.group_ids)]
        )
        train_mask = ~test_mask

        self.X_train, self.y_train = self.X_full[train_mask], self.y_full[train_mask]
        self.X_test, self.y_test = self.X_full[test_mask], self.y_full[test_mask]

        # Default = training set
        self.X, self.y = self.X_train, self.y_train

    def set_split(self, split="train"):
        if split == "train":
            self.X, self.y = self.X_train, self.y_train
        elif split == "test":
            self.X, self.y = self.X_test, self.y_test
        else:
            raise ValueError("split must be 'train' or 'test'")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = int(self.y[idx])
        for transform in self.online_transform:
            x = transform(x)
        return torch.tensor(x), torch.tensor(y)