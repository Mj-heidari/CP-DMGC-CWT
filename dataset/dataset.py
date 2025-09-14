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
        fold_idx,
        n_folds,
        online_transforms: List[Callable] = None,
        offline_transforms: List[Callable] = None,
    ):
        """
        Args:
            X (ndarray): EEG data, shape (N, C, T)
            y (ndarray): labels, shape (N,)
            group_ids (ndarray): group labels (e.g. 'preictal_1', 'interictal_2')
            fold_idx (int): which seizure fold to leave out (0 <= fold_idx < n_folds)
            n_folds (int): total number of seizures (preictal groups)
            transform: optional transform on X
        """
        self.online_transform = online_transforms or []

        # Encode labels to 0/1
        y = np.array([1 if label == "preictal" else 0 for label in y])

        # Split preictal groups
        preictal_groups = sorted(set(g for g in group_ids if g.startswith("preictal")))
        interictal_idx = [
            i for i, g in enumerate(group_ids) if g.startswith("interictal")
        ]

        # Shuffle interictal and split into N parts
        interictal_idx = shuffle(interictal_idx, random_state=42)
        interictal_chunks = np.array_split(interictal_idx, n_folds)

        # Select test groups
        test_preictal_group = preictal_groups[fold_idx]
        test_interictal_idx = interictal_chunks[fold_idx]

        # Build masks
        test_mask = np.array(
            [
                (g == test_preictal_group) or (i in test_interictal_idx)
                for i, g in enumerate(group_ids)
            ]
        )
        train_mask = ~test_mask

        for transform in offline_transforms or []:
            X= transform(X)

        self.X_train = X[train_mask]
        self.y_train = y[train_mask]
        self.X_test = X[test_mask]
        self.y_test = y[test_mask]

        # By default use training set
        self.X = self.X_train
        self.y = self.y_train

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
