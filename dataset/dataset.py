import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from typing import Callable, List
import glob
import os
from .utils import invert_uint16_scaling
from tqdm import tqdm


class CHBMITDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str = "data/BIDS_CHB-MIT",
        use_uint16: bool = False,
        subject_id: str = "01",
        online_transforms: List[Callable] = None,
        offline_transforms: List[Callable] = None,
    ):
        """
        Args:
            dataset_dir (string): path to the processed BIDS_CHB-MIT dataset
            use_uint16 (boolean): if true uses
            subject_id (str): use "*" to include all subjects,
        """
        subject_dirs = sorted(glob.glob(os.path.join(dataset_dir, f"sub-{subject_id}")))

        for subj_path in tqdm(subject_dirs):
            subj_id = os.path.basename(subj_path)

            # Collect all sessions for this subject
            suffix = "*uint16.npz" if use_uint16 else "*segments.npz"
            ses_paths = glob.glob(os.path.join(subj_path, "ses-*", "eeg", suffix))
            if ses_paths == []:
                continue

            subj_X, subj_y, subj_group_ids = [], [], []
            for ses_path in ses_paths:
                data = np.load(ses_path)

                X_temp = data["X"]
                if use_uint16:
                    scales = data["scales"]
                    X_temp = invert_uint16_scaling(X_temp, scales)

                subj_X.append(X_temp)
                subj_y.append(data["y"])
                subj_group_ids.append(data["group_ids"])

            # Concatenate sessions of this subject
            X = np.concatenate(subj_X, axis=0)
            y = np.concatenate(subj_y, axis=0)
            group_ids = np.concatenate(subj_group_ids, axis=0)

        self.online_transform = online_transforms or []

        # Encode labels to 0/1
        self.y = np.array([1 if label == "preictal" else 0 for label in y]).astype(
            np.float32
        )
        self.X = X.astype(np.float32)
        self.group_ids = group_ids

        # Apply offline transforms once
        for transform in offline_transforms or []:
            transformed_X = []
            for i in range(self.X.shape[0]):
                transformed_X.append(transform(eeg=self.X[i]))
            self.X = np.stack(transformed_X, axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        g = self.group_ids[idx]
        for transform in self.online_transform:
            x = transform(x)
        return torch.tensor(x), torch.tensor(y), g


def leave_one_preictal_group_out(dataset, shuffle=True, random_state=0):
    """
    Cross-validation splitter:
      - Each fold leaves one preictal group out for testing.
      - Remaining preictal groups go to training.
      - Interictal samples are split into the same number of folds.
    """
    y = np.array(dataset.y)
    group_id = np.array(dataset.group_ids)

    # Masks
    pre_mask = y == 1
    inter_mask = ~pre_mask

    # Unique preictal groups
    pre_groups = np.unique(group_id[pre_mask])
    n_splits = len(pre_groups)

    # Shuffle interictal indices reproducibly
    rng = np.random.default_rng(seed=random_state)
    inter_indices = np.where(inter_mask)[0]
    rng.shuffle(inter_indices)

    # Divide interictal into n_splits chunks
    inter_chunks = np.array_split(inter_indices, n_splits)

    # Build folds
    for fold, test_group in enumerate(pre_groups):
        # Preictal split
        pre_test_mask = group_id == test_group
        pre_train_mask = pre_mask & ~pre_test_mask

        pre_train_idx = np.where(pre_train_mask)[0]
        pre_test_idx = np.where(pre_test_mask)[0]

        # Interictal split
        inter_test_idx = inter_chunks[fold]
        inter_train_idx = np.hstack(
            [chunk for i, chunk in enumerate(inter_chunks) if i != fold]
        )

        # Combine
        train_idx = np.concatenate([pre_train_idx, inter_train_idx]).tolist()
        test_idx = np.concatenate([pre_test_idx, inter_test_idx]).tolist()

        yield Subset(dataset, train_idx), Subset(dataset, test_idx)
