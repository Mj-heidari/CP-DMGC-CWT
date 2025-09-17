import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from typing import Callable, List
import glob
import os
from .utils import invert_uint16_scaling
from tqdm import tqdm
from collections import defaultdict


class CHBMITDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str = "data/BIDS_CHB-MIT",
        use_uint16: bool = True,
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
            np.long
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
        # g = self.group_ids[idx]
        for transform in self.online_transform:
            x = transform(x)
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)




def leave_one_preictal_group_out(dataset, shuffle=True, random_state=0):
    """
    Cross-validation splitter:
      - Each fold leaves one preictal group out for testing.
      - Remaining preictal groups go to training.
      - Interictal samples are split into the same number of folds.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        base_ds = dataset.dataset
        idx = dataset.indices
        idx.sort()
        y = base_ds.y[idx]
        group_id = base_ds.group_ids[idx]


    else:
        y, group_id = dataset.y, dataset.group_ids

    # Masks
    pre_mask = y == 1
    inter_mask = ~pre_mask

    # Unique preictal groups in order of appearance
    pre_groups = np.unique(group_id[pre_mask])

    # Indices
    pre_indices = np.where(pre_mask)[0]
    inter_indices = np.where(inter_mask)[0]

    # Compute start/end of each preictal group
    pre_bounds = []
    for g in pre_groups:
        indices = np.where(group_id == g)[0]
        pre_bounds.append((indices[0], indices[-1]))

    # Assign each interictal sample to nearest preictal group
    inter_assignment = []

    for idx in inter_indices:
        # idx is the position in the original dataset
        if idx <= pre_bounds[0][0]:
            inter_assignment.append(pre_groups[0])
        elif idx >= pre_bounds[-1][1]:
            inter_assignment.append(pre_groups[-1])
        else:
            for j in range(len(pre_bounds)-1):
                mid = (pre_bounds[j][1] + pre_bounds[j+1][0]) // 2
                if idx <= mid and idx >= pre_bounds[j][1]:
                    inter_assignment.append(pre_groups[j])
                    break
                elif idx > mid and idx <= pre_bounds[j+1][1]:
                    inter_assignment.append(pre_groups[j+1])
                    break

    # Group interictal indices by assigned preictal group
    inter_chunks = defaultdict(list)
    for idx, assigned_group in zip(inter_indices, inter_assignment):
        inter_chunks[assigned_group].append(idx)

    for pre_group in pre_groups:
        if pre_group not in inter_chunks.keys():
            pre_mask[group_id == pre_group] = 0

    # Build folds
    for test_group in inter_chunks.keys():
        # Preictal split
        pre_test_mask = group_id == test_group
        pre_train_mask = pre_mask & ~pre_test_mask

        pre_train_idx = np.where(pre_train_mask)[0]
        pre_test_idx = np.where(pre_test_mask)[0]

        # Interictal split for this fold
        inter_test_idx = np.array(inter_chunks[test_group])
        inter_train_idx = np.hstack(
            [inter_chunks[g] for g in pre_groups if g != test_group and g in inter_chunks.keys()]
        ).astype('int')

        train_idx = np.concatenate([pre_train_idx, inter_train_idx]).tolist()
        test_idx = np.concatenate([pre_test_idx, inter_test_idx]).tolist()

        yield Subset(dataset, train_idx), Subset(dataset, test_idx)
