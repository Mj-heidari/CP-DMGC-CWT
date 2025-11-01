import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from typing import Callable, List
import glob
import os
import math
from .utils import invert_uint16_scaling
from tqdm import tqdm
from collections import defaultdict
from sklearn import utils
from sklearn.model_selection import StratifiedKFold

class CHBMITDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str = "data/BIDS_CHB-MIT",
        use_uint16: bool = False,
        subject_id: str = "01",
        online_transforms: List[Callable] = None,
        offline_transforms: List[Callable] = None,
        suffix: str = "zscore_T",
    ):
        """
        Args:
            dataset_dir (string): path to the processed BIDS_CHB-MIT dataset
            use_uint16 (boolean): if true uses uint16 format
            subject_id (str): use "*" to include all subjects,
        """
        subject_dirs = sorted(glob.glob(os.path.join(dataset_dir, f"sub-{subject_id}")))

        for subj_path in tqdm(subject_dirs):
            # Collect all sessions for this subject
            suffix_pattern = (
                f"*{suffix}_uint16.npz" if use_uint16 else f"*{suffix}_float.npz"
            )
            ses_paths = glob.glob(
                os.path.join(subj_path, "ses-*", "eeg", suffix_pattern)
            )
            if ses_paths == []:
                continue

            subj_X, subj_y, subj_group_ids = [], [], []
            for i, ses_path in enumerate(ses_paths):
                data = np.load(ses_path)

                X_temp = data["X"]
                if use_uint16:
                    scales = data["scales"]
                    X_temp = invert_uint16_scaling(X_temp, scales)

                subj_X.append(X_temp)
                subj_y.append(data["y"])
                subj_group_ids.append(
                    np.array([gid + f"_{i}" for gid in data["group_ids"]])
                )

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
            for i in tqdm(range(self.X.shape[0])):
                transformed_X.append(transform(eeg=self.X[i]))
            self.X = np.stack(transformed_X, axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        for transform in self.online_transform:
            x = transform(x)
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)

    def get_class_indices(self):
        """Return indices for each class - useful for undersampling"""
        preictal_indices = np.where(self.y == 1)[0]
        interictal_indices = np.where(self.y == 0)[0]
        return preictal_indices, interictal_indices


class SubsetWithInfo(Subset):
    """A Subset that maintains instances' classes and groups information"""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        if isinstance(dataset, SubsetWithInfo):
            self.base_dataset: CHBMITDataset = dataset.base_dataset
            self.base_indices = np.array(dataset.base_indices)[indices]
        else:
            self.base_dataset: CHBMITDataset = dataset
            self.base_indices = indices

        self.y = self.base_dataset.y[self.base_indices]
        self.group_ids = self.base_dataset.group_ids[self.base_indices]

    def get_class_indices(self):
        """Return indices for each class within this subset"""
        preictal_indices = np.where(self.y == 1)[0]
        interictal_indices = np.where(self.y == 0)[0]
        return preictal_indices, interictal_indices


class UnderSampledDataLoader:
    """Custom DataLoader that undersamples interictal data each epoch"""
    
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Get indices for each class
        self.preictal_indices = []
        self.interictal_indices = []
        
        for i in range(len(dataset)):
            if dataset.y[i] == 1:  # preictal
                self.preictal_indices.append(i)
            else:  # interictal
                self.interictal_indices.append(i)
        
        self.preictal_indices = np.array(self.preictal_indices)
        self.interictal_indices = np.array(self.interictal_indices)
        self.count_seen = {idx: 0 for idx in self.interictal_indices}
        self.all_indices = self.get_indices()

    def get_indices(self):    
        # Randomly undersample interictal to match preictal count
        n_preictal = len(self.preictal_indices)
        n_interictal = len(self.interictal_indices)
        
        if n_interictal > n_preictal:
            # Convert to arrays for easy masking
            interictal_array = np.array(list(self.count_seen.keys()))
            seen_counts = np.array(list(self.count_seen.values()))

            min_count = seen_counts.min()
            min_mask = seen_counts == min_count
            min_indices = interictal_array[min_mask]

            if len(min_indices) < n_preictal:
                remaining = n_preictal - len(min_indices)
                not_min_indices = interictal_array[~min_mask]
                selected_extra = np.random.choice(not_min_indices, size=remaining, replace=False)
                selected_interictal = np.concatenate([min_indices, selected_extra])
            else:
                selected_interictal = np.random.choice(min_indices, size=n_preictal, replace=False)
        else:
            selected_interictal = self.interictal_indices

        for idx in selected_interictal:
            self.count_seen[idx] += 1

        all_indices = np.concatenate([self.preictal_indices, selected_interictal])
        if self.shuffle:
            np.random.shuffle(all_indices)
        return all_indices
        
    def __iter__(self):
        self.all_indices = self.get_indices()
        
        # Create batches
        for i in range(0, len(self.all_indices), self.batch_size):
            batch_indices = self.all_indices[i:i + self.batch_size]
            batch_data = []
            batch_labels = []
            
            for idx in batch_indices:
                data, label = self.dataset[idx]
                batch_data.append(data)
                batch_labels.append(label)
            
            yield torch.stack(batch_data), torch.stack(batch_labels)
    
    def __len__(self):
        return (len(self.all_indices) + self.batch_size - 1) // self.batch_size


class MilDataloader:
    """DataLoader that builds random MIL bags each epoch."""

    def __init__(self, dataset: SubsetWithInfo, batch_size=32, shuffle=True, bag_size=8, balance=True, seed=42):
        if seed is not None:
            np.random.seed(seed)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bag_size = bag_size
        self.balance = balance

        # Pre-group indices by class and group_id
        self.preictal_indices_grouped = defaultdict(list)
        self.interictal_indices = []

        for i in range(len(dataset)):
            if dataset.y[i] == 1:
                self.preictal_indices_grouped[dataset.group_ids[i]].append(i)
            else:
                self.interictal_indices.append(i)
        
        self.count_seen = {idx: 0 for idx in self.interictal_indices}

        self.build_bags()
    
    def build_bags(self):
        preictal_bags = []
        interictal_bags = []

        # --- Build preictal bags ---
        total_preictal_bags = 0
        for group_inds in self.preictal_indices_grouped.values():
            inds = np.array(group_inds)
            np.random.shuffle(inds)
            n_bags = len(inds) // self.bag_size
            if n_bags > 0:
                bags = np.array_split(inds[: n_bags * self.bag_size], n_bags)
                preictal_bags.extend([(b, 1) for b in bags])
                total_preictal_bags += n_bags

        # --- Build interictal bags (prefer unseen first) ---
        interictal_array = np.array(list(self.count_seen.keys()))
        seen_counts = np.array(list(self.count_seen.values()))

        # Prefer indices that have been used less often
        min_count = seen_counts.min()
        min_mask = seen_counts == min_count
        candidate_indices = interictal_array[min_mask]

        # Determine how many interictal samples we actually need
        # Initially match number of preictal bags if balance=True
        if self.balance:
            n_needed_samples = total_preictal_bags * self.bag_size
        else:
            n_needed_samples = len(self.interictal_indices)

        # Randomize within each usage level
        np.random.shuffle(candidate_indices)

        # If not enough unseen samples, fill with slightly more seen ones
        if len(candidate_indices) < n_needed_samples:
            remaining = n_needed_samples - len(candidate_indices)
            others = interictal_array[~min_mask]
            np.random.shuffle(others)
            selected = np.concatenate([candidate_indices, others[:remaining]])
        else:
            selected = candidate_indices[:n_needed_samples]

        # Update usage counter for selected indices
        for idx in selected:
            self.count_seen[idx] += 1

        # Split selected indices into bags
        n_bags_inter = len(selected) // self.bag_size
        if n_bags_inter > 0:
            bags = np.array_split(selected[: n_bags_inter * self.bag_size], n_bags_inter)
            interictal_bags.extend([(b, 0) for b in bags])

        # --- Combine & shuffle all bags ---
        self.all_bags = preictal_bags + interictal_bags
        if self.shuffle:
            np.random.shuffle(self.all_bags)

    def __iter__(self):
        self.build_bags()
        # --- Yield mini-batches of bags ---
        for i in range(0, len(self.all_bags), self.batch_size):
            batch = self.all_bags[i : i + self.batch_size]
            batch_data, batch_labels = [], []

            for bag_indices, bag_label in batch:
                bag_data, instance_labels = [], []
                for idx in bag_indices:
                    x, y = self.dataset[idx]
                    bag_data.append(x)
                    instance_labels.append(y)

                bag_data = torch.stack(bag_data)
                instance_labels = torch.tensor(instance_labels)

                if not torch.all(instance_labels == bag_label):
                    raise ValueError("Mixed labels in bag")

                batch_data.append(bag_data)
                batch_labels.append(bag_label)

            yield torch.stack(batch_data), torch.tensor(batch_labels)

    def __len__(self):
        return math.ceil(len(self.all_bags) / self.batch_size)


def leave_one_preictal_group_out(dataset, method="balanced", random_state=0):
    """
    Cross-validation splitter for seizure prediction datasets.

    In each fold:
      - One preictal group (seizure event) is held out for testing.
      - Remaining preictal groups are used for training.
      - Interictal (non-seizure) samples are partitioned among folds
        according to the selected method.

    Parameters
    ----------
    dataset : Dataset or Subset
        A dataset with attributes:
          - y : array-like of shape (n_samples,)
                Binary labels (1 = preictal, 0 = interictal).
          - group_ids : array-like of shape (n_samples,)
                Identifiers for preictal groups (all samples from the same
                seizure share the same ID).

    method : {"balanced", "balanced_shuffled"}, default="balanced"
        Strategy to assign interictal samples to folds:
          - "balanced":
              Interictal samples are split into N equal parts,
              where N = number of preictal groups.
              Each part is assigned to one group.
          - "balanced_shuffled":
              Same as "balanced", but interictal samples are shuffled
              randomly before splitting. Shuffling is controlled by
              `random_state` for reproducibility.

    random_state : int, default=0
        Random seed for reproducible shuffling (used only if
        method="balanced_shuffled").

    Yields
    ------
    (train_subset, test_subset) : tuple of SubsetWithInfo
        - train_subset: all preictal samples except the test group, plus
          interictal samples assigned to the remaining groups.
        - test_subset: preictal samples of the held-out group, plus
          interictal samples assigned to that group.

    Example
    -------
    >>> for train_set, test_set in leave_one_preictal_group_out(dataset, method="nearest"):
    ...     # train model on train_set
    ...     # evaluate on test_set
    """
    y, group_id = dataset.y, dataset.group_ids

    # Masks
    pre_mask = y == 1
    inter_mask = ~pre_mask

    # Unique preictal groups in order of appearance
    pre_groups = np.unique(group_id[pre_mask])

    # Indices
    inter_indices = np.where(inter_mask)[0]

    # split interictal indices
    inter_chunks = defaultdict(list)

    if method == "balanced" or method == "balanced_shuffled":
        n_splits = len(pre_groups)
        if method == "balanced_shuffled":
            rng = np.random.default_rng(seed=random_state)
            rng.shuffle(inter_indices)
        chunks = np.array_split(inter_indices, n_splits)
        for i, group in enumerate(pre_groups):
            inter_chunks[group] = chunks[i]

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
            [
                inter_chunks[g]
                for g in pre_groups
                if g != test_group and g in inter_chunks.keys()
            ]
        ).astype("int")

        train_idx = np.concatenate([pre_train_idx, inter_train_idx]).tolist()
        test_idx = np.concatenate([pre_test_idx, inter_test_idx]).tolist()

        yield SubsetWithInfo(dataset, train_idx), SubsetWithInfo(dataset, test_idx)

def split_into_strata(base_indices, group_ids, groups, N=5, M=3):
    group_splits = {}
    for gid in groups:
        inds = base_indices[group_ids == gid]
        inds = np.sort(inds)  # chronological order
        if len(inds) == 0:
            continue
        # Assign M consecutive samples to each fold in round-robin
        splits = [[] for _ in range(N)]
        i = 0
        while i < len(inds):
            for fold in range(N):
                if i + M <= len(inds):
                    splits[fold].extend(inds[i:i+M])
                else:
                    # last chunk: assign remaining samples evenly across folds
                    remaining = len(inds) - i
                    per_fold = remaining // (N - fold)
                    splits[fold].extend(inds[i:i+per_fold])
                    i += per_fold - 1  # -1 because will increment at end
                i += M

        # convert lists to arrays
        group_splits[gid] = [np.array(s) for s in splits]
    return group_splits


def cross_validation(dataset, shuffle=True, n_fold=5, random_state=0):
    """
    Custom cross-validation splitter:
      - Preictal groups are each split chronologically into n_fold strata.
        In fold i, one stratum per group becomes validation; the rest training.
      - Interictal samples are pooled, shuffled, and split globally into n_fold parts.
      - Works with both CHBMITDataset and SubsetWithInfo.
    """
    rng = np.random.RandomState(random_state)
    y = np.array(dataset.y)
    group_ids = np.array(dataset.group_ids)
    base_indices = np.arange(len(dataset))

    # Separate preictal / interictal
    pre_mask = y == 1
    inter_mask = ~pre_mask

    pre_groups = np.unique(group_ids[pre_mask])
    inter_groups = np.unique(group_ids[inter_mask])

    # For each interictal and preictal group, split chronologically into n_fold strata
    int_group_splits = split_into_strata(base_indices, group_ids, inter_groups, N=n_fold)
    pre_group_splits = split_into_strata(base_indices, group_ids, pre_groups, N=n_fold)

    # Generate folds
    for i_fold in range(n_fold):
        val_indices = []
        train_indices = []

        # Add interictal split for this fold
        for gid, splits in int_group_splits.items():
            val_indices.extend(splits[i_fold])
            train_indices.extend(np.concatenate([x for j, x in enumerate(splits) if j != i_fold]))

        # Add preictal splits group by group
        for gid, splits in pre_group_splits.items():
            val_indices.extend(splits[i_fold])
            train_indices.extend(np.concatenate([x for j, x in enumerate(splits) if j != i_fold]))

        # Convert to numpy arrays
        val_indices = np.array(val_indices)
        train_indices = np.array(train_indices)

        # Optionally shuffle train set (to randomize interictal order)
        if shuffle:
            train_indices = utils.shuffle(train_indices, random_state=rng)
            val_indices = utils.shuffle(val_indices, random_state=rng)

        # Yield the two SubsetWithInfo objects
        yield SubsetWithInfo(dataset, train_indices), SubsetWithInfo(dataset, val_indices)

def old_cross_validation(dataset, shuffle=False, n_fold=5, random_state=0):
    """
    Cross-validation splitter:
      - Splits dataset into n_fold stratified folds based on y.
      - Handles Dataset and Subset objects.
    """ 
    y = np.array(dataset.y)

    if not shuffle:
        random_state = None
    skf = StratifiedKFold(n_splits=n_fold, shuffle=shuffle, random_state=random_state)

    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
        yield SubsetWithInfo(dataset, train_idx), SubsetWithInfo(dataset, test_idx)

def make_cv_splitter(dataset, mode="stratified", **kwargs):
    """
    Wrapper for cross-validation splitters.

    Parameters
    ----------
    dataset : Dataset or Subset
        Input dataset.

    mode : {"stratified", "leave_one_preictal"}
        - "stratified": Standard stratified K-fold CV.
        - "leave_one_preictal": Leave-one-preictal-group-out CV.

    kwargs : dict
        Extra arguments passed to the underlying splitter:
        - For "stratified": shuffle, n_fold, random_state
        - For "leave_one_preictal": method, random_state

    Yields
    ------
    (train_subset, test_subset) : tuple of SubsetWithInfo
    """
    if mode == "stratified":
        return cross_validation(dataset, **kwargs)
    elif mode == "leave_one_preictal":
        return leave_one_preictal_group_out(dataset, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    dataset = CHBMITDataset(
        "data/BIDS_CHB-MIT",
        use_uint16=True,
        offline_transforms=[],
        online_transforms=[],
        suffix="zscore_F_T",
        subject_id="01",
    )

    train_val_dataset, test_dataset = next(
        make_cv_splitter(dataset, "leave_one_preictal",method = "balanced_shuffled")
    )
    train_dataset, val_dataset = next(
        make_cv_splitter(train_val_dataset, "stratified")
    )
    print("------------")
    print(dataset.__len__())
    print(
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        len(train_dataset) + len(val_dataset) + len(test_dataset),
    )
    print("------------")
    print(np.unique(dataset.group_ids[train_dataset.base_indices]))
    print(np.unique(dataset.group_ids[val_dataset.base_indices]))
    print(np.unique(dataset.group_ids[test_dataset.base_indices]))
    print("------------")
    print(np.unique(train_dataset.group_ids))
    print(np.unique(val_dataset.group_ids))
    print(np.unique(test_dataset.group_ids))
    print("------------")
    print(np.unique(train_dataset.group_ids[train_dataset.get_class_indices()[0]]))
    print(np.unique(val_dataset.group_ids[val_dataset.get_class_indices()[0]]))
    print(np.unique(test_dataset.group_ids[test_dataset.get_class_indices()[0]]))
    print("------------")
    print(train_dataset.group_ids[train_dataset.get_class_indices()[0]].__len__())
    dataloader = UnderSampledDataLoader(test_dataset)
    # dataloader = MilDataloader(train_dataset, batch_size=16, bag_size=16)
    print(dataloader.__len__())
    print(val_dataset.base_indices[0:30])
    for batch_data, batch_labels in iter(dataloader):
        print(batch_data.shape)
        print(batch_labels)
        break
    
    print("============ INNER-FOLD DISTRIBUTION CHECK ============")
    from collections import Counter

    # Recreate the inner folds iterator for reproducibility
    n_folds = 5  # or match your make_cv_splitter setting
    inner_folds = list(make_cv_splitter(train_val_dataset, "stratified", n_fold=n_folds))

    fold_stats = []
    for fold_idx, (train_idx, val_idx) in enumerate(inner_folds):
        train_samples = len(train_idx)
        val_samples = len(val_idx)

        train_preictal_idx = train_dataset.get_class_indices()[0]
        val_preictal_idx = val_dataset.get_class_indices()[0]

        n_train_preictal = len(train_preictal_idx)
        n_train_interictal = train_samples - n_train_preictal
        n_val_preictal = len(val_preictal_idx)
        n_val_interictal = val_samples - n_val_preictal

        print(f"\n--- Inner Fold {fold_idx+1}/{len(inner_folds)} ---")
        print(f"Train: {train_samples} samples  (preictal={n_train_preictal}, interictal={n_train_interictal})")
        print(f"Val:   {val_samples} samples  (preictal={n_val_preictal}, interictal={n_val_interictal})")

        # Unique groups
        train_groups = np.unique(train_dataset.group_ids[train_preictal_idx])
        val_groups = np.unique(val_dataset.group_ids[val_preictal_idx])
        print(f"Train preictal groups: {train_groups}")
        print(f"Val preictal groups:   {val_groups}")

        # Group counts table
        train_counts = Counter(train_dataset.group_ids[train_preictal_idx])
        val_counts = Counter(val_dataset.group_ids[val_preictal_idx])

        print("\nTrain preictal group counts:")
        for g in sorted(train_counts.keys()):
            print(f"  {g:<12}: {train_counts[g]}")

        print("Val preictal group counts:")
        for g in sorted(val_counts.keys()):
            print(f"  {g:<12}: {val_counts[g]}")

    print("\n============ SUMMARY ============")
    for i, (train_idx, val_idx) in enumerate(inner_folds):
        n_train_preictal = len(train_dataset.get_class_indices()[0])
        n_val_preictal = len(val_dataset.get_class_indices()[0])
        print(f"Fold {i}: train preictal={n_train_preictal}, val preictal={n_val_preictal}, train groups={len(train_groups)}, val groups={len(val_groups)}")