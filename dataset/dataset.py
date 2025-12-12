import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import pandas as pd
from typing import Callable, List, Literal
import glob
import os
import math
from utils import invert_uint16_scaling
from tqdm import tqdm
from collections import defaultdict
from sklearn import utils
from collections import Counter


class CHBMITDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str = "data/BIDS_CHB-MIT",
        use_uint16: bool = False,
        subject_id: str = "01",
        online_transforms: List[Callable] = None,
        offline_transforms: List[Callable] = None,
        suffix: str = "fd_5s_szx5_prex5",
        task: Literal["prediction", "detection"] = "prediction",
        print_events: bool = True,
    ):
        """
        This class does three things:
        1. Loads the processed BIDS_CHB-MIT data and its metadata from .npz files for the specified subject
        2. Determine the binary labels and which samples are used for training
        3. Applies offline and online transforms to the data

        metadata fields:
        "event_id", "label", "epoch_index_within_event", "global_epoch_id"
        "n_segments_in_event", "start_time_in_event", "augmented", "pp_mean", "pp_max"
        "sd_mean", "sd_max", "onset_sec", "duration_sec"

        Args:
            dataset_dir (string): path to the processed BIDS_CHB-MIT dataset
            use_uint16 (boolean): if true uses uint16 format
            subject_id (str): use "*" to include all subjects,
        """
        # Task settings
        if task == "detection":
            target_label = "seizure"
            background_labels_training = [
                "interictal",
                "preictal",
                "post_buffer",
                "pre_buffer",
            ]
        else:
            target_label = "preictal"
            background_labels_training = ["interictal"]

        # Subject directory
        subject_dir = os.path.join(dataset_dir, f"sub-{subject_id}/")
        print(f"Loading data for subject: {subject_id} from {subject_dir}")
        if not os.path.isdir(subject_dir):
            raise ValueError(f"Subject directory not found: {subject_dir}")

        # Find the NPZ session files
        suffix_pattern = (
            f"*{suffix}_uint16.npz" if use_uint16 else f"*{suffix}_float.npz"
        )
        search_pattern = os.path.join(subject_dir, "ses-*", "eeg", suffix_pattern)
        ses_paths = glob.glob(search_pattern)
        print(f"Found {len(ses_paths)} session files using pattern: {search_pattern}")

        if len(ses_paths) == 0:
            raise ValueError(
                f"No processed NPZ files found for subject_id={subject_id}"
            )

        all_X, all_y, all_group_ids, all_metadata = [], [], [], []

        # Load each session
        for i, ses_path in enumerate(ses_paths):
            data = np.load(ses_path, allow_pickle=True)

            # Load X
            X_temp = data["X"]
            if use_uint16:
                scales = data["scales"]
                X_temp = invert_uint16_scaling(X_temp, scales)
            all_X.append(X_temp)

            # Load labels
            all_y.append(data["y"])

            # Load metadata
            meta_obj = data["meta_df"]
            meta_dict = meta_obj.item() if hasattr(meta_obj, "item") else dict(meta_obj)
            meta_df = pd.DataFrame(meta_dict)

            # event_id is used as a group label
            event_ids = [f"{v}_{i}" for v in meta_df["event_id"].values]
            all_group_ids.append(event_ids)

            # store list of dicts
            all_metadata.extend(meta_df.to_dict(orient="records"))

        # Concatenate sessions
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        group_ids = np.concatenate(all_group_ids, axis=0)

        if print_events:
            counter = Counter(group_ids.tolist())
            print("\nData samples per event (group_id):")
            for g in sorted(counter.keys()):
                print(f"  {g:<12}: {counter[g]}")
            print("")

        if len(y) == 0:
            raise ValueError(
                f"No data samples found for subject_id={subject_id} in {dataset_dir}"
            )

        # Prepare tensors and labels
        self.X = X.astype(np.float32)

        # Convert textual labels to 0/1
        self.y = np.array(
            [1 if label == target_label else 0 for label in y],
            dtype=np.int64,
        )

        self.group_ids = group_ids
        self.metadata = all_metadata

        # Which samples are used in training
        allowed = {target_label} | set(background_labels_training)
        self.is_used_in_train = np.array([label in allowed for label in y])

        # Offline transforms (applied once)
        self.online_transform = online_transforms or []
        for transform in offline_transforms or []:
            transformed = []
            for i in tqdm(range(self.X.shape[0]), desc="Applying offline transforms"):
                transformed.append(transform(eeg=self.X[i]))
            self.X = np.stack(transformed, axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        meta = self.metadata[idx]

        for transform in self.online_transform:
            x = transform(x)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            meta,
        )

    def get_class_indices(self):
        """Return indices for each class"""
        target_indices = np.where(self.y == 1)[0]
        baseline_indices = np.where(self.y == 0)[0]
        return target_indices, baseline_indices


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
        self.metadata = [self.base_dataset.metadata[i] for i in self.base_indices]

    def get_class_indices(self):
        """Return indices for each class within this subset"""
        target_indices = np.where(self.y == 1)[0]
        baseline_indices = np.where(self.y == 0)[0]
        return target_indices, baseline_indices


class UnderSampledDataLoader:
    """Custom DataLoader that undersamples interictal data each epoch"""

    def __init__(self, dataset: SubsetWithInfo, batch_size=32, shuffle=True, random_state=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(random_state)

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
                selected_extra = self.rng.choice(
                    not_min_indices, size=remaining, replace=False
                )
                selected_interictal = np.concatenate([min_indices, selected_extra])
            else:
                selected_interictal = self.rng.choice(
                    min_indices, size=n_preictal, replace=False
                )
        else:
            selected_interictal = self.interictal_indices

        for idx in selected_interictal:
            self.count_seen[idx] += 1

        all_indices = np.concatenate([self.preictal_indices, selected_interictal])
        if self.shuffle:
            self.rng.shuffle(all_indices)
        return all_indices

    def __iter__(self):
        self.all_indices = self.get_indices()

        # Create batches
        for i in range(0, len(self.all_indices), self.batch_size):
            batch_indices = self.all_indices[i : i + self.batch_size]
            batch_data = []
            batch_labels = []
            batch_metas = []
            for idx in batch_indices:
                data, label, meta = self.dataset[idx]
                batch_data.append(data)
                batch_labels.append(label)
                batch_metas.append(meta)

            yield torch.stack(batch_data), torch.stack(batch_labels), batch_metas

    def __len__(self):
        return (len(self.all_indices) + self.batch_size - 1) // self.batch_size


class MilDataloader:
    """DataLoader that builds random MIL bags each epoch."""

    def __init__(
        self,
        dataset: SubsetWithInfo,
        batch_size=32,
        shuffle=True,
        bag_size=8,
        balance=True,
        random_state=0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bag_size = bag_size
        self.balance = balance
        self.rng = np.random.RandomState(random_state)

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
            self.rng.shuffle(inds)
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
        self.rng.shuffle(candidate_indices)

        # If not enough unseen samples, fill with slightly more seen ones
        if len(candidate_indices) < n_needed_samples:
            remaining = n_needed_samples - len(candidate_indices)
            others = interictal_array[~min_mask]
            self.rng.shuffle(others)
            selected = np.concatenate([candidate_indices, others[:remaining]])
        else:
            selected = candidate_indices[:n_needed_samples]

        # Update usage counter for selected indices
        for idx in selected:
            self.count_seen[idx] += 1

        # Split selected indices into bags
        n_bags_inter = len(selected) // self.bag_size
        if n_bags_inter > 0:
            bags = np.array_split(
                selected[: n_bags_inter * self.bag_size], n_bags_inter
            )
            interictal_bags.extend([(b, 0) for b in bags])

        # --- Combine & shuffle all bags ---
        self.all_bags = preictal_bags + interictal_bags
        if self.shuffle:
            self.rng.shuffle(self.all_bags)

    def __iter__(self):
        self.build_bags()
        # --- Yield mini-batches of bags ---
        for i in range(0, len(self.all_bags), self.batch_size):
            batch = self.all_bags[i : i + self.batch_size]
            batch_data, batch_labels, batch_metas = [], [], []

            for bag_indices, bag_label in batch:
                bag_data, instance_labels, bag_metas = [], [], []
                for idx in bag_indices:
                    x, y, meta = self.dataset[idx]
                    bag_data.append(x)
                    instance_labels.append(y)
                    bag_metas.append(meta)

                bag_data = torch.stack(bag_data)
                instance_labels = torch.tensor(instance_labels, dtype=torch.long)

                if not torch.all(instance_labels == bag_label):
                    raise ValueError("Mixed labels in bag")

                batch_data.append(bag_data)
                batch_labels.append(bag_label)
                batch_metas.append(bag_metas)

            yield torch.stack(batch_data), torch.tensor(batch_labels), batch_metas

    def __len__(self):
        return math.ceil(len(self.all_bags) / self.batch_size)


def leave_one_out(dataset: CHBMITDataset, shuffle=False, random_state=0):
    """
    Cross-validation splitter.
    This function does three things:
    1. Select samples that are included in training (based on dataset.is_used_in_train)
    2. Identifies preictal groups (or seizure events) using group_ids.
    3. Creates folds by leaving one preictal group (seizure event) out for testing.

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
                Binary labels (1 = preictal (or seizure), 0 = interictal (or non-seizure)).
          - group_ids : array-like of shape (n_samples,)
                Identifiers for preictal groups (all samples from the same
                seizure share the same ID).

    shuffle : bool, default=False
        Whether to shuffle baseline samples before splitting

    random_state : int, default=0
        Random seed for reproducible shuffling (used only if
        method="balanced_shuffled").

    Yields
    ------
    (train_subset, test_subset) : tuple of SubsetWithInfo
        - train_subset: all preictal (or seziure) samples except the test group, plus
          interictal (or non-seizure) samples assigned to the remaining groups.
        - test_subset: preictal (or seziure) samples of the held-out group, plus
          interictal (or non-seizure) samples assigned to that group.

    Example
    -------
    >>> for train_set, test_set in leave_one_preictal_group_out(dataset, shuffle=True, random_state=0):
    ...     # train model on train_set
    ...     # evaluate on test_set
    """
    # Select samples used for training
    if hasattr(dataset, "is_used_in_train"):
        train_mask = dataset.is_used_in_train
        dataset = SubsetWithInfo(dataset, np.where(train_mask)[0])

    y, group_id = dataset.y, dataset.group_ids

    # Masks
    pre_mask = y == 1
    inter_mask = ~pre_mask

    # Unique preictal groups in order of appearance
    pre_groups = np.unique(group_id[pre_mask])

    # Indices
    inter_indices = np.where(inter_mask)[0]

    # split interictal indices
    if shuffle:
        # It is highly to not shuffle interictal (or non-seizure) samples
        # to preserve temporal structure. Since in the test phase
        # where moving windows are used, using shuffled interictal samples
        # cause some problems.
        # it either breaks the temporal structure of interictal samples
        # or causes data leakage since some interictal samples from training
        # appear in the window.
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(inter_indices)

    n_splits = len(pre_groups)
    chunks = np.array_split(inter_indices, n_splits)

    inter_chunks = {
        g: np.asarray(chunks[i], dtype=int) for i, g in enumerate(pre_groups)
    }

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
            [inter_chunks[g] for g in pre_groups if g != test_group]
        ).astype("int")

        train_idx = np.concatenate([pre_train_idx, inter_train_idx]).astype("int")
        test_idx = np.concatenate([pre_test_idx, inter_test_idx]).astype("int")

        yield SubsetWithInfo(dataset, train_idx), SubsetWithInfo(dataset, test_idx)


def split_into_strata(indices, N=5, M=10):
    """
    This function splits the given indices into N strata.

    Parameters
    ----------
    indices : array-like
        The indices to be split into strata.
    N : int
        The number of strata to create.
    M : int
        How many samples are assigned to each stratum in each iteration.

    Returns
    -------
    splits : list of np.ndarray
        A list containing N arrays, each representing a stratum with assigned indices.

    Notes
    -----
    It is better to use larger M to reduce the effect of temporal correlation and
    also when the samples have overlapping windows to reduce the data leakage.
    """

    max_M = math.ceil(len(indices) / (N + 1))
    if M > max_M:
        raise ValueError(
            f"M={M} is too large for event length={len(indices)} and N={N} folds. "
            f"Maximum valid M is {max_M}."
        )

    indices = np.sort(indices)
    if len(indices) == 0:
        return [np.array([]) for _ in range(N)]

    splits = [[] for _ in range(N)]
    i = 0

    while i < len(indices):
        for fold in range(N):
            if i >= len(indices):
                break
            end = min(i + M, len(indices))
            splits[fold].extend(indices[i:end])
            i = end

    return [np.array(s) for s in splits if len(s) > 0]


def KFold(
    dataset,
    shuffle=True,
    n_fold=5,
    random_state=0,
    mode: Literal["random_split", "split" , "strata", "per_event_strata"] = "per_event_strata",
    M=10,
):
    """
    Custom cross-validation splitter:
      - Preictal groups are each split chronologically into n_fold strata.
        In fold i, one stratum per group becomes validation; the rest training.
      - Interictal samples are pooled, shuffled, and split globally into n_fold parts.
      - Works with both CHBMITDataset and SubsetWithInfo.
    Parameters
    ----------
    dataset : Dataset or Subset
        Input dataset.
    shuffle : bool, default=True
        Whether to shuffle training and validation indices before yielding.
    n_fold : int, default=5
        Number of folds.
    random_state : int, default=0
        Random seed for reproducible shuffling.
    mode : {"random_split", "split, "strata", "per_event_strata"}, default="per_event_strata"
        - "random_split": Randomly split each class into n_fold
        - "split": Split each class into n_fold chronologically
        - "strata": Split each class into n_fold strata chronologically
        - "per_event_strata": Split each class one-by-one per event into n_fold strata chronologically
    M : int, default=10
        Number of samples that are assigned to each stratum in each iteration.

    Yields
    ------
    (train_subset, val_subset) : tuple of SubsetWithInfo
        - train_subset: training samples for the fold.
        - val_subset: validation samples for the fold.

    Notes
    -----
    - Using larger M helps reduce temporal correlation and data leakage
      when samples have overlapping windows.
    - If some events have few samples, the per_event_strata mode may not work as intended.
    """
    rng = np.random.RandomState(random_state)

    y = np.array(dataset.y)
    group_ids = np.array(dataset.group_ids)
    base_indices = np.arange(len(dataset))

    # ------------------------------------
    # Separate classes
    # ------------------------------------
    class0 = base_indices[y == 0]
    class1 = base_indices[y == 1]

    # Unique seizure groups
    seizure_groups = np.unique(group_ids[y == 1])
    background_groups = np.unique(group_ids[y == 0])

    # ---------- MODE 1: RANDOM ----------
    if mode == "random_split" or mode == "split":
        if mode == "random_split":
            rng.shuffle(class0)
            rng.shuffle(class1)

        cls0_splits = np.array_split(class0, n_fold)
        cls1_splits = np.array_split(class1, n_fold)

    # ---------- MODE 2: STRATA ----------
    elif mode == "strata":
        cls0_splits = split_into_strata(class0, N=n_fold, M=M)
        cls1_splits = split_into_strata(class1, N=n_fold, M=M)

    # ---------- MODE 3: PER-EVENT STRATA ----------
    elif mode == "per_event_strata":
        # Split background events one-by-one
        bg_splits = []
        for gid in background_groups:
            inds = base_indices[(group_ids == gid) & (y == 0)]
            bg_splits.append(split_into_strata(inds, N=n_fold, M=M))

        # Combine background stratification across events
        combined_bg_splits = []
        for k in range(n_fold):
            combined_bg_splits.append(np.concatenate([event[k] for event in bg_splits]))

        # Split seizure events one-by-one
        sz_splits = []
        for gid in seizure_groups:
            inds = base_indices[(group_ids == gid) & (y == 1)]
            sz_splits.append(split_into_strata(inds, N=n_fold, M=M))

        # Combine seizure stratification across events
        combined_sz_splits = []
        for k in range(n_fold):
            combined_sz_splits.append(np.concatenate([event[k] for event in sz_splits]))

        cls0_splits = combined_bg_splits
        cls1_splits = combined_sz_splits

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Use random, strata, per_event_strata."
        )

    for k in range(n_fold):
        val_idx = np.concatenate([cls0_splits[k], cls1_splits[k]])
        train_idx = np.concatenate(
            [
                np.concatenate([cls0_splits[i] for i in range(n_fold) if i != k]),
                np.concatenate([cls1_splits[i] for i in range(n_fold) if i != k]),
            ]
        )

        if shuffle:
            train_idx = utils.shuffle(train_idx, random_state=rng)
            val_idx = utils.shuffle(val_idx, random_state=rng)

        yield (SubsetWithInfo(dataset, train_idx), SubsetWithInfo(dataset, val_idx))


def make_cv_splitter(dataset, method="LOO", **kwargs):
    """
    Wrapper for cross-validation splitters.

    This function does two things:
    1. Depending on the mode, it selects the appropriate CV splitter.
    2. It returns the samples not used for training as a separate subset if it exists.

    Parameters
    ----------
    dataset : Dataset or Subset
        Input dataset.

    method : {"KFold", "LOO"}
        - "KFold": Stratified/Random K-Fold CV.
        - "LOO": Leave-One-Out CV.

    kwargs : dict
        Extra arguments passed to the underlying splitter:
        - For "KFold": shuffle, n_fold, random_state.
        - For "LOO": method, random_state.

    Yields
    ------
    (train_subset, test_subset) : tuple of SubsetWithInfo
    """
    not_used = get_not_used_subset(dataset)

    # Select only samples that are allowed to be used for training
    if hasattr(dataset, "is_used_in_train"):
        used_idx = np.where(dataset.is_used_in_train)[0]
        cv_dataset = SubsetWithInfo(dataset, used_idx)
    else:
        cv_dataset = dataset  # fallback

    if method == "KFold":
        splits = KFold(cv_dataset, **kwargs)
    elif method == "LOO":
        splits = leave_one_out(cv_dataset, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    return splits, not_used


def get_not_used_subset(dataset):
    if hasattr(dataset, "is_used_in_train"):
        not_used_mask = ~dataset.is_used_in_train
        not_used_indices = np.where(not_used_mask)[0]
        if len(not_used_indices) > 0:
            return SubsetWithInfo(dataset, not_used_indices)
    return None


if __name__ == "__main__":
    # print("\n=== TEST: PREDICTION TASK ===")
    # ds_pred = CHBMITDataset(
    #     dataset_dir="data/BIDS_CHB-MIT",
    #     use_uint16=False,
    #     offline_transforms=[],
    #     online_transforms=[],
    #     suffix="fd_5s_szx5_prex5",
    #     subject_id="01",
    #     task="prediction",
    # )

    # print(f"Total samples: {len(ds_pred)}")
    # print(f"X shape: {ds_pred.X.shape}")
    # print(f"y distribution: {np.bincount(ds_pred.y)}  (0=interictal, 1=preictal)")
    # print(f"Number used for training: {ds_pred.is_used_in_train.sum()}")

    # print("\n--- Showing 3 random prediction samples ---")
    # for i in np.random.choice(len(ds_pred), 3, replace=False):
    #     x, y, meta = ds_pred[i]
    #     print(f"\nSample {i}:")
    #     print(f"  Label: {y.item()}")
    #     print(f"  X shape: {x.shape}")
    #     print(f"  Event ID: {meta['event_id']}")
    #     print(f"  Epoch index in event: {meta['epoch_index_within_event']}")
    #     print(
    #         f"  Onset sec: {meta.get('onset_sec')}, Duration: {meta.get('duration_sec')}"
    #     )
    #     print(f"  metadata values: {meta.values()}")

    print("\n\n=== TEST: DETECTION TASK ===")
    ds_det = CHBMITDataset(
        dataset_dir="data/BIDS_CHB-MIT",
        use_uint16=False,
        offline_transforms=[],
        online_transforms=[],
        suffix="fd_5s_szx5_prex5",
        subject_id="01",
        task="detection",
    )

    print(f"Total samples: {len(ds_det)}")
    print(f"X shape: {ds_det.X.shape}")
    print(f"y distribution: {np.bincount(ds_det.y)}  (0=non-seizure, 1=seizure)")
    print(f"Number used for training: {ds_det.is_used_in_train.sum()}")

    print("\n--- Showing 3 random detection samples ---")
    for i in np.random.choice(len(ds_det), 3, replace=False):
        x, y, meta = ds_det[i]
        print(f"\nSample {i}:")
        print(f"  Label: {y.item()}")
        print(f"  X shape: {x.shape}")
        print(f"  Event ID: {meta['event_id']}")
        print(f"  Epoch index in event: {meta['epoch_index_within_event']}")
        print(
            f"  Onset sec: {meta.get('onset_sec')}, Duration: {meta.get('duration_sec')}"
        )
        print(f"  metadata values: {meta.values()}")

    dataset = ds_det
    mode = "strata"
    train_val_dataset, test_dataset = next(
        make_cv_splitter(dataset, "LOO", shuffle=True, random_state=0)[0]
    )
    train_dataset, val_dataset = next(
        make_cv_splitter(train_val_dataset, "KFold", n_fold=5, mode=mode)[0]
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
    for batch_data, batch_labels, meta in iter(dataloader):
        print("------------start")
        print(batch_data.shape)
        print(batch_labels)
        print(len(meta))
        print(meta[0])
        print("------------end")
        break

    print("============ INNER-FOLD DISTRIBUTION CHECK ============")
    from collections import Counter

    n_folds = 5  # match your CV setting
    mode = "strata"  # or "random", "per_event_strata"

    # Get the generator for inner folds
    inner_folds = make_cv_splitter(train_val_dataset, "KFold", n_fold=n_folds, mode=mode)[0]

    # Iterate over each fold
    for fold_idx, (train_subset, val_subset) in enumerate(inner_folds):
        # Total samples
        train_samples = len(train_subset)
        val_samples = len(val_subset)

        # Preictal and interictal indices for this fold
        train_preictal_idx, train_inter_idx = train_subset.get_class_indices()
        val_preictal_idx, val_inter_idx = val_subset.get_class_indices()

        n_train_preictal = len(train_preictal_idx)
        n_train_interictal = len(train_inter_idx)
        n_val_preictal = len(val_preictal_idx)
        n_val_interictal = len(val_inter_idx)

        print(f"\n--- Inner Fold {fold_idx + 1}/{n_folds} ---")
        print(f"Train: {train_samples} samples  (preictal={n_train_preictal}, interictal={n_train_interictal})")
        print(f"Val:   {val_samples} samples  (preictal={n_val_preictal}, interictal={n_val_interictal})")

        # Unique preictal groups in this fold
        train_groups = np.unique(train_subset.group_ids[train_preictal_idx])
        val_groups = np.unique(val_subset.group_ids[val_preictal_idx])
        print(f"Train preictal groups: {train_groups}")
        print(f"Val preictal groups:   {val_groups}")

        # Preictal group counts
        train_counts = Counter(train_subset.group_ids[train_preictal_idx])
        val_counts = Counter(val_subset.group_ids[val_preictal_idx])

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
        print(
            f"Fold {i}: train preictal={n_train_preictal}, val preictal={n_val_preictal}, train groups={len(train_groups)}, val groups={len(val_groups)}"
        )
