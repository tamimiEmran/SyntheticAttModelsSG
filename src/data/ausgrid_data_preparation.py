from sgcc_data_preparation import (
    DatasetGroup, assign_equal_attackTypes, DatasetSplit)
from preprocessing import preprocess_tsg
from typing import (
    Hashable, Sequence, Tuple, Optional)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Paths to the raw data files (HDF5 format)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
data_dir = r'data\processed'
ausgrid = r'ausgrid_attacked.h5'
ausgrid_attacked_path = os.path.join(root_dir, data_dir, ausgrid)

if not os.path.exists(ausgrid_attacked_path):
    raise FileNotFoundError(f"Path does not exist: {ausgrid_attacked_path}")

def load_dataframe(attack_type: str) -> pd.DataFrame:
    """
    Load the DataFrame for the specified attack type from the HDF5 file.
    """
    if attack_type == "original":
        return pd.read_hdf(ausgrid_attacked_path, key="original")
    else:
        return pd.read_hdf(ausgrid_attacked_path, key=attack_type)


# ---------------------------------------------------------------------------
# Balanced synthetic-vs-normal dataset, stratified on multi-class labels
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from typing import Sequence, Hashable, Tuple
import numpy as np
import pandas as pd

def ausgrid_set(
    fold: int,
    *,
    n_folds: int,
    attackTypes: Sequence[Hashable],
    train_attackTypes: Optional[Sequence[Hashable]] = None,
    test_attackTypes:  Optional[Sequence[Hashable]] = None,
    val_frac: float = 0.20,
    random_state_base: int = 42,
) -> DatasetGroup:
    """
    Cross-validated Ausgrid split.

    Parameters
    ----------
    fold : int
        1-based fold index that will serve as **test** split.
    n_folds : int
        Total number of folds (e.g. 3 ⇒ test_frac = 1/3 ≈ 0.333).
    attackTypes : list/hashable
        Default list of attack keys to use for both train/val and test
        unless overridden by the two *_attackTypes* kwargs.
    train_attackTypes, test_attackTypes : list/hashable | None
        If given, override the attack pool for the corresponding splits.
    val_frac : float
        Fraction of the **train + val** pool that becomes validation.
    """
    train_attackTypes = attackTypes if train_attackTypes is None else train_attackTypes
    test_attackTypes  = attackTypes if test_attackTypes  is None else test_attackTypes

    # ── load original dataframe ------------------------------------------
    df_orig = load_dataframe("original")
    ids     = np.asarray(df_orig.index, dtype=object)
    if ids.size == 0:
        raise RuntimeError("Ausgrid ‘original’ dataframe is empty.")

    # ── half → thieves, half → normals -----------------------------------
    rng = np.random.default_rng(random_state_base)
    rng.shuffle(ids)
    n_thieves  = ids.size // 2
    thief_ids  = ids[:n_thieves]
    normal_ids = ids[n_thieves:]

    # binary flag for stratification
    thief_flag = np.zeros(ids.size, dtype=int)
    thief_flag[:n_thieves] = 1

    # ── K-fold split ------------------------------------------------------
    if not (1 <= fold <= n_folds):
        raise ValueError(f"`fold` must be in 1…{n_folds}")

    skf = StratifiedKFold(
        n_splits   = n_folds,
        shuffle    = True,
        random_state = random_state_base,
    )
    for i, (trainval_idx, test_idx) in enumerate(skf.split(ids, thief_flag), 1):
        if i == fold:
            break  # found our test fold

    trainval_ids = ids[trainval_idx]
    test_ids     = ids[test_idx]

    trainval_thieves = np.intersect1d(trainval_ids, thief_ids, assume_unique=True)
    test_thieves     = np.intersect1d(test_ids,     thief_ids, assume_unique=True)

    # ── assign attack types separately to the two thief pools -------------
    att_train = assign_equal_attackTypes(
        trainval_thieves, train_attackTypes, seed=random_state_base + fold
    )
    att_test  = assign_equal_attackTypes(
        test_thieves,  test_attackTypes,  seed=random_state_base + fold + 99
    )
    att_series = pd.concat([att_train, att_test])

    # ── build combined dataframe -----------------------------------------
    need_types   = pd.unique(att_series).tolist()
    df_thieves   = pd.concat(
        [load_dataframe(att).loc[att_series.index[att_series.eq(att)]] for att in need_types],
        axis=0,
    )
    df_normals   = df_orig.loc[normal_ids]
    df_full      = pd.concat([df_thieves, df_normals], axis=0)

    # ── multi-class labels ("normal", "attack_<key>") ---------------------
    labels_full = pd.Series("normal", index=df_full.index, name="target", dtype=object)
    labels_full.update("attack_" + att_series.astype(str))

    # ── train ↔ val split inside the trainval pool ------------------------
    train_ids, val_ids = train_test_split(
        trainval_ids,
        test_size    = val_frac,
        random_state = random_state_base + fold,
        shuffle      = True,
        stratify     = labels_full[trainval_ids].values,
    )

    # ── helper: dataframe → (X, y) ---------------------------------------
    def _prep(idx: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        if len(idx) == 0:
            raise ValueError("Empty split.")

        df_part         = preprocess_tsg(df_full.loc[idx])
        consumer_labels = labels_full.loc[idx]

        arr     = df_part.to_numpy(dtype=np.float32, copy=False)
        n_days  = arr.shape[1] // 48
        if n_days == 0:
            raise RuntimeError("Need at least 48 half-hour slots per consumer.")

        arr = arr[:, : n_days * 48].reshape(arr.shape[0], n_days, 48)
        valid_mask = ~np.isnan(arr).any(axis=2)

        X_list, y_list = [], []
        for i, cid in enumerate(df_part.index):
            good = arr[i, valid_mask[i]]
            if good.size:
                X_list.append(good)
                y_list.append(np.full(len(good), consumer_labels[cid], dtype=object))

        if not X_list:
            raise RuntimeError("All days contained NaNs.")
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y

    X_train, y_train = _prep(train_ids)
    X_val,   y_val   = _prep(val_ids)
    X_test,  y_test  = _prep(test_ids)

    ds = DatasetGroup(
        train=DatasetSplit(X_train, y_train),
        val  = DatasetSplit(X_val,   y_val),
        test = DatasetSplit(X_test,  y_test),
    ).shuffled(seed = random_state_base + fold + 42)
    return ds

if __name__ == "__main__":
    print("Ausgrid Data Preparation")
    # Example usage
    fold = 1
    attack_types = []
    train_attack_types = ["attack_1", "attack_2", "attack_3", "attack_4", "attack_5"]
    test_attack_types = ["attack_6", "attack_7", "attack_8", "attack_9", "attack_10"]
    dataset = ausgrid_set(fold, attack_types, train_attackTypes=train_attack_types, test_attackTypes=test_attack_types)

    # print the counts and proportions of each label in the train, val, and test sets
    print("Train label counts:", pd.Series(dataset.train.y).value_counts(normalize=True))
    print("Val label counts:", pd.Series(dataset.val.y).value_counts(normalize=True))
    print("Test label counts:", pd.Series(dataset.test.y).value_counts(normalize=True))
