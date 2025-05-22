from __future__ import annotations
# -*- coding: utf-8 -*-
"""data_pipeline.py

End-to-end utilities that generate **real** and **synthetic** train/validation/test
splits for the energy-theft detection project.  All heavy numerical work
(number-crunching, attack injection, feature extraction) lives in **three** small
helpers; everything else is bookkeeping and I/O.

External dependencies expected to exist in the runtime:
    • load_dataframe(attack_type: int | str) -> pd.DataFrame
    • examplesFromDF(df: pd.DataFrame)          -> np.ndarray
    • utils.fold(k)  # returns the (train_ids, test_ids) pair for fold *k*

If those functions live in other modules, simply adjust the import section.
"""



from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union, Hashable

import numpy as np
import pandas as pd
from sklearn.utils import shuffle as sk_shuffle
import pickle
import os
import numpy as np
import pandas as pd
from .preprocessing import create_monthly_examples, preprocess_tsg, oversample
from sklearn.model_selection import train_test_split
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Paths to the raw data files (HDF5 format)
current_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

data_dir = r'data\processed'
sgcc_labels = r"original_labels.h5"
sgcc = r"sgcc_attacked.h5"
sgcc_attacked_path = os.path.join(root_dir, data_dir, sgcc)
sgcc_labels_path = os.path.join(root_dir, data_dir, sgcc_labels)

# test if the paths exist
for path in [sgcc_labels_path, sgcc_attacked_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")



ATTACK_TYPES_DEFAULT: Sequence[str] = [f"attack_{i}" for i in range(13) ] + ["attack_ieee"]
ZERO_THRESHOLD = 10               # max zero-readings allowed per consumer-month
TOL = 1e-6                        # tolerance used in np.isclose(…, 0.0)

# ---------------------------------------------------------------------------
# Small typed containers - for safer plumbing
# ---------------------------------------------------------------------------
@dataclass
class DatasetSplit:
    """A single (X, y) split."""

    X: np.ndarray  # shape: (n_examples, n_features)
    y: np.ndarray  # shape: (n_examples,)

    def shuffled(self, seed: int | None = None) -> "DatasetSplit":
        """Return a *new* split with rows shuffled in unison."""
        X_shuf, y_shuf = sk_shuffle(self.X, self.y, random_state=seed)
        return DatasetSplit(X_shuf, y_shuf)

@dataclass
class dataframeGroup:
    """train / validation / test bundle"""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
@dataclass
class labelsGroup:
    """train / validation / test bundle"""

    train: pd.Series
    val: pd.Series
    test: pd.Series


@dataclass
class DatasetGroup:
    """train / validation / test bundle"""

    train: DatasetSplit = None 
    val: DatasetSplit = None
    test: DatasetSplit = None
    dataframe: dataframeGroup = None
    labels: labelsGroup = None

    def shuffled(self, seed: int | None = None) -> "DatasetGroup":
        return DatasetGroup(
            train=self.train.shuffled(seed),
            val=self.val.shuffled(seed + 1),
            test=self.test.shuffled(seed + 2),
            dataframe=self.dataframe,
            labels=self.labels,
        )
    
    def combined(self) -> DatasetSplit:
        """Return a single DatasetSplit with all data combined."""
        return DatasetSplit(
            X=np.concatenate([self.train.X, self.val.X, self.test.X], axis=0),
            y=np.concatenate([self.train.y, self.val.y, self.test.y], axis=0),
        )


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _listHDFkeys(path: str | Path) -> list[str]:
    """Return a list of all keys in the HDF5 file at *path*."""
    with pd.HDFStore(path) as store:
        keys = store.keys()
    return [key[1:] for key in keys]  # strip leading '/'

def filter_zero_rows(arr: np.ndarray, max_zeros: int, tol: float = TOL) -> np.ndarray:
    """Return a view of *arr* that keeps only rows with ≤ *max_zeros* zeros."""
    zero_counts = np.sum(np.isclose(arr, 0.0, atol=tol), axis=1)
    return arr[zero_counts <= max_zeros]

def load_dataframe(attack_type: int | str) -> pd.DataFrame:
    """Load the attack-specific DataFrame from the HDF5 file.

    Parameters
    ----------
    attack_type : int | str
        Attack type ID (0-12) or "ieee" for the IEEE attack.

    Returns
    -------
    pd.DataFrame
        DataFrame with time series consumption data (time as columns, consumers as rows).
    """
    #sometimes the attack_type is given as a string starting with "attack_"
    if isinstance(attack_type, str) and attack_type.startswith("attack_"):
        return pd.read_hdf(sgcc_attacked_path, key=attack_type)


    if isinstance(attack_type, int):
        key = f"attack_{attack_type}"
    
    elif isinstance(attack_type, str) and attack_type.lower() == "original":
        key = "original"
    elif isinstance(attack_type, str):
        if (attack_type.lower() == "ieee") or (attack_type == "attack_ieee"):
            key = "attack_ieee"
        elif int(attack_type):
            key = f"attack_{attack_type}"
    else:
        raise ValueError("attack_type must be an int (0-12) or 'ieee'.")

    df = pd.read_hdf(sgcc_attacked_path, key=key)
    return df

# ---------------------------------------------------------------------------
# Index helpers (fold handling)
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Union, Optional

def make_cv_label_dataframe(
    labels: pd.Series,
    *,
    n_folds: int = 10,
    thief_value: Union[str, int, bool] = 1,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame that assigns every consumer to one of four textual
    categories for each cross-validation fold:

    * realDataset-thieves         - the true thieves (value == `thief_value`)
      • appears in **all** folds for those indices
    * realDataset-normal          - non-thieves that serve as the *test* set
      • varies by fold
    * syntheticDataset-thieves    - 50 % of the *training* set (random split)
    * syntheticDataset-normal     - the other 50 % of that training set

    Parameters
    ----------
    labels : pd.Series
        Index = consumer IDs (strings).  
        Values = original class labels; value equal to `thief_value`
        marks a real thief.
    n_folds : int, default 10
        Number of K-Fold splits to create (≥ 2).
    thief_value : str | int | bool, default 1
        The label that denotes a real thief.
    random_state : int | None, default None
        Seed used for reproducible shuffles in both
        * K-Fold generation and
        * the 50 / 50 training split inside each fold.

    Returns
    -------
    pd.DataFrame
        Index  : consumer IDs (same order as input `labels`).
        Columns: "fold 1", …, "fold n".
        Cell values are the four category strings above.
    """
    # --- basic preparation --------------------------------------------------
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    labels = labels.copy()
    thieves_mask = labels == thief_value
    thief_ids = labels.index[thieves_mask]
    normal_ids = labels.index[~thieves_mask]

    # set up the empty output frame
    col_names = [f"fold {i}" for i in range(1, n_folds + 1)]
    out = pd.DataFrame(index=labels.index, columns=col_names, dtype=object)

    # every real thief is flagged in every fold column
    out.loc[thief_ids, :] = "realDataset-thieves"

    # KFold only on normals
    kf = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )
    normal_ids = np.array(normal_ids)

    # loop over folds
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(normal_ids), start=1):
        col = f"fold {fold_idx}"

        # --- test split ------------------------------------------------------
        test_ids = normal_ids[test_idx]
        out.loc[test_ids, col] = "realDataset-normal"

        # --- training split (to be halved) -----------------------------------
        train_ids = normal_ids[train_idx]
        rng = np.random.default_rng(
            None if random_state is None else random_state + fold_idx
        )
        rng.shuffle(train_ids)

        half = len(train_ids) // 2
        # Handle odd counts: randomly assign the extra sample
        if len(train_ids) % 2 == 1:
            if rng.random() < 0.5:
                half += 1  # extra goes to synthetic thieves
        synth_thief_ids = train_ids[:half]
        synth_normal_ids = train_ids[half:]

        out.loc[synth_thief_ids, col] = "syntheticDataset-thieves"
        out.loc[synth_normal_ids, col] = "syntheticDataset-normal"

    return out




def print_fold_label_counts(df: pd.DataFrame, *,
                            sort_index: bool = True) -> pd.DataFrame:
    counts = df.apply(lambda col: col.value_counts()).fillna(0).astype(int)

    if sort_index:
        counts = counts.sort_index()

    print(counts.to_string())
    return counts

def assign_equal_attackTypes(
    index: pd.Index | Sequence[str],
    attackTypes: Sequence[Hashable],
    *,
    seed: int = 42,
) -> pd.Series:
    """
    Assigns every consumer in `index` one attack-type so that the counts of
    each attack-type are as equal as mathematically possible.

    Parameters
    ----------
    index
        Iterable of consumer IDs (e.g. `df.index`).
    attackTypes
        Iterable of labels to distribute (strings, ints, Enums …).
    seed
        Seed for the reproducible shuffle of consumer IDs.

    Returns
    -------
    pd.Series
        Index = original consumer IDs, values = assigned attack-type.
    """
    # --------------------------- validation ---------------------------------
    if len(attackTypes) == 0:
        raise ValueError("`attackTypes` must contain at least one element")
    if len(index) == 0:
        return pd.Series([], index=pd.Index([], name="consumer_id"), dtype=object)

    # ---------------- reproducible shuffle of IDs ---------------------------
    ids = np.asarray(index, dtype=object).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    # ------------------- equal-as-possible allocation -----------------------
    k = len(attackTypes)
    base  = len(ids) // k          # guaranteed minimum per attack-type
    extra = len(ids) %  k          # how many attack-types get one extra

    # counts[i] = how many consumers will receive attackTypes[i]
    counts = np.full(k, base, dtype=int)

    if extra > 0:
        extra_attackTypes = rng.choice(k, size=extra, replace=False)
        counts[extra_attackTypes] += 1     # bump the chosen types once

    # ------------------- build assignments ----------------------------------
    assigned = np.empty_like(ids, dtype=object)
    start = 0
    for attType, n in zip(attackTypes, counts):
        assigned[start : start + n] = attType
        start += n

    # `ids` is shuffled, so pair & then restore original ordering
    attType_series = pd.Series(assigned, index=ids, name="attType")
    attType_series = attType_series.reindex(index)

    return attType_series
# ---------------------------------------------------------------------------
# to make a real set
# load the dataframe.
# preprocess it with the function `preprocess_tsg`. and create the monthly examples.
# do not oversample the data.

def real_set(
    cv: pd.DataFrame,
    fold: int,
    *,
    test_frac: float = 0.30,
    val_frac: float = 0.20,
    random_state_base: int = 42,
) -> DatasetGroup:
    """
    Parameters
    ----------
    cv : pd.DataFrame
        Output of `make_cv_label_dataframe`.
    fold : int
        1-based fold index (column "fold {fold}" is used to select consumers).
    test_frac : float, default 0.20
        Fraction of *real* consumers reserved for the final **test** split.
    val_frac : float, default 0.20
        Fraction of the **training** consumers (after test removal) reserved
        for **validation**.
    random_state_base : int, default 42
        Base seed; the effective seed is `random_state_base + fold`.

    Returns
    -------
    DatasetGroup
        With `train`, `val`, and `test` attributes of type `DatasetSplit`.
    """



    fold_col = cv[f"fold {fold}"]
    real_ids = fold_col.index[fold_col.str.startswith("realDataset")]

    df_raw   = load_dataframe("original").loc[real_ids]  
    labels_s = pd.read_hdf(sgcc_labels_path, key="original").loc[real_ids]

    if real_ids.empty:
        raise ValueError(f"No real-dataset consumers found for fold {fold}")

    seed = random_state_base + fold  


    stratify_labels = labels_s[real_ids].values

    train_ids, test_ids = train_test_split(
        real_ids,
        test_size=test_frac,
        random_state=seed,
        shuffle=True,
        stratify=stratify_labels 
    )
    stratify_labels_train = labels_s[train_ids].values
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=val_frac,
        random_state=seed,  
        shuffle=True,
        stratify=stratify_labels_train
    )


    def _prep(ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        if len(ids) == 0:
            raise ValueError("Split is empty — check the split fractions.")
        df_part = preprocess_tsg(df_raw.loc[ids])
        X, y = create_monthly_examples(
            df_part,
            labels_s.loc[ids],
            days_per_example=31
        )
        return X, y

    X_train, y_train = _prep(train_ids)
    X_val,   y_val   = _prep(val_ids)
    X_test,  y_test  = _prep(test_ids)

    return DatasetGroup(
        train=DatasetSplit(X_train, y_train),
        val  = DatasetSplit(X_val,   y_val),
        test = DatasetSplit(X_test,  y_test),
    )
def combine_DatasetGroup(dataset: DatasetGroup) -> DatasetSplit:
    X_combined = np.concatenate([dataset.train.X, dataset.val.X, dataset.test.X], axis=0)
    y_combined = np.concatenate([dataset.train.y, dataset.val.y, dataset.test.y], axis=0)
    return DatasetSplit(X_combined, y_combined)

def combine_DatasetGroup_leave_for_threshold(dataset: DatasetGroup) -> DatasetGroup:
    X_combined_no_val = np.concatenate([dataset.train.X, dataset.test.X], axis=0)
    y_combined_no_val = np.concatenate([dataset.train.y, dataset.test.y], axis=0)

    x_val, y_val = dataset.val.X, dataset.val.y

    return DatasetGroup(
        train= None,
        test =DatasetSplit(X = X_combined_no_val, y = y_combined_no_val),
        val = DatasetSplit(X = x_val, y = y_val)
    )



# to make a synthetic set
# load the dataframe with the attack types.
# preprocess it with the function `preprocess_tsg`. and create the monthly examples.
# do not oversample the data.
def synthetic_set(
    cv: pd.DataFrame,
    fold: int,
    attackTypes: Sequence[Hashable] = ATTACK_TYPES_DEFAULT,
    *,
    test_frac: float = 0.30,
    val_frac: float = 0.20,
    random_state_base: int = 42,
) -> DatasetGroup:
    """
    Parameters
    ----------
    cv : pd.DataFrame
        Output of `make_cv_label_dataframe`.
    fold : int
        1-based fold index (column "fold {fold}" is used to select consumers).
    test_frac : float, default 0.30
        Fraction of *real* consumers reserved for the final **test** split.
    val_frac : float, default 0.20
        Fraction of the **training** consumers (after test removal) reserved
        for **validation**.
    random_state_base : int, default 42
        Base seed; the effective seed is `random_state_base + fold`.

    Returns
    -------
    DatasetGroup
        With `train`, `val`, and `test` attributes of type `DatasetSplit`.
    """
    fold_col = cv[f"fold {fold}"]
    synth_ids = fold_col.index[fold_col.str.startswith("syntheticDataset")]

    df_raw   = load_dataframe("original").loc[synth_ids]  
    

    if synth_ids.empty:
        raise ValueError(f"No synthetic-dataset consumers found for fold {fold}")

    synth_ids_thieves = fold_col.index[fold_col.eq("syntheticDataset-thieves")]
    
    attType_series = assign_equal_attackTypes(synth_ids_thieves, attackTypes)
    # example output:
    """
    CONS_NO
    1ECF28CAF36C19132B0673378B2B8AA3    attack_10
    B40585F95C419203FF46350D2360B630    attack_11
    B6E8F3547A23BF3F899ABEF08398EEF5     attack_1
    590251F1A3B8283C482F26331E3AB694     attack_9
    61E50BE6D7EDAF382C8E4EA3E0B490A8     attack_6
                                        ...    
    5FCE503B6B616D5023B3C46173383107    attack_12
    F3C8BBCD2DC26C1E0249DEEF6A4256B7     attack_1
    A9A0FE83467A680FBFB0DBFC910DF227     attack_3
    D9A6ADA018FA46A55D5438370456AA45     attack_6
    F3406636BAD1E6E0826E8EDDC9A1BF00     attack_7
    Name: attType, Length: 32757, dtype: object
        """

    attackType_dataframes = {
        attType: load_dataframe(f"{attType}").loc[attType_series.index[attType_series.eq(attType)]]
        for attType in attackTypes
    }

    # make all of them a single dataframe
    df_synthetic_attacks = pd.concat(attackType_dataframes.values(), axis=0)

    #each row is a thief so labels are all 1
    labels_synthetic_thieves = pd.Series(
        1, index=df_synthetic_attacks.index, name="FLAG", dtype=int
    )

    synth_ids_normals = fold_col.index[fold_col.eq("syntheticDataset-normal")]
    df_synthetic_normal = load_dataframe("original").loc[synth_ids_normals]
    labels_synthetic_normal = pd.Series(
        0, index=df_synthetic_normal.index, name="FLAG", dtype=int
    )

    
    df_synthetic = pd.concat([df_synthetic_attacks, df_synthetic_normal], axis=0)
    labels_synthetic = pd.concat([labels_synthetic_thieves, labels_synthetic_normal], axis=0)

    assert len(df_synthetic) == len(labels_synthetic), "Dataframe and labels length mismatch"
    assert df_synthetic.index.equals(labels_synthetic.index), "indices of dataframe and labels mismatch"

    seed = random_state_base + fold  

    stratify_labels = labels_synthetic.values
    train_ids, test_ids = train_test_split(
        df_synthetic.index,
        test_size=test_frac,
        random_state=seed,
        shuffle=True,
        stratify=stratify_labels 
    )
    stratify_labels_train = labels_synthetic[train_ids].values
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=val_frac,
        random_state= seed + 1,  
        shuffle=True,
        stratify=stratify_labels_train
    )

    def _prep(ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        if len(ids) == 0:
            raise ValueError("Split is empty — check the split fractions.")
        df_part = preprocess_tsg(df_synthetic.loc[ids])
        X, y = create_monthly_examples(
            df_part,
            labels_synthetic.loc[ids],
            days_per_example=31
        )
        return X, y
    

    X_train, y_train = _prep(train_ids)
    X_val,   y_val   = _prep(val_ids)
    X_test,  y_test  = _prep(test_ids)

    return DatasetGroup(
        train=DatasetSplit(X_train, y_train),
        val  = DatasetSplit(X_val,   y_val),
        test = DatasetSplit(X_test,  y_test),
    )


# experiment 1 dataset.
# load the whole dataframe as is. 
# preprocess it with the function `preprocess_tsg`.
# oversample the data.

def sgcc_wholeConsumer(
    test_frac: float = 0.30,
    val_frac: float = 0.20,
    to_oversample: bool = True   ) -> DatasetGroup:
    """ 

    Parameters
    fold : int
    """
    dataframe = load_dataframe("original")
    labels = pd.read_hdf(sgcc_labels_path, key='original')

   
    # sklearn train_test_split = train_test_split(

    train, test = train_test_split(
        dataframe.index,
        test_size=test_frac,
        stratify= labels.values,
        random_state=42
    )


    
    train_df = preprocess_tsg(df = dataframe.loc[train])
    test_df = preprocess_tsg(df = dataframe.loc[test])
    if to_oversample:
        X_train, y_train = oversample(
            x = train_df.to_numpy(),
            y = labels.loc[train].to_numpy()
        )
    
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train, y_train,
    test_size=val_frac,
    stratify= y_train,
    random_state=43
    )

    X_val_final,   y_val_final = pd.DataFrame(X_val_final), y_val_final

    X_test, y_test = test_df, labels.loc[test].to_numpy()

    return DatasetGroup(
        train=DatasetSplit(X_train_final, y_train_final),
        val=DatasetSplit(X_val_final,   y_val_final),
        test=DatasetSplit(X_test,  y_test),
        )


# experiment 1B dataset.
# load the whole dataframe as is.
# preprocess it with the function `preprocess_tsg`.
# create the monthly examples.
# oversample the data.
def sgcc_monthly(
    test_frac: float = 0.30,
    val_frac: float = 0.20,
    to_oversample: bool = True   ) -> DatasetGroup:
    """ 

    Parameters
    fold : int
    """
    dataframe = load_dataframe("original")
    labels = pd.read_hdf(sgcc_labels_path, key='original')

   
    # sklearn train_test_split = train_test_split(

    train, test = train_test_split(
        dataframe.index,
        test_size=test_frac,
        stratify= labels.values,
        random_state=42
    )

    train , val = train_test_split(
        train,
        test_size=val_frac,
        stratify= labels.loc[train].values,
        random_state=42
    )
    
    train_df = preprocess_tsg(df = dataframe.loc[train])
    val_df = preprocess_tsg(df = dataframe.loc[val])
    test_df = preprocess_tsg(df = dataframe.loc[test]) # no cross contamination between train and test. Preprocessing is user based.

    dfsGroup = dataframeGroup(
        train = train_df,
        val = val_df,
        test = test_df
    )

    x_train, y_train = create_monthly_examples(
        df = train_df,
        labels = labels.loc[train]
    )
    x_val, y_val = create_monthly_examples(
        df = val_df,
        labels = labels.loc[val]
    )
    x_test, y_test = create_monthly_examples(
        df = test_df,
        labels = labels.loc[test]
    )

    if to_oversample:
        X_train, y_train = oversample(
            x = x_train,
            y = y_train
        )

    return   DatasetGroup(
        train=DatasetSplit(X_train, y_train),
        val=DatasetSplit(x_val,   y_val),
        test=DatasetSplit(x_test,  y_test),
        dataframe = dfsGroup,
        labels = labelsGroup(
            train = labels.loc[train],
            val = labels.loc[val],
            test = labels.loc[test]
        )
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage of the functions in this module
    fold_id = 1
    sgcc_labels = pd.read_hdf(sgcc_labels_path, key='original')
    cv = make_cv_label_dataframe(sgcc_labels, n_folds=10, random_state=42)

    # test the real_set function

    group = real_set(cv, fold_id, test_frac=0.3, val_frac=0.2)
    print("Real set:")
    print(f"Train shape: {group.train.X.shape}, Val shape: {group.val.X.shape}, Test shape: {group.test.X.shape}")
    print(f"Train labels shape: {group.train.y.shape}, Val labels shape: {group.val.y.shape}, Test labels shape: {group.test.y.shape}")
    print("Train labels counts:")
    print(pd.Series(group.train.y).value_counts())
    print("Val labels counts:")
    print(pd.Series(group.val.y).value_counts())
    print("Test labels counts:")
    print(pd.Series(group.test.y).value_counts())

    syn_group = synthetic_set(cv, fold_id, test_frac=0.3, val_frac=0.2)
    print("Synthetic set:")
    print(f"Train shape: {syn_group.train.X.shape}, Val shape: {syn_group.val.X.shape}, Test shape: {syn_group.test.X.shape}")
    print(f"Train labels shape: {syn_group.train.y.shape}, Val labels shape: {syn_group.val.y.shape}, Test labels shape: {syn_group.test.y.shape}")

    print("Train labels counts:")
    print(pd.Series(syn_group.train.y).value_counts())
    print("Val labels counts:")
    print(pd.Series(syn_group.val.y).value_counts())
    print("Test labels counts:")
    print(pd.Series(syn_group.test.y).value_counts())

    # test the sgcc_wholeConsumer function
    whole_group = sgcc_wholeConsumer(test_frac=0.3, val_frac=0.2, to_oversample=True)
    print("SGCC Whole Consumer set:")
    print(f"Train shape: {whole_group.train.X.shape}, Val shape: {whole_group.val.X.shape}, Test shape: {whole_group.test.X.shape}")
    print(f"Train labels shape: {whole_group.train.y.shape}, Val labels shape: {whole_group.val.y.shape}, Test labels shape: {whole_group.test.y.shape}")
    print("Train labels counts:")
    print(pd.Series(whole_group.train.y).value_counts())
    print("Val labels counts:")
    print(pd.Series(whole_group.val.y).value_counts())
    print("Test labels counts:")
    print(pd.Series(whole_group.test.y).value_counts())


    #test the sgcc_monthly function
    monthly_group = sgcc_monthly(test_frac=0.3, val_frac=0.2, to_oversample=True)
    print("SGCC Monthly set:")
    print(f"Train shape: {monthly_group.train.X.shape}, Val shape: {monthly_group.val.X.shape}, Test shape: {monthly_group.test.X.shape}")
    print(f"Train labels shape: {monthly_group.train.y.shape}, Val labels shape: {monthly_group.val.y.shape}, Test labels shape: {monthly_group.test.y.shape}")
    print("Train labels counts:")
    print(pd.Series(monthly_group.train.y).value_counts())
    print("Val labels counts:")
    print(pd.Series(monthly_group.val.y).value_counts())
    print("Test labels counts:")
    print(pd.Series(monthly_group.test.y).value_counts())
