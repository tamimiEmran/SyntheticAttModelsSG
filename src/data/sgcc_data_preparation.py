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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils import shuffle as sk_shuffle

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ATTACK_TYPES_DEFAULT: Sequence[Union[int, str]] = (*range(13), "ieee")
LABELS_FILE = Path("labels.h5")  # HDF5 with a single key "df"
ZERO_THRESHOLD = 10               # max zero-readings allowed per consumer-month
TOL = 1e-6                        # tolerance used in np.isclose(…, 0.0)

# ---------------------------------------------------------------------------
# Small typed containers – for safer plumbing
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
class DatasetGroup:
    """train / validation / test bundle"""

    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def filter_zero_rows(arr: np.ndarray, max_zeros: int, tol: float = TOL) -> np.ndarray:
    """Return a view of *arr* that keeps only rows with ≤ *max_zeros* zeros."""
    zero_counts = np.sum(np.isclose(arr, 0.0, atol=tol), axis=1)
    return arr[zero_counts <= max_zeros]


# ---------------------------------------------------------------------------
# Index helpers (fold handling)
# ---------------------------------------------------------------------------

def ten_fold_indices(columns: Sequence, *, cache_file: str | Path | None = "10folds.pkl") -> list[Tuple[list, list]]:
    """Create (or load) a 10-fold split of *columns*.

    Each element i in the returned list is a tuple ``(train_cols, test_cols)``
    suitable for fold-i cross-validation.
    """
    cache_path = Path(cache_file) if cache_file is not None else None
    if cache_path and cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    cols = np.array(columns)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(cols)
    folds = np.array_split(cols, 10)

    result: list[Tuple[list, list]] = []
    for i in range(10):
        test_cols = list(folds[i])
        train_cols = list(np.concatenate([folds[j] for j in range(10) if j != i]))
        result.append((train_cols, test_cols))

    if cache_path:
        cache_path.write_bytes(pickle.dumps(result))
    return result


def get_fold_data(fold_id: int) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """Return three *index* lists for the chosen fold.

    • real-positive IDs  (label == 1) — never manipulated
    • synthetic IDs      — candidates for *attack injection*
    • real-negative IDs  — honest consumers (label == 0)
    """
    if not 1 <= fold_id <= 10:
        raise ValueError("fold_id must be in 1…10, inclusive.")

    train_ids, test_ids = utils.fold(fold_id)  # type: ignore – external util
    # utils.fold returns *two* ID lists; we treat the first as "synthetic pool"
    synthetic_pool = pd.Index(train_ids)
    real_negative_pool = pd.Index(test_ids)

    labels_ser: pd.Series = pd.read_hdf(LABELS_FILE, key="df")
    real_positive_pool = labels_ser[labels_ser == 1].index

    return real_positive_pool, synthetic_pool, real_negative_pool


# ---------------------------------------------------------------------------
# Synthetic attack generation
# ---------------------------------------------------------------------------

def build_attack_examples(
    ids: Sequence,
    attack_types: Sequence[Union[int, str]],
    *,
    zero_threshold: int = ZERO_THRESHOLD,
) -> np.ndarray:
    """Return a feature matrix **X** where each row is an attack-augmented
    consumer-month example.

    Each consumer ID is paired with exactly **one** attack type (round-robin
    assignment achieved with ``np.array_split``).
    """
    if len(ids) == 0:
        return np.empty((0, 0))

    id_chunks = np.array_split(np.array(ids), len(attack_types))
    matrices: list[np.ndarray] = []

    for chunk, atk in zip(id_chunks, attack_types):
        df_atk = load_dataframe(atk)  # attack-specific traces
        df_chunk = df_atk.loc[chunk]
        X_chunk = examplesFromDF(df_chunk)
        X_chunk = filter_zero_rows(X_chunk, zero_threshold)
        matrices.append(X_chunk)

    return np.vstack(matrices) if matrices else np.empty((0, 0))


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def build_dataset_for_fold(
    fold_id: int,
    *,
    attack_types: Sequence[Union[int, str]] = ATTACK_TYPES_DEFAULT,
    zero_threshold: int = ZERO_THRESHOLD,
    shuffle_seed: int | None = None,
) -> Tuple[DatasetGroup, DatasetGroup]:
    """Return ``synthetic_dataset`` and ``real_dataset`` for the requested fold.

    Each dataset is a :class:`DatasetGroup` holding three :class:`DatasetSplit`s
    (train, val, test).  The **synthetic** variant contains attack-augmented
    positives; the **real** variant uses genuine thief consumers as positives.
    """
    rp_ids, synt_ids, rn_ids = get_fold_data(fold_id)

    df_clean = load_dataframe(-1)  # unmodified raw meter readings

    # ------------------------------------------------------------------
    # Helper to turn (pos_ids, neg_ids) → DatasetSplit, while respecting
    # the train/val/test triple structure returned by a hidden util
    # ------------------------------------------------------------------
    def _make_group(
        pos_splits: Tuple[Sequence, Sequence, Sequence],
        neg_splits: Tuple[Sequence, Sequence, Sequence],
        pos_builder: callable,
        neg_builder: callable,
    ) -> DatasetGroup:
        out_splits: list[DatasetSplit] = []
        for pos_ids, neg_ids in zip(pos_splits, neg_splits):
            X_pos = pos_builder(pos_ids)
            X_neg = neg_builder(neg_ids)
            X = np.vstack((X_pos, X_neg))
            y = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_neg))))
            out_splits.append(DatasetSplit(X, y).shuffled(shuffle_seed))
        return DatasetGroup(*out_splits)

    # ids already partitioned by some hidden util (train/val/test)
    realP_ids, realN_ids, synthP_ids, synthN_ids = tr_val_tst_DF_indices(
        fold_id, attack_types=None  # type: ignore – external util
    )

    # --- builders ------------------------------------------------------
    examples_from_clean = lambda ids: filter_zero_rows(
        examplesFromDF(df_clean.loc[ids]), zero_threshold
    )

    examples_from_attack = lambda ids: build_attack_examples(
        ids, attack_types, zero_threshold=zero_threshold
    )

    # Synthetic dataset: positives = ATTACK, negatives = clean traces
    synthetic_dataset = _make_group(
        synthP_ids,
        synthN_ids,
        examples_from_attack,
        examples_from_clean,
    )

    # Real dataset: positives = real thieves, negatives = clean honest users
    real_dataset = _make_group(
        realP_ids,
        realN_ids,
        examples_from_clean,
        examples_from_clean,
    )

    return synthetic_dataset, real_dataset


# ---------------------------------------------------------------------------
# Convenience wrapper for external code
# ---------------------------------------------------------------------------

def load_attack_data(
    fold_id: int,
    dataset_type: str,
    *,
    attack_types: Sequence[Union[int, str]] = ATTACK_TYPES_DEFAULT,
    zero_threshold: int = ZERO_THRESHOLD,
    shuffle_seed: int | None = None,
) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    """Public function mirroring the original signature.

    Parameters
    ----------
    fold_id : int
        1 … 10
    dataset_type : {'real', 'synthetic'}
        Which flavour to return.
    attack_types : sequence
        Passed straight to :func:`build_dataset_for_fold`.
    zero_threshold : int
        Filter threshold for zero readings (per consumer-month row).

    Returns
    -------
    (train, val, test) : tuple[DatasetSplit, DatasetSplit, DatasetSplit]
    """
    synthetic_ds, real_ds = build_dataset_for_fold(
        fold_id,
        attack_types=attack_types,
        zero_threshold=zero_threshold,
        shuffle_seed=shuffle_seed,
    )

    dataset_type_lower = dataset_type.lower()
    if dataset_type_lower == "synthetic":
        chosen = synthetic_ds
    elif dataset_type_lower == "real":
        chosen = real_ds
    else:  # pragma: no cover – guardrail
        raise ValueError("dataset_type must be 'synthetic' or 'real'.")

    return chosen.train, chosen.val, chosen.test


# ---------------------------------------------------------------------------
# EOF
# ---------------------------------------------------------------------------
