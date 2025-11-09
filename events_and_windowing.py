from constants import set_seeds, configure_tensorflow
set_seeds()
configure_tensorflow()
import pandas as pd
import numpy as np
import random
import glob
import os
from itertools import permutations, islice
from math import factorial, floor
from typing import List, Dict, Tuple, Optional, Sequence
import random
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import re


# count number of beep_schedules_folders:
def _natural_key(path_or_name: str):
    """
    Split digits so names like 'beep_schedules_Claire10' sort after '...Claire2'.
    Works with full paths or basenames.
    """
    name = os.path.basename(os.fspath(path_or_name)).lower()
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', name)]

def folders_tot(type, list_comb):
    if type == 'train':
        endings = tuple(list_comb[:-2])  # All except last two
    elif type == 'val':
        nr = list_comb[-2]
        endings = (nr,)
    elif type == 'test':
        nr = list_comb[-1]
        endings = (nr,)
    else:
        raise ValueError("Invalid type. Choose 'train', 'val', or 'test'.")

    all_matching_folders = []

    # Iterate through each required ending (e.g., '0' and '1' for 'train')
    for ending in endings:
        # Build the pattern using an f-string to insert the current ending
        # Example: 'beep_schedules_*0'
        DATA_ROOT = os.path.join(os.getcwd(), "data")
        pattern = os.path.join(DATA_ROOT, f'beep_schedules_{ending}*')
        
        # Find all paths matching the specific pattern
        current_matches = sorted(glob.glob(pattern), key=_natural_key)
        
        # Filter to ensure the match is actually a directory (folder)
        folder_paths = [path for path in current_matches if os.path.isdir(path)]
        
        # Add the valid folders to the main list
        all_matching_folders.extend(folder_paths)

    num_beep_schedules_folders = len(all_matching_folders)
    
    return all_matching_folders, num_beep_schedules_folders

def std_mean(mean, std):
    for k in std.keys():
        var = std[k] - mean[k] * mean[k]
        std[k] = float(np.sqrt(var if var > 0 else 0.0))
    print("mean stats")
    for k, v in mean.items():
        print(f"  {k}: {v:.6f}")
    print("std stats")
    for k, v in std.items():
        print(f"  {k}: {v:.6f}")

def count_all_zero_windows(X):
    """
    X: np.ndarray of shape (N, 400, 14)
    Prints how many windows X[s, :, :] are entirely zeros.
    Returns (count, indices_array).
    """
    zero_mask = np.all(X == 0, axis=(1, 2))  # True where whole window is zeros
    count = int(zero_mask.sum())
    idxs = np.flatnonzero(zero_mask)
    print(f"{count} all-zero windows out of {X.shape[0]}")
    if count:
        print("indices:", idxs.tolist())
    return count, idxs

def count_labels(y, labels=(0, 1, 2), verbose=True):
    y = np.asarray(y).ravel().astype(int)
    total = y.size
    counts = {lbl: 0 for lbl in labels}
    vals, cnts = np.unique(y, return_counts=True)
    for v, c in zip(vals, cnts):
        if v in counts:
            counts[v] = int(c)

    if verbose:
        for lbl in labels:
            n = counts[lbl]
            pct = (n / total * 100) if total else 0.0
            print(f"label {lbl}: {n} ({pct:.1f}%)")
        print(f"total: {total}")
    return counts

def remove_columns(df_imu, columns, save_path=None, case_insensitive=False):
    """
    Remove specified columns from a DataFrame. Missing columns are ignored.
    If save_path is given, the updated DataFrame is saved to that CSV (overwrites).

    Args:
        df_imu (pd.DataFrame): The IMU dataframe.
        columns (Iterable[str]): Column names to remove, e.g. ['pitch_rad', 'acc_x'].
        save_path (str | pathlib.Path | None): If provided, write CSV to this path.
        case_insensitive (bool): If True, match names ignoring case and surrounding spaces.

    Returns:
        pd.DataFrame: The same DataFrame object with columns removed.
    """
    import pandas as pd

    if not columns:
        if save_path:
            df_imu.to_csv(save_path, index=False)
        return df_imu

    # Normalize input
    cols_wanted = [str(c).strip() for c in columns]

    if case_insensitive:
        lookup = {c.strip().lower(): c for c in df_imu.columns}
        to_drop = [lookup[c.lower()] for c in cols_wanted if c.lower() in lookup]
    else:
        to_drop = [c for c in cols_wanted if c in df_imu.columns]

    # Drop if any match; ignore missing
    if to_drop:
        df_imu.drop(columns=to_drop, inplace=True)

    if save_path is not None:
        df_imu.to_csv(save_path, index=False)

    return df_imu

def add_pitch_to_df(df_imu, out_col="pitch_rad",
                    quat_cols=("quat_x","quat_y","quat_z","quat_w")):
    """
    Add a robust head/torso pitch estimate in radians.

    Strategy:
      1) If grav_x/grav_y/grav_z exist, compute pitch from gravity:
           pitch = atan2(-grav_x, sqrt(grav_y^2 + grav_z^2))
         This is scale-invariant and avoids gimbal-lock artifacts.
      2) Otherwise, compute pitch from quaternion (x,y,z,w) **after row-wise normalization**:
           sinp = 2*(w*y - z*x)
           pitch = arcsin(clip(sinp, -1, 1))

    The column is inserted after 'grav_z' if present, else after 'quat_w', else at the end.
    The function modifies df_imu in place and returns it for convenience.
    """
    import numpy as np

    # ---- Path A: prefer gravity when available (robust, scale-invariant) ----
    if {"grav_x", "grav_y", "grav_z"}.issubset(df_imu.columns):
        gx = df_imu["grav_x"].to_numpy(dtype=np.float64, copy=False)
        gy = df_imu["grav_y"].to_numpy(dtype=np.float64, copy=False)
        gz = df_imu["grav_z"].to_numpy(dtype=np.float64, copy=False)

        # Pitch: rotation about device's left-right axis; negative is forward flexion if gx>0
        pitch = np.arctan2(-gx, np.sqrt(gy*gy + gz*gz))

    # ---- Path B: fall back to quaternion (normalize first) -------------------
    elif all(c in df_imu.columns for c in quat_cols):
        x = df_imu[quat_cols[0]].to_numpy(dtype=np.float64, copy=False)
        y = df_imu[quat_cols[1]].to_numpy(dtype=np.float64, copy=False)
        z = df_imu[quat_cols[2]].to_numpy(dtype=np.float64, copy=False)
        w = df_imu[quat_cols[3]].to_numpy(dtype=np.float64, copy=False)

        # Row-wise normalization to avoid arcsin saturation from non-unit quats
        qn = np.sqrt(x*x + y*y + z*z + w*w)
        good = qn > 1e-12
        x = np.where(good, x / qn, np.nan)
        y = np.where(good, y / qn, np.nan)
        z = np.where(good, z / qn, np.nan)
        w = np.where(good, w / qn, np.nan)

        # Tait–Bryan: Z-Y-X (yaw-pitch-roll), pitch about Y
        sinp = 2.0 * (w*y - z*x)
        sinp = np.clip(sinp, -1.0, 1.0)  # numerical safety
        pitch = np.arcsin(sinp)

    else:
        # Neither gravity nor the specified quaternion columns are present: no-op
        return df_imu

    # ---- Insert/replace column in a stable location -------------------------
    if out_col in df_imu.columns:
        df_imu.drop(columns=[out_col], inplace=True)

    cols = list(df_imu.columns)
    if "grav_z" in cols:
        insert_at = cols.index("grav_z") + 1
    elif "quat_w" in cols:
        insert_at = cols.index("quat_w") + 1
    else:
        insert_at = len(cols)

    df_imu.insert(insert_at, out_col, pitch)
    return df_imu

def add_roll_to_df(df_imu, out_col="roll_rad",
                   quat_cols=("quat_x","quat_y","quat_z","quat_w")):
    """
    Add a robust head/torso roll estimate in radians.

    Strategy (mirrors pitch):
      1) If grav_x/grav_y/grav_z exist, compute roll from gravity (scale-invariant):
           roll = atan2(grav_y, grav_z)
      2) Otherwise, compute roll from quaternion (x,y,z,w) **after row-wise normalization**:
           t0 = 2*(w*x + y*z)
           t1 = 1 - 2*(x^2 + y^2)
           roll = atan2(t0, t1)

    The angle is wrapped to (-pi, pi].
    Column is inserted after 'grav_z' if present, else after 'quat_w', else at end.
    Modifies df_imu in place and returns it.
    """
    import numpy as np

    def wrap_to_pi(a):
        return (a + np.pi) % (2.0*np.pi) - np.pi

    # ---- Path A: prefer gravity (drift-free, scale-invariant) ----
    if {"grav_x", "grav_y", "grav_z"}.issubset(df_imu.columns):
        gy = df_imu["grav_y"].to_numpy(dtype=np.float64, copy=False)
        gz = df_imu["grav_z"].to_numpy(dtype=np.float64, copy=False)
        roll = np.arctan2(gy, gz)

    # ---- Path B: normalized quaternion fallback ----
    elif all(c in df_imu.columns for c in quat_cols):
        x = df_imu[quat_cols[0]].to_numpy(dtype=np.float64, copy=False)
        y = df_imu[quat_cols[1]].to_numpy(dtype=np.float64, copy=False)
        z = df_imu[quat_cols[2]].to_numpy(dtype=np.float64, copy=False)
        w = df_imu[quat_cols[3]].to_numpy(dtype=np.float64, copy=False)

        qn = np.sqrt(x*x + y*y + z*z + w*w)
        good = qn > 1e-12
        if not good.any():
            return df_imu
        x = np.where(good, x/qn, np.nan)
        y = np.where(good, y/qn, np.nan)
        z = np.where(good, z/qn, np.nan)
        w = np.where(good, w/qn, np.nan)

        t0 = 2.0 * (w*x + y*z)
        t1 = 1.0 - 2.0 * (x*x + y*y)
        roll = np.arctan2(t0, t1)

    else:
        return df_imu  # neither gravity nor quaternion columns available

    roll = wrap_to_pi(roll)

    # ---- Insert/replace column in a stable location ----
    if out_col in df_imu.columns:
        df_imu.drop(columns=[out_col], inplace=True)

    cols = list(df_imu.columns)
    if "grav_z" in cols:
        insert_at = cols.index("grav_z") + 1
    elif "quat_w" in cols:
        insert_at = cols.index("quat_w") + 1
    else:
        insert_at = len(cols)

    df_imu.insert(insert_at, out_col, roll)
    return df_imu

def add_yaw_to_df(df_imu, out_col="yaw_rad",
                  quat_cols=("quat_x","quat_y","quat_z","quat_w")):
    """
    Add yaw (heading) estimate in radians.

    Strategy (analogous style to pitch fallback):
      - Compute yaw from quaternion (x,y,z,w) **after row-wise normalization**.
        Gravity alone cannot provide yaw.
        Z-Y-X (yaw-pitch-roll) conversion:
           t3 = 2*(w*z + x*y)
           t4 = 1 - 2*(y^2 + z^2)
           yaw = atan2(t3, t4)

    The angle is wrapped to (-pi, pi].
    Column is inserted after 'grav_z' if present, else after 'quat_w', else at end.
    Modifies df_imu in place and returns it.
    """
    import numpy as np

    def wrap_to_pi(a):
        return (a + np.pi) % (2.0*np.pi) - np.pi

    if not all(c in df_imu.columns for c in quat_cols):
        return df_imu  # cannot compute yaw without quaternions

    x = df_imu[quat_cols[0]].to_numpy(dtype=np.float64, copy=False)
    y = df_imu[quat_cols[1]].to_numpy(dtype=np.float64, copy=False)
    z = df_imu[quat_cols[2]].to_numpy(dtype=np.float64, copy=False)
    w = df_imu[quat_cols[3]].to_numpy(dtype=np.float64, copy=False)

    qn = np.sqrt(x*x + y*y + z*z + w*w)
    good = qn > 1e-12
    if not good.any():
        return df_imu
    x = np.where(good, x/qn, np.nan)
    y = np.where(good, y/qn, np.nan)
    z = np.where(good, z/qn, np.nan)
    w = np.where(good, w/qn, np.nan)

    t3 = 2.0 * (w*z + x*y)
    t4 = 1.0 - 2.0 * (y*y + z*z)
    yaw = np.arctan2(t3, t4)
    yaw = wrap_to_pi(yaw)

    # ---- Insert/replace column in a stable location ----
    if out_col in df_imu.columns:
        df_imu.drop(columns=[out_col], inplace=True)

    cols = list(df_imu.columns)
    if "grav_z" in cols:
        insert_at = cols.index("grav_z") + 1
    elif "quat_w" in cols:
        insert_at = cols.index("quat_w") + 1
    else:
        insert_at = len(cols)

    df_imu.insert(insert_at, out_col, yaw)
    return df_imu

def find_combinations(names: List[str], fraction: float = 1.0) -> Tuple[Dict[int, List[str]],
                                                                        Dict[str, float],
                                                                        Dict[str, float]]:
    """
    Build permutations where:
      - First n-2 items are the TRAIN split (order canonicalized to the original 'names' order)
      - (n-1)th is VAL
      - nth is TEST

    Canonicalizing the training portion ensures that once a particular set of n-2 names has been
    used in the first n-2 positions, it will not appear again in another order.

    Universe size: n * (n - 1)  (all ordered (val, test) pairs), rather than n!.
    'fraction' is applied to this universe.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1 inclusive.")

    # De-duplicate while preserving order
    seen = set()
    unique: List[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique.append(name)

    n = len(unique)
    # Edge cases: not enough names to form val/test
    if n < 2:
        return {}, {"loss": 0.0, "balanced_acc": 0.0}, {"loss": 0.0, "balanced_acc": 0.0}

    # New universe: all ordered (val, test) choices; training is the rest in canonical order
    total = n * (n - 1)
    k = floor(total * fraction)

    combinations: Dict[int, List[str]] = {}
    if k:
        # Sample indices uniformly without replacement from [0, total)
        sampled = random.sample(range(total), k)

        perms: List[List[str]] = []
        for r in sampled:
            # Map r -> (val_idx, test_idx) without materializing the O(n^2) list of pairs
            val_idx = r // (n - 1)
            rem = r % (n - 1)
            test_idx = rem if rem < val_idx else rem + 1  # skip the diagonal

            # Canonical training list = all others in original order
            if val_idx < test_idx:
                train = unique[:val_idx] + unique[val_idx + 1:test_idx] + unique[test_idx + 1:]
            else:
                train = unique[:test_idx] + unique[test_idx + 1:val_idx] + unique[val_idx + 1:]

            val = unique[val_idx]
            test = unique[test_idx]
            perms.append(train + [val, test])

        # Randomize display order of the sampled combinations
        random.shuffle(perms)
        combinations = {i: p for i, p in enumerate(perms)}

    mean = {"loss": 0.0, "BA_no_upr": 0.0, "f1_sl": 0.0}
    std = {"loss": 0.0, "BA_no_upr": 0.0, "f1_sl": 0.0}
    return combinations, mean, std

def fix_length(df_imu: pd.DataFrame, target_len: int = 18_000,
               time_col_idx: int = 0, fs_hint: float = 50.0) -> None:
    """
    In-place: ensure df_imu has exactly `target_len` rows.
      - If longer: crop trailing rows.
      - If shorter: append rows created by mirroring the tail for all *non-time* columns,
        and extend the time column linearly using the median dt (fallback 1/fs_hint).
    Returns None and mutates df_imu.
    """
    if not isinstance(df_imu, pd.DataFrame):
        raise TypeError("fix_length expects a pandas DataFrame")

    n = len(df_imu)
    if n == 0:
        raise ValueError("Cannot pad an empty DataFrame to a target length.")

    # --- Crop (in place) ---
    if n > target_len:
        df_imu.drop(df_imu.index[target_len:], inplace=True)
        df_imu.reset_index(drop=True, inplace=True)
        return

    if n == target_len:
        return  # nothing to do

    # --- Compute dt from existing time column (robust to non-numeric) ---
    time_col = df_imu.columns[time_col_idx]
    t = pd.to_numeric(df_imu.iloc[:, time_col_idx], errors="coerce").to_numpy()
    finite = np.isfinite(t)
    if finite.sum() >= 2:
        diffs = np.diff(t[finite])
        diffs = diffs[np.isfinite(diffs)]
        dt = float(np.median(diffs)) if diffs.size > 0 else (1.0 / fs_hint)
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0 / fs_hint
    else:
        dt = 1.0 / fs_hint
    last_t = float(t[finite][-1]) if finite.any() else 0.0

    # --- Build mirrored padding for all columns, then overwrite time column ---
    need = target_len - n
    tail_rev = df_imu.iloc[::-1].reset_index(drop=True)

    if need <= n:
        pad = tail_rev.iloc[:need].copy()
    else:
        q, r = divmod(need, n)
        blocks = [tail_rev] * q + ([tail_rev.iloc[:r].copy()] if r else [])
        pad = pd.concat(blocks, ignore_index=True)

    # Ensure exact column order
    pad = pad[df_imu.columns]

    # Extend time linearly
    pad_times = last_t + dt * np.arange(1, need + 1, dtype=float)
    pad.loc[:, time_col] = pad_times

    # --- Append rows one-by-one (works in all pandas versions) ---
    # Using itertuples(name=None) gives a plain tuple per row (fast and clean).
    start = n
    for i, row in enumerate(pad.itertuples(index=False, name=None)):
        df_imu.loc[start + i] = row

    # Normalize the index
    df_imu.reset_index(drop=True, inplace=True)

    # Safety
    if len(df_imu) != target_len:
        raise RuntimeError(f"fix_length failed: len={len(df_imu)} != target_len={target_len}")
    
def verify_lengths(root="data"):
    bad = []
    for folder in sorted(p for p in glob.glob(os.path.join(root, "beep_schedules_*")) if os.path.isdir(p)):
        for csv_path in sorted(glob.glob(os.path.join(folder, "airpods_motion_*.csv"))):
            n = pd.read_csv(csv_path, nrows=0).shape[0]  # header only
            n = pd.read_csv(csv_path).shape[0]
            if n != 18_000:
                bad.append((csv_path, n))
    if bad:
        print("Files not at 18,000 rows:")
        for p, n in bad:
            print(f"  {n:5d}  {p}")
    else:
        print("All IMU CSVs have exactly 18,000 rows.")

def normalize_chanels(df_imu: pd.DataFrame, time_col_idx: int = 0, stats: dict = None, eps: float = 1e-8) -> None:
    """
    In-place column-wise z-score normalization for all channels except the time column.
      - If `stats` is None: compute mean/std from this df and apply (per-channel).
      - If `stats` is provided: use `stats['mean']` and `stats['std']` for the listed columns.
    Mutates df_imu and returns None.

    `stats` format (optional):
      {'mean': {col: float, ...}, 'std': {col: float, ...}}
    """
    if not isinstance(df_imu, pd.DataFrame):
        raise TypeError("normalize_chanels expects a pandas DataFrame")

    cols = list(df_imu.columns)
    if not (0 <= time_col_idx < len(cols)):
        raise IndexError("time_col_idx out of range")

    # features = all columns except the time column
    feat_cols = [c for i, c in enumerate(cols) if i != time_col_idx]

    # to numeric (safe), ignore NaNs in stats
    X = df_imu[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)

    if stats is None:
        mean = np.nanmean(X, axis=0)
        std  = np.nanstd(X,  axis=0)
    else:
        # Use provided stats in the exact feat_cols order; fallback to per-file stats if missing
        mean = np.array([stats['mean'].get(c, np.nanmean(X[:, j])) for j, c in enumerate(feat_cols)], dtype=np.float64)
        std  = np.array([stats['std' ].get(c, np.nanstd( X[:, j])) for j, c in enumerate(feat_cols)], dtype=np.float64)

    # guard std
    bad = ~np.isfinite(std) | (std < eps)
    std[bad] = 1.0

    # normalize and write back (float32 is fine for ML)
    Xn = (X - mean) / std
    df_imu.loc[:, feat_cols] = Xn.astype(np.float32)
    # (timestamp column left untouched)

##### DELTA CALCULATION ###
def _detect_timestamp_col(df, candidates=("t_sec", "timestamp", "time", "time_sec", "epoch", "epoch_s")):
    # Prefer common names
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: if the first column is numeric & monotonic, assume it's time
    first = df.columns[0]
    s = pd.to_numeric(df[first], errors="coerce")
    if s.notna().all() and s.is_monotonic_increasing:
        return first
    return None

def make_delta_csv_path(csv_path: str) -> str:
    """
    Return the canonical delta path for a motion CSV.
    Ensures EXACTLY ONE 'd' after the second underscore.

      airpods_motion_1760734866.csv   -> airpods_motion_d1760734866.csv
      airpods_motion_d1760734866.csv  -> airpods_motion_d1760734866.csv
      airpods_motion_dd1760734866.csv -> airpods_motion_d1760734866.csv

    For other filenames, appends a single '_d' before the extension (idempotent).
    """
    folder, fname = os.path.split(csv_path)
    parts = fname.split("_", 2)  # e.g. ["airpods","motion","1760734866.csv"]
    if len(parts) >= 3:
        # Strip ANY number of leading 'd' chars from the tail, then re-add ONE 'd'
        tail = parts[2]
        tail = re.sub(r"^d+", "", tail)  # compress d's to zero, we'll add one back
        new_fname = f"{parts[0]}_{parts[1]}_d{tail}"
    else:
        # Generic fallback: normalize to a single '_d' suffix before extension
        base, ext = os.path.splitext(fname)
        base = re.sub(r"_d+$", "", base)  # drop any trailing '_d' run
        new_fname = f"{base}_d{ext or '.csv'}"
    return os.path.join(folder, new_fname)

def _choose_events_file_for(csv_path: str,
                            df_imu: pd.DataFrame,
                            timestamp_col: Optional[str] = None) -> Optional[str]:
    """
    In the IMU csv's folder, pick the events_inferred_*.csv that best overlaps
    the IMU time range. If only one exists, pick that.
    """
    folder = os.path.dirname(csv_path)
    candidates = sorted(glob.glob(os.path.join(folder, "events_inferred_*.csv")))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # detect timestamp column if not provided
    if not timestamp_col:
        timestamp_col = _detect_timestamp_col(df_imu)
    if not timestamp_col or timestamp_col not in df_imu.columns:
        return None

    # compute IMU range
    t = pd.to_numeric(df_imu[timestamp_col], errors="coerce").dropna()
    if t.empty:
        return None
    tmin, tmax = float(t.min()), float(t.max())
    imu_mid = 0.5 * (tmin + tmax)

    def file_ts_hint(path: str) -> Optional[float]:
        m = re.search(r"events_inferred_(\d+(?:\.\d+)?)", os.path.basename(path))
        return float(m.group(1)) if m else None

    best = None
    best_overlap = -1.0
    best_hint_dist = float("inf")

    for ev_path in candidates:
        try:
            ev = pd.read_csv(ev_path, usecols=["t_sec"])
            ev_t = pd.to_numeric(ev["t_sec"], errors="coerce").dropna()
            if ev_t.empty:
                continue
            emin, emax = float(ev_t.min()), float(ev_t.max())
            overlap = max(0.0, min(tmax, emax) - max(tmin, emin))
            hint = file_ts_hint(ev_path)
            hint_dist = abs(imu_mid - hint) if hint is not None else float("inf")
            if overlap > best_overlap or (overlap == best_overlap and hint_dist < best_hint_dist):
                best, best_overlap, best_hint_dist = ev_path, overlap, hint_dist
        except Exception:
            continue

    return best or candidates[0]

def is_delta_motion_file(path: str) -> bool:
    """True if this looks like a delta motion csv (airpods_motion_d*.csv)."""
    return os.path.basename(path).startswith("airpods_motion_d")

def calculate_deltas(
    df_imu: pd.DataFrame,
    *,
    csv_path: Optional[str] = None,
    events_path: Optional[str] = None,
    timestamp_col: Optional[str] = None,   # auto-detected if None
    id_cols: Optional[List[str]] = None,   # preserved (not baseline-subtracted) for NUMERIC channels
    baseline_method: str = "mean",         # "mean" or "median" for NUMERIC channels
    min_window_seconds: float = 0.5,
    scope: str = "per_upright_epoch",      # kept for backward-compat; see behavior below
    apply_global_to_gaps: bool = True,     # apply global upright baseline outside epochs
    reuse_last_baseline: bool = True,      # if an epoch has no upright samples
    add_epoch_id: bool = False,            # optional debug column
    verbose: bool = True,
    # NEW: do NOT delta these (kept raw)
    no_delta_cols: Optional[List[str]] = None,
    # NEW: when forming upright baselines, trim 1s off both ends by default
    edge_trim_seconds: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, float], List[Tuple[float, float]]]:
    """
    Convert IMU channels to deltas and compute relative quaternions.

    Key behavior:
    - Numeric channels: per-epoch delta relative to an upright baseline,
      EXCEPT for any in `no_delta_cols` (which are kept raw).
    - Relative quaternions: for each epoch, compute the baseline quaternion
      from the *pre-movement upright* window, trimmed by `edge_trim_seconds`
      at both start and end, then apply q_rel = q_sample ⊗ conj(q_base_epoch).
    - Epochs are defined only by *valid pre-movement uprights*:
        upright_start < movement_start < next_upright_start
      Any "upright" occurring inside a movement is ignored.
    - Global fallbacks are built from the union of all trimmed upright windows.
      If none exist, fall back to the whole recording.

    Returns
    -------
    df_delta : DataFrame
        Transformed IMU with numeric deltas, quat_rel_* columns, and without raw quat_*.
    global_numeric_baseline : Dict[str, float]
        Global numeric baselines used as fallback.
    windows_trimmed : List[Tuple[float,float]]
        The trimmed upright windows actually used to build baselines.
    """

    # ---------------- helpers: quaternion math ----------------
    def _has_quats(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        qnames = {"quat_x", "quat_y", "quat_z", "quat_w"}
        lower_map = {c.lower(): c for c in df.columns}
        present = [lower_map[name] for name in qnames if name in lower_map]
        return (len(present) == 4, present)

    def _q_normalize(q: np.ndarray) -> np.ndarray:
        eps = 1e-12
        n = np.linalg.norm(q, axis=-1, keepdims=True)
        n = np.where(n < eps, 1.0, n)
        return q / n

    def _q_conj(q: np.ndarray) -> np.ndarray:
        qc = q.copy()
        qc[..., 0:3] *= -1.0  # negate vector part (x,y,z)
        return qc

    def _q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply quaternions in (x,y,z,w) form: q = q1 ⊗ q2
        """
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        w = w1*w2 - (x1*x2 + y1*y2 + z1*z2)
        x = w1*x2 + w2*x1 + (y1*z2 - z1*y2)
        y = w1*y2 + w2*y1 + (z1*x2 - x1*z2)
        z = w1*z2 + w2*z1 + (x1*y2 - y1*x2)
        out = np.stack([x, y, z, w], axis=-1)
        return _q_normalize(out)

    def _q_mean(qs: np.ndarray) -> Optional[np.ndarray]:
        """
        Approximate mean quaternion (Markley-style, hemisphere aligned).
        qs: (N,4) with possible NaNs; returns (4,) unit quat or None if insufficient data.
        """
        if qs.size == 0:
            return None
        mask = ~np.isnan(qs).any(axis=1)
        qs = qs[mask]
        if qs.shape[0] == 0:
            return None
        qs = _q_normalize(qs)
        q0 = qs[0]
        dots = np.sum(qs * q0, axis=1)
        qs[dots < 0] *= -1.0
        q_mean = np.mean(qs, axis=0)
        return _q_normalize(q_mean)

    def _ensure_continuity(qs: np.ndarray) -> np.ndarray:
        """
        Flip sign to enforce continuity along time: qs[i] ~ qs[i-1].
        """
        if qs.shape[0] == 0:
            return qs
        for i in range(1, qs.shape[0]):
            if np.dot(qs[i], qs[i - 1]) < 0:
                qs[i] *= -1.0
        return qs

    def _pre_movement_uprights(
        events_df: pd.DataFrame,
        t_end: float,
        *,
        edge_trim_seconds: float = 1.0,
        min_window_seconds: float = 0.5
    ) -> Tuple[Dict[float, Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Identify 'true' uprights that precede a movement and build:
          - baseline_window_by_start:  {upright_start -> (upright_start+trim, movement_start-trim)}
          - windows_trimmed:           list of (baseline_start, baseline_end) across all trials
          - segments:                  [(u0, u1), (u1, u2), ..., (u_last, t_end)]
                                       built only from those valid pre-movement uprights.

        A 'movement start' is any event whose name ends with '_START' and is not 'UPRIGHT_HOLD_START'.
        A valid upright is one whose next movement start occurs *before* the next upright start:
            upright_start < movement_start < next_upright_start
        This ignores 'upright' detections that occur inside an ongoing movement.
        """
        df = events_df.copy()
        if "t_sec" not in df.columns or "event" not in df.columns:
            return {}, [], []

        df = df.dropna(subset=["t_sec", "event"]).copy()
        df["t_sec"] = pd.to_numeric(df["t_sec"], errors="coerce")
        df = df.dropna(subset=["t_sec"]).sort_values("t_sec")

        ev_upper = df["event"].astype(str).str.upper()
        is_upright = (ev_upper == "UPRIGHT_HOLD_START")
        uprights = df.loc[is_upright, "t_sec"].to_numpy(dtype=float)

        # movement start = any *_START except UPRIGHT_HOLD_START
        is_start = ev_upper.str.endswith("_START") & ~is_upright
        mov_starts = df.loc[is_start, "t_sec"].to_numpy(dtype=float)

        uprights = np.sort(uprights)
        mov_starts = np.sort(mov_starts)

        baseline_window_by_start: Dict[float, Tuple[float, float]] = {}
        windows_trimmed: List[Tuple[float, float]] = []
        valid_starts: List[float] = []

        for i, u in enumerate(uprights):
            u_next = uprights[i + 1] if (i + 1) < len(uprights) else t_end
            # next movement strictly after u
            j = np.searchsorted(mov_starts, u, side="right")
            m = mov_starts[j] if j < len(mov_starts) else None

            # valid pre-movement upright: u < m < u_next
            if m is None or not (u < m < u_next):
                continue

            s_trim = u + float(edge_trim_seconds)
            e_trim = m - float(edge_trim_seconds)
            if e_trim - s_trim >= max(min_window_seconds, 0.0):
                baseline_window_by_start[u] = (s_trim, e_trim)
                windows_trimmed.append((s_trim, e_trim))
                valid_starts.append(u)

        # segments are formed only by valid uprights
        segments: List[Tuple[float, float]] = []
        for i, s in enumerate(valid_starts):
            e = valid_starts[i + 1] if (i + 1) < len(valid_starts) else t_end
            if e > s:
                segments.append((float(s), float(e)))

        return baseline_window_by_start, windows_trimmed, segments

    # ---------------- detect timestamp & events ----------------
    if not timestamp_col:
        timestamp_col = _detect_timestamp_col(df_imu)
    if not timestamp_col or timestamp_col not in df_imu.columns:
        raise ValueError("Could not detect a timestamp column (looked for 't_sec', 'timestamp', 'time', ...).")

    ev_path = events_path
    if ev_path is None and csv_path is not None:
        ev_path = _choose_events_file_for(csv_path, df_imu, timestamp_col=timestamp_col)

    events_df = None
    if ev_path is not None and os.path.isfile(ev_path):
        try:
            events_df = pd.read_csv(ev_path, usecols=["t_sec", "event"])
        except Exception:
            events_df = None

    # ---------------- time vector ----------------
    t_series = pd.to_numeric(df_imu[timestamp_col], errors="coerce").dropna()
    if t_series.empty:
        raise ValueError(f"Timestamp column '{timestamp_col}' contains no numeric values.")
    t_end = float(t_series.max())
    t = pd.to_numeric(df_imu[timestamp_col], errors="coerce").values

    # ---------------- columns ----------------
    has_quat, quat_cols = _has_quats(df_imu)
    if id_cols is None:
        id_cols = [timestamp_col]
    id_set = set(id_cols)
    numeric_cols = df_imu.select_dtypes(include=[np.number]).columns.tolist()

    # keep these raw (no delta)
    if no_delta_cols is None:
        no_delta_cols = ["acc_x", "acc_y", "acc_z", "rot_x", "rot_y", "rot_z"]
    lower_map = {c.lower(): c for c in df_imu.columns}
    protected = {lower_map[c.lower()] for c in no_delta_cols if c.lower() in lower_map}

    # numeric delta candidates = numeric minus ids, quats, protected
    delta_cols = [c for c in numeric_cols
                  if (c not in id_set) and (not has_quat or c not in quat_cols) and (c not in protected)]

    # ---------------- discover pre-movement upright windows & segments ----------------
    baseline_window_by_start: Dict[float, Tuple[float, float]] = {}
    windows_trimmed: List[Tuple[float, float]] = []
    segments: List[Tuple[float, float]] = []

    if events_df is not None and not events_df.empty:
        baseline_window_by_start, windows_trimmed, segments = _pre_movement_uprights(
            events_df,
            t_end,
            edge_trim_seconds=edge_trim_seconds,
            min_window_seconds=min_window_seconds,
        )

    # ---------------- numeric baseline aggregator ----------------
    def _agg_numeric(df_sub: pd.DataFrame) -> Dict[str, float]:
        if not delta_cols:
            return {}
        if baseline_method == "median":
            return df_sub[delta_cols].median(numeric_only=True).to_dict()
        return df_sub[delta_cols].mean(numeric_only=True).to_dict()

    # ---------------- global numeric & quaternion baselines ----------------
    if windows_trimmed:
        mask_upright_all = np.zeros(len(df_imu), dtype=bool)
        for (s, e) in windows_trimmed:
            mask_upright_all |= (t >= s) & (t < e)
        global_numeric_baseline = _agg_numeric(df_imu.loc[mask_upright_all]) if mask_upright_all.any() else _agg_numeric(df_imu)
        if has_quat:
            Q_all = df_imu.loc[mask_upright_all, quat_cols].to_numpy(dtype=float) if mask_upright_all.any() \
                    else df_imu[quat_cols].to_numpy(dtype=float)
            q_base_global = _q_mean(Q_all)
        else:
            q_base_global = None
    else:
        global_numeric_baseline = _agg_numeric(df_imu)
        q_base_global = _q_mean(df_imu[quat_cols].to_numpy(dtype=float)) if has_quat else None

    df_delta = df_imu.copy()

    # ---------------- scope: GLOBAL (simple path) ----------------
    if scope == "global" or not segments or events_df is None or events_df.empty:
        # numeric deltas
        for c in delta_cols:
            b = global_numeric_baseline.get(c, 0.0)
            if pd.notna(b):
                df_delta[c] = df_delta[c] - b

        # relative quaternions (global baseline)
        if has_quat and q_base_global is not None:
            Q = df_imu[quat_cols].to_numpy(dtype=float)
            valid = ~np.isnan(Q).any(axis=1)
            if np.any(valid):
                Qn = _q_normalize(Q[valid])
                q_base_conj = _q_conj(_q_normalize(q_base_global))
                Qrel = _q_mul(Qn, q_base_conj)
                Qrel = _ensure_continuity(Qrel)
                rel_cols = ["quat_rel_x", "quat_rel_y", "quat_rel_z", "quat_rel_w"]
                for i, name in enumerate(rel_cols):
                    df_delta[name] = np.nan
                    df_delta.loc[valid, name] = Qrel[:, i]
            # drop raw quats
            if has_quat:
                df_delta.drop(columns=quat_cols, inplace=True, errors="ignore")

        if add_epoch_id:
            df_delta["epoch_id"] = -1

        if verbose:
            source = "upright_trimmed" if windows_trimmed else "global"
            print(f"Δ-baseline: {baseline_method} / global from {source} "
                  f"(windows={len(windows_trimmed)}, samples_used={len(df_imu)}). "
                  f"Transformed {len(delta_cols)} channels (kept {len(protected)} raw). "
                  f"Quat_rel={'yes' if has_quat else 'no'}. Time='{timestamp_col}'.")
        return df_delta, global_numeric_baseline, windows_trimmed

    # ---------------- scope: PER PRE-MOVEMENT UPRIGHT EPOCH ----------------
    touched = np.zeros(len(df_imu), dtype=bool)
    epoch_ids = np.full(len(df_imu), -1, dtype=int)
    last_numeric_baseline: Optional[Dict[str, float]] = None
    last_q_base: Optional[np.ndarray] = None
    epochs_applied = 0

    # Prepare output columns for relative quaternions
    if has_quat:
        rel_cols = ["quat_rel_x", "quat_rel_y", "quat_rel_z", "quat_rel_w"]
        for name in rel_cols:
            df_delta[name] = np.nan
        last_rel_tail: Optional[np.ndarray] = None  # for sign continuity across blocks

    for epoch_idx, (seg_start, seg_end) in enumerate(segments):
        # ----- numeric baseline for this epoch from trimmed pre-movement upright -----
        bw = baseline_window_by_start.get(seg_start)
        if bw is not None:
            b_start, b_end = bw
            m_b = (t >= b_start) & (t < b_end)
            numeric_baseline = _agg_numeric(df_imu.loc[m_b]) if m_b.any() else None
            if numeric_baseline:
                last_numeric_baseline = numeric_baseline
            else:
                numeric_baseline = last_numeric_baseline if (reuse_last_baseline and last_numeric_baseline is not None) else global_numeric_baseline
        else:
            numeric_baseline = last_numeric_baseline if (reuse_last_baseline and last_numeric_baseline is not None) else global_numeric_baseline

        # apply numeric deltas to the entire epoch
        m_seg = (t >= seg_start) & (t < seg_end)
        if m_seg.any() and delta_cols:
            for c in delta_cols:
                b = numeric_baseline.get(c, 0.0)
                if pd.notna(b):
                    df_delta.loc[m_seg, c] = df_delta.loc[m_seg, c] - b

        # ----- quaternion baseline & relative for this epoch -----
        if has_quat:
            q_base_epoch = None
            if bw is not None:
                m_b = (t >= b_start) & (t < b_end)
                if m_b.any():
                    Qb = df_imu.loc[m_b, quat_cols].to_numpy(dtype=float)
                    q_base_epoch = _q_mean(Qb)
                    if q_base_epoch is not None:
                        last_q_base = q_base_epoch
            if q_base_epoch is None:
                q_base_epoch = last_q_base if (reuse_last_baseline and last_q_base is not None) else q_base_global

            if q_base_epoch is not None and m_seg.any():
                Qseg = df_imu.loc[m_seg, quat_cols].to_numpy(dtype=float)
                valid = ~np.isnan(Qseg).any(axis=1)
                if np.any(valid):
                    Qn = _q_normalize(Qseg[valid])
                    q_base_conj = _q_conj(_q_normalize(q_base_epoch))
                    Qrel = _q_mul(Qn, q_base_conj)
                    Qrel = _ensure_continuity(Qrel)

                    # align epoch start to previous block for cross-block continuity
                    if 'last_rel_tail' in locals() and last_rel_tail is not None and Qrel.shape[0] > 0:
                        if np.dot(Qrel[0], last_rel_tail) < 0:
                            Qrel *= -1.0

                    # write back
                    for j, name in enumerate(rel_cols):
                        arr = df_delta.loc[m_seg, name].to_numpy(dtype=float)
                        arr[:] = np.nan
                        arr[valid] = Qrel[:, j]
                        df_delta.loc[m_seg, name] = arr

                    # remember last rel value for cross-block continuity
                    last_rel_tail = Qrel[-1].copy()

        if m_seg.any():
            touched |= m_seg
            epoch_ids[m_seg] = epoch_idx
            epochs_applied += 1

    # ----- gaps: numeric & quaternion with global baseline -----
    if apply_global_to_gaps:
        m_gap = ~touched
        if np.any(m_gap):
            # numeric
            for c in delta_cols:
                b = global_numeric_baseline.get(c, 0.0)
                if pd.notna(b):
                    df_delta.loc[m_gap, c] = df_delta.loc[m_gap, c] - b
            # quaternion
            if has_quat and q_base_global is not None:
                Qgap = df_imu.loc[m_gap, quat_cols].to_numpy(dtype=float)
                valid = ~np.isnan(Qgap).any(axis=1)
                if np.any(valid):
                    Qn = _q_normalize(Qgap[valid])
                    Qrel = _q_mul(Qn, _q_conj(_q_normalize(q_base_global)))
                    Qrel = _ensure_continuity(Qrel)
                    if 'last_rel_tail' in locals() and last_rel_tail is not None and Qrel.shape[0] > 0:
                        if np.dot(Qrel[0], last_rel_tail) < 0:
                            Qrel *= -1.0
                    rel_cols = ["quat_rel_x", "quat_rel_y", "quat_rel_z", "quat_rel_w"]
                    for j, name in enumerate(rel_cols):
                        col = df_delta.loc[m_gap, name].to_numpy(dtype=float)
                        col[:] = np.nan
                        col[valid] = Qrel[:, j]
                        df_delta.loc[m_gap, name] = col
                    last_rel_tail = Qrel[-1].copy()

    # ----- drop raw quaternion columns; keep only quat_rel_* -----
    if has_quat:
        df_delta.drop(columns=quat_cols, inplace=True, errors="ignore")

    if add_epoch_id:
        df_delta["epoch_id"] = epoch_ids

    if verbose:
        kept = len(protected)
        print(f"Δ-baseline: {baseline_method} / per_pre_movement_upright "
              f"(epochs_applied={epochs_applied}, trimmed_windows={len(windows_trimmed)}). "
              f"Gaps->global={apply_global_to_gaps}. "
              f"Transformed {len(delta_cols)} numeric channels; kept {kept} raw "
              f"({', '.join(sorted(protected)) if kept else '—'}). "
              f"Quat_rel={'yes' if has_quat else 'no'}. Time='{timestamp_col}'.")

    return df_delta, global_numeric_baseline, windows_trimmed

def compute_and_save_delta_once(
    df_imu: pd.DataFrame,
    *,
    csv_path: str,
    baseline_method: str = "median",
    min_window_seconds: float = 1.0,
    verbose: bool = False,
) -> Tuple[str, bool]:
    """
    Compute delta for the given IMU dataframe and save to the canonical
    delta path *only if it does not already exist*.

    Returns: (out_csv_path, created_bool)
      - created_bool=False means an existing delta file was left untouched.
    """
    out_csv = make_delta_csv_path(csv_path)  # e.g., airpods_motion_d*.csv

    # Fast path: if it exists, don't touch it.
    if os.path.exists(out_csv):
        if verbose:
            print(f"⏭️  Delta exists; leaving as-is: {out_csv}")
        return out_csv, False

    # Compute deltas
    df_delta, _, _ = calculate_deltas(
        df_imu,
        csv_path=csv_path,
        baseline_method=baseline_method,
        min_window_seconds=min_window_seconds,
        verbose=verbose,
    )

    # Exclusive create to be race-safe. If another process created it meanwhile,
    # this will raise FileExistsError and we will leave it alone.
    try:
        with open(out_csv, "x", newline="") as fh:
            df_delta.to_csv(fh, index=False)
        if verbose:
            print(f"✅ Wrote delta: {out_csv}")
        return out_csv, True
    except FileExistsError:
        if verbose:
            print(f"⏭️  Delta appeared concurrently; skipping: {out_csv}")
        return out_csv, False

##### SIGMA CALCULATION ###
def make_delta_scaled_csv_path(csv_path: str) -> str:
    """
    Return the canonical scaled-delta path for a motion CSV.
    Ensures EXACTLY 'ds' after the second underscore.

      airpods_motion_1760.csv    -> airpods_motion_ds1760.csv
      airpods_motion_d1760.csv   -> airpods_motion_ds1760.csv
      airpods_motion_ds1760.csv  -> airpods_motion_ds1760.csv
      airpods_motion_dds1760.csv -> airpods_motion_ds1760.csv

    For other filenames, appends a single '_ds' before the extension (idempotent).
    """
    folder, fname = os.path.split(csv_path)
    parts = fname.split("_", 2)  # ["airpods","motion","1760.csv"] etc.
    if len(parts) >= 3:
        tail = parts[2]
        # Strip any leading 'd'/'s' run, then re-add 'ds'
        tail = re.sub(r"^[ds]+", "", tail, flags=re.IGNORECASE)
        new_fname = f"{parts[0]}_{parts[1]}_ds{tail}"
    else:
        base, ext = os.path.splitext(fname)
        base = re.sub(r"_ds+$", "", base, flags=re.IGNORECASE)
        new_fname = f"{base}_ds{ext or '.csv'}"
    return os.path.join(folder, new_fname)

def compute_and_save_delta_scaled_once(
    *,
    csv_path: str,
    channels: Optional[List[str]] = None,
    sigma_method: str = "std",   # currently only "std" implemented
    ddof: int = 0,               # population std by default
    verbose: bool = False,
) -> Tuple[str, bool, Dict[str, float]]:
    """
    Read the delta CSV for this raw csv_path, compute per-column sigma for selected channels,
    divide those columns by sigma, and save as airpods_motion_ds*.csv.

    Returns: (out_ds_csv_path, created_bool, sigma_dict)
    """
    import numpy as np

    # Source = delta file; Target = ds file
    delta_csv = make_delta_csv_path(csv_path)
    out_csv   = make_delta_scaled_csv_path(csv_path)

    if not os.path.exists(delta_csv):
        if verbose:
            print(f"⚠️  No delta file found; expected at: {delta_csv}")
        return out_csv, False, {}

    # Idempotent create: if ds already exists, leave it
    if os.path.exists(out_csv):
        if verbose:
            print(f"⏭️  Scaled-delta exists; leaving as-is: {out_csv}")
        return out_csv, False, {}

    try:
        df = pd.read_csv(delta_csv, low_memory=False)
    except Exception as e:
        if verbose:
            print(f"❌ Failed to read delta CSV: {delta_csv}\n   ↳ {e}")
        return out_csv, False, {}

    # Determine which columns to scale
    # Default: only these channels if present -> acc_*, rot_*, grav_* (or gravity_*)
    if channels is None:
        wanted = [
            "acc_x", "acc_y", "acc_z",
            "rot_x", "rot_y", "rot_z",
            "grav_x", "grav_y", "grav_z",
            "gravity_x", "gravity_y", "gravity_z",
        ]
        # keep only those that actually exist in df
        channels = [c for c in wanted if c in df.columns]

    # Explicitly exclude any quaternion columns even if provided
    exclude_prefixes = ("quat_", "quatrel_", "quat_rel_")
    channels = [c for c in channels
                if not any(c.lower().startswith(p) for p in exclude_prefixes)]

    # Compute sigmas and scale
    sigmas: Dict[str, float] = {}
    eps = 1e-12
    for c in channels:
        s = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if sigma_method == "std":
            sigma = float(np.nanstd(s, ddof=ddof))
        else:
            # Future hook for MAD/IQR if needed
            sigma = float(np.nanstd(s, ddof=ddof))
        if not np.isfinite(sigma) or sigma < eps:
            # Avoid division by ~0; leave column unchanged
            sigma = 1.0
        df[c] = s / sigma
        sigmas[c] = sigma

    # Write the scaled csv atomically (race-safe exclusive create)
    try:
        with open(out_csv, "x", newline="") as fh:
            df.to_csv(fh, index=False)
        if verbose:
            sc = ", ".join(f"{k}={v:.4g}" for k, v in sigmas.items())
            print(f"✅ Wrote scaled-delta: {out_csv}\n   σ: {sc}")
        return out_csv, True, sigmas
    except FileExistsError:
        if verbose:
            print(f"⏭️  Scaled-delta appeared concurrently; skipping: {out_csv}")
        return out_csv, False, sigmas

#########################

def edit_csv():
    """
    Walk data/beep_schedules_*/, open each airpods_motion_*.csv,
    add derived columns in place (e.g., pitch_rad), and save back.
    """
    DATA_ROOT = os.path.join(os.getcwd(), "data")
    folder_glob = os.path.join(DATA_ROOT, "beep_schedules_*")
    folders = sorted(p for p in glob.glob(folder_glob) if os.path.isdir(p))

    if not folders:
        print(f"⚠️ No folders found with pattern: {folder_glob}")
        return

    for folder_path in folders:
        imu_glob = os.path.join(folder_path, "airpods_motion_[0-9]*.csv")
        imu_files = sorted(glob.glob(imu_glob))
        if not imu_files:
            print(f"⚠️ No IMU CSVs in {folder_path}")
            continue

        for csv_path in imu_files:
            try:
                df_imu = pd.read_csv(csv_path, low_memory=False)
            except Exception as e:
                print(f"❌ Skipping (read error): {csv_path}\n   ↳ {e}")
                continue

            # --- add/refresh derived columns (idempotent) ---
            fix_length(df_imu, target_len=18_000)
            add_pitch_to_df(df_imu)  # modifies df_imu in place
            remove_columns(df_imu, ['roll_rad', 'yaw_rad'])
            #add_roll_to_df(df_imu)
            #add_yaw_to_df(df_imu)
            #normalize_chanels(df_imu)

            try:
                df_imu.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"❌ Failed to write: {csv_path}\n   ↳ {e}")

            ### NEW FILE WITH DELTA CALCULATION ###

            try:
                compute_and_save_delta_once(
                    df_imu,
                    csv_path=csv_path,
                    baseline_method="median",
                    min_window_seconds=1.0,
                    verbose=False,
                )
            except Exception as e:
                print(f"⚠️ Delta conversion skipped for: {csv_path}\n   ↳ {e}")

            try:
                out_delta, _ = compute_and_save_delta_once(
                    df_imu,
                    csv_path=csv_path,
                    baseline_method="median",   # or "mean" as you like
                    min_window_seconds=1.0,
                    verbose=False,
                )
            except Exception as e:
                print(f"⚠️ Delta conversion skipped for: {csv_path}\n   ↳ {e}")
                continue

            ### NEW FILE WITH SCALED-DELTA (per-file σ) ###
            try:
                compute_and_save_delta_scaled_once(
                    csv_path=csv_path,
                    # channels=None -> defaults to acc/rot/grav only, skips quats
                    sigma_method="std",
                    ddof=0,
                    verbose=False,
                )
            except Exception as e:
                print(f"⚠️ Scaled-delta skipped for: {csv_path}\n   ↳ {e}")

def individual_accuracy(model, X_test, y_test, classes=(0, 1, 2)):
    """
    Print per-class accuracy (i.e., recall per class) for a 3-class classifier.
    Handles y_test in shape (N,), (N,1), or one-hot (N,3), and predictions in (N,3) or (N,1,3).
    """

    # --- Normalize y_true to shape (N,) with integer class IDs ---
    y_true = np.asarray(y_test)
    if y_true.ndim > 1 and y_true.shape[-1] > 1:
        # one-hot -> class ids
        y_true = np.argmax(y_true, axis=-1)
    y_true = np.squeeze(y_true).astype(int).ravel()  # ensure 1-D

    # --- Predict and normalize to class ids (N,) ---
    y_prob = model.predict(X_test, verbose=0)

    # common TCN cases: (N,3) or (N,1,3)
    if y_prob.ndim == 3 and y_prob.shape[1] == 1:
        y_prob = y_prob[:, 0, :]       # squeeze time/channel dim
    elif y_prob.ndim == 3:
        # If your model outputs (N,T,C) with T>1, decide how to reduce (e.g., last timestep):
        y_prob = y_prob[:, -1, :]      # take last timestep; adjust if you prefer avg/max

    if y_prob.ndim == 2:
        y_pred = np.argmax(y_prob, axis=-1)
    elif y_prob.ndim == 1:
        # already class ids
        y_pred = y_prob.astype(int)
    else:
        raise ValueError(f"Unexpected prediction shape: {y_prob.shape}")

    y_pred = np.squeeze(y_pred).astype(int).ravel()  # ensure 1-D

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(f"Length mismatch: y_pred={y_pred.shape}, y_true={y_true.shape}")

    # --- Per-class accuracy (recall) ---
    for c in classes:
        mask = (y_true == c)           # 1-D boolean mask
        n = int(mask.sum())
        if n == 0:
            print(f"Class {c}: no samples (n=0)")
            continue
        acc = float(np.mean(y_pred[mask] == y_true[mask]))
        print(f"Class {c}: Accuracy={acc:.3f} (n={n})")


# --- New Version of function: label_at_time to clean data ---

def label_at_time(t, names, times, m):
    """
    3-class labeling at timestamp t with margin m (seconds) treated as human delay.

    Intervals (inclusive on the left, exclusive on the right):
        Upright (0):            UPRIGHT_HOLD_START + m → SLOUCH_START + m
        Transition (1):         SLOUCH_START + m   → SLOUCHED_HOLD_START
                                RECOVERY_START + m → next UPRIGHT_HOLD_START + 2*m
        Slouched (2):           SLOUCHED_HOLD_START → RECOVERY_START + m

    Non-slouch markers ('NOSLOUCH_START', 'NOSLOUCHED_HOLD_START') are always upright (0),
    except that we still mark transitions before/after the non-slouch hold:
        NOSLOUCH_START → NOSLOUCHED_HOLD_START  ==  transition (1)
        RECOVERY_START  → UPRIGHT_HOLD_START    ==  transition (1)

    Parameters
    ----------
    t : float
    names : sequence of str (sorted by time)
    times : sequence of float (same length as names; non-decreasing)
    m : float  (>=0)
    """
    if len(names) != len(times):
        raise ValueError("`names` and `times` must have the same length.")
    times = np.asarray(times, dtype=float)
    if len(times) and np.any(np.diff(times) < 0):
        raise ValueError("`times` must be sorted in non-decreasing order.")
    m = max(0.0, float(m))

    # Index of last event at or before t
    k = int(np.searchsorted(times, t, side='right')) - 1

    # Before first event -> upright (no anticipation)
    if k < 0:
        return 0

    prev_ev, prev_t = names[k], times[k]
    next_ev = names[k + 1] if (k + 1) < len(names) else None
    next_t  = times[k + 1] if (k + 1) < len(times) else np.inf
    prev_prev_ev = names[k - 1] if (k - 1) >= 0 else None

    # ---------- Non-slouch onset/hold ----------
    if prev_ev == 'NOSLOUCH_START':
        # Before delay m -> upright
        if t < prev_t + m:
            return 0
        # After delay until NOSLOUCHED_HOLD_START -> transition
        if next_ev == 'NOSLOUCHED_HOLD_START':
            return 1
        return 0

    if prev_ev == 'NOSLOUCHED_HOLD_START':
        # For m seconds after NOSLOUCHED_HOLD_START, still transition (closing the onset band)
        if prev_prev_ev == 'NOSLOUCH_START' and t < prev_t + m:
            return 1
        return 0  # the non-slouch hold itself is upright

    # ---------- Upright hold ----------
    if prev_ev == 'UPRIGHT_HOLD_START':
        # After a recovery (either slouch or non-slouch), keep m (slouch) or up to 2m as transition close-out
        if prev_prev_ev == 'RECOVERY_START' and t < prev_t + 2*m:
            return 1
        return 0

    # ---------- Slouch onset ----------
    if prev_ev == 'SLOUCH_START':
        if t < prev_t + m:
            return 0  # still upright during delay
        return 1      # transition until SLOUCHED_HOLD_START

    # ---------- Slouch hold ----------
    if prev_ev == 'SLOUCHED_HOLD_START':
        return 2

    # ---------- Recovery ----------
    if prev_ev == 'RECOVERY_START':
        # Case A: recovering from a true slouch hold
        if prev_prev_ev == 'SLOUCHED_HOLD_START':
            if t < prev_t + m:
                return 2      # still slouched during delay
            return 1          # transition until next UPRIGHT_HOLD_START

        # Case B: recovering from a non-slouch hold -> we still want a transition band
        if prev_prev_ev == 'NOSLOUCHED_HOLD_START':
            return 1          # entire interval until UPRIGHT_HOLD_START is transition

        # Anything else: default upright
        return 0

    # Fallback: upright
    return 0

#---- Confusion Matrix
class ConfusionMatrixAverager:
    """
    Accumulates raw confusion matrices over multiple runs and saves
    a single averaged confusion matrix as a PNG (always full n_classes x n_classes).

    Display normalization:
        * normalize="true":  rows sum to 1 (per-true-class recall)
        * normalize="pred":  columns sum to 1 (per-predicted-class precision)
        * normalize="all":   entire matrix sums to 1
        * normalize=None:    raw counts

    Parameters
    ----------
    n_classes : Optional[int]
        If None, inferred from history.model.output_shape[-1] on the first add(...).
    class_names : Optional[Sequence[str]]
        If None and n_classes==3, defaults to ["upright", "transition", "slouch"].
        Else defaults to ["0","1",...].
    save_dir : str
        Directory to save figures into (created if missing).
    """

    def __init__(self,
                 n_classes: Optional[int] = None,
                 class_names: Optional[Sequence[str]] = None,
                 save_dir: str = "confusion_matrix"):
        self.n_classes = None if n_classes is None else int(n_classes)
        self.class_names = list(class_names) if class_names is not None else None
        self.save_dir = save_dir

        self._cm_counts = None  # np.ndarray[n_classes, n_classes]
        self._runs = 0

    # ---------- public API ----------
    def add(self, history, X, y, batch_size: int = 256, verbose: int = 0):
        """
        Add one run's confusion matrix (computed on X,y) to the accumulator.

        Parameters
        ----------
        history : tf.keras.callbacks.History
            The History returned by model.fit(...). We use history.model for prediction.
        X : np.ndarray
            Feature array (e.g., your test split), shape [N, T, C].
        y : np.ndarray
            Integer class labels, shape [N] or [N,1].
        """
        model = getattr(history, "model", None)
        if model is None:
            raise ValueError("The provided 'history' has no .model; pass the History from model.fit(...).")

        X = np.asarray(X)
        y_true = np.asarray(y).squeeze().astype(np.int64)

        self._ensure_initialized(history, y_true)

        y_probs = model.predict(X, batch_size=batch_size, verbose=verbose)
        y_pred = np.argmax(y_probs, axis=-1).astype(np.int64)

        cm_counts = confusion_matrix(
            y_true, y_pred, labels=np.arange(self.n_classes), normalize=None
        )
        if self._cm_counts is None:
            self._cm_counts = cm_counts.astype(np.int64)
        else:
            self._cm_counts += cm_counts.astype(np.int64)

        self._runs += 1
        return cm_counts

    def save_figure(self,
                    model_tag: str = "cnn",
                    normalize: str = "true",  # "true", "pred", "all", or None
                    dpi: int = 220) -> str:
        """
        Save the averaged confusion matrix as a PNG and return the path.

        Parameters
        ----------
        model_tag : {"tcn","cnn",...}
            Used in the filename, e.g. cnn_19-10-2025_14-37.png.
        normalize : {"true","pred","all",None}
            - "true": rows sum to 1 (per-class recall)
            - "pred": columns sum to 1 (per-class precision)
            - "all":  all entries sum to 1
            - None:   raw counts (summed across runs)
        dpi : int
            Figure DPI.

        Returns
        -------
        path : str
            Filesystem path of the saved PNG.
        """
        if self._cm_counts is None or self._runs == 0:
            raise ValueError("No runs added. Call add(...) at least once before saving.")

        cm_plot = self._normalized(self._cm_counts, normalize=normalize)
        ba_all = self._balanced_accuracy_from_counts(self._cm_counts)
        ba_no_upr = self._balanced_accuracy_no_upright(self._cm_counts)  # uses rows 1 & 2 from full CM

        # Plot
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Proportion" if normalize is not None else "Count",
                           rotation=90, va="bottom")

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(np.arange(self.n_classes))
        ax.set_yticks(np.arange(self.n_classes))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.class_names)

        norm_label = { "true": "row-normalized",
                       "pred": "col-normalized",
                       "all":  "global-normalized",
                       None:   "counts" }.get(normalize, "row-normalized")

        title_top = f"Averaged Confusion Matrix ({model_tag.upper()})"
        if self.n_classes >= 3:
            subtitle = (f"Runs: {self._runs} • BA(all): {ba_all*100:.1f}% • "
                        f"BA(no_upright: classes 1&2): {ba_no_upr*100:.1f}% • {norm_label}")
        else:
            subtitle = f"Runs: {self._runs} • BA(all): {ba_all*100:.1f}% • {norm_label}"
        ax.set_title(f"{title_top}\n{subtitle}")

        # Annotate each cell
        fmt = ".2f" if normalize is not None else "d"
        valid_max = np.nanmax(cm_plot) if np.any(np.isfinite(cm_plot)) else 0.0
        thresh = valid_max / 2.0
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                val = cm_plot[i, j]
                text = format(val, fmt) if np.isfinite(val) else "nan"
                ax.text(j, i, text,
                        ha="center", va="center",
                        color="white" if np.isfinite(val) and val > thresh else "black")

        fig.tight_layout()

        # Safe timestamp for filenames: dd-mm-yyyy_HH-MM (no seconds)
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
        os.makedirs(self.save_dir, exist_ok=True)
        filename = f"{model_tag.lower()}_{timestamp}.png"
        out_path = os.path.join(self.save_dir, filename)

        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ---------- helpers ----------
    def _ensure_initialized(self, history, y_true: np.ndarray):
        if self.n_classes is None:
            try:
                self.n_classes = int(history.model.output_shape[-1])
            except Exception:
                self.n_classes = int(np.max(y_true)) + 1

        if self.class_names is None:
            if self.n_classes == 3:
                self.class_names = ["upright", "transition", "slouch"]
            else:
                self.class_names = [str(i) for i in range(self.n_classes)]

    @staticmethod
    def _normalized(cm_counts: np.ndarray, normalize: Optional[str]):
        if normalize == "true":
            # Row-normalize (per-class recall)
            row_sums = cm_counts.sum(axis=1, keepdims=True).astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                cm = np.divide(cm_counts, row_sums, where=row_sums > 0)
                cm[row_sums.squeeze() == 0] = 0.0
            return cm
        elif normalize == "pred":
            # Column-normalize (per-class precision)
            col_sums = cm_counts.sum(axis=0, keepdims=True).astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                cm = np.divide(cm_counts, col_sums, where=col_sums > 0)
                zero_cols = (col_sums.squeeze() == 0)
                if np.any(zero_cols):
                    cm[:, zero_cols] = 0.0
            return cm
        elif normalize == "all":
            total = cm_counts.sum().astype(np.float64)
            return cm_counts / total if total > 0 else cm_counts.astype(np.float64)
        else:
            return cm_counts.astype(np.float64)

    @staticmethod
    def _balanced_accuracy_from_counts(cm_counts: np.ndarray) -> float:
        """Balanced accuracy over ALL classes (mean row recall over rows with support > 0)."""
        row_sums = cm_counts.sum(axis=1).astype(np.float64)
        diag = np.diag(cm_counts).astype(np.float64)
        mask = row_sums > 0
        if not np.any(mask):
            return float("nan")
        recalls = np.zeros_like(row_sums, dtype=np.float64)
        recalls[mask] = diag[mask] / row_sums[mask]
        return float(np.mean(recalls[mask]))

    @staticmethod
    def _balanced_accuracy_no_upright(cm_counts: np.ndarray) -> float:
        """
        BA over classes 1 & 2 using the FULL matrix (no slicing/plotting subset).
        Equivalent to mean([recall_class1, recall_class2]).
        """
        if cm_counts.shape[0] < 3:
            return float("nan")
        row_sums = cm_counts.sum(axis=1).astype(np.float64)
        diag = np.diag(cm_counts).astype(np.float64)
        mask12 = np.array([False, row_sums[1] > 0, row_sums[2] > 0])
        recalls = np.zeros(3, dtype=np.float64)
        # Safe division; if a class has no support, it contributes 0 and will be excluded in denom
        recalls[1] = diag[1] / row_sums[1] if row_sums[1] > 0 else 0.0
        recalls[2] = diag[2] / row_sums[2] if row_sums[2] > 0 else 0.0
        denom = int(mask12[1]) + int(mask12[2])
        return float((recalls[1] + recalls[2]) / denom) if denom > 0 else float("nan")

def save_confusion_matrix_for_run(history, X, y,
                                  model_tag: str = "cnn",
                                  save_dir: str = "confusion_matrix",
                                  dpi: int = 220,
                                  normalize: str = "true") -> str:
    """
    Convenience helper: compute & save a single-run confusion matrix (full matrix).
    Returns the PNG path.
    """
    cma = ConfusionMatrixAverager(save_dir=save_dir)
    cma.add(history, X, y)
    return cma.save_figure(model_tag=model_tag, dpi=dpi, normalize=normalize)
#----------------------------  

def find_shapes():
    # pick the first IMU file under data/*/
    DATA_ROOT = os.path.join(os.getcwd(), "data")
    imu_glob = os.path.join(DATA_ROOT, 'beep_schedules_*', 'airpods_motion_*.csv')
    imu_list = glob.glob(imu_glob)
    if not imu_list:
        raise FileNotFoundError(f"No IMU CSVs found with pattern: {imu_glob}")

    df_imu = pd.read_csv(imu_list[0])  # use first found file

    t = df_imu.iloc[:, 0].astype(float).to_numpy()
    dt = np.diff(t)
    dt_med = float(np.median(dt)) if dt.size > 0 else (1.0/50.0)
    fs = (1.0 / dt_med) if dt_med > 0 else 50.0
    stride = 0.5
    len_window_sec = 1.5

    win_len_frames = int(round(len_window_sec * fs))      # samples per window
    stride_frames = int(round(stride * fs))    # samples per stride

    # Recompute windows_per_rec from sample counts (more robust than using seconds)
    N = df_imu.shape[0]                           # 18'000
    windows_per_rec = max(0, 1 + (N - win_len_frames) // stride_frames)

    return t, dt_med, fs, win_len_frames, stride_frames, N, windows_per_rec, stride, len_window_sec

def remove_edge_windows(X, y, drop_labels=(0, 2)):
    """
    Remove the first and last sample of every consecutive run whose label is in `drop_labels`.
    By default this trims runs of 0 (upright) and 2 (slouch) and leaves runs of 1 (transition) intact.

    Parameters
    ----------
    X : np.ndarray
        Feature tensor (N, ...). Only the first dimension is used for indexing.
    y : np.ndarray
        Labels of shape (N,) or (N, 1) with values in {0,1,2}.
    drop_labels : tuple[int], optional
        Labels whose run edges should be removed (default: (0, 2)).

    Returns
    -------
    X_out : np.ndarray
        X with trimmed windows removed.
    y_out : np.ndarray
        y with the same windows removed. Keeps column shape (M, 1) if input was (N, 1).

    Notes
    -----
    - If a run has length 1, removing "first and last" removes that single sample.
    - If a run has length 2, both samples are removed.
    """
    X = np.asarray(X)
    y_arr = np.asarray(y)

    # Keep track if y was a column vector to preserve shape on return
    y_was_col = (y_arr.ndim == 2 and y_arr.shape[1] == 1)
    y1d = y_arr.reshape(-1)

    if X.shape[0] != y1d.size:
        raise ValueError(f"X and y must have the same number of windows (got {X.shape[0]} vs {y1d.size}).")

    n = y1d.size
    if n == 0:
        return X, y

    # Identify run starts/ends
    changes = np.flatnonzero(np.diff(y1d) != 0)
    starts = np.r_[0, changes + 1]
    ends   = np.r_[changes, n - 1]

    # Decide which runs to trim (those whose label is in drop_labels)
    run_labels = y1d[starts]
    to_trim = np.isin(run_labels, drop_labels)

    # Build keep mask and drop first/last of the selected runs
    mask = np.ones(n, dtype=bool)
    mask[starts[to_trim]] = False
    mask[ends[to_trim]]   = False

    X_out = X[mask]
    y_out = y_arr[mask, :] if y_was_col else y_arr[mask]

    return X_out, y_out

def drop_quaternion_channels(X_tot: np.ndarray, y_tot: np.ndarray, n_quat: int = 4, copy: bool = False):
    """
    Remove the last `n_quat` feature channels from X_tot (the quaternion channels).
    y_tot is passed through unchanged.

    Parameters
    ----------
    X_tot : np.ndarray
        Shape (N_windows, win_len_frames, n_features)
    y_tot : np.ndarray
        Shape (N_windows, 1) or similar. Returned unchanged.
    n_quat : int, default 4
        Number of trailing channels to drop.
    copy : bool, default False
        If True, returns a copy; otherwise returns a view.

    Returns
    -------
    X_noquat : np.ndarray
        X_tot with the last `n_quat` channels removed.
    y_tot : np.ndarray
        Unchanged.
    """
    if X_tot is None:
        return X_tot, y_tot
    if X_tot.ndim != 3:
        raise ValueError(f"X_tot must be 3D (N, T, C), got shape {X_tot.shape}")
    C = X_tot.shape[-1]
    if C < n_quat:
        raise ValueError(f"X_tot has only {C} channels; cannot drop {n_quat} quaternion channels.")
    X_noquat = X_tot[..., :C - n_quat]
    if copy:
        X_noquat = X_noquat.copy()
    return X_noquat, y_tot

def drop_pitch_channel(X_tot: np.ndarray, y_tot: np.ndarray, copy: bool = False):
    """
    Remove the last feature channel from X_tot (assumed to be the 'pitch' channel).
    y_tot is passed through unchanged.

    Parameters
    ----------
    X_tot : np.ndarray
        Shape (N_windows, win_len_frames, n_features)
    y_tot : np.ndarray
        Shape (N_windows, 1) or similar. Returned unchanged.
    copy : bool, default False
        If True, returns a copy; otherwise returns a view.

    Returns
    -------
    X_nopitch : np.ndarray
        X_tot with the last channel removed.
    y_tot : np.ndarray
        Unchanged.
    """
    if X_tot is None:
        return X_tot, y_tot
    if X_tot.ndim != 3:
        raise ValueError(f"X_tot must be 3D (N, T, C), got shape {X_tot.shape}")

    C = X_tot.shape[-1]
    if C < 1:
        raise ValueError("X_tot has zero channels; cannot drop pitch channel.")

    X_nopitch = X_tot[..., :C - 1]
    if copy:
        X_nopitch = X_nopitch.copy()
    return X_nopitch, y_tot   

def keep_only_gravity_channels(X_tot: np.ndarray, y_tot: np.ndarray, copy: bool = False):
    """
    Keep only grav_x, grav_y, grav_z from X_tot and drop all other channels.
    y_tot is passed through unchanged.

    Assumes the _d* file column order provided. After removing the timestamp column
    (which your code already does via df_imu.iloc[:, 1:]), the per-frame feature
    order is expected to be:
        [rot_x, rot_y, rot_z, acc_x, acc_y, acc_z,
         grav_x, grav_y, grav_z,
         pitch_rad, quat_rel_x, quat_rel_y, quat_rel_z, quat_rel_w]
    i.e., C == 14 and gravity sits at indices 6:9.

    If a timestamp channel accidentally remains in X_tot (C == 15), the function
    auto-adjusts and keeps indices 7:10.

    Parameters
    ----------
    X_tot : np.ndarray
        Shape (N_windows, win_len_frames, n_features)
    y_tot : np.ndarray
        Shape (N_windows, 1) or similar. Returned unchanged.
    copy : bool, default False
        If True, returns a copy; otherwise returns a view.

    Returns
    -------
    X_grav : np.ndarray
        X_tot reduced to just [grav_x, grav_y, grav_z] (shape: N x T x 3).
    y_tot : np.ndarray
        Unchanged.
    """
    if X_tot is None:
        return X_tot, y_tot
    if X_tot.ndim != 3:
        raise ValueError(f"X_tot must be 3D (N, T, C), got shape {X_tot.shape}")

    C = X_tot.shape[-1]
    if C == 14:
        start = 6      # grav_x at 6, grav_y at 7, grav_z at 8
    elif C == 15:
        start = 7      # timestamp still present in X_tot; shift by +1
    else:
        raise ValueError(
            f"Unexpected feature count C={C}. Expected 14 (no timestamp) or 15 (timestamp present) "
            "per the _d* format: timestamp, rot[3], acc[3], grav[3], pitch, quat[4]."
        )

    end = start + 3
    if end > C:
        raise ValueError("Not enough channels to select grav_x..grav_z with the expected layout.")

    # Use a contiguous slice so the default is a view; honor copy flag if requested.
    X_grav = X_tot[..., start:end]
    if copy:
        X_grav = X_grav.copy()
    return X_grav, y_tot

# ----------SCALE LARGE CHANELS DOWN------------
def calculate_std(matching_folders):
    """
    Compute per-channel standard deviations from airpods_motion_d*.csv (NOT ds files),
    aggregate them per participant and overall, and DISPLAY a plot of participants'
    average std per channel.

    Changes vs. previous version:
    - Excludes 'pitch_rad' from ALL std computations and from the plot.
    - Still excludes the 4 relative quaternion channels ('quat_rel_x','quat_rel_y',
      'quat_rel_z','quat_rel_w') from the plot (but they are included in stats if present).

    Parameters
    ----------
    matching_folders : list[str]
        A list of participant base names (e.g., ["Abi", "Ben"]) OR explicit session folder
        paths (e.g., ["data/beep_schedules_Abi0", ...]). For each name, all matching
        session folders (e.g., Abi0..Abi3) are collected under ./data/beep_schedules_*.

    Returns
    -------
    per_file_std : dict
        { participant_display_name :
            { session_folder_basename : { channel : std, ... }, ... },
          ...
        }

    per_participant_avg_std : dict
        For each participant and channel, includes BOTH the average std (across that
        participant’s sessions) and the std-of-std across that participant’s sessions.
        {
          participant_display_name : {
              channel : {
                  "mean_std" : float,
                  "std_of_std" : float,
                  "n_sessions" : int
              }, ...
          }, ...
        }

    overall_avg_std : dict
        For each channel, includes BOTH the overall mean std across participants and the
        std-of-std *across participants* (computed over the per-participant mean_std values).
        {
          channel : {
              "mean_std" : float,
              "std_of_std" : float,
              "n_participants" : int
          }, ...
        }

    Notes
    -----
    * Only files matching airpods_motion_d[0-9]*.csv are used (explicitly excludes ds files).
    * The timestamp column is excluded automatically (common names: t_sec, timestamp, time, ...).
    * Only numeric columns are considered (non-numeric coerced to NaN and ignored).
    * All std computations use population std (ddof=0) and ignore NaNs.
    * 'pitch_rad' is excluded from statistics and from the plot.
    * The plot pops up (plt.show()) and is NOT saved.
    * The 4 relative quaternion channels are excluded from the plot.
    """
    import os
    import re
    import glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # ---- configuration ----
    DATA_ROOT = os.path.join(os.getcwd(), "data")
    BEEP_GLOB = os.path.join(DATA_ROOT, "beep_schedules_*")  # where sessions live
    PITCH_EXCLUDE = {"pitch_rad"}  # case-insensitive match handled below
    REL_QUAT_EXCLUDE_FROM_PLOT = {"quat_rel_x", "quat_rel_y", "quat_rel_z", "quat_rel_w"}

    # ---- helpers ----
    def _natural_key(path_or_name: str):
        """Split digits so names like 'Claire10' sort after 'Claire2'."""
        name = os.path.basename(os.fspath(path_or_name)).lower()
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', name)]

    def _detect_time_col(df, candidates=("t_sec", "timestamp", "time", "time_sec", "epoch", "epoch_s")):
        lower = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in lower:
                return lower[c.lower()]
        return df.columns[0]  # fallback to first column

    def _participant_key_from_basename(basename: str) -> str:
        """
        Extract a stable participant key from a session folder name like 'beep_schedules_Abi0' -> 'abi'.
        """
        b = os.path.basename(basename)
        b = re.sub(r'^(beep_schedules[_\-]*)', '', b, flags=re.IGNORECASE)
        b = re.sub(r'[_\-]+$', '', b)
        b = re.sub(r'[_\-]?[0-9]{1,2}$', '', b)     # strip trailing small integer tag
        b = re.sub(r'^[0-9]{1,2}[_\-]?', '', b)     # strip leading small integer tag if any
        return b.strip("_- ").lower()

    def _gather_all_session_dirs():
        return sorted(
            [p for p in glob.glob(BEEP_GLOB) if os.path.isdir(p)],
            key=_natural_key
        )

    def _resolve_sessions_for_name(name_key: str, all_dirs):
        """
        Map a participant base name (case-insensitive) to all its session folders.
        """
        key = name_key.strip().lower()
        out = [d for d in all_dirs if _participant_key_from_basename(d) == key]
        return sorted(out, key=_natural_key)

    # ---- build participant -> [session folders] map ----
    all_session_dirs = _gather_all_session_dirs()
    participants = {}  # canonical_key -> {"display": display_name, "sessions": [dirs...]}

    for entry in matching_folders:
        if os.path.isdir(entry):
            base = os.path.basename(os.path.normpath(entry))
            canon = _participant_key_from_basename(base)
            sessions = _resolve_sessions_for_name(canon, all_session_dirs)
            display = re.sub(r'[_\-]+$', '', re.sub(r'^(beep_schedules[_\-]*)', '', base, flags=re.IGNORECASE))
        else:
            canon = entry.strip().lower()
            sessions = _resolve_sessions_for_name(canon, all_session_dirs)
            display = entry  # keep caller's casing

        if not sessions:
            continue

        if canon not in participants:
            participants[canon] = {"display": display, "sessions": []}
        # dedupe while preserving order
        seen = set(participants[canon]["sessions"])
        for s in sessions:
            if s not in seen:
                participants[canon]["sessions"].append(s)
                seen.add(s)

    # Early exit if nothing matched
    if not participants:
        return {}, {}, {}

    # ---- compute per-file stds ----
    per_file_std = {}
    for canon_key in sorted(participants.keys(), key=lambda k: participants[k]["display"].lower()):
        display_name = participants[canon_key]["display"]
        session_dirs = participants[canon_key]["sessions"]

        file_stats_for_participant = {}
        for sess_dir in session_dirs:
            # only d[0-9]* csv (exclude ds*)
            csv_files = sorted(
                glob.glob(os.path.join(sess_dir, "airpods_motion_d[0-9]*.csv")),
                key=_natural_key
            )
            if not csv_files:
                continue

            csv_path = csv_files[0]  # take first in natural order
            try:
                df = pd.read_csv(csv_path, low_memory=False)
            except Exception:
                continue  # skip unreadable

            if df.shape[0] == 0 or df.shape[1] == 0:
                continue

            time_col = _detect_time_col(df)

            # --- feature columns excluding the time column AND 'pitch_rad' (case-insensitive) ---
            lower_map = {c.lower(): c for c in df.columns}
            feat_cols_all = [c for c in df.columns if c != time_col]
            feat_cols = [c for c in feat_cols_all if c.lower() not in PITCH_EXCLUDE]
            if not feat_cols:
                continue

            # numeric only; errors -> NaN; compute population std ignoring NaN
            X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
            valid_cols = [c for c in feat_cols if X[c].notna().any()]
            if not valid_cols:
                continue

            stats = {}
            for c in valid_cols:
                col = X[c].to_numpy(dtype=float, copy=False)
                sigma = float(np.nanstd(col, ddof=0))
                if np.isfinite(sigma):
                    stats[c] = sigma

            if stats:
                file_stats_for_participant[os.path.basename(sess_dir)] = dict(
                    sorted(stats.items(), key=lambda kv: kv[0].lower())
                )

        if file_stats_for_participant:
            per_file_std[display_name] = file_stats_for_participant

    # ---- per-participant aggregates: mean_std and std_of_std (across that participant's sessions) ----
    from collections import defaultdict as _dd
    per_participant_avg_std = {}
    for disp_name, files_dict in per_file_std.items():
        acc = _dd(list)
        for _sess, ch_stats in files_dict.items():
            for ch, v in ch_stats.items():
                if np.isfinite(v):
                    acc[ch].append(float(v))
        if not acc:
            continue

        ch_agg = {}
        for ch, vals in acc.items():
            arr = np.array(vals, dtype=float)
            mean_std = float(np.nanmean(arr)) if arr.size else float("nan")
            std_of_std = float(np.nanstd(arr, ddof=0)) if arr.size else float("nan")
            ch_agg[ch] = {
                "mean_std": mean_std,
                "std_of_std": std_of_std,
                "n_sessions": int(arr.size),
            }

        per_participant_avg_std[disp_name] = {k: ch_agg[k] for k in sorted(ch_agg.keys(), key=lambda s: s.lower())}

    # ---- overall aggregates across participants (computed over participant mean_std values) ----
    channel_to_participant_means = _dd(list)
    for _pname, ch_dict in per_participant_avg_std.items():
        for ch, d in ch_dict.items():
            val = d.get("mean_std", np.nan)
            if np.isfinite(val):
                channel_to_participant_means[ch].append(float(val))

    overall_avg_std = {}
    for ch, vals in channel_to_participant_means.items():
        arr = np.array(vals, dtype=float)
        if arr.size == 0:
            continue
        overall_avg_std[ch] = {
            "mean_std": float(np.nanmean(arr)),
            "std_of_std": float(np.nanstd(arr, ddof=0)),
            "n_participants": int(arr.size),
        }

    # ---- sort for deterministic returned dicts ----
    def _sort_nested(d):
        return {k: d[k] for k in sorted(d.keys(), key=lambda x: str(x).lower())}

    per_file_std = {pk: _sort_nested(per_file_std[pk]) for pk in sorted(per_file_std.keys(), key=lambda x: str(x).lower())}
    per_participant_avg_std = {pk: _sort_nested(per_participant_avg_std[pk]) for pk in sorted(per_participant_avg_std.keys(), key=lambda x: str(x).lower())}
    overall_avg_std = _sort_nested(overall_avg_std)

    return per_file_std, per_participant_avg_std, overall_avg_std


def X_and_y(type, list_comb, label_anchor):
    """
    Build X (windows) and y (labels) across selected folders.

    Parameters
    ----------
    type : {'train','val','test'}
    list_comb : list[str]
    label_anchor : {'start','center','end'}, default 'end'
        - 'start' : label at the window start time  (t = t0 + i*stride)
        - 'center': label at the window mid time    (t = t0 + i*stride + 0.5*len_window_sec)
        - 'end'   : label at the window end time    (t = t0 + i*stride + len_window_sec)
                     (this matches your original function)
    """
    # --- normalize anchor choice & map to offset in SECONDS ---
    anchor = str(label_anchor).strip().lower()
    if anchor not in {"start", "center", "end"}:
        print(f"[warn] Unknown label_anchor={anchor!r}; falling back to 'end'.")
        anchor = "end"

    matching_folders, n_folders = folders_tot(type, list_comb)
    _, _, _, win_len_frames, stride_frames, _, windows_per_rec, stride, len_window_sec = find_shapes()
    
    ## Calculate std from 'train' participants
    # ...

    # seconds offset to add to t0 + i*stride for the label timestamp
    if anchor == "start":
        anchor_offset_sec = 0.0
    elif anchor == "center":
        anchor_offset_sec = 0.5 * float(len_window_sec)
    else:  # "end"
        anchor_offset_sec = float(len_window_sec)

    X_tot = None
    y_tot = np.zeros((n_folders * windows_per_rec, 1), dtype=int)

    for index, folder_path in enumerate(matching_folders):
        if not os.path.isdir(folder_path):
            continue

        # use the new inferred files (single canonical one per session)
        event_files = sorted(glob.glob(os.path.join(folder_path, 'events_inferred_*.csv')), key=_natural_key)
        # allow both d* and non-d* IMU filenames
        imu_files   = sorted(glob.glob(os.path.join(folder_path, 'airpods_motion_d*.csv')), key=_natural_key)
        if not event_files or not imu_files:
            print(f"⚠️ Missing files in {folder_path}")
            continue
        
        df_event = pd.read_csv(event_files[0])
        df_imu   = pd.read_csv(imu_files[0])

        # --- IMU time axis (optional: sanitize to non-decreasing) ---
        t_imu = df_imu.iloc[:, 0].astype(float).to_numpy()
        if np.any(np.diff(t_imu) < 0):
            t_imu = np.maximum.accumulate(t_imu)
        t0 = float(t_imu[0])

        # --- Events are ALREADY on the IMU time axis -> no re-alignment ---
        ev = df_event[['t_sec','event']].copy()
        ev = ev.sort_values('t_sec', kind='mergesort').reset_index(drop=True)

        # ensure initial upright at t0 (harmless if already present)
        if ev.iloc[0]['event'] != 'UPRIGHT_HOLD_START':
            ev = pd.concat([
                pd.DataFrame({'t_sec':[t0], 'event':['UPRIGHT_HOLD_START']}),
                ev
            ], ignore_index=True)
            ev = ev.sort_values('t_sec', kind='mergesort').reset_index(drop=True)

        times = ev['t_sec'].astype(float).to_numpy()
        names = ev['event'].astype(str).tolist()
        
        # --- Labels: state at chosen ANCHOR time on IMU axis (seconds) ---
        # Using seconds like your original avoids tail indexing issues.
        m = 0.0  # manual offsets already encode margins
        labels_array = np.zeros((windows_per_rec, 1), dtype=int)
        for i in range(windows_per_rec):
            current_time = t0 + i * float(stride) + anchor_offset_sec
            labels_array[i, 0] = label_at_time(current_time, names, times, m)
        
        y_tot[index*windows_per_rec:(index+1)*windows_per_rec, 0:1] = labels_array

        # --- Features: drop the time column by position (your current approach) ---
        Xsig = df_imu.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
        n_ch = Xsig.shape[1]

        if X_tot is None:
            X_tot = np.zeros((n_folders * windows_per_rec, win_len_frames, n_ch), dtype=np.float32)

        # --- Windowing ---
        base = index * windows_per_rec
        max_crop = max(0, min(stride_frames - 1, win_len_frames - 1))
        for i in range(windows_per_rec):
            start = i * stride_frames
            end = start + win_len_frames
            win = Xsig[start:end, :]

            # pad if needed
            if win.shape[0] == 0:
                last = Xsig[-1:, :]
                win = np.repeat(last, win_len_frames, axis=0)
            elif win.shape[0] < win_len_frames:
                need = win_len_frames - win.shape[0]
                win = np.concatenate([win, np.repeat(win[-1:, :], need, axis=0)], axis=0)

            # optional left-mirror augmentation
            r = random.randint(0, max_crop) if max_crop > 0 else 0
            if r > 0:
                left = win[:r, :]
                win = np.concatenate([np.flip(left, axis=0), win[r:, :]], axis=0)

            X_tot[base + i, :, :] = win

    #return X_tot, y_tot
    X_tot, y_tot = drop_quaternion_channels(X_tot, y_tot)  # 14 → 10 chans
    X_tot, y_tot = drop_pitch_channel(X_tot, y_tot)        # 10 → 9 chans
    #X_tot, y_tot = keep_only_gravity_channels(X_tot, y_tot)  # 14 → 3

    def scale_group_to_base_std(
        base,
        to_be_scaled,
        *,
        agg="mean",                     # "mean" or "median" across participants
        weight_by_sessions=True,        # if mean, weight each participant by #selected sessions
        eps=1e-8,                       # floor for denominator
        clip=10.0,                      # clamp ratios to [1/clip, clip]; set None to disable
        verbose=True,
        return_info=False               # set True to get (X_scaled, info)
    ):
        """
        Returns a scaled copy of X_tot, multiplying ONLY the windows belonging to the
        explicitly selected sessions in `to_be_scaled`.

        You can pass items like:
            base        = ["Abi[0-3]", "Ben[0-3]", "Cara[0-3]"]
            to_be_scaled= ["Claire[0-2]"]        # scales 0,1,2 but not 3

        Notation in square brackets supports:
            - Ranges:   [0-3]
            - Lists:    [0,2,3]
            - Mix:      [0,2-3]
        If no brackets are provided (e.g., "Abi"), all sessions for that participant
        are considered for the stats (and scaled if in to_be_scaled).

        Relies on outer-scope variables of X_and_y:
            - X_tot (n_windows x win_len x n_channels)
            - matching_folders (list of session folder paths, natural-sorted)
            - windows_per_rec (int)

        Returns
        -------
        X_scaled : np.ndarray
            Scaled copy of X_tot.
        (optional) info : dict
            Diagnostics including scale factors, group stds, and missing selections.
        """
        import os, re, glob
        import numpy as np
        import pandas as pd
        from collections import defaultdict

        # ---------- canonicalizers ----------
        def _canon_name(s):
            b = str(s)
            b = re.sub(r'^(beep_schedules[_\-]*)', '', b, flags=re.IGNORECASE)
            b = re.sub(r'[_\-]+$', '', b)
            b = re.sub(r'[_\-]?[0-9]{1,2}$', '', b)
            b = re.sub(r'^[0-9]{1,2}[_\-]?', '', b)
            return b.strip("_- ").lower()

        def _canon_ch(s):
            return str(s).strip().lower()

        def _natural_key(path_or_name: str):
            name = os.path.basename(os.fspath(path_or_name)).lower()
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', name)]

        def _session_index_from_basename(basename: str):
            m = re.search(r'(\d+)$', os.path.basename(str(basename)))
            return int(m.group(1)) if m else None

        # ---------- parse specs like "Abi[0-3]" / "Ben[1,3]" ----------
        def _parse_spec_list(items):
            """
            Returns a dict:
              { canon_name : {"display": <name>, "indices": None or set[int]} }
            """
            out = {}
            for raw in items:
                s = str(raw).strip()
                m = re.match(r'^\s*(?:beep_schedules[_\-]*)?([A-Za-z0-9_\-]+)(?:\[(.*?)\])?\s*$', s)
                if m:
                    name = m.group(1)
                    sel  = m.group(2)
                else:
                    name = s
                    sel  = None

                indices = None
                if sel is not None and sel != "":
                    indices = set()
                    for tok in sel.split(","):
                        tok = tok.strip()
                        if not tok:
                            continue
                        if "-" in tok:
                            a, b = tok.split("-", 1)
                            try:
                                ai, bi = int(a), int(b)
                            except Exception:
                                continue
                            if ai <= bi:
                                rng = range(ai, bi + 1)
                            else:
                                rng = range(bi, ai + 1)
                            indices.update(rng)
                        else:
                            try:
                                indices.add(int(tok))
                            except Exception:
                                pass

                canon = _canon_name(name)
                if canon in out:
                    old = out[canon]["indices"]
                    if old is None or indices is None:
                        out[canon]["indices"] = None
                    else:
                        out[canon]["indices"].update(indices)
                else:
                    out[canon] = {"display": name, "indices": indices}
            return out

        base_map = _parse_spec_list(base)
        tbs_map  = _parse_spec_list(to_be_scaled)

        # ---------- channel names after your drops (quat_rel_* + pitch_rad) ----------
        first_cols = None
        for fp in matching_folders:
            imu_files = sorted(glob.glob(os.path.join(fp, 'airpods_motion_d*.csv')), key=_natural_key)
            if not imu_files:
                continue
            try:
                cols = pd.read_csv(imu_files[0], nrows=1).columns.tolist()
            except Exception:
                continue
            if len(cols) >= 2:
                first_cols = cols[1:]  # drop time column
                break

        if first_cols is None:
            if verbose:
                print("[std-scale] Could not read any IMU header; skipping scaling.")
            return (X_tot, {}) if return_info else X_tot

        drop_set = {"quat_rel_x", "quat_rel_y", "quat_rel_z", "quat_rel_w", "pitch_rad"}
        channel_names_after_drops = [c for c in first_cols if _canon_ch(c) not in drop_set]

        if len(channel_names_after_drops) != X_tot.shape[-1]:
            if verbose:
                print(f"[std-scale] Header-derived channel count ({len(channel_names_after_drops)}) "
                      f"does not match X_tot last dim ({X_tot.shape[-1]}). Skipping scaling.")
            return (X_tot, {}) if return_info else X_tot

        # ---------- compute per-file stds for union of requested participants ----------
        names_for_stats = sorted({d["display"] for d in base_map.values()} |
                                 {d["display"] for d in tbs_map.values()})
        try:
            per_file_std, _pp, _ov = calculate_std(names_for_stats)
        except Exception as e:
            if verbose:
                print(f"[std-scale] calculate_std failed: {e}; skipping scaling.")
            return (X_tot, {}) if return_info else X_tot

        # Canonicalize per_file_std keys
        per_file_std_by_canon = defaultdict(dict)  # {canon: {sess_basename: {ch:std}}}
        for disp, sess_dict in per_file_std.items():
            canon = _canon_name(disp)
            per_file_std_by_canon[canon].update(sess_dict)

        # ---------- aggregate group stds over SELECTED sessions ----------
        def _aggregate_group_from_selected(per_file_canon, spec_map):
            ch_to_vals = defaultdict(list)  # per-channel list of per-participant means
            ch_to_w    = defaultdict(list)  # weights per participant (num selected sessions with that channel)
            missing = []                    # participants with no selected sessions found
            missing_sessions = {}           # {display: sorted(list-of-indices-that-were-requested-but-missing)}

            for canon, d in spec_map.items():
                sess_stats = per_file_canon.get(canon, {})
                allowed = d["indices"]  # None -> all sessions
                # collect per-session values per channel for selected sessions
                ch_to_session_vals = defaultdict(list)
                got_any = False
                present_indices = set()

                for sess_base, ch_stats in sess_stats.items():
                    sidx = _session_index_from_basename(sess_base)
                    if allowed is None or (sidx is not None and sidx in allowed):
                        got_any = True
                        if sidx is not None:
                            present_indices.add(sidx)
                        for ch, v in ch_stats.items():
                            if np.isfinite(v):
                                ch_to_session_vals[_canon_ch(ch)].append(float(v))

                if not got_any:
                    missing.append(d["display"])
                    if d["indices"] not in (None, set()):
                        # report missing indices exactly
                        missing_sessions[d["display"]] = sorted(list(d["indices"])) if d["indices"] else []
                    continue

                if d["indices"]:
                    miss_idx = sorted(list(d["indices"] - present_indices))
                    if miss_idx:
                        missing_sessions[d["display"]] = miss_idx

                # per-participant mean over their selected sessions
                for chn, vals in ch_to_session_vals.items():
                    mean_std = float(np.mean(vals))
                    w = float(len(vals))
                    ch_to_vals[chn].append(mean_std)
                    ch_to_w[chn].append(w)

            # finalize group aggregation across participants
            group_std = {}
            for chn, vals in ch_to_vals.items():
                arr = np.array(vals, dtype=float)
                if agg == "median":
                    group_std[chn] = float(np.median(arr))
                else:
                    if weight_by_sessions:
                        ws = np.array(ch_to_w[chn], dtype=float)
                        sw = ws.sum()
                        group_std[chn] = float((arr * ws).sum() / sw) if sw > 0 else float(np.mean(arr))
                    else:
                        group_std[chn] = float(np.mean(arr))

            return group_std, missing, missing_sessions

        base_std, miss_base, miss_base_sess = _aggregate_group_from_selected(per_file_std_by_canon, base_map)
        tbs_std,  miss_tbs,  miss_tbs_sess  = _aggregate_group_from_selected(per_file_std_by_canon, tbs_map)

        # ---------- ratios base / to_be_scaled, aligned to X_tot order ----------
        common = set(base_std.keys()).intersection(tbs_std.keys())
        ratio_by_ch = {}
        for ch in common:
            b = base_std[ch]; t = tbs_std[ch]
            if not np.isfinite(b): b = 0.0
            if not np.isfinite(t): t = 0.0
            if t <= eps and b <= eps:
                r = 1.0
            elif t <= eps:
                r = b / eps
            else:
                r = b / max(t, eps)
            if clip is not None and clip > 1:
                r = min(max(r, 1.0/clip), clip)
            ratio_by_ch[ch] = float(r)

        scale_vec = np.ones(len(channel_names_after_drops), dtype=float)
        scale_per_channel = {}
        channels_not_scaled = []
        for i, name in enumerate(channel_names_after_drops):
            key = _canon_ch(name)
            if key in ratio_by_ch:
                scale_vec[i] = ratio_by_ch[key]
                scale_per_channel[name] = ratio_by_ch[key]
            else:
                channels_not_scaled.append(name)

        # ---------- build window mask ONLY for selected sessions in to_be_scaled ----------
        n_windows = X_tot.shape[0]
        mask = np.zeros(n_windows, dtype=bool)
        for idx, folder_path in enumerate(matching_folders):
            base_name = os.path.basename(os.path.normpath(folder_path))
            canon = _canon_name(base_name)
            sidx  = _session_index_from_basename(base_name)
            spec  = tbs_map.get(canon)
            if spec is None:
                continue
            allowed = spec["indices"]
            selected = (allowed is None) or (sidx is not None and sidx in allowed)
            if selected:
                start = idx * windows_per_rec
                end   = min(start + windows_per_rec, n_windows)
                mask[start:end] = True

        # ---------- apply scaling ----------
        X_scaled = X_tot.copy()
        scale_tensor = scale_vec[np.newaxis, np.newaxis, :].astype(X_scaled.dtype, copy=False)
        X_scaled[mask, :, :] *= scale_tensor

        if verbose:
            print(f"[std-scale] Scaled {mask.sum()} of {mask.size} windows "
                  f"({mask.sum()/max(1,mask.size):.1%}). "
                  f"Channels scaled: {len(scale_per_channel)}; skipped: {len(channels_not_scaled)}.")

        info = {
            "scale_per_channel": scale_per_channel,
            "base_group_std":    {c: base_std.get(_canon_ch(c), np.nan) for c in channel_names_after_drops},
            "tbs_group_std":     {c: tbs_std.get(_canon_ch(c), np.nan)  for c in channel_names_after_drops},
            "missing_participants": {"base": miss_base, "to_be_scaled": miss_tbs},
            "missing_sessions": {"base": miss_base_sess, "to_be_scaled": miss_tbs_sess},
            "channels_not_scaled": channels_not_scaled
        }
        return (X_scaled, info) if return_info else X_scaled
    
    X_tot = scale_group_to_base_std(
        base=["Abi[0-3]","Claire[0-3]"],
        to_be_scaled=["Dario[0-3]","David[0-3]","Ivan[0-3]","Mohid[0-2]"]
    )

    return remove_edge_windows(X_tot, y_tot)
