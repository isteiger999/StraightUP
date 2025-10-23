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
        current_matches = glob.glob(pattern)
        
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

    mean = {"loss": 0.0, "balanced_acc": 0.0}
    std = {"loss": 0.0, "balanced_acc": 0.0}
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
    Insert a 'd' after the second underscore:
      airpods_motion_1760734866.csv -> airpods_motion_d1760734866.csv
    Fallback: file.csv -> file_d.csv
    """
    folder, fname = os.path.split(csv_path)
    parts = fname.split("_", 2)  # ["airpods", "motion", "1760734866.csv"]
    if len(parts) >= 3:
        new_fname = f"{parts[0]}_{parts[1]}_d{parts[2]}"
    else:
        base, ext = os.path.splitext(fname)
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

def _extract_upright_windows(events_df: pd.DataFrame,
                             t_end: float,
                             start_event: str = "UPRIGHT_HOLD_START",
                             end_events: Tuple[str, ...] = ("SLOUCH_START", "UPRIGHT_HOLD_START"),
                             min_window_seconds: float = 0.5) -> List[Tuple[float, float]]:
    """
    Scan events chronologically and return (start, end) windows for upright holds.
    A window starts at UPRIGHT_HOLD_START and ends at the next SLOUCH_START
    (or at the next UPRIGHT_HOLD_START if it appears earlier). If the recording
    ends while upright, close the last window at t_end.
    """
    df = events_df.copy()
    if "t_sec" not in df.columns or "event" not in df.columns:
        return []

    df["t_sec"] = pd.to_numeric(df["t_sec"], errors="coerce")
    df = df.dropna(subset=["t_sec"]).sort_values("t_sec")

    windows: List[Tuple[float, float]] = []
    current_start: Optional[float] = None

    for _, row in df.iterrows():
        ev = str(row["event"])
        ts = float(row["t_sec"])

        if ev == start_event:
            # if we already were in an upright window, close it here (defensive)
            if current_start is not None and ts > current_start:
                dur = ts - current_start
                if dur >= min_window_seconds:
                    windows.append((current_start, ts))
            current_start = ts
        elif ev in end_events and current_start is not None and ts > current_start:
            dur = ts - current_start
            if dur >= min_window_seconds:
                windows.append((current_start, ts))
            current_start = None

    # If recording ends while upright, close the last window at t_end
    if current_start is not None and t_end > current_start:
        dur = t_end - current_start
        if dur >= min_window_seconds:
            windows.append((current_start, t_end))

    return windows

def _upright_segments(events_df: pd.DataFrame, t_end: float) -> List[Tuple[float, float]]:
    """
    Return epoch segments driven by UPRIGHT_HOLD_START:
      [(u0, u1), (u1, u2), ..., (u_last, t_end)]
    Each segment receives the baseline computed from its own upright-only window.
    """
    if "t_sec" not in events_df.columns or "event" not in events_df.columns:
        return []

    df = events_df[events_df["event"] == "UPRIGHT_HOLD_START"].copy()
    if df.empty:
        return []

    u = pd.to_numeric(df["t_sec"], errors="coerce").dropna().sort_values().tolist()
    segments: List[Tuple[float, float]] = []
    for i, s in enumerate(u):
        e = u[i + 1] if i + 1 < len(u) else t_end
        if e > s:
            segments.append((float(s), float(e)))
    return segments

def calculate_deltas(
    df_imu: pd.DataFrame,
    *,
    csv_path: Optional[str] = None,
    events_path: Optional[str] = None,
    timestamp_col: Optional[str] = None,   # auto-detected if None
    id_cols: Optional[List[str]] = None,   # preserved (not baseline-subtracted)
    baseline_method: str = "mean",         # "mean" or "median"
    min_window_seconds: float = 0.5,
    scope: str = "per_upright_epoch",      # default: per-epoch rebaselining
    apply_global_to_gaps: bool = True,     # apply global upright baseline outside epochs
    reuse_last_baseline: bool = True,      # if an epoch has no upright samples
    add_epoch_id: bool = False,            # optional debug column
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float], List[Tuple[float, float]]]:
    """
    Convert IMU channels to deltas relative to an upright baseline.

    scope="per_upright_epoch" (default):
        For each UPRIGHT_HOLD_START at time u_i:
          • Baseline = aggregate (mean/median) over the upright-only window that begins at u_i
            and ends at the next SLOUCH_START or next UPRIGHT_HOLD_START (whichever comes first).
          • Apply that baseline to the entire epoch [u_i, u_{i+1}) (upright + following slouch).
        Rows not covered by any epoch receive the global upright baseline if apply_global_to_gaps=True.

    scope="global":
        One baseline per channel = aggregate over ALL upright windows (fallback: whole file).
        Apply everywhere.

    Returns
    -------
    df_delta : DataFrame
        Copy of df_imu with numeric (non-id) columns baseline-subtracted.
        If add_epoch_id=True, includes an 'epoch_id' column (-1 for gap/global rows).
    baseline : dict
        The global upright baseline used (still useful for inspection/fallbacks).
    windows : list[(start, end)]
        Upright-only windows used to compute per-epoch baselines (and the global one).
    """
    # --- detect timestamp column if not provided ---
    if not timestamp_col:
        timestamp_col = _detect_timestamp_col(df_imu)
    if not timestamp_col or timestamp_col not in df_imu.columns:
        raise ValueError("Could not detect a timestamp column (looked for 't_sec', 'timestamp', 'time', ...).")

    # Identify events file (auto-pick best overlap in same folder when not given)
    ev_path = events_path
    if ev_path is None and csv_path is not None:
        ev_path = _choose_events_file_for(csv_path, df_imu, timestamp_col=timestamp_col)

    events_df = None
    if ev_path is not None and os.path.isfile(ev_path):
        try:
            events_df = pd.read_csv(ev_path, usecols=["t_sec", "event"])
        except Exception:
            events_df = None

    # Time vector & boundaries
    t_series = pd.to_numeric(df_imu[timestamp_col], errors="coerce").dropna()
    if t_series.empty:
        raise ValueError(f"Timestamp column '{timestamp_col}' contains no numeric values.")
    t_min = float(t_series.min()); t_end = float(t_series.max())
    t = pd.to_numeric(df_imu[timestamp_col], errors="coerce").values

    # Upright-only windows (for baselines) and epoch segments (for application)
    windows: List[Tuple[float, float]] = []
    segments: List[Tuple[float, float]] = []
    if events_df is not None and not events_df.empty:
        windows = _extract_upright_windows(events_df, t_end=t_end, min_window_seconds=min_window_seconds)
        segments = _upright_segments(events_df, t_end=t_end)

    # Columns to delta-transform
    if id_cols is None:
        id_cols = [timestamp_col]
    id_set = set(id_cols)
    numeric_cols = df_imu.select_dtypes(include=[np.number]).columns.tolist()
    delta_cols = [c for c in numeric_cols if c not in id_set]

    def agg(df_sub: pd.DataFrame) -> Dict[str, float]:
        if not delta_cols:
            return {}
        if baseline_method == "median":
            return df_sub[delta_cols].median(numeric_only=True).to_dict()
        return df_sub[delta_cols].mean(numeric_only=True).to_dict()

    # Global upright baseline (used for gaps and as fallback)
    if windows:
        mask_upright_all = np.zeros(len(df_imu), dtype=bool)
        for (s, e) in windows:
            mask_upright_all |= (t >= s) & (t < e)
        global_baseline = agg(df_imu.loc[mask_upright_all]) if mask_upright_all.any() else agg(df_imu)
    else:
        global_baseline = agg(df_imu)

    df_delta = df_imu.copy()

    # --- scope: global (simple path) ---
    if scope == "global" or not segments or events_df is None or events_df.empty:
        for c in delta_cols:
            b = global_baseline.get(c, 0.0)
            if pd.notna(b):
                df_delta[c] = df_delta[c] - b
        if add_epoch_id:
            df_delta["epoch_id"] = -1
        if verbose:
            source = "upright_windows" if windows else "global"
            print(f"Δ-baseline: {baseline_method} / global from {source} "
                  f"(windows={len(windows)}, samples_used={len(df_imu)}). "
                  f"Transformed {len(delta_cols)} channels. Time='{timestamp_col}'.")
        return df_delta, global_baseline, windows

    # --- scope: per_upright_epoch ---
    baseline_window_by_start: Dict[float, Tuple[float, float]] = {s: (s, e) for (s, e) in windows}
    touched = np.zeros(len(df_imu), dtype=bool)
    epoch_ids = np.full(len(df_imu), -1, dtype=int)  # -1 = not in any epoch
    last_baseline_vals: Optional[Dict[str, float]] = None
    epochs_applied = 0

    for epoch_idx, (seg_start, seg_end) in enumerate(segments):
        # Baseline window for this epoch (upright-only)
        bw = baseline_window_by_start.get(seg_start)
        if bw is not None:
            b_start, b_end = bw
            m_b = (t >= b_start) & (t < b_end)
            if m_b.any():
                baseline_vals = agg(df_imu.loc[m_b])
                last_baseline_vals = baseline_vals
            else:
                baseline_vals = last_baseline_vals if (reuse_last_baseline and last_baseline_vals is not None) else global_baseline
        else:
            baseline_vals = last_baseline_vals if (reuse_last_baseline and last_baseline_vals is not None) else global_baseline

        # Apply baseline to the entire epoch [seg_start, seg_end)
        m_seg = (t >= seg_start) & (t < seg_end)
        if m_seg.any():
            for c in delta_cols:
                b = baseline_vals.get(c, 0.0)
                if pd.notna(b):
                    df_delta.loc[m_seg, c] = df_delta.loc[m_seg, c] - b
            touched |= m_seg
            epoch_ids[m_seg] = epoch_idx
            epochs_applied += 1

    # Apply global upright baseline to gaps (outside any epoch)
    if apply_global_to_gaps:
        m_gap = ~touched
        if m_gap.any():
            for c in delta_cols:
                b = global_baseline.get(c, 0.0)
                if pd.notna(b):
                    df_delta.loc[m_gap, c] = df_delta.loc[m_gap, c] - b
            # epoch_ids for gaps remain -1

    if add_epoch_id:
        df_delta["epoch_id"] = epoch_ids

    if verbose:
        print(f"Δ-baseline: {baseline_method} / per_upright_epoch "
              f"(epochs_applied={epochs_applied}, windows={len(windows)}). "
              f"Gaps->global={apply_global_to_gaps}. "
              f"Transformed {len(delta_cols)} channels. Time='{timestamp_col}'.")

    return df_delta, global_baseline, windows
############################

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
        imu_glob = os.path.join(folder_path, "airpods_motion_*.csv")
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
                df_delta, _, _ = calculate_deltas(
                    df_imu,
                    csv_path=csv_path,
                    id_cols=[_detect_timestamp_col(df_imu), "quat_x", "quat_y", "quat_z", "quat_w"],
                    min_window_seconds=1.0,
                    verbose=False
                )
                out_csv = make_delta_csv_path(csv_path)
                df_delta.to_csv(out_csv, index=False)
            except Exception as e:
                print(f"⚠️ Delta conversion skipped for: {csv_path}\n   ↳ {e}")

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

def label_at_time(t, names, times, m):
    """
    3-class labeling at timestamp t with margin m (seconds).

    Emits transition (1) **only** for the slouch cycle:
      UPRIGHT_HOLD_START  --(near next SLOUCH_START - m)-->  1
      SLOUCH_START                                       --> 1
      SLOUCHED_HOLD_START (first m sec)                  --> 1
      SLOUCHED_HOLD_START (after  m sec)                 --> 2
      RECOVERY_START *if recovering from a slouch hold*  --> 1

    Non-slouch events ('NOSLOUCH_START', 'NOSLOUCHED_HOLD_START') never
    yield transition or slouched; they are treated as upright (0).
    """
    # index of the last event at or before t
    k = int(np.searchsorted(times, t, side='right')) - 1

    # before first event: only treat as transition if first event is SLOUCH_START
    if k < 0:
        if len(names) and names[0] == 'SLOUCH_START' and t >= times[0] - m:
            return 1
        return 0

    prev_ev, prev_t = names[k], times[k]
    next_ev = names[k+1] if (k + 1) < len(names) else None
    next_t  = times[k+1]  if (k + 1) < len(times) else np.inf
    prev_prev_ev = names[k-1] if (k - 1) >= 0 else None

    # ---- Non-slouch events are always upright ----
    if prev_ev in ('NOSLOUCH_START', 'NOSLOUCHED_HOLD_START'):
        return 0

    # ---- Slouch-cycle labeling ----
    if prev_ev == 'UPRIGHT_HOLD_START':
        # Only near an upcoming true SLOUCH_START do we call transition
        if next_ev == 'SLOUCH_START' and t >= next_t - m:
            return 1
        return 0

    if prev_ev == 'SLOUCH_START':
        return 1

    if prev_ev == 'SLOUCHED_HOLD_START':
        # transition for m right after entering slouched, then slouched
        if t < prev_t + m:
            return 1
        return 2

    if prev_ev == 'RECOVERY_START':
        # Transition only if we are recovering from a real slouch hold
        if prev_prev_ev == 'SLOUCHED_HOLD_START':
            return 1
        return 0

    # default
    return 0

class ConfusionMatrixAverager:
    """
    Accumulates raw confusion matrices over multiple runs and saves
    a single averaged confusion matrix as a PNG.

    - Uses raw counts per run (no normalization) and sums them.
      The final figure can be normalized in different ways for display:
        * normalize="true":  rows sum to 1 (per-true-class recall)
        * normalize="pred":  columns sum to 1 (per-predicted-class precision)
        * normalize="all":   entire matrix sums to 1
        * normalize=None:    raw counts

      Summing raw counts corresponds to pooling all test samples across runs,
      which is robust when each split has different class supports.

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

        self._cm_counts = None  # will be np.ndarray[n_classes, n_classes]
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
            raise ValueError(
                "The provided 'history' has no .model; pass the History returned by model.fit(...)."
            )

        X = np.asarray(X)
        y_true = np.asarray(y).squeeze().astype(np.int64)

        # Initialize shapes/names lazily on first call
        self._ensure_initialized(history, y_true)

        # Predict and compute raw-count confusion matrix for this run
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
                    model_tag: str = "tcn",
                    normalize: str = "true",  # "true", "pred", "all", or None
                    dpi: int = 220) -> str:
        """
        Save the averaged confusion matrix as a PNG and return the path.

        Parameters
        ----------
        model_tag : {"tcn","cnn",...}
            Used in the filename, e.g. tcn_19-10-2025_14-37.png.
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
        balanced_acc = self._balanced_accuracy_from_counts(self._cm_counts)

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
        subtitle = f"Runs: {self._runs} • Balanced Acc: {balanced_acc*100:.1f}% • {norm_label}"
        ax.set_title(f"{title_top}\n{subtitle}")

        # Annotate each cell
        fmt = ".2f" if normalize is not None else "d"
        # Use nanmax to avoid warnings when matrix may contain NaNs (e.g., empty rows after normalization)
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

        # Safe timestamp for filenames: dd-mm-yyyy_HH-MM (no seconds, filesystem-safe)
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
        os.makedirs(self.save_dir, exist_ok=True)
        filename = f"{model_tag.lower()}_{timestamp}.png"
        out_path = os.path.join(self.save_dir, filename)

        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ---------- helpers ----------
    def _ensure_initialized(self, history, y_true: np.ndarray):
        # Infer n_classes from model output size if missing
        if self.n_classes is None:
            try:
                self.n_classes = int(history.model.output_shape[-1])
            except Exception:
                self.n_classes = int(np.max(y_true)) + 1

        # Default class names
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
            # Column-normalize (per-class precision across predictions)
            col_sums = cm_counts.sum(axis=0, keepdims=True).astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                cm = np.divide(cm_counts, col_sums, where=col_sums > 0)
                # For columns with zero predictions, keep zeros
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
        # Balanced accuracy = mean of per-class recall over classes with support > 0
        row_sums = cm_counts.sum(axis=1).astype(np.float64)
        diag = np.diag(cm_counts).astype(np.float64)
        mask = row_sums > 0
        if not np.any(mask):
            return float("nan")
        recalls = np.zeros_like(row_sums, dtype=np.float64)
        recalls[mask] = diag[mask] / row_sums[mask]
        return float(np.mean(recalls[mask]))

def save_confusion_matrix_for_run(history, X, y,
                                  model_tag: str = "tcn",
                                  save_dir: str = "confusion_matrix",
                                  dpi: int = 220,
                                  normalize: str = "true") -> str:
    """
    Convenience helper: compute & save a single-run confusion matrix.
    (Internally just uses ConfusionMatrixAverager once.)

    Returns the PNG path.
    """
    cma = ConfusionMatrixAverager(save_dir=save_dir)
    cma.add(history, X, y)
    return cma.save_figure(model_tag=model_tag, dpi=dpi, normalize=normalize)

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


def X_and_y(type, list_comb):
    matching_folders, n_folders = folders_tot(type, list_comb)
    _, _, _, win_len_frames, stride_frames, _, windows_per_rec, stride, len_window_sec = find_shapes()
    
    X_tot = None
    y_tot = np.zeros((n_folders * windows_per_rec, 1), dtype=int)

    for index, folder_path in enumerate(matching_folders):
        if not os.path.isdir(folder_path):
            continue

        event_files = glob.glob(os.path.join(folder_path, 'events_inferred_*.csv'))
        imu_files   = glob.glob(os.path.join(folder_path, 'airpods_motion_dd*.csv'))
        if not event_files or not imu_files:
            print(f"⚠️ Missing files in {folder_path}")
            continue
        
        df_event = pd.read_csv(event_files[0])
        df_imu   = pd.read_csv(imu_files[0])

        # --- Align event times to IMU time axis ---
        t_imu = df_imu.iloc[:, 0].astype(float).to_numpy()
        t0 = float(t_imu[0])

        ev = df_event[['t_sec','event']].copy()
        ev = ev.sort_values('t_sec').reset_index(drop=True)

        # shift events so their first timestamp maps to IMU t0
        ev['t_aligned'] = ev['t_sec'].astype(float) - float(ev['t_sec'].iloc[0]) + t0

        # ensure initial upright AT t0, then keep list sorted by aligned time
        if ev.iloc[0]['event'] != 'UPRIGHT_HOLD_START':
            ev = pd.concat([
                pd.DataFrame({'t_sec':[ev['t_sec'].iloc[0]], 'event':['UPRIGHT_HOLD_START'], 't_aligned':[t0]}),
                ev
            ], ignore_index=True)

        ev = ev.sort_values('t_aligned', kind='mergesort').reset_index(drop=True)

        times = ev['t_aligned'].to_numpy()
        names = ev['event'].astype(str).tolist()
        
        # --- Labels: state at window END on IMU axis ---
        '''
        WORK ON THIS LATER TO LABEL 'TRANSITION' RIGHT
        m = 0.2
        labels_array = np.zeros((windows_per_rec, 1), dtype=int)
        for i in range(windows_per_rec):
            t_end   = t0 + len_window_sec + i * stride
            t_start = t_end - len_window_sec
            labels_array[i, 0] = label_window_interval_centered(
                t_start, t_end, names, times, m,
                n_samples=win_len_frames,      # 75 for your setup
                center_band=(0.4, 0.6),      # only count transitions near the middle
                min_total_trans_frac=0.12,     # ~9/75 frames
                min_center_trans_frac=0.08     # ~6/75 frames
            )
        '''
        m = 0.2
        labels_array = np.zeros((windows_per_rec, 1), dtype=int)
        for i in range(windows_per_rec):
            current_time = t0 + len_window_sec + i * stride
            labels_array[i, 0] = label_at_time(current_time, names, times, m)
        
        y_tot[index*windows_per_rec:(index+1)*windows_per_rec, 0:1] = labels_array

        # --- Features: drop the time column by position (your current approach) ---
        Xsig = df_imu.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
        n_ch = Xsig.shape[1]
        N = Xsig.shape[0]

        # lazy allocation once we know n_ch
        if X_tot is None:
            X_tot = np.zeros((n_folders * windows_per_rec, win_len_frames, n_ch), dtype=np.float32)

        # --- Windowing ---
        base = index * windows_per_rec
        max_crop = max(0, min(stride_frames - 1, win_len_frames - 1))
        for i in range(windows_per_rec):
            start = i * stride_frames
            end = start + win_len_frames
            win = Xsig[start:end, :]

            # If start is beyond the signal (empty slice), repeat the last sample
            if win.shape[0] == 0:
                last = Xsig[-1:, :]
                win = np.repeat(last, win_len_frames, axis=0)
            elif win.shape[0] < win_len_frames:
                need = win_len_frames - win.shape[0]
                pad_tail = np.repeat(win[-1:, :], need, axis=0)
                win = np.concatenate([win, pad_tail], axis=0)

            r = random.randint(0, max_crop) if max_crop > 0 else 0
            if r > 0:
                left = win[:r, :]
                left_mirror = np.flip(left, axis=0)
                win = np.concatenate([left_mirror, win[r:, :]], axis=0)

            X_tot[base + i, :, :] = win

    return X_tot, y_tot





