import pandas as pd
import numpy as np
import random
import glob
import os


# count number of beep_schedules_folders:
def folders_tot(type):
    if type == 'train':
        endings = ('0', '1')
    elif type == 'val':
        endings = ('2',)
    elif type == 'test':
        endings = ('3',)
    else:
        raise ValueError("Invalid type. Choose 'train', 'val', or 'test'.")

    all_matching_folders = []

    # Iterate through each required ending (e.g., '0' and '1' for 'train')
    for ending in endings:
        # Build the pattern using an f-string to insert the current ending
        # Example: 'beep_schedules_*0'
        DATA_ROOT = os.path.join(os.getcwd(), "data")
        pattern = os.path.join(DATA_ROOT, f'beep_schedules_*{ending}')
        
        # Find all paths matching the specific pattern
        current_matches = glob.glob(pattern)
        
        # Filter to ensure the match is actually a directory (folder)
        folder_paths = [path for path in current_matches if os.path.isdir(path)]
        
        # Add the valid folders to the main list
        all_matching_folders.extend(folder_paths)

    num_beep_schedules_folders = len(all_matching_folders)
    
    return all_matching_folders, num_beep_schedules_folders

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

def label_at_time(t, names, times, m):
    """Return 0=upright, 1=transition, 2=slouched at time t, with margin m."""
    # find prev and next event indices
    k = int(np.searchsorted(times, t, side='right')) - 1
    if k < 0:
        # before first event: upright, unless within m of first SLOUCH_START
        if names[0] == 'SLOUCH_START' and t >= times[0] - m:
            return 1
        return 0
    prev_ev, prev_t = names[k], times[k]
    next_ev = names[k+1] if k + 1 < len(names) else None
    next_t = times[k+1] if k + 1 < len(times) else np.inf

    # intervals with margin logic
    if prev_ev == 'UPRIGHT_HOLD_START':
        # upright until slouch_start - m; then transition
        if next_ev == 'SLOUCH_START' and t >= next_t - m:
            return 1
        # also treat just after a fresh upright as transition for m
        if t < prev_t + m:
            return 1
        return 0

    if prev_ev == 'SLOUCH_START':
        # transition until slouched_hold (+m handled in next segment)
        return 1

    if prev_ev == 'SLOUCHED_HOLD_START':
        # transition for m after hold, then slouched
        if t < prev_t + m:
            return 1
        # slouched until recovery - m (handled by recovery segment)
        return 2

    if prev_ev == 'RECOVERY_START':
        # transition until next upright (+m after upright handled above)
        return 1

    # default fallback
    return 0

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
    Add roll (radians) as a RELATIVE angle to the first valid frame.

    Strategy:
      A) If grav_x/grav_y/grav_z exist (preferred, drift-free):
           roll_abs = atan2(grav_y, grav_z)           # in (-pi, pi]
           roll     = wrap_to_pi(roll_abs - roll_abs[0_valid])
      B) Else if quaternions exist:
           Normalize each quaternion row.
           Let q0 = first valid normalized quaternion.
           q_rel(t) = conj(q0) * q(t)
           roll     = atan2(2*(w*x + y*z), 1 - 2*(x^2 + y^2))   # from q_rel

    No unwrapping is used; results stay in (-pi, pi] and return to ~0 when you
    return to the starting pose. The column is inserted after 'grav_z' if present,
    else after 'quat_w', else at the end. Modifies df_imu in place and returns it.
    """
    import numpy as np

    def wrap_to_pi(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    # -------- Path A: gravity (robust to drift) --------
    if {"grav_x", "grav_y", "grav_z"}.issubset(df_imu.columns):
        gy = df_imu["grav_y"].to_numpy(dtype=np.float64, copy=False)
        gz = df_imu["grav_z"].to_numpy(dtype=np.float64, copy=False)
        roll_abs = np.arctan2(gy, gz)  # (-pi, pi]

        valid = np.isfinite(roll_abs)
        if not valid.any():
            return df_imu
        i0 = int(np.flatnonzero(valid)[0])
        roll0 = roll_abs[i0]
        roll = wrap_to_pi(roll_abs - roll0)

    # -------- Path B: quaternion (relative orientation) --------
    elif all(c in df_imu.columns for c in quat_cols):
        x = df_imu[quat_cols[0]].to_numpy(dtype=np.float64, copy=False)
        y = df_imu[quat_cols[1]].to_numpy(dtype=np.float64, copy=False)
        z = df_imu[quat_cols[2]].to_numpy(dtype=np.float64, copy=False)
        w = df_imu[quat_cols[3]].to_numpy(dtype=np.float64, copy=False)

        qn = np.sqrt(x*x + y*y + z*z + w*w)
        good = qn > 1e-12
        if not good.any():
            return df_imu

        # Normalize rows
        x = np.where(good, x / qn, np.nan)
        y = np.where(good, y / qn, np.nan)
        z = np.where(good, z / qn, np.nan)
        w = np.where(good, w / qn, np.nan)

        i0 = int(np.flatnonzero(good)[0])
        x0, y0, z0, w0 = x[i0], y[i0], z[i0], w[i0]

        # q_rel = conj(q0) * q  (xyzw order; conj(q0) = (-x0, -y0, -z0, w0))
        xr = w0*x - x0*w - y0*z + z0*y
        yr = w0*y + x0*z - y0*w - z0*x
        zr = w0*z - x0*y + y0*x - z0*w
        wr = w0*w + x0*x + y0*y + z0*z

        # Euler roll from q_rel (Z-Y-X convention)
        t0 = 2.0 * (wr*xr + yr*zr)
        t1 = 1.0 - 2.0 * (xr*xr + yr*yr)
        roll = np.arctan2(t0, t1)  # already in (-pi, pi]

    else:
        return df_imu

    # -------- Insert/replace column in a stable location --------
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
    Add yaw (radians) as a RELATIVE angle to the first valid frame.

    Yaw cannot be derived from gravity alone. We use quaternions:
        Normalize each quaternion row.
        Let q0 = first valid normalized quaternion.
        q_rel(t) = conj(q0) * q(t)
        yaw      = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))  # from q_rel

    No unwrapping; yaw stays in (-pi, pi] and returns to ~0 whenever the user
    returns to the starting pose (aside from true sensor-fusion drift).
    Column placement mirrors pitch/roll functions. Modifies df_imu in place and returns it.
    """
    import numpy as np

    if not all(c in df_imu.columns for c in quat_cols):
        return df_imu  # cannot compute yaw reliably without quaternions

    x = df_imu[quat_cols[0]].to_numpy(dtype=np.float64, copy=False)
    y = df_imu[quat_cols[1]].to_numpy(dtype=np.float64, copy=False)
    z = df_imu[quat_cols[2]].to_numpy(dtype=np.float64, copy=False)
    w = df_imu[quat_cols[3]].to_numpy(dtype=np.float64, copy=False)

    qn = np.sqrt(x*x + y*y + z*z + w*w)
    good = qn > 1e-12
    if not good.any():
        return df_imu

    # Normalize rows
    x = np.where(good, x / qn, np.nan)
    y = np.where(good, y / qn, np.nan)
    z = np.where(good, z / qn, np.nan)
    w = np.where(good, w / qn, np.nan)

    i0 = int(np.flatnonzero(good)[0])
    x0, y0, z0, w0 = x[i0], y[i0], z[i0], w[i0]

    # q_rel = conj(q0) * q  (xyzw order; conj(q0) = (-x0, -y0, -z0, w0))
    xr = w0*x - x0*w - y0*z + z0*y
    yr = w0*y + x0*z - y0*w - z0*x
    zr = w0*z - x0*y + y0*x - z0*w
    wr = w0*w + x0*x + y0*y + z0*z

    # Euler yaw from q_rel (Z-Y-X convention)
    t3 = 2.0 * (wr*zr + xr*yr)
    t4 = 1.0 - 2.0 * (yr*yr + zr*zr)
    yaw = np.arctan2(t3, t4)  # in (-pi, pi]

    # -------- Insert/replace column in a stable location --------
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
            add_roll_to_df(df_imu)
            add_yaw_to_df(df_imu)

            #normalize_chanels(df_imu)

            try:
                df_imu.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"❌ Failed to write: {csv_path}\n   ↳ {e}")

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



def X_and_y(type):
    matching_folders, n_folders = folders_tot(type)
    _, _, _, win_len_frames, stride_frames, _, windows_per_rec, stride, len_window_sec = find_shapes()
    X_tot = None
    y_tot = np.zeros((n_folders * windows_per_rec, 1), dtype=int)

    for index, folder_path in enumerate(matching_folders):
        if not os.path.isdir(folder_path):
            continue

        event_files = glob.glob(os.path.join(folder_path, 'events_inferred_*.csv'))
        imu_files   = glob.glob(os.path.join(folder_path, 'airpods_motion_*.csv'))
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





