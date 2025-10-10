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
    Compute pitch (radians) from quaternions and INSERT it into df_imu in place.
    Column is placed after 'grav_z' if present, else after 'quat_w', else at end.
    Does not write to disk. Returns df_imu for convenience.
    """
    # Ensure quaternion columns exist
    missing = [c for c in quat_cols if c not in df_imu.columns]
    if missing:
        # quietly do nothing if quats missing
        return df_imu

    # Vectorized pitch computation (Tait–Bryan, pitch about Y)
    x = df_imu[quat_cols[0]].to_numpy(dtype=np.float64, copy=False)
    y = df_imu[quat_cols[1]].to_numpy(dtype=np.float64, copy=False)
    z = df_imu[quat_cols[2]].to_numpy(dtype=np.float64, copy=False)
    w = df_imu[quat_cols[3]].to_numpy(dtype=np.float64, copy=False)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)  # radians

    # If column already exists, drop it so we can control its position
    if out_col in df_imu.columns:
        df_imu.drop(columns=[out_col], inplace=True)

    cols = list(df_imu.columns)
    if "grav_z" in cols:
        insert_at = cols.index("grav_z") + 1
    elif "quat_w" in cols:
        insert_at = cols.index("quat_w") + 1
    else:
        insert_at = len(cols)

    # Insert in place
    df_imu.insert(insert_at, out_col, pitch)
    return df_imu

def fix_length(df_imu: pd.DataFrame, target_len: int = 18_000, time_col_idx: int = 0, fs_hint: float = 50.0) -> None:
    """
    In-place: ensure df_imu has exactly `target_len` rows.
      - If longer: drop trailing rows.
      - If shorter: append mirrored rows from the tail for all *non-time* columns,
        but extend the time column linearly using the median dt (fallback 1/fs_hint).

    Parameters
    ----------
    df_imu : pd.DataFrame
        IMU dataframe (first column assumed to be time in seconds).
    target_len : int
        Desired number of rows (default 18_000).
    time_col_idx : int
        Index of the time column (default 0).
    fs_hint : float
        Fallback sample rate in Hz if dt can't be inferred (default 50.0).

    Returns
    -------
    None  (mutates df_imu in place)
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
        # nothing to do
        return

    # --- Compute dt from existing time column ---
    time_col = df_imu.columns[time_col_idx]
    t = df_imu.iloc[:, time_col_idx].astype(float).to_numpy()
    if t.size >= 2:
        diffs = np.diff(t)
        diffs = diffs[np.isfinite(diffs)]
        dt = float(np.median(diffs)) if diffs.size > 0 else (1.0 / fs_hint)
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0 / fs_hint
    else:
        dt = 1.0 / fs_hint
    last_t = float(t[-1])

    # --- Build mirrored padding for all columns, then overwrite time column ---
    need = target_len - n
    q, r = divmod(need, n)
    tail_rev = df_imu.iloc[::-1].reset_index(drop=True)

    pad_blocks = []
    if q:
        pad_blocks.extend([tail_rev] * q)
    if r:
        pad_blocks.append(tail_rev.iloc[:r])

    pad = pd.concat(pad_blocks, ignore_index=True) if pad_blocks else df_imu.iloc[0:0].copy()
    # Ensure column order matches exactly
    pad = pad[df_imu.columns]

    # Replace time column with linear extension at same sample rate
    pad_times = last_t + dt * np.arange(1, need + 1, dtype=float)
    pad.loc[:, time_col] = pad_times

    # --- Append in place via .loc (expands the frame) ---
    start_idx = n
    end_idx = n + need - 1
    df_imu.loc[start_idx:end_idx, df_imu.columns] = pad.to_numpy()

    # Normalize the index
    df_imu.reset_index(drop=True, inplace=True)

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
            add_pitch_to_df(df_imu)  # modifies df_imu in place
            fix_length(df_imu, target_len=18_000)

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

        event_files = glob.glob(os.path.join(folder_path, 'events_inferred_template_*.csv'))
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





