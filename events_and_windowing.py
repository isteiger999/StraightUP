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
        pattern = os.path.join(os.getcwd(), f'beep_schedules_*{ending}')
        
        # Find all paths matching the specific pattern
        current_matches = glob.glob(pattern)
        
        # Filter to ensure the match is actually a directory (folder)
        folder_paths = [path for path in current_matches if os.path.isdir(path)]
        
        # Add the valid folders to the main list
        all_matching_folders.extend(folder_paths)

    num_beep_schedules_folders = len(all_matching_folders)
    
    return all_matching_folders, num_beep_schedules_folders

def find_shapes():
    # use first IMU recording for shape definition of X_tot and y_tot
    df_imu = pd.read_csv(r"beep_schedules_Ivan0/airpods_motion_1759863949.csv")
    t = df_imu.iloc[:, 0].astype(float).to_numpy()
    dt_med = float(np.median(np.diff(t)))
    fs = (1.0 / dt_med) if dt_med > 0 else 50.0
    a = 1.5
    stride = 0.5
    len_window_sec = 1.5

    win_len_frames = int(round(len_window_sec * fs))      # samples per window
    stride_frames = int(round(stride * fs))    # samples per stride

    Xsig = df_imu.iloc[:, 1:].to_numpy()        # exclude time colum
    n_ch = Xsig.shape[1]                        # nr. of chanels 

    # Recompute windows_per_rec from sample counts (more robust than using seconds)
    N = Xsig.shape[0]                           # 18'000
    windows_per_rec = max(0, 1 + (N - win_len_frames) // stride_frames)

    return t, dt_med, fs, win_len_frames, stride_frames, n_ch, N, windows_per_rec, stride, len_window_sec

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

def drop_timestamp_inplace_from_files(imu_files):
    """
    imu_files: list of file paths (e.g., from glob.glob)
    Removes 'timestamp' column if present and overwrites each CSV.
    """
    updated = 0
    for csv_path in imu_files:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"Skipping {csv_path}: read error: {e}")
            continue

        if "timestamp" in df.columns:
            df.drop(columns=["timestamp"], inplace=True)
            df.to_csv(csv_path, index=False)
            updated += 1

def add_pitch_in_memory(df_imu, out_col="pitch_rad",
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


def X_and_y(type):
    matching_folders, num_beep_schedules_folders = folders_tot(type)
    df_imu0 = pd.read_csv(r"beep_schedules_Ivan0/airpods_motion_1759863949.csv")  # (see note below)
    t, dt_med, fs, win_len_frames, stride_frames, n_ch, N, windows_per_rec, stride, len_window_sec = find_shapes()
    X_tot = np.zeros((num_beep_schedules_folders*windows_per_rec, win_len_frames, n_ch), dtype=np.float32)
    y_tot = np.zeros((num_beep_schedules_folders*windows_per_rec, 1), dtype=int)

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

        # Add pitch (in place)
        add_pitch_in_memory(df_imu)

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

        # --- Windowing ---
        base = index * windows_per_rec
        max_crop = max(0, min(stride_frames - 1, win_len_frames - 1))
        for i in range(windows_per_rec):
            start = i * stride_frames
            end = start + win_len_frames
            win = Xsig[start:end, :]

            if win.shape[0] < win_len_frames:
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





