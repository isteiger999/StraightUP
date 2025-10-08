import pandas as pd
import numpy as np
import random
import glob
import os

# count number of beep_schedules_folders:
def folders_tot():
    pattern = os.path.join(os.getcwd(), 'beep_schedules_*')
    matching_folders = glob.glob(pattern)
    folder_paths = [path for path in matching_folders if os.path.isdir(path)]
    num_beep_schedules_folders = len(folder_paths)
    return num_beep_schedules_folders

def find_shapes():
    # use first IMU recording for shape definition of X_tot and y_tot
    df_imu = pd.read_csv(r"beep_schedules_Ivan0/airpods_motion_1759863949.csv")
    t = df_imu.iloc[:, 0].astype(float).to_numpy()
    dt_med = float(np.median(np.diff(t)))
    fs = (1.0 / dt_med) if dt_med > 0 else 50.0

    win_len = int(round(len_window * fs))      # samples per window
    stride_frames = int(round(stride * fs))    # samples per stride

    Xsig = df_imu.iloc[:, 1:].to_numpy()        # exclude time colum
    n_ch = Xsig.shape[1]                        # nr. of chanels 

    # Recompute windows_per_rec from sample counts (more robust than using seconds)
    N = Xsig.shape[0]                           # 18'000
    windows_per_rec = max(0, 1 + (N - win_len) // stride_frames)

    return t, dt_med, fs, win_len, stride_frames, n_ch, N, windows_per_rec

def label_at_time(t):
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


def X_tot_y_tot():
    num_beep_schedules_folders = folders_tot()
    t, dt_med, fs, win_len, stride_frames, n_ch, N, windows_per_rec = find_shapes(df_imu)
    X_tot = np.zeros((num_beep_schedules_folders*windows_per_rec, win_len, n_ch), dtype=np.float32)
    y_tot = np.zeros((num_beep_schedules_folders*windows_per_rec, 1), dtype=np.float32)

    for index, folder_path in enumerate(matching_paths):
        if os.path.isdir(folder_path):

            event_pattern = os.path.join(folder_path, 'events_inferred_template_*.csv')
            event_files = glob.glob(event_pattern)

            imu_pattern = os.path.join(folder_path, 'airpods_motion_*.csv')
            imu_files = glob.glob(imu_pattern)

            if event_files:
                event_file_path = event_files[0]
                print(f"  Found event file: {os.path.basename(event_file_path)}")
                df_event = pd.read_csv(event_file_path)
            else:
                print("⚠️ Event file not found in this folder.")
                continue  

            if imu_files:
                imu_file_path = imu_files[0]
                print(f"  Found IMU file: {os.path.basename(imu_file_path)}")
                df_imu = pd.read_csv(imu_file_path)
            else:
                print("  ⚠️ IMU file not found in this folder.")
                continue  

            # assume first column in IMU is time (t_sec). If not, rename accordingly.
            t0 = float(df_imu.iloc[0, 0])
            t1 = float(df_imu.iloc[-1, 0])

            len_window = 1.5
            stride = 0.5
            m = 0.2  # margin around starts treated as transition

            labels_array = np.zeros((windows_per_rec, 1), dtype=int)  # 0=upright,1=transition,2=slouched

            # sort events and ensure we have an initial upright at t0
            events = df_event[['t_sec','event']].sort_values('t_sec').to_numpy().tolist()
            if not events or events[0][1] != 'UPRIGHT_HOLD_START':
                events = [[t0, 'UPRIGHT_HOLD_START']] + events
            times = np.array([float(t) for t, _ in events])
            names = [e for _, e in events]

            # label comes from the state at the end of the window (causal)
            for i in range(windows_per_rec):
                current_time = t0 + len_window + i * stride
                labels_array[i, 0] = label_at_time(current_time)

            y_tot[index:(index+1)*windows_per_rec, 0] = labels_array

            ## --------------------------------------------------------------------------------------------------
            # Now create actual windows
            # Random left-crop (0..stride_frames-1) then mirror-pad on the left
            max_crop = max(0, min(stride_frames - 1, win_len - 1))

            for i in range(windows_per_rec):
                start = i * stride_frames
                end = start + win_len
                win = Xsig[start:end, :]

                # If we’re a few samples short at the very end, pad by repeating the last row
                if win.shape[0] < win_len:
                    need = win_len - win.shape[0]
                    pad_tail = np.repeat(win[-1:, :], need, axis=0)
                    win = np.concatenate([win, pad_tail], axis=0)

                # Random crop amount (can be 0)
                r = random.randint(0, max_crop) if max_crop > 0 else 0
                if r > 0:
                    left = win[:r, :]                     # the part we "remove"
                    left_mirror = np.flip(left, axis=0)   # mirror it
                    win = np.concatenate([left_mirror, win[r:, :]], axis=0)  # pad-left + keep rest

                X_tot[i, :, :] = win

    return X_tot, y_tot





