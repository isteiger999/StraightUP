import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path   # to cycle through csv files in folders
from sklearn.preprocessing import StandardScaler   # for the normalization of X_train
from sklearn.model_selection import train_test_split
from constants import FEATURE_ORDER

def train_test(X_tot, y_tot):
    # 1. Split the dataset into Train + Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_tot, y_tot,
        test_size=0.25,          # 20% test
        stratify=y_tot,         # keep 1/0 ratio the same
        shuffle=True,           # break your class-ordered rows
        random_state=42         # reproducible
    )
    
    X_train = X_train.astype("float32")
    X_test  = X_test.astype("float32")

    return X_train, X_test, y_train, y_test

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

def plot_data():
    data_test = pd.read_csv(r"slouch_data/slouch0.csv", na_values=['NA'])

    offset_time = data_test.iloc[0, 0]
    data_test['timestamp'] = data_test['timestamp'] - offset_time

    acc_x = data_test['acc_x']
    acc_y = data_test['acc_y']
    acc_z = data_test['acc_z']
    acc_tot = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    grav_x = data_test['grav_x']
    grav_y = data_test['grav_y']
    grav_z = data_test['grav_z']
    grav_tot = np.sqrt(grav_x**2 + grav_y**2 + grav_z**2)

    plt.plot(data_test['timestamp'].iloc[-500:], data_test['grav_y'].iloc[-500:], label='grav_y')
    plt.xlabel("Time")
    plt.ylabel("Acceleration in x")
    plt.title('total recording')
    plt.legend()
    plt.show()

def _assert_and_order(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")
    return df[FEATURE_ORDER]  # <-- enforce exact order

def quat_pitch_xyzw(x, y, z, w):
    # Tait-Bryan pitch (rotation about Y), consistent with common iOS convention
    sinp = 2.0 * (w*y - z*x)
    if np.abs(sinp) >= 1:
        return np.sign(sinp) * (np.pi/2)  # clamp
    return np.arcsin(sinp)

def drop_timestamp_inplace(folders):
    for folder in map(Path, folders):
        for csv_path in folder.glob("*.csv"):
            df = pd.read_csv(csv_path, low_memory=False)
            if "timestamp" in df.columns:
                df = df.drop(columns=["timestamp"])
                df.to_csv(csv_path, index=False)
            else:
                pass

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

def obtain_windows():
    f_sample = 50      # [Hz].  (actually 50.07 Hz) 180sekunden --> 12.6 frames = 0.252s
    window_length = 8  # [sec]
    timestep_window = f_sample * window_length
    windows_tot = 22
    samples_slouch = sum(1 for _ in Path("slouch_data").glob("*.csv")) * 11
    samples_noslouch = sum(1 for _ in Path("no_slouch_data").glob("*.csv")) * 22
    samples_tot = samples_slouch + samples_noslouch
    chanels = 14            # 13 + pitch_rad

    X_tot = np.zeros((samples_tot, timestep_window, chanels)) # shape: 50'000, timesteps(8*50), channels(14)
    y_tot = np.zeros((samples_tot, 1))


    ## for SLOUCH data ##
    slouch_count = 0
    folder = Path("slouch_data")
    for rec in folder.glob("*.csv"):
        df = pd.read_csv(rec, na_values=['NA'])
        # add 14th signal (Tate-Bryan angle)
        px, py, pz, pw = df["quat_x"], df["quat_y"], df["quat_z"], df["quat_w"]
        df["pitch_rad"] = [quat_pitch_xyzw(x,y,z,w) for x,y,z,w in zip(px,py,pz,pw)]
        df = _assert_and_order(df)

        for index in range(windows_tot):
            if index % 2 != 0 and index != 0:
                X_tot[slouch_count, :, :] = df.iloc[index*timestep_window:(index+1)*timestep_window, :].values
                y_tot[slouch_count] = 1
                slouch_count += 1

    ## for NO_SLOUCH data ##. (can use all 22 windows instead of only 11)
    folder = Path("no_slouch_data")
    no_slouch_count = 0
    index2 = 0
    for rec in folder.glob("*.csv"):
        df = pd.read_csv(rec, na_values=['NA'])
        # add 14th signal (Tate-Bryan angle)
        px, py, pz, pw = df["quat_x"], df["quat_y"], df["quat_z"], df["quat_w"]
        df["pitch_rad"] = [quat_pitch_xyzw(x,y,z,w) for x,y,z,w in zip(px,py,pz,pw)]
        df = _assert_and_order(df)
        for index in range(windows_tot):
            X_tot[index2 + slouch_count + index, :, :] = df.iloc[index*timestep_window:(index+1)*timestep_window, :].values
            y_tot[index2 + slouch_count + index] = 0
            no_slouch_count += 1
        index2 += 22

    return X_tot, y_tot