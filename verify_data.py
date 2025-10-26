import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import numpy as np
import tensorflow as tf
import platform
import pandas as pd
import matplotlib.pyplot as plt
from events_and_windowing import X_and_y
from matplotlib.patches import Patch
from typing import Optional, Sequence, Tuple

# ---------------- Filtering ----------------
def ema_alpha(fc, fs):
    """For a 1st-order causal low-pass: y[n]=(1-α)x[n]+αy[n-1], α=exp(-2πfc/fs)."""
    if fc <= 0 or fs <= 0:
        raise ValueError("fc and fs must be > 0")
    return float(np.exp(-2.0 * np.pi * float(fc) / float(fs)))

def ema_1d(x, alpha, dtype=np.float32):
    """Causal EMA for a 1D signal."""
    x = np.asarray(x, dtype=dtype)
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.shape[0]):
        y[i] = (1.0 - alpha) * x[i] + alpha * y[i - 1]
    return y

def detect_fs(df):
    """Try to infer sampling rate from a time column; fall back to 50 Hz."""
    c = "timestamp"
    if c in df.columns:
        t = pd.to_numeric(df[c], errors="coerce").to_numpy()
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            return float(1.0 / np.median(dt))
    return 50.0

# ---------------- FOR PLOTTING TO CHECK DELTAS ----------------
def plot_reconstructed_signal(
    X_train,
    y_train,
    windows: int,
    chanel: int,
    *,
    stride: Optional[int] = None,
    num_windows: Optional[int] = None,
    start_window: int = 0,
    label_names: Optional[Sequence[str]] = ("upright", "transition", "slouch"),
    show_bands: bool = True,
    show_vlines: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
    return_signal: bool = False,
    show: bool = True,                 # <-- NEW: display the plot when running as a script
    savepath: Optional[str] = None,    # <-- NEW: save figure to file
):
    X = np.asarray(X_train)
    y = np.asarray(y_train).reshape(-1).astype(int)
    if X.ndim not in (2, 3):
        raise ValueError(f"X_train must be 2D or 3D, got {X.ndim}D.")

    # Extract channel -> (n_windows_total, window_len)
    if X.ndim == 2:
        data = X
        if data.shape[1] != windows:
            windows = int(data.shape[1])
    else:
        if X.shape[1] == windows:               # (nW, win, nCh)
            n_channels = X.shape[2]
            if not (0 <= chanel < n_channels):
                raise IndexError(f"chanel {chanel} out of range [0, {n_channels-1}]")
            data = X[:, :, chanel]
        elif X.shape[2] == windows:             # (nW, nCh, win)
            n_channels = X.shape[1]
            if not (0 <= chanel < n_channels):
                raise IndexError(f"chanel {chanel} out of range [0, {n_channels-1}]")
            data = X[:, chanel, :]
        else:
            if X.shape[1] >= X.shape[2]:        # infer (nW, win, nCh)
                n_channels = X.shape[2]
                if not (0 <= chanel < n_channels):
                    raise IndexError(f"chanel {chanel} out of range [0, {n_channels-1}]")
                data = X[:, :, chanel]
                windows = int(X.shape[1])
            else:                                # infer (nW, nCh, win)
                n_channels = X.shape[1]
                if not (0 <= chanel < n_channels):
                    raise IndexError(f"chanel {chanel} out of range [0, {n_channels-1}]")
                data = X[:, chanel, :]
                windows = int(X.shape[2])

    n_windows_total, win_len = data.shape
    if win_len != windows:
        windows = win_len
    if y.shape[0] != n_windows_total:
        raise ValueError(f"y_train length ({y.shape[0]}) != number of windows ({n_windows_total}).")
    if start_window < 0 or start_window >= n_windows_total:
        raise ValueError(f"start_window must be within [0, {n_windows_total-1}], got {start_window}.")

    n_plot = (n_windows_total - start_window) if (num_windows is None) else num_windows
    if n_plot <= 0 or start_window + n_plot > n_windows_total:
        raise ValueError("Invalid num_windows/start_window slice.")

    data_plot = data[start_window : start_window + n_plot]
    y_plot = y[start_window : start_window + n_plot]

    if stride is None:
        stride = max(1, windows // 3)
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")

    total_len = (n_plot - 1) * stride + windows
    recon = np.zeros(total_len, dtype=float)
    weights = np.zeros(total_len, dtype=float)
    for k in range(n_plot):
        s, e = k * stride, k * stride + windows
        recon[s:e] += data_plot[k]
        weights[s:e] += 1.0
    weights[weights == 0] = 1.0
    recon /= weights

    x = np.arange(total_len)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, recon, linewidth=1.2)

    palette = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}  # upright / transition / slouch
    if show_bands or show_vlines:
        for k, lab in enumerate(y_plot):
            s, e = k * stride, k * stride + windows
            c = palette.get(int(lab), "0.5")
            if show_bands:
                ax.axvspan(s, e, color=c, alpha=0.12, lw=0)
            if show_vlines:
                ax.axvline(s, color=c, alpha=0.8, lw=0.8)

    ax.set_xlim(0, total_len - 1)
    ax.set_xlabel("Frame")
    ax.set_ylabel(f"Channel {chanel}")
    if title is None:
        end_idx = start_window + n_plot - 1
        title = (f"Reconstructed signal — ch {chanel}, window={windows}, stride={stride} | "
                 f"windows {start_window}–{end_idx}")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Legend
    unique_labels = sorted(set(int(v) for v in y_plot))
    handles = [Patch(facecolor=palette.get(k, "0.7"),
                     edgecolor=palette.get(k, "0.7"),
                     alpha=0.25,
                     label=f"{k} – {(label_names[k] if label_names and 0<=k<len(label_names) else str(k))}")
               for k in unique_labels]
    if handles:
        ax.legend(handles=handles, loc="upper right", title="Window labels")

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return (x, recon) if return_signal else None

def main():
    X_train, y_train = X_and_y("train", ['Ivan', 'Dario', 'David', 'Claire', 'Mohid'])
    plot_reconstructed_signal(X_train, y_train, windows=75, chanel=4, num_windows=200,
                              start_window=0, show=True)


    '''
    df = pd.read_csv(r"data/beep_schedules_Claire0/airpods_motion_d1760629578.csv")
    duration = 1600
    type = 'pitch_rad'
    signal = df[type]
    signal = signal[:duration]
    # pull the signal, clean NaNs for a clean plot
    s = pd.to_numeric(df[type], errors="coerce").interpolate(limit_direction="both")
    s = s.iloc[:duration].to_numpy(dtype=np.float32)

    fs = detect_fs(df)   # uses time column if available, else 50 Hz
    fc = 1.5             # cutoff (try 5–8 Hz for posture)
    a  = ema_alpha(fc, fs)
    s_filt = ema_1d(s, a)

    plt.plot(np.arange(len(signal)), signal)
    plt.plot(np.arange(len(signal)), s_filt)
    plt.xlabel("Time/Frames")
    plt.ylabel("delta signal")
    plt.title("Visualization Delta Signal")
    plt.legend()
    plt.show()
    '''

if __name__ == '__main__':
    main()