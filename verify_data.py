from constants import set_seeds, configure_tensorflow
set_seeds()
configure_tensorflow()
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
from typing import Optional, Sequence, Tuple, Union
from matplotlib.widgets import Slider
from matplotlib.patches import Patch
import os
import glob
import re

# ---- Delete delta files ----
def delete_airpods_motion_d_csvs(root: str = "data", dry_run: bool = True):
    """
    Delete ONLY files matching:
        data/beep_schedules_*/airpods_motion_d<digits>.csv

    Safety:
      - Uses a strict regex (airpods_motion_d\\d+\\.csv).
      - Set dry_run=False to actually delete.

    Returns a list of file paths (matched; deleted if dry_run=False).
    """
    pattern = os.path.join(root, "beep_schedules_*", "airpods_motion_d*.csv")
    files = sorted(glob.glob(pattern))
    rx = re.compile(r"^airpods_motion_d\d+\.csv\Z")

    matched = []
    for p in files:
        name = os.path.basename(p)
        if rx.fullmatch(name) and os.path.isfile(p):
            matched.append(p)

    if not matched:
        print("No matching files found.")
        return []

    if dry_run:
        print(f"[DRY RUN] Would delete {len(matched)} file(s):")
        for p in matched:
            print(" -", p)
        return matched

    deleted = []
    for p in matched:
        try:
            os.remove(p)
            deleted.append(p)
        except Exception as e:
            print(f"Failed to delete {p}: {e}")

    print(f"Deleted {len(deleted)} file(s).")
    for p in deleted:
        print(" -", p)
    return deleted
# -------------------

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

def plot_label_verlauf(y_train, length):
    plt.plot(np.arange(y_train[:length].shape[0]), y_train[:length])
    plt.xlabel("windows")
    plt.ylabel("Label")
    plt.show()

def plot_reconstructed_signal_slider(
    X_train,
    y_train,
    windows: int,
    chanel,
    *,
    stride: Optional[int] = None,
    view_windows: int = 100,        # how many windows to show at once
    start_window: int = 0,          # initial position
    label_names: Optional[Sequence[str]] = ("upright", "transition", "slouch"),
    show_bands: bool = True,
    show_vlines: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
):
    """
    Interactive viewer: same visualization as plot_reconstructed_signal,
    but with a horizontal slider to move the start window. Always shows
    exactly `view_windows` windows at a time.

    Adds thick separators after every 718 windows to mark participant changes,
    and even thicker blue separators every 2872 windows.
    """
    # ---- config for participant separators ----
    SEP_EVERY = 718              # black separators every 718 windows
    SEP_COLOR = "black"
    SEP_LW = 3.0
    SEP_ALPHA = 0.9

    # ---- additional thicker blue separators every 2872 windows ----
    SUPER_SEP_EVERY = 2872
    SUPER_SEP_COLOR = "#1f77b4"  # matplotlib's default blue
    SUPER_SEP_LW = 4.5           # a bit thicker than the 718-line
    SUPER_SEP_ALPHA = 0.95

    # --- normalize channel argument (keep your original 'chanel' spelling) ---
    if isinstance(chanel, (int, np.integer)):
        channels = [int(chanel)]
    else:
        channels = [int(c) for c in chanel]
        seen = set(); _uniq = []
        for c in channels:
            if c not in seen:
                _uniq.append(c); seen.add(c)
        channels = _uniq

    # --- basic arrays ---
    X = np.asarray(X_train)
    y = np.asarray(y_train).reshape(-1).astype(int)

    if X.ndim not in (2, 3):
        raise ValueError(f"X_train must be 2D or 3D, got {X.ndim}D.")

    # --- extract to (nW, win, n_sel) exactly (same logic as your function) ---
    if X.ndim == 2:
        if len(channels) != 1 or channels[0] != 0:
            raise ValueError("For 2D X_train (single-channel), use chanel=[0] or chanel=0.")
        data3 = X[:, :, None]
        if data3.shape[1] != windows:
            windows = int(data3.shape[1])
    else:
        if X.shape[1] == windows:
            n_channels_total = X.shape[2]
            if not all(0 <= c < n_channels_total for c in channels):
                raise IndexError(f"Some indices in {channels} are out of range [0, {n_channels_total-1}].")
            data3 = X[:, :, channels]
        elif X.shape[2] == windows:
            n_channels_total = X.shape[1]
            if not all(0 <= c < n_channels_total for c in channels):
                raise IndexError(f"Some indices in {channels} are out of range [0, {n_channels_total-1}].")
            data3 = np.transpose(X[:, channels, :], (0, 2, 1))
        else:
            if X.shape[1] >= X.shape[2]:
                n_channels_total = X.shape[2]
                if not all(0 <= c < n_channels_total for c in channels):
                    raise IndexError(f"Some indices in {channels} are out of range [0, {n_channels_total-1}].")
                data3 = X[:, :, channels]
                windows = int(X.shape[1])
            else:
                n_channels_total = X.shape[1]
                if not all(0 <= c < n_channels_total for c in channels):
                    raise IndexError(f"Some indices in {channels} are out of range [0, {n_channels_total-1}].")
                data3 = np.transpose(X[:, channels, :], (0, 2, 1))
                windows = int(X.shape[2])

    n_windows_total, win_len, n_sel = data3.shape
    if win_len != windows:
        windows = win_len

    if y.shape[0] != n_windows_total:
        raise ValueError(f"y_train length ({y.shape[0]}) != number of windows ({n_windows_total}).")

    if stride is None:
        stride = max(1, windows // 3)
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")

    view_windows = int(view_windows)
    if not (1 <= view_windows <= n_windows_total):
        view_windows = min(100, n_windows_total)

    max_start = n_windows_total - view_windows
    start_window = int(np.clip(int(start_window), 0, max_start))

    # global y-limits so the vertical scale doesn't jump when sliding
    global_min = float(np.nanmin(data3))
    global_max = float(np.nanmax(data3))
    pad = 0.05 * max(1e-9, global_max - global_min)
    y_min, y_max = global_min - pad, global_max + pad

    palette = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}  # upright/transition/slouch

    # --- plotting & slider ---
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.22)  # room for the slider

    def draw_slice(start_idx: int):
        ax.clear()
        k0, k1 = start_idx, start_idx + view_windows
        data_plot3 = data3[k0:k1, :, :]
        y_plot = y[k0:k1]
        n_plot = data_plot3.shape[0]

        total_len = (n_plot - 1) * stride + windows
        recon = np.zeros((total_len, n_sel), dtype=float)
        weights = np.zeros(total_len, dtype=float)
        for k in range(n_plot):
            s, e = k * stride, k * stride + windows
            recon[s:e, :] += data_plot3[k, :, :]
            weights[s:e] += 1.0
        weights[weights == 0] = 1.0
        recon = recon / weights[:, None]
        x = np.arange(total_len)

        # channels
        channel_lines = []
        for j in range(n_sel):
            (line,) = ax.plot(x, recon[:, j], linewidth=1.2, label=f"ch {channels[j]}")
            channel_lines.append(line)

        # label decorations
        if show_bands or show_vlines:
            for k, lab in enumerate(y_plot):
                s, e = k * stride, k * stride + windows
                c = palette.get(int(lab), "0.5")
                if show_bands:
                    ax.axvspan(s, e, color=c, alpha=0.12, lw=0)
                if show_vlines:
                    ax.axvline(s, color=c, alpha=0.8, lw=0.8)

        # --- participant separators (every 718 windows) ---
        if SEP_EVERY > 0:
            boundaries = np.arange(SEP_EVERY, n_windows_total, SEP_EVERY, dtype=int)
            boundaries_in_view = boundaries[(boundaries >= k0) & (boundaries < k1)]
            for b in boundaries_in_view:
                x_sep = (b - k0) * stride
                ax.axvline(x_sep, color=SEP_COLOR, lw=SEP_LW, alpha=SEP_ALPHA)

        # --- thicker blue separators (every 2872 windows) ---
        if SUPER_SEP_EVERY > 0:
            boundaries2 = np.arange(SUPER_SEP_EVERY, n_windows_total, SUPER_SEP_EVERY, dtype=int)
            boundaries2_in_view = boundaries2[(boundaries2 >= k0) & (boundaries2 < k1)]
            for b in boundaries2_in_view:
                x_sep = (b - k0) * stride
                ax.axvline(x_sep, color=SUPER_SEP_COLOR, lw=SUPER_SEP_LW, alpha=SUPER_SEP_ALPHA)

        ax.set_xlim(0, total_len - 1)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Signal value")
        end_idx = start_idx + n_plot - 1
        ttl = title or (f"Reconstructed signal — channels {channels}, window={windows}, stride={stride} | "
                        f"windows {start_idx}–{end_idx}")
        ax.set_title(ttl)
        ax.grid(True, alpha=0.3)

        # legends
        legend_channels = ax.legend(handles=channel_lines, loc="upper left", title="Channels")
        unique_labels = sorted(set(int(v) for v in y_plot))
        handles_labels = [Patch(facecolor=palette.get(k, "0.7"),
                                edgecolor=palette.get(k, "0.7"),
                                alpha=0.25,
                                label=f"{k} – {(label_names[k] if label_names and 0<=k<len(label_names) else str(k))}")
                         for k in unique_labels]
        if handles_labels:
            legend_labels = ax.legend(handles=handles_labels, loc="upper right", title="Window labels")
            ax.add_artist(legend_channels)  # keep both legends
        fig.canvas.draw_idle()

    # initial draw
    draw_slice(start_window)

    # slider under the plot
    ax_slider = fig.add_axes([0.12, 0.08, 0.76, 0.04])
    s_start = Slider(ax_slider, "start_window", 0, max_start, valinit=start_window, valstep=1)

    def on_change(val):
        draw_slice(int(s_start.val))

    s_start.on_changed(on_change)
    plt.show()
    return fig, ax, s_start

def main():
    
    X_train, y_train = X_and_y("train", ['David', 'Ivan', 'Claire', 'Mohid', 'Dario'],
                               label_anchor='center') #['Ivan', 'Dario', 'David', 'Claire', 'Mohid']
    #plot_label_verlauf(y_train, length=700)

    plot_reconstructed_signal_slider(
        X_train, y_train,
        windows=75,
        chanel=[0, 7, 11, 12], #4
        view_windows=100,
        start_window=0
    )
    
    ##
    # Delete delta files:
    # delete_airpods_motion_d_csvs("data", dry_run=False)  # actually delete
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