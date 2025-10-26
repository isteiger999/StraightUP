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
    chanel: Union[int, Sequence[int]],   # <-- now supports list/tuple/ndarray of ints
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
    show: bool = True,
    savepath: Optional[str] = None,
):
    """
    Reconstruct a continuous signal from overlapped windows and visualize
    which window corresponds to which label. Can overlay multiple channels.

    Parameters
    ----------
    X_train : array-like
        Shapes supported:
        - (n_windows, window_len, n_channels)
        - (n_windows, n_channels, window_len)
        - (n_windows, window_len)  # single-channel only
    y_train : array-like of int, length n_windows
        Label per window (0: upright, 1: transition, 2: slouch).
    windows : int
        Window length in frames (e.g., 75).
    chanel : int or sequence of int
        Channel index(es) to plot. If multiple, all are overlaid in the same axes.
        For 2-D X (single-channel), this must be a single index (0).
    stride : int, optional
        Frame step between consecutive window starts. Defaults to windows//3.
    num_windows : int, optional
        Number of consecutive windows to plot from `start_window`. If None, plots all.
    start_window : int
        Starting window index.
    label_names : sequence of str
        Names for labels [0,1,2].
    show_bands : bool
        Draw translucent bands for each window span colored by its label.
    show_vlines : bool
        Draw a vertical line at each window start colored by its label.
    title : str, optional
    figsize : (int, int)
    return_signal : bool
        If True, returns (x, recon) where:
            - recon has shape (total_len,) for a single channel,
            - recon has shape (total_len, n_selected_channels) for multiple channels.
    show : bool
        Call plt.show() if True.
    savepath : str, optional
        If provided, saves the figure to this path.

    Returns
    -------
    (x, recon) if return_signal else None
    """
    # --- normalize channel argument to a clean, unique list of ints ---
    if isinstance(chanel, (int, np.integer)):
        channels = [int(chanel)]
    else:
        try:
            channels = [int(c) for c in chanel]
        except Exception as e:
            raise TypeError("`chanel` must be an int or a sequence of ints.") from e
        # de-duplicate while preserving order
        seen = set(); _uniq = []
        for c in channels:
            if c not in seen:
                _uniq.append(c); seen.add(c)
        channels = _uniq

    X = np.asarray(X_train)
    y = np.asarray(y_train).reshape(-1).astype(int)

    if X.ndim not in (2, 3):
        raise ValueError(f"X_train must be 2D or 3D, got {X.ndim}D.")

    # --- Extract requested channels -> data3 with shape (n_windows_total, window_len, n_sel_channels) ---
    if X.ndim == 2:
        # Single-channel case: (n_windows_total, window_len)
        if len(channels) != 1 or channels[0] != 0:
            raise ValueError("For 2D X_train (single-channel), use chanel=[0] or chanel=0.")
        data3 = X[:, :, None]  # add a 'channel' axis of size 1
        if data3.shape[1] != windows:
            windows = int(data3.shape[1])
    else:
        # 3D input
        if X.shape[1] == windows:
            # (n_windows_total, window_len, n_channels)
            n_channels_total = X.shape[2]
            if not all(0 <= c < n_channels_total for c in channels):
                raise IndexError(f"Some indices in {channels} are out of range [0, {n_channels_total-1}].")
            data3 = X[:, :, channels]  # -> (nW, win, n_sel)
        elif X.shape[2] == windows:
            # (n_windows_total, n_channels, window_len)
            n_channels_total = X.shape[1]
            if not all(0 <= c < n_channels_total for c in channels):
                raise IndexError(f"Some indices in {channels} are out of range [0, {n_channels_total-1}].")
            data3 = X[:, channels, :]              # (nW, n_sel, win)
            data3 = np.transpose(data3, (0, 2, 1)) # -> (nW, win, n_sel)
        else:
            # Infer which axis is window_len
            if X.shape[1] >= X.shape[2]:
                # Assume (nW, win, nCh)
                n_channels_total = X.shape[2]
                if not all(0 <= c < n_channels_total for c in channels):
                    raise IndexError(f"Some indices in {channels} are out of range [0, {n_channels_total-1}].")
                data3 = X[:, :, channels]
                windows = int(X.shape[1])
            else:
                # Assume (nW, nCh, win)
                n_channels_total = X.shape[1]
                if not all(0 <= c < n_channels_total for c in channels):
                    raise IndexError(f"Some indices in {channels} are out of range [0, {n_channels_total-1}].")
                data3 = X[:, channels, :]
                windows = int(X.shape[2])
                data3 = np.transpose(data3, (0, 2, 1))

    n_windows_total, win_len, n_sel = data3.shape
    if win_len != windows:
        windows = win_len

    if y.shape[0] != n_windows_total:
        raise ValueError(f"y_train length ({y.shape[0]}) != number of windows ({n_windows_total}).")

    if not (0 <= start_window < n_windows_total):
        raise ValueError(f"start_window must be within [0, {n_windows_total-1}], got {start_window}.")

    n_plot = (n_windows_total - start_window) if (num_windows is None) else num_windows
    if n_plot <= 0 or start_window + n_plot > n_windows_total:
        raise ValueError("Invalid num_windows/start_window slice.")

    data_plot3 = data3[start_window : start_window + n_plot, :, :]  # (n_plot, win, n_sel)
    y_plot = y[start_window : start_window + n_plot]

    if stride is None:
        stride = max(1, windows // 3)
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")

    # --- Overlap-add reconstruction for all selected channels at once ---
    total_len = (n_plot - 1) * stride + windows
    recon = np.zeros((total_len, n_sel), dtype=float)
    weights = np.zeros(total_len, dtype=float)

    for k in range(n_plot):
        s, e = k * stride, k * stride + windows
        recon[s:e, :] += data_plot3[k, :, :]   # add all channels for this window
        weights[s:e] += 1.0

    weights[weights == 0] = 1.0
    recon = recon / weights[:, None]          # average overlaps, channel-wise

    # --- Visualization ---
    x = np.arange(total_len)
    fig, ax = plt.subplots(figsize=figsize)

    # plot each selected channel
    channel_lines = []
    for j in range(n_sel):
        (line,) = ax.plot(x, recon[:, j], linewidth=1.2, label=f"ch {channels[j]}")
        channel_lines.append(line)

    # window label decorations
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
    ax.set_ylabel("Signal value")
    if title is None:
        end_idx = start_window + n_plot - 1
        title = (f"Reconstructed signal — channels {channels}, window={windows}, stride={stride} | "
                 f"windows {start_window}–{end_idx}")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Legends: channels (left) + window labels (right)
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

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Return 1D for single-channel to preserve backward compatibility
    if return_signal:
        if n_sel == 1:
            return x, recon[:, 0]
        return x, recon
    return None

def main():
    X_train, y_train = X_and_y("train", ['Ivan', 'Dario', 'David', 'Claire', 'Mohid'])
    plot_reconstructed_signal(
        X_train, y_train,
        windows=75,
        chanel=[4, 9],
        num_windows=200,
        start_window=0,
        show=True
    )


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