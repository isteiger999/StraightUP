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

def delete_airpods_motion_ds_csvs(root: str = "data", dry_run: bool = True):
    """
    Delete ONLY files matching:
        data/beep_schedules_*/airpods_motion_ds<digits>.csv

    Safety:
      - Uses a strict regex (airpods_motion_ds\\d+\\.csv).
      - Set dry_run=False to actually delete.

    Returns a list of file paths (matched; deleted if dry_run=False).
    """
    pattern = os.path.join(root, "beep_schedules_*", "airpods_motion_ds*.csv")
    files = sorted(glob.glob(pattern))
    rx = re.compile(r"^airpods_motion_ds\d+\.csv\Z")

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

    # ---- plotting: participants' average std per channel (exclude quat_rel_* and pitch_rad) ----
    channels = set()
    for _pname, ch_dict in per_participant_avg_std.items():
        channels.update(ch_dict.keys())
    channels_to_plot = [
        c for c in sorted(channels, key=lambda s: s.lower())
        if c.lower() not in REL_QUAT_EXCLUDE_FROM_PLOT and c.lower() not in PITCH_EXCLUDE
    ]

    if channels_to_plot and per_participant_avg_std:
        participants_order = sorted(per_participant_avg_std.keys(), key=lambda s: s.lower())
        x = np.arange(len(channels_to_plot))

        fig = plt.figure(figsize=(max(8.0, 0.7 * len(channels_to_plot)), 6.0))
        ax = plt.gca()

        nP = len(participants_order)
        jitter = 0.8 / max(nP, 1)

        for i, pname in enumerate(participants_order):
            y_vals, x_idx = [], []
            for j, ch in enumerate(channels_to_plot):
                v = per_participant_avg_std[pname].get(ch, {}).get("mean_std", np.nan)
                if np.isfinite(v):
                    y_vals.append(v)
                    x_idx.append(x[j] + (i - (nP - 1) / 2.0) * jitter)
            if y_vals:
                ax.scatter(x_idx, y_vals, label=pname, alpha=0.9, s=28)

        ax.set_xticks(x)
        ax.set_xticklabels(channels_to_plot, rotation=45, ha="right")
        ax.set_ylabel("Average std (per participant)")
        ax.set_title("Per-participant average std by channel (from airpods_motion_d*.csv)")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
        if len(participants_order) <= 12:
            ax.legend(title="Participant", fontsize=9)
        fig.tight_layout()
        plt.show()

    # ---- sort for deterministic returned dicts ----
    def _sort_nested(d):
        return {k: d[k] for k in sorted(d.keys(), key=lambda x: str(x).lower())}

    per_file_std = {pk: _sort_nested(per_file_std[pk]) for pk in sorted(per_file_std.keys(), key=lambda x: str(x).lower())}
    per_participant_avg_std = {pk: _sort_nested(per_participant_avg_std[pk]) for pk in sorted(per_participant_avg_std.keys(), key=lambda x: str(x).lower())}
    overall_avg_std = _sort_nested(overall_avg_std)

    return per_file_std, per_participant_avg_std, overall_avg_std

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
    per_file_std, per_participant_avg_std, overall_avg_std = calculate_std(['Abi', 'Ivan', 'Dario', 'Mohid', 'Claire', 'David', 'Svetlana'])
    #print(f"Per file std {per_file_std}")
    #print(f"Per participant std {per_participant_avg_std}")
    #print(f"Overall std {overall_avg_std}")
    '''
    X_train, y_train = X_and_y("train", ['Svetlana', 'Claire', 'Dario', 'Mohid', 'Ivan', 'David', 'Abi'],
                               label_anchor='center') #['Ivan', 'Dario', 'David', 'Claire', 'Mohid']
    #plot_label_verlauf(y_train, length=2900)

    plot_reconstructed_signal_slider(
        X_train, y_train,
        windows=75,
        chanel=[9], #4
        view_windows=100,
        start_window=0
    )
    
    ##
    # Delete delta files:
    #delete_airpods_motion_d_csvs("data", dry_run=False)  # actually delete
    #delete_airpods_motion_ds_csvs("data", dry_run=False)
    '''
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
    
    df_claire = pd.read_csv("data/beep_schedules_David0/airpods_motion_d1760039321.csv")
    df_claire_grav_z = df_claire["grav_z"]
    df_claire_grav_z = df_claire_grav_z.iloc[:500]
    df_claire_grav_y = df_claire["grav_y"]
    df_claire_grav_y = df_claire_grav_y.iloc[:500]
    df_claire_grav_x = df_claire["grav_x"]
    df_claire_grav_x = df_claire_grav_x.iloc[:500]
    '''
    df_david = pd.read_csv("data/beep_schedules_David0/airpods_motion_d1760039321.csv")
    df_david = df_david["grav_z"]
    df_david = df_david.iloc[:500]
    df_abi = pd.read_csv("data/beep_schedules_Abi0/airpods_motion_d1762015680.csv")
    df_abi = df_abi["grav_z"]
    df_abi = df_abi.iloc[:500]
    '''
    
    plt.plot(df_claire_grav_z.values, label="claire grav_z")
    plt.plot(df_claire_grav_y.values, label="claire grav_y")
    plt.plot(df_claire_grav_x.values, label="claire grav_x")
    plt.legend()
    plt.show()
    '''
    df_mohid0 = pd.read_csv("data/beep_schedules_Mohid0/airpods_motion_d1760172227.csv")
    df_mohid0 = df_mohid0["acc_y"]
    df_mohid0 = df_mohid0.iloc[:500]
    df_mohid1 = pd.read_csv("data/beep_schedules_Mohid1/airpods_motion_d1760172227.csv")
    df_mohid1 = df_mohid1["acc_y"]
    df_mohid1 = df_mohid1.iloc[:500]
    df_mohid2 = pd.read_csv("data/beep_schedules_Mohid2/airpods_motion_d1760174060.csv")
    df_mohid2 = df_mohid2["acc_y"]
    df_mohid2 = df_mohid2.iloc[:500]
    df_mohid3 = pd.read_csv("data/beep_schedules_Mohid3/airpods_motion_d1760174615.csv")
    df_mohid3 = df_mohid3["acc_y"]
    df_mohid3 = df_mohid3.iloc[:500]
    '''
    '''
    df_test1 = pd.read_csv("airpods_motion_1762617123.csv")
    df_test2 = pd.read_csv("airpods_motion_1762618651.csv")
    df_test3 = pd.read_csv("airpods_motion_1762621605.csv")
    df_test4 = pd.read_csv("airpods_motion_1762622942.csv")
    df_test5 = pd.read_csv("airpods_motion_1762624794.csv")
    df_test5 = df_test5['acc_y']

    df_test1 = df_test1['acc_y']
    df_test2 = df_test2['acc_y']
    df_test3 = df_test3['acc_y']
    df_test_gravz = df_test4['grav_z']
    df_test_gravy = df_test4['grav_y']
    df_test_gravx = df_test4['grav_x']
    #plt.plot(df_mohid0.values, label="Mohid0")
    #plt.plot(df_mohid1.values, label="Mohid1")
    #plt.plot(df_mohid2.values, label="Mohid2")
    #plt.plot(df_test1.values, label="test1 Ivan 8. Nov")
    #plt.plot(df_test2.values, label="test2 Ivan 8. Nov")
    plt.plot(df_test5.values, label="test5 Ivan 8. Nov")
    #plt.plot(df_test_gravz.values, label="grav_z Ivan 8. Nov")
    #plt.plot(df_test_gravy.values, label="grav_y Ivan 8. Nov")
    #plt.plot(df_test_gravx.values, label="grav_x Ivan 8. Nov")
    #plt.plot(np.arange(len(df_test4)), np.sqrt(df_test_gravz**2 + df_test_gravy**2 + df_test_gravx**2))

    #plt.plot(df_claire.values, label="Claire")
    #plt.plot(df_abi.values, label="Abi")
    #plt.plot(df_svetlana.values, label="Svetlana")
    #plt.plot(df_david.values)
    plt.legend()
    plt.show()
    '''

if __name__ == '__main__':
    main()