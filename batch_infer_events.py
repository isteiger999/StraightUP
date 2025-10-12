#!/usr/bin/env python3
"""
Batch-infer events for every airpods_motion_*.csv under ./data by default.

Behavior:
- Recursively finds all IMU CSVs matching: airpods_motion_*.csv (under --root, default: ./data)
- For each IMU, hunts for an appropriate beep_schedule_*.csv (same folder first, then globally under --root);
  best match is chosen by (1) filename similarity, then (2) closest modified time.
- Runs a 2-pass axis-adaptive detection (strict then relaxed).
- ALWAYS seeds an initial UPRIGHT_HOLD_START at the first IMU timestamp (t0).
  * Its "value" (type=...) comes from the first slouch movement if available; otherwise it falls back to the
    first positive beep's movement, else "normal_slouch".
- Always overwrites outputs placed NEXT TO the IMU file:
    events_inferred_<imu-stem>.csv
    events_debug_<imu-stem>.csv
- If no schedule is found:
    We still write an events file with just the initial UPRIGHT_HOLD_START at t0 (and a header-only debug CSV).

Usage:
    python batch_infer_events.py --verbose
"""

import argparse, csv, math, os, re, sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

RAD2DEG = 180.0 / math.pi

# ====== Tunables ============================================================
# windows / gates
BASELINE_PRE_S   = 2.0      # seconds before beep for baseline stats
SEARCH_WINDOW_S  = 6.5      # seconds after beep to search for onset
CONFIRM_LEN_S    = 0.50     # seconds after onset to measure angle change
CALM_LEN_S       = 0.18     # seconds of "calm" we require for holds

# onset/hold easing
RELAX_FACTOR     = 0.70     # relaxed pass multipliers (lower = easier)
GYRO_STD_STRICT  = 2.0      # baseline_gm + k*std for onset gate
GYRO_STD_CALM    = 3.0      # baseline_gm + k*std for "calm" during holds
UPRIGHT_NEAR_DEG = 6.0      # closeness to baseline angle to call upright (unused but kept for reference)

# angle thresholds (absolute degrees; sign-agnostic)
DTHETA_IN = {
    "roll":         3.5,    # lateral (left/right)
    "pitch_micro":  2.0,    # micro slouch
    "pitch":        6.0,    # normal slouch
    "pitch_fast":   5.0,    # fast slouch
}
THETA_HOLD = {
    "roll":         5.5,
    "pitch_micro":  3.0,
    "pitch":        8.0,
    "pitch_fast":   8.0,
}

# fallback synthetic labels (guarantee) — (onset_offset_s, hold_offset_s) from beep
FALLBACK_OFFSETS = {
    "fast_slouch":   (0.25, 0.70),
    "normal_slouch": (0.45, 0.90),
    "micro_slouch":  (0.60, 1.00),
    "lateral_left":  (0.45, 0.90),
    "lateral_right": (0.45, 0.90),
}

WRITE_DEBUG = True  # set False to stop writing events_debug_*.csv
# ===========================================================================

# ---------- helpers ----------
def find_col(df: "pd.DataFrame", patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in df.columns:
            if c == pat or c.lower() == pat.lower() or rx.fullmatch(c) or rx.search(c):
                return c
    return None

def _choose_time_scale_by_plausible_dt(dtn: np.ndarray) -> Tuple[float, str]:
    """
    Given raw diffs (same units as the raw time column), choose a scale -> seconds so that
    median dt lands in a plausible IMU range (0.1 ms .. 1 s).
    """
    dtn = np.asarray(dtn, dtype=float)
    dtn = dtn[np.isfinite(dtn) & (dtn != 0)]
    if dtn.size == 0:
        return 1.0, "unknown"

    candidates = [("s", 1.0), ("ms->s", 1e-3), ("us->s", 1e-6), ("ns->s", 1e-9)]
    for label, scale in candidates:
        dt_med = float(np.median(np.abs(dtn) * scale))
        if 1e-4 <= dt_med <= 1.0:   # 0.1 ms .. 1 s
            return scale, label
    return 1.0, "s"

def normalize_time_units_by_dt(t_raw: np.ndarray) -> Tuple[np.ndarray, str]:
    t = t_raw.astype(float)
    tn = t[np.isfinite(t)]
    if len(tn) < 3:
        return t, "unknown"
    dtn = np.diff(tn)
    scale, label = _choose_time_scale_by_plausible_dt(dtn)
    return t * scale, label

def get_time_seconds(df: "pd.DataFrame") -> Tuple[np.ndarray, str]:
    tcol = None
    for cand in ["t_sec", "timestamp", "time", r".*time.*"]:
        tcol = find_col(df, [cand])
        if tcol: break
    if tcol is None:
        raise ValueError("No time column found (t_sec/timestamp/time).")
    t_sec, units = normalize_time_units_by_dt(df[tcol].to_numpy())
    return t_sec, units

def get_gyro(df: "pd.DataFrame") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = df[find_col(df, ["gx", r"rotationRate\.x", r".*rotation.*x", r"rot.*x"])].astype(float).to_numpy()
    gy = df[find_col(df, ["gy", r"rotationRate\.y", r".*rotation.*y", r"rot.*y"])].astype(float).to_numpy()
    gz = df[find_col(df, ["gz", r"rotationRate\.z", r".*rotation.*z", r"rot.*z"])].astype(float).to_numpy()
    return gx, gy, gz

def euler_from_quat_arrays(qw, qx, qy, qz):
    t0 = +2.0*(qw*qx + qy*qz); t1 = +1.0 - 2.0*(qx*qx + qy*qy); roll_x = np.arctan2(t0, t1)
    t2 = +2.0*(qw*qy - qz*qx); t2 = np.clip(t2, -1.0, +1.0);   pitch_y = np.arcsin(t2)
    t3 = +2.0*(qw*qz + qx*qy); t4 = +1.0 - 2.0*(qy*qy + qz*qz); yaw_z   = np.arctan2(t3, t4)
    return yaw_z*RAD2DEG, pitch_y*RAD2DEG, roll_x*RAD2DEG

def get_pitch_roll(df: "pd.DataFrame") -> Tuple[np.ndarray, np.ndarray]:
    pc = find_col(df, ["pitch", r".*pitch.*"]); rc = find_col(df, ["roll", r".*roll.*"])
    if pc and rc:
        return df[pc].astype(float).to_numpy(), df[rc].astype(float).to_numpy()
    gcx = find_col(df, ["grav_x", r"gravity\.x", r".*gravity.*x"])
    gcy = find_col(df, ["grav_y", r"gravity\.y", r".*gravity.*y"])
    gcz = find_col(df, ["grav_z", r"gravity\.z", r".*gravity.*z"])
    if gcx and gcy and gcz:
        gx = df[gcx].astype(float).to_numpy(); gy = df[gcy].astype(float).to_numpy(); gz = df[gcz].astype(float).to_numpy()
        pitch = np.arctan2(-gx, np.sqrt(gy*gy + gz*gz)) * RAD2DEG
        roll  = np.arctan2(gy, gz) * RAD2DEG
        return pitch, roll
    qw = find_col(df, ["qw","q_w", r".*quaternion.*w"])
    qx = find_col(df, ["qx","q_x", r".*quaternion.*x"])
    qy = find_col(df, ["qy","q_y", r".*quaternion.*y"])
    qz = find_col(df, ["qz","q_z", r".*quaternion.*z"])
    if qw and qx and qy and qz:
        w = df[qw].astype(float).to_numpy(); x = df[qx].astype(float).to_numpy()
        y = df[qy].astype(float).to_numpy(); z = df[qz].astype(float).to_numpy()
        _, pitch, roll = euler_from_quat_arrays(w,x,y,z)
        return pitch, roll
    raise ValueError("Could not find pitch/roll or gravity or quaternion columns.")

def lowpass_ma(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return x
    pad = np.pad(x, (k-1, 0), mode="edge"); cs = np.cumsum(pad)
    return (cs[k:] - cs[:-k]) / k

def first_run_above(x: np.ndarray, th: float, run_len: int) -> Optional[int]:
    cnt = 0
    for i, v in enumerate(x):
        if v > th:
            cnt += 1
            if cnt >= run_len:
                return i - run_len + 1
        else:
            cnt = 0
    return None

# ---------- axis-adaptive detection ----------
def detect_for_axis(angle, dangle, gyromag, t, fs, i_beep, i0b, i1s,
                    dtheta_in: float, theta_hold: float,
                    strict: bool, axis_name: str):
    base_ang = float(np.nanmean(angle[max(0,i0b):i_beep]))
    base_gm  = float(np.nanmean(gyromag[max(0,i0b):i_beep]))
    std_gm   = float(np.nanstd(gyromag[max(0,i0b):i_beep])) + 1e-6

    run_len = max(6, int(round(0.12*fs)))
    th_g = base_gm + (GYRO_STD_STRICT if strict else (GYRO_STD_STRICT*RELAX_FACTOR)) * std_gm
    onset_rel = first_run_above(gyromag[i_beep:i1s], th_g, run_len)

    if onset_rel is None:
        pre_deriv = np.abs(dangle[max(0,i0b):i_beep])
        med_abs = float(np.nanmedian(pre_deriv)) if pre_deriv.size else 0.0
        k = 8.0 if strict else 6.0
        th_d = med_abs * k + 1e-6
        onset_rel = first_run_above(np.abs(dangle[i_beep:i1s]), th_d, run_len)

    if onset_rel is None:
        return {"found": False, "onset_idx": None, "hold_idx": None, "onset_score": 0.0,
                "dtheta": 0.0, "base_ang": base_ang, "base_gm": base_gm, "std_gm": std_gm, "axis": axis_name}

    i_on = i_beep + onset_rel
    i_c  = min(len(t)-1, i_on + int(round(CONFIRM_LEN_S * fs)))
    dtheta = abs(float(angle[i_c] - angle[i_on]))
    need = dtheta_in if strict else dtheta_in * RELAX_FACTOR
    onset_ok = (dtheta >= need)

    # Hold detection (only if onset passed)
    i_hold = None
    if onset_ok:
        calm_len = max(int(round(CALM_LEN_S * fs)), 3)
        th_hold_g = base_gm + GYRO_STD_CALM * std_gm
        need_hold = theta_hold if strict else theta_hold * RELAX_FACTOR
        for j in range(i_on, i1s):
            j_end_idx = min(j + calm_len - 1, len(t) - 1)   # inclusive end
            seg = gyromag[j : j_end_idx + 1]
            if seg.size < calm_len:
                break
            calm = bool(np.all(seg < th_hold_g))
            high = abs(angle[j_end_idx] - base_ang) >= need_hold
            if calm and high:
                i_hold = j_end_idx
                break

    score = (dtheta / need) if need > 1e-6 else 0.0
    return {"found": onset_ok, "onset_idx": i_on, "hold_idx": i_hold,
            "onset_score": score, "dtheta": dtheta,
            "base_ang": base_ang, "base_gm": base_gm, "std_gm": std_gm,
            "axis": axis_name}

# ---------- one-session inference ----------
def infer_events_for_session(imu_csv: str, schedule_csv: str, out_csv: str,
                             debug_csv: str, verbose: bool=False) -> dict:
    imu = pd.read_csv(imu_csv)
    if len(imu.index) < 2:
        raise RuntimeError(f"IMU CSV appears empty or header-only: {imu_csv}")

    # Time cleanup
    t_sec, t_units = get_time_seconds(imu)
    valid = np.isfinite(t_sec)
    if valid.sum() < 5:
        raise RuntimeError("Too few valid time samples after dropping NaNs.")
    imu = imu.loc[valid].reset_index(drop=True)
    t_sec = t_sec[valid]

    # sort by time, then drop exact-duplicate timestamps to keep gradient stable
    order = np.argsort(t_sec, kind="mergesort")
    if not np.all(order == np.arange(len(order))):
        imu = imu.iloc[order].reset_index(drop=True); t_sec = t_sec[order]
    dts_all = np.diff(t_sec)
    keep = np.hstack(([True], dts_all > 0))
    if not np.all(keep):
        imu = imu.loc[keep].reset_index(drop=True)
        t_sec = t_sec[keep]
        dts_all = np.diff(t_sec)  # recompute after filtering

    dts_pos = dts_all[dts_all > 0]
    dt = float(np.median(dts_pos)) if dts_pos.size else 0.0
    fs = (1.0/dt) if dt > 0 else 50.0

    # Gyro magnitude (optional)
    try:
        gx, gy, gz = get_gyro(imu); gyromag = np.sqrt(gx*gx + gy*gy + gz*gz)
    except Exception:
        gyromag = np.zeros_like(t_sec)

    # Angles
    pitch, roll = get_pitch_roll(imu)
    k = max(1, int(round(0.14 * fs)))
    pitch_s = lowpass_ma(pitch, k); roll_s = lowpass_ma(roll, k)

    # Align lengths
    n = min(len(t_sec), len(pitch_s), len(roll_s), len(gyromag))
    t = t_sec[:n]; pitch_s = pitch_s[:n]; roll_s = roll_s[:n]; gyromag = gyromag[:n]
    dpitch = np.gradient(pitch_s, t); droll = np.gradient(roll_s, t)
    t0 = float(t[0])

    # Schedule & alignment
    sch = pd.read_csv(schedule_csv)
    if "t_sec" not in sch.columns: raise RuntimeError("Schedule missing t_sec.")
    s0, s1 = float(sch["t_sec"].min()), float(sch["t_sec"].max())
    i0, i1 = float(t[0]), float(t[-1])
    shift = 0.0
    if (s1 < i0) or (s0 > i1) or (s0 < 120.0 and i0 > 1e6):
        shift = i0 - s0; sch["t_sec"] = sch["t_sec"] + shift

    if verbose:
        ss0, ss1 = float(sch["t_sec"].min()), float(sch["t_sec"].max())
        print(f"[debug] IMU t range: {i0:.3f}..{i1:.3f} (dt~{(1/fs) if fs>0 else float('nan'):.3f}s @fs~{fs:.1f}Hz, units={t_units})")
        print(f"[debug] SCH t range: {ss0:.3f}..{ss1:.3f} (shift={shift:.3f})")

    # Beeps
    beeps = sch[sch["event"].isin(["BEEP_SLOUCH","BEEP_RECOVER"])].copy()

    # Count only positives as the denominator (skip hard negatives)
    def movement_type(val: str) -> str:
        return val.split("type=",1)[1] if "type=" in val else val

    hard_negs = {"neck_only","reach_left","reach_right","reach","twist"}
    beeps["mv"] = beeps["value"].astype(str).apply(movement_type)
    beeps["is_neg"] = beeps["mv"].isin(hard_negs)
    beeps_pos = beeps[~beeps["is_neg"]].copy()
    beeps_pos_total = int(beeps_pos.shape[0])

    # Determine movement label to attach to the synthetic initial upright
    first_slouch_beeps = beeps_pos[beeps_pos["event"] == "BEEP_SLOUCH"].sort_values("t_sec")
    if not first_slouch_beeps.empty:
        mv_for_upright = str(first_slouch_beeps["mv"].iloc[0])
    elif not beeps_pos.empty:
        mv_for_upright = str(beeps_pos.sort_values("t_sec")["mv"].iloc[0])  # earliest positive beep's movement
    else:
        mv_for_upright = "normal_slouch"

    def idx(ts): return int(np.searchsorted(t, ts, side="left"))

    out_rows = []; dbg_rows = []
    holds_total = 0; holds_detected = 0; beeps_in_range = 0

    # If there are no slouch/recover beeps at all, still write initial upright at t0 and return.
    if beeps.empty or beeps_pos_total == 0:
        os.makedirs(Path(out_csv).parent, exist_ok=True)
        # Always put an upright at t0
        out_rows.append([t0, "UPRIGHT_HOLD_START", mv_for_upright, 0.0])
        # Write outputs
        def _sort_key(r):  # UPRIGHT first if same timestamp
            return (r[0], 0 if r[1] == "UPRIGHT_HOLD_START" else 1)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t_sec","event","value","confidence"])
            for r in sorted(out_rows, key=_sort_key):
                w.writerow([f"{r[0]:.3f}", r[1], f"type={r[2]}", f"{r[3]:.3f}"])
        if WRITE_DEBUG:
            with open(debug_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["t_beep","event","type","axis_chosen","onset_time","hold_time","onset_score","dtheta","max_pitch","max_roll"]
                )
        if verbose:
            print("[warn] No usable beeps; wrote initial UPRIGHT_HOLD_START at t0.")
        return {"events": len(out_rows), "holds_total": holds_total, "holds_detected": holds_detected,
                "beeps_in_range": beeps_in_range, "beeps_total": beeps_pos_total, "shift": shift}

    # Normal path: detect around beeps
    for _, r in beeps_pos.iterrows():
        ev = str(r["event"]); mv = str(r["mv"])
        t_beep = float(r["t_sec"])
        i_beep = idx(t_beep); i0b = idx(t_beep-BASELINE_PRE_S); i1s = idx(t_beep+SEARCH_WINDOW_S)
        if i_beep <= 0 or i1s <= i_beep or i0b >= i_beep:
            continue
        beeps_in_range += 1

        # thresholds by axis/mode
        if "micro" in mv:
            need_pitch = ("pitch_micro", DTHETA_IN["pitch_micro"], THETA_HOLD["pitch_micro"])
        elif "fast" in mv:
            need_pitch = ("pitch_fast", DTHETA_IN["pitch_fast"], THETA_HOLD["pitch_fast"])
        else:
            need_pitch = ("pitch", DTHETA_IN["pitch"], THETA_HOLD["pitch"])
        need_roll = ("roll", DTHETA_IN["roll"], THETA_HOLD["roll"])

        # strict pass
        res_p_strict = detect_for_axis(pitch_s, dpitch, gyromag, t, fs, i_beep, i0b, i1s,
                                       need_pitch[1], need_pitch[2], True,  "pitch")
        res_r_strict = detect_for_axis(roll_s,  droll,  gyromag, t, fs, i_beep, i0b, i1s,
                                       need_roll[1],  need_roll[2],  True,  "roll")
        cand = max([res_p_strict, res_r_strict], key=lambda d: d["onset_score"])

        # relaxed pass if strict failed
        if not cand["found"]:
            res_p_relax = detect_for_axis(pitch_s, dpitch, gyromag, t, fs, i_beep, i0b, i1s,
                                          need_pitch[1]*RELAX_FACTOR, need_pitch[2]*RELAX_FACTOR, False, "pitch")
            res_r_relax = detect_for_axis(roll_s,  droll,  gyromag, t, fs, i_beep, i0b, i1s,
                                          need_roll[1]*RELAX_FACTOR,  need_roll[2]*RELAX_FACTOR,  False, "roll")
            cand = max([cand, res_p_relax, res_r_relax], key=lambda d: d["onset_score"])

        # debug maxima per beep
        max_pitch = float(np.max(np.abs(pitch_s[i_beep:i1s] - np.mean(pitch_s[max(0,i0b):i_beep])))) if i1s>i_beep and i_beep>i0b else 0.0
        max_roll  = float(np.max(np.abs(roll_s[i_beep:i1s]  - np.mean(roll_s[max(0,i0b):i_beep]))))  if i1s>i_beep and i_beep>i0b else 0.0

        axis_chosen = cand.get("axis", "none")
        used_fallback = False

        if cand["found"] and cand["onset_idx"] is not None:
            # ---- Detected onset ----
            i_on = cand["onset_idx"]
            conf_on = float((gyromag[i_on]-cand["base_gm"]) / (cand["std_gm"] if cand["std_gm"]>1e-6 else 1.0))

            if ev == "BEEP_SLOUCH":
                out_rows.append([t[i_on], "SLOUCH_START", mv, conf_on])
                # hold: detected or fallback aligned to onset spacing
                if cand["hold_idx"] is not None:
                    hold_time = t[cand["hold_idx"]]; hold_conf = float(cand["dtheta"])
                    holds_detected += 1
                else:
                    d_on, d_hold = FALLBACK_OFFSETS.get(mv, (0.45, 0.90))
                    hold_time = t[i_on] + (d_hold - d_on); hold_conf = 0.0
                    used_fallback = True
                out_rows.append([hold_time, "SLOUCHED_HOLD_START", mv, hold_conf])
                holds_total += 1

            else:  # RECOVER
                out_rows.append([t[i_on], "RECOVERY_START", mv, conf_on])
                if cand["hold_idx"] is not None:
                    hold_time = t[cand["hold_idx"]]; hold_conf = float(cand["dtheta"])  # unified semantics
                    holds_detected += 1
                else:
                    d_on, d_hold = FALLBACK_OFFSETS.get(mv, (0.45, 0.90))
                    hold_time = t[i_on] + (d_hold - d_on); hold_conf = 0.0
                    used_fallback = True
                out_rows.append([hold_time, "UPRIGHT_HOLD_START", mv, hold_conf])
                holds_total += 1

            if WRITE_DEBUG:
                dbg_rows.append([t_beep, ev, mv,
                                 (axis_chosen + ("+hold_fallback" if used_fallback else "")),
                                 t[i_on], hold_time, float(cand["onset_score"]), float(cand["dtheta"]), max_pitch, max_roll])

        else:
            # ---- Fallback: synthesize both events from beep ----
            d_on, d_hold = FALLBACK_OFFSETS.get(mv, (0.45, 0.90))
            if ev == "BEEP_SLOUCH":
                out_rows.append([t_beep + d_on,   "SLOUCH_START",         mv, 0.0])
                out_rows.append([t_beep + d_hold, "SLOUCHED_HOLD_START",  mv, 0.0])
            else:
                out_rows.append([t_beep + d_on,   "RECOVERY_START",       mv, 0.0])
                out_rows.append([t_beep + d_hold, "UPRIGHT_HOLD_START",   mv, 0.0])
            holds_total += 1
            if WRITE_DEBUG:
                dbg_rows.append([t_beep, ev, mv, "fallback",
                                 np.nan, np.nan, 0.0, 0.0, max_pitch, max_roll])

    # --- Ensure an initial upright exactly at t0 -----------------------------
    # Avoid duplicating if one already sits at t0.
    already_at_t0 = any((abs(r[0] - t0) <= 1e-9) and (r[1] == "UPRIGHT_HOLD_START") for r in out_rows)
    if not already_at_t0:
        out_rows.append([t0, "UPRIGHT_HOLD_START", mv_for_upright, 0.0])

    # write outputs (always overwrite)
    os.makedirs(Path(out_csv).parent, exist_ok=True)

    def _sort_key(r):
        # Sort by time; if same timestamp, make sure UPRIGHT_HOLD_START comes first
        return (r[0], 0 if r[1] == "UPRIGHT_HOLD_START" else 1)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_sec","event","value","confidence"])
        for r in sorted(out_rows, key=_sort_key):
            w.writerow([f"{r[0]:.3f}", r[1], f"type={r[2]}", f"{r[3]:.3f}"])

    if WRITE_DEBUG:
        with open(debug_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(
                ["t_beep","event","type","axis_chosen","onset_time","hold_time","onset_score","dtheta","max_pitch","max_roll"]
            )
            for row in dbg_rows:
                w.writerow(row)

    return {
        "events": len(out_rows),
        "holds_total": holds_total,
        "holds_detected": holds_detected,
        "beeps_in_range": beeps_in_range,
        "beeps_total": beeps_pos_total,
        "shift": shift
    }

# ---------- IMU / schedule discovery ----------
def looks_like_non_imu(name: str) -> bool:
    lname = name.lower()
    return (
        lname.startswith("beep_schedule_")
        or lname.startswith("events_inferred")
        or lname.startswith("events_debug")
        or lname.startswith("readme")
        or lname.endswith(".wav")
    )

def quick_is_valid_imu(path: Path) -> bool:
    if looks_like_non_imu(path.name): return False
    try:
        df = pd.read_csv(path, nrows=12)
        if df.empty: return False
        has_time = any(re.search(r"time|t_sec|timestamp", c, re.IGNORECASE) for c in df.columns)
        has_gyro = any(re.search(r"\bg(?:x|y|z)\b|rotationRate", c, re.IGNORECASE) for c in df.columns)
        has_grav = any(re.search(r"grav|gravity", c, re.IGNORECASE) for c in df.columns)
        has_quat = any(re.search(r"^q[wxyz]_?|quaternion", c, re.IGNORECASE) for c in df.columns)
        return bool(has_time and (has_gyro or has_grav or has_quat))
    except Exception:
        return False

def list_imus(root: Path, pattern: str="airpods_motion_*.csv") -> List[Path]:
    files = sorted(root.rglob(pattern))
    return [p for p in files if p.is_file() and quick_is_valid_imu(p)]

def _name_core(stem: str, prefix: str) -> str:
    return stem[len(prefix):] if stem.startswith(prefix) else stem

def _schedule_match_score(imu_path: Path, sch_path: Path) -> Tuple[int, float]:
    """
    Returns a tuple (name_score, -mtime_diff_seconds).
    Higher is better. name_score: 2 exact, 1 partial, 0 none.
    """
    imu_core = _name_core(imu_path.stem, "airpods_motion_")
    sch_core = _name_core(sch_path.stem, "beep_schedule_")
    name_score = 0
    if sch_core == imu_core:
        name_score = 2
    elif sch_core in imu_path.stem or imu_core in sch_path.stem:
        name_score = 1
    mtime_diff = abs(imu_path.stat().st_mtime - sch_path.stat().st_mtime)
    return (name_score, -mtime_diff)

def find_schedule_for_imu(imu_path: Path, all_schedules: List[Path]) -> Optional[Path]:
    # Prefer schedules in the same folder as the IMU first
    local = [p for p in imu_path.parent.glob("beep_schedule_*.csv") if p.is_file()]
    candidates = local if local else all_schedules
    if not candidates:
        return None
    scored = sorted(candidates, key=lambda p: _schedule_match_score(imu_path, p), reverse=True)
    return scored[0]

def write_upright_only_outputs(imu_csv: str, out_csv: str, debug_csv: str, verbose: bool):
    """When no schedule is found, still write a single UPRIGHT_HOLD_START at IMU t0."""
    try:
        df = pd.read_csv(imu_csv)
        t_sec, _ = get_time_seconds(df)
        valid = np.isfinite(t_sec)
        if valid.sum() == 0:
            raise RuntimeError("No valid IMU timestamps.")
        t0 = float(np.sort(t_sec[valid])[0])
    except Exception as e:
        # Fall back to t0=0 if something is deeply wrong
        if verbose:
            print(f"[warn] Could not read IMU to get t0 ({e}); writing t0=0.000", file=sys.stderr)
        t0 = 0.0

    os.makedirs(Path(out_csv).parent, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_sec","event","value","confidence"])
        w.writerow([f"{t0:.3f}", "UPRIGHT_HOLD_START", "type=normal_slouch", "0.000"])

    if WRITE_DEBUG:
        with open(debug_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["t_beep","event","type","axis_chosen","onset_time","hold_time","onset_score","dtheta","max_pitch","max_roll"]
            )

# ---------- Batch driver ----------
def main():
    ap = argparse.ArgumentParser(description="Batch infer events for every airpods_motion_*.csv under the given root (default: ./data).")
    ap.add_argument("--root", type=str, default="data", help="Root folder to search (default: ./data).")
    ap.add_argument("--glob", type=str, default="airpods_motion_*.csv", help="Glob for IMU files (default: airpods_motion_*.csv).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[error] Root folder does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    # Index all schedules under root once
    all_schedules = sorted(root.rglob("beep_schedule_*.csv"))

    imus = list_imus(root, args.glob)
    if not imus:
        print("[error] No IMU files found under", root, "matching pattern:", args.glob, file=sys.stderr)
        sys.exit(1)

    summary = []
    for imu_path in imus:
        imu_csv = str(imu_path)
        stem_core = _name_core(imu_path.stem, "airpods_motion_")
        out_csv   = str(imu_path.parent / f"events_inferred_{stem_core}.csv")
        debug_csv = str(imu_path.parent / f"events_debug_{stem_core}.csv")

        sch_path = find_schedule_for_imu(imu_path, all_schedules)
        if sch_path is None:
            print(f"[run] {imu_path.relative_to(root)}  SCHED=<none>  -> {Path(out_csv).name} (upright@t0 only)")
            write_upright_only_outputs(imu_csv, out_csv, debug_csv, args.verbose)
            # Record minimal summary
            summary.append((str(imu_path.relative_to(root)), 1, 0, 0, 0, 0, 0.0, True))
            continue

        schedule_csv = str(sch_path)
        print(f"[run] {imu_path.relative_to(root)}  SCHED={sch_path.relative_to(root)}  -> {Path(out_csv).name}")

        try:
            stats = infer_events_for_session(imu_csv, schedule_csv, out_csv, debug_csv, verbose=args.verbose)
            ok = True
        except Exception as e:
            ok = False
            print(f"[fail] {imu_path.name}: {e}", file=sys.stderr)
            # As a fallback on error, still produce an upright-only file so downstream doesn't break
            write_upright_only_outputs(imu_csv, out_csv, debug_csv, args.verbose)
            stats = {"events":1,"holds_total":0,"holds_detected":0,"beeps_in_range":0,"beeps_total":0,"shift":0.0}

        summary.append((
            str(imu_path.relative_to(root)),
            stats["events"], stats["holds_total"], stats["holds_detected"],
            stats["beeps_in_range"], stats["beeps_total"], stats["shift"], ok
        ))

        if ok:
            print(f"[ok ] {imu_path.name}: events={stats['events']} holds_total={stats['holds_total']} "
                  f"holds_detected={stats['holds_detected']} beeps_in_range={stats['beeps_in_range']}/{stats['beeps_total']} "
                  f"shift={stats['shift']:.3f}")

    if summary:
        print("\n=== Summary ===")
        for relpath, ev, htot, hdet, bir, bt, sh, ok in summary:
            mark = "OK" if ok else "ERR"
            print(f"{mark:>3}  {relpath}: events={ev:4d} holds_total={htot:3d} "
                  f"holds_detected={hdet:3d} beeps_in_range={bir:2d}/{bt:2d} shift={sh:7.3f}")
    else:
        print("[warn] No files processed.")

if __name__ == "__main__":
    main()