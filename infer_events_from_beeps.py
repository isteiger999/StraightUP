#!/usr/bin/env python3
import argparse, math, csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

try:
    import numpy as np
    import pandas as pd
except Exception as e:
    print("This script requires numpy and pandas installed.", file=sys.stderr)
    raise

RAD2DEG = 180.0 / math.pi

def euler_from_quat(w, x, y, z):
    # yaw-pitch-roll (Z-Y-X). We'll use pitch (about Y) and roll (about X).
    # Source: standard conversion
    t0 = +2.0*(w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + y*y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0*(w*y - z*x)
    t2 = +1.0 if t2>+1.0 else t2
    t2 = -1.0 if t2<-1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0*(w*z + x*y)
    t4 = +1.0 - 2.0*(y*y + z*z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z*RAD2DEG, pitch_y*RAD2DEG, roll_x*RAD2DEG

def compute_pitch_roll(df: "pd.DataFrame") -> Tuple[np.ndarray, np.ndarray]:
    # Prefer gravity if present
    cols = df.columns
    has_grav = all(c in cols for c in ["grav_x","grav_y","grav_z"])
    has_quat = all(c in cols for c in ["qw","qx","qy","qz"]) or all(c in cols for c in ["q_w","q_x","q_y","q_z"])
    if "pitch" in cols and "roll" in cols:
        return df["pitch"].to_numpy(float), df["roll"].to_numpy(float)
    if has_grav:
        gx = df["grav_x"].to_numpy(float)
        gy = df["grav_y"].to_numpy(float)
        gz = df["grav_z"].to_numpy(float)
        pitch = np.arctan2(-gx, np.sqrt(gy*gy + gz*gz)) * RAD2DEG
        roll  = np.arctan2(gy, gz) * RAD2DEG
        return pitch, roll
    if has_quat:
        qw = df["qw"].to_numpy(float) if "qw" in cols else df["q_w"].to_numpy(float)
        qx = df["qx"].to_numpy(float) if "qx" in cols else df["q_x"].to_numpy(float)
        qy = df["qy"].to_numpy(float) if "qy" in cols else df["q_y"].to_numpy(float)
        qz = df["qz"].to_numpy(float) if "qz" in cols else df["q_z"].to_numpy(float)
        yaws, pitchs, rolls = zip(*(euler_from_quat(w,x,y,z) for w,x,y,z in zip(qw,qx,qy,qz)))
        return np.array(pitchs, dtype=float), np.array(rolls, dtype=float)
    raise ValueError("No gravity, quaternion, or pitch/roll columns found. Provide grav_* or q* or pitch/roll.")

def lowpass_ma(x: np.ndarray, k: int) -> np.ndarray:
    if k<=1: return x
    k = int(k)
    pad = np.pad(x, (k-1, 0), mode="edge")
    cumsum = np.cumsum(pad)
    out = (cumsum[k:] - cumsum[:-k]) / k
    return out

def zscore(x: np.ndarray):
    m = np.nanmean(x); s = np.nanstd(x) + 1e-9
    return (x - m)/s, m, s

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

def parse_type(val: str) -> str:
    # value looks like "type=normal_slouch"
    if isinstance(val, str) and "type=" in val:
        return val.split("type=",1)[1].strip()
    return str(val)

def infer_events(imu_csv: str, schedule_csv: str, out_csv: str,
                 fs_hint: float = 50.0,
                 search_window: float = 3.0,
                 baseline_pre: float = 2.0,
                 settle_ms: int = 250):
    df = pd.read_csv(imu_csv)
    if "t_sec" not in df.columns:
        # try to find a time column
        if "timestamp" in df.columns:
            df["t_sec"] = df["timestamp"].astype(float)
        else:
            raise ValueError("IMU CSV must contain 't_sec' or 'timestamp' column (seconds).")

    t = df["t_sec"].to_numpy(float)
    if len(t) < 10:
        raise ValueError("IMU CSV too short.")
    # estimate fs
    dt = np.median(np.diff(t))
    fs = 1.0/dt if dt>0 else fs_hint

    # gyro magnitude if available
    has_g = all(c in df.columns for c in ["gx","gy","gz"])
    if has_g:
        gyromag = np.sqrt(df["gx"]**2 + df["gy"]**2 + df["gz"]**2).to_numpy(float)
    else:
        gyromag = np.zeros_like(t)  # fallback

    pitch, roll = compute_pitch_roll(df)
    # smooth pitch/roll lightly
    k = max(1, int(round(0.14 * fs)))  # ~140ms MA
    pitch_s = lowpass_ma(pitch, k)
    roll_s  = lowpass_ma(roll, k)
    # align sizes if MA shortened it
    min_len = min(len(t), len(pitch_s), len(roll_s), len(gyromag))
    t, pitch_s, roll_s, gyromag = t[:min_len], pitch_s[:min_len], roll_s[:min_len], gyromag[:min_len]

    # derivatives
    dpitch = np.gradient(pitch_s, t)
    droll  = np.gradient(roll_s, t)

    schedule = pd.read_csv(schedule_csv)
    out_rows = []

    # helper to convert time to index
    def ti(time_s: float) -> int:
        return int(np.searchsorted(t, time_s, side="left"))

    for idx, row in schedule.iterrows():
        ev = row["event"]
        val = row.get("value", "")
        mv = parse_type(val)

        if ev not in ("BEEP_SLOUCH","BEEP_RECOVER"):
            continue

        t_beep = float(row["t_sec"])

        # types considered "hard negatives"
        is_negative = mv in ("neck_only","reach_right","reach_left","reach","twist")

        # choose angle channel
        use_roll = mv in ("lateral_left","lateral_right")
        angle = roll_s if use_roll else pitch_s
        dangle = droll if use_roll else dpitch

        # windows
        i0 = max(0, ti(t_beep - baseline_pre))
        i_beep = ti(t_beep)
        i1 = min(len(t)-1, ti(t_beep + search_window))

        # adaptive baselines from pre-beep window
        base_ang = float(np.nanmean(angle[i0:i_beep])) if i_beep>i0 else float(angle[max(0,i_beep-5):i_beep].mean())
        base_gm  = float(np.nanmean(gyromag[i0:i_beep])) if i_beep>i0 else float(gyromag[max(0,i_beep-5):i_beep].mean())
        std_gm   = float(np.nanstd(gyromag[i0:i_beep])) + 1e-6

        # thresholds by type
        if use_roll:
            dtheta_in = 6.0  # degrees
            theta_hold = 8.0
        elif mv == "micro_slouch":
            dtheta_in = 3.5
            theta_hold = 5.0
        else:
            dtheta_in = 7.0
            theta_hold = 10.0

        # detect onset
        if ev == "BEEP_SLOUCH" and not is_negative:
            # motion onset + angle change positive
            run_len = max(6, int(round(0.12*fs)))
            th = base_gm + 3.0*std_gm
            onset_rel = first_run_above(gyromag[i_beep:i1], th, run_len)
            if onset_rel is not None:
                i_on = i_beep + onset_rel
                # confirm angle rise within 0.5 s
                i_confirm = min(len(t)-1, i_on + int(round(0.5*fs)))
                dtheta = float(angle[i_confirm] - angle[i_on])
                if dtheta >= dtheta_in * 0.8:  # a little slack
                    out_rows.append([t[i_on], "SLOUCH_START", mv, float((gyromag[i_on]-base_gm)/std_gm)])
                    # find settled hold
                    settle_len = max( int(round(0.25*fs)), 3)
                    th_hold = base_gm + 1.0*std_gm
                    # scan forward for sustained calm + angle above threshold
                    for j in range(i_on, i1):
                        j2 = min(j+settle_len, len(t)-1)
                        calm = np.all(gyromag[j:j2] < th_hold)
                        high = (angle[j2] - base_ang) >= theta_hold
                        if calm and high:
                            out_rows.append([t[j2], "SLOUCHED_HOLD_START", mv, float((angle[j2]-base_ang))])
                            break
        elif ev == "BEEP_RECOVER" and not is_negative:
            run_len = max(6, int(round(0.12*fs)))
            th = base_gm + 3.0*std_gm
            onset_rel = first_run_above(gyromag[i_beep:i1], th, run_len)
            if onset_rel is not None:
                i_on = i_beep + onset_rel
                # confirm angle drop within 0.5 s
                i_confirm = min(len(t)-1, i_on + int(round(0.5*fs)))
                dtheta = float(angle[i_on] - angle[i_confirm])
                if dtheta >= dtheta_in * 0.6:
                    out_rows.append([t[i_on], "RECOVERY_START", mv, float((gyromag[i_on]-base_gm)/std_gm)])
                    # upright settle
                    settle_len = max( int(round(0.30*fs)), 3)
                    th_hold = base_gm + 1.0*std_gm
                    for j in range(i_on, i1):
                        j2 = min(j+settle_len, len(t)-1)
                        calm = np.all(gyromag[j:j2] < th_hold)
                        near = abs(angle[j2] - base_ang) <= 3.0
                        if calm and near:
                            out_rows.append([t[j2], "UPRIGHT_HOLD_START", mv, float(3.0-abs(angle[j2]-base_ang))])
                            break
        else:
            # negatives: skip creating slouch/hold; remain upright
            pass

    # Write out
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_sec","event","value","confidence"])
        for r in sorted(out_rows, key=lambda x: x[0]):
            w.writerow([f"{r[0]:.3f}", r[1], f"type={r[2]}", f"{r[3]:.3f}"])

    print(f"Wrote inferred events: {out_path}")
    return str(out_path)

def main():
    ap = argparse.ArgumentParser(description="Infer posture events from IMU + beep schedule")
    ap.add_argument("--imu-csv", required=True, help="CSV with at least t_sec and gyro (gx,gy,gz) + gravity or quaternion")
    ap.add_argument("--schedule-csv", required=True, help="CSV from generate_beep_schedule")
    ap.add_argument("--out-csv", required=True, help="Output CSV for inferred events")
    ap.add_argument("--fs-hint", type=float, default=50.0)
    args = ap.parse_args()
    infer_events(args.imu_csv, args.schedule_csv, args.out_csv, fs_hint=args.fs_hint)

if __name__ == "__main__":
    main()
