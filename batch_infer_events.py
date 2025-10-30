#!/usr/bin/env python3
"""
Batch-generate events for every airpods_motion_*.csv under ./data (default)
using *manual offsets* from the beep schedule.

Key behavior:
- Normalizes the session ID by stripping any leading 'd' from the IMU stem.
- Writes EXACTLY two files per IMU (overwriting if present):
    events_inferred_<session>.csv
    events_debug_<session>.csv
- Removes stray variants with a leading 'd' (events_*_d<session>.csv, etc.).
- Manual-offset synthesis:
    BEEP_SLOUCH  -> onset at (t_beep - pre), hold at (t_beep + post)
    BEEP_RECOVER -> onset at (t_beep - pre), hold at (t_beep + post)
  Slouch vs. non-slouch sets define event names.

Usage:
    python batch_infer_events.py --verbose --pre 0.2 --post 1.0 --offset 2.0
"""

import argparse, csv, os, re, sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ====== Movement sets =======================================================
SLOUCH_MOVES   = {"normal_slouch", "fast_slouch", "micro_slouch", "neck_only"}
NOSLOUCH_MOVES = {"lateral_left", "lateral_right", "reach_left", "reach_right"}

# ---------- helpers: file/name/time utilities ----------
def find_col(df: "pd.DataFrame", patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in df.columns:
            if c == pat or c.lower() == pat.lower() or rx.fullmatch(c) or rx.search(c):
                return c
    return None

def _choose_time_scale_by_plausible_dt(dtn: np.ndarray) -> Tuple[float, str]:
    dtn = np.asarray(dtn, dtype=float)
    dtn = dtn[np.isfinite(dtn) & (dtn != 0)]
    if dtn.size == 0:
        return 1.0, "unknown"
    candidates = [("s", 1.0), ("ms->s", 1e-3), ("us->s", 1e-6), ("ns->s", 1e-9)]
    for label, scale in candidates:
        dt_med = float(np.median(np.abs(dtn) * scale))
        if 1e-4 <= dt_med <= 1.0:
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
        return bool(has_time)
    except Exception:
        return False

def list_imus(root: Path, pattern: str="airpods_motion_*.csv") -> List[Path]:
    files = sorted(root.rglob(pattern))
    return [p for p in files if p.is_file() and quick_is_valid_imu(p)]

def _name_core(stem: str, prefix: str) -> str:
    return stem[len(prefix):] if stem.startswith(prefix) else stem

def normalize_core(core: str) -> str:
    """Strip leading 'd' characters from a stem; e.g., 'd1760...' -> '1760...'."""
    return re.sub(r'^d+', '', core)

def _schedule_match_score(imu_path: Path, sch_path: Path) -> Tuple[int, int, float]:
    """
    Sort candidates by (norm_match, raw_match, -mtime_diff).
    """
    imu_core = _name_core(imu_path.stem, "airpods_motion_")
    sch_core = _name_core(sch_path.stem, "beep_schedule_")
    imu_norm = normalize_core(imu_core)
    sch_norm = normalize_core(sch_core)

    raw_match  = 2 if sch_core == imu_core else (1 if (sch_core in imu_path.stem or imu_core in sch_path.stem) else 0)
    norm_match = 1 if sch_norm == imu_norm else 0
    mtime_diff = abs(imu_path.stat().st_mtime - sch_path.stat().st_mtime)
    return (norm_match, raw_match, -mtime_diff)

def find_schedule_for_imu(imu_path: Path, all_schedules: List[Path]) -> Optional[Path]:
    local = [p for p in imu_path.parent.glob("beep_schedule_*.csv") if p.is_file()]
    candidates = local if local else all_schedules
    if not candidates:
        return None
    scored = sorted(candidates, key=lambda p: _schedule_match_score(imu_path, p), reverse=True)
    return scored[0]

def cleanup_variant_event_files(folder: Path, session: str) -> None:
    patterns = [
        f"events_inferred_d{session}.csv",
        f"events_debug_d{session}.csv",
    ]
    for prefix in ["events_inferred_", "events_debug_"]:
        for k in range(2, 6):  # dd..dddd
            patterns.append(f"{prefix}{'d'*k}{session}.csv")
    for pat in patterns:
        p = folder / pat
        if p.exists():
            try:
                p.unlink()
                print(f"[clean] removed stray {p.name}")
            except Exception as e:
                print(f"[warn] could not remove {p.name}: {e}", file=sys.stderr)

# ---------- alignment helpers ----------
def align_schedule_times(sch: pd.DataFrame, t0: float, t1: float, verbose: bool=False) -> Tuple[pd.DataFrame, float, str]:
    """
    Align schedule times to IMU axis.
    Modes:
      - 'relative+add_t0'   : times look small (seconds), IMU looks epoch-like -> sch.t_sec += t0
      - 'absolute-overlap'  : schedule already overlaps IMU range -> leave as-is
      - 'fallback-first-to-t0': rare; align first schedule time to t0 (legacy)
    Returns: (aligned_df, shift_used, mode)
    """
    sch = sch.copy()
    s = pd.to_numeric(sch["t_sec"], errors="coerce")
    s0, s1 = float(s.min()), float(s.max())

    # IMU looks like epoch?
    imu_epochish = (t0 > 1e6)
    # Schedule looks like "small seconds"?
    sch_small = (s1 - s0) < 1e6 and s1 < 1e6

    # Case 1: already overlaps with IMU time -> keep
    if (s0 <= t1 and s1 >= t0):
        mode, shift = "absolute-overlap", 0.0
        sch["t_sec"] = s

    # Case 2: likely relative schedule -> add t0
    elif imu_epochish and sch_small:
        mode, shift = "relative+add_t0", float(t0)
        sch["t_sec"] = s + t0

    # Case 3: fallback: align first schedule time to t0
    else:
        shift = t0 - s0
        mode  = "fallback-first-to-t0"
        sch["t_sec"] = s + shift

    if verbose:
        print(f"[align] mode={mode} s0={s0:.3f} s1={s1:.3f} t0={t0:.3f} t1={t1:.3f} shift={shift:.3f}")
    return sch, float(shift), mode

# ---------- main synthesis (manual offsets) ----------
def movement_type_from_value(val: str) -> str:
    val = str(val) if pd.notna(val) else ""
    return val.split("type=", 1)[1] if "type=" in val else val

def write_upright_only_outputs(imu_csv: str, out_csv: str, debug_csv: str, verbose: bool):
    try:
        df = pd.read_csv(imu_csv)
        t_sec, _ = get_time_seconds(df)
        valid = np.isfinite(t_sec)
        if valid.sum() == 0:
            raise RuntimeError("No valid IMU timestamps.")
    except Exception as e:
        if verbose:
            print(f"[warn] Could not read IMU to get t0 ({e}); writing t0=0.000", file=sys.stderr)
        t0 = 0.0
    else:
        order = np.argsort(t_sec, kind="mergesort")
        t_sec = t_sec[order]
        dts = np.diff(t_sec)
        keep = np.hstack(([True], dts > 0))
        t0 = float(t_sec[keep][0])

    os.makedirs(Path(out_csv).parent, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_sec","event","value"])
        w.writerow([f"{t0:.3f}", "UPRIGHT_HOLD_START", "type=normal_slouch"])
    with open(debug_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(
            ["t_beep","event","type","onset_event","onset_time","hold_event","hold_time"]
        )
    if verbose:
        print("[warn] No schedule; wrote initial UPRIGHT_HOLD_START at t0.")

def synthesize_events_from_schedule(imu_csv: str, schedule_csv: str, out_csv: str,
                                    debug_csv: str, pre_s: float, post_s: float,
                                    offset_s: float = 0.0, verbose: bool=False) -> dict:
    # --- IMU time axis ---
    imu = pd.read_csv(imu_csv)
    t_sec, t_units = get_time_seconds(imu)
    valid = np.isfinite(t_sec)
    if valid.sum() < 1:
        raise RuntimeError("Too few valid IMU time samples.")
    order = np.argsort(t_sec, kind="mergesort")
    if not np.all(order == np.arange(len(order))):
        imu = imu.iloc[order].reset_index(drop=True); t_sec = t_sec[order]
    dts = np.diff(t_sec)
    keep = np.hstack(([True], dts > 0))
    if not np.all(keep):
        imu = imu.loc[keep].reset_index(drop=True)
        t_sec = t_sec[keep]
    t0 = float(t_sec[0]); t1 = float(t_sec[-1])

    # --- Schedule & alignment ---
    sch = pd.read_csv(schedule_csv)
    if "t_sec" not in sch.columns:
        raise RuntimeError("Schedule missing t_sec column.")

    # NEW: apply user-provided offset (subtract seconds from every schedule row)
    if offset_s != 0.0:
        sch = sch.copy()
        sch["t_sec"] = pd.to_numeric(sch["t_sec"], errors="coerce") - float(offset_s)
        if verbose:
            print(f"[align] schedule: subtracted offset {offset_s:.3f}s from t_sec")

    sch_aligned, shift, mode = align_schedule_times(sch, t0, t1, verbose=verbose)

    if verbose:
        ss0, ss1 = float(sch_aligned["t_sec"].min()), float(sch_aligned["t_sec"].max())
        print(f"[debug] IMU t range: {t0:.3f}..{t1:.3f} (units={t_units})")
        print(f"[debug] SCH t range: {ss0:.3f}..{ss1:.3f}")

    beeps = sch_aligned[sch_aligned["event"].isin(["BEEP_SLOUCH","BEEP_RECOVER"])].copy()
    if beeps.empty:
        write_upright_only_outputs(imu_csv, out_csv, debug_csv, verbose)
        return {"events": 1, "beeps": 0, "shift": shift}

    beeps = beeps.sort_values("t_sec").reset_index(drop=True)
    beeps["mv"] = beeps["value"].apply(movement_type_from_value)

    first_slouch = beeps[(beeps["event"]=="BEEP_SLOUCH") & (beeps["mv"].isin(SLOUCH_MOVES))]
    mv_for_upright = str(first_slouch["mv"].iloc[0]) if not first_slouch.empty else str(beeps["mv"].iloc[0])

    # --- Generate events using manual offsets ---
    out_rows = []      # [t, event, "type=<mv>"]
    debug_rows = []    # [t_beep, event, type, onset_event, onset_time, hold_event, hold_time]

    def start_event_for(mv: str) -> str:
        return "SLOUCH_START" if mv in SLOUCH_MOVES else "NOSLOUCH_START"
    def hold_event_for(mv: str) -> str:
        return "SLOUCHED_HOLD_START" if mv in SLOUCH_MOVES else "NOSLOUCHED_HOLD_START"

    for _, r in beeps.iterrows():
        ev = str(r["event"]); mv = str(r["mv"]); t_beep = float(r["t_sec"])
        if ev == "BEEP_SLOUCH":
            onset_event = start_event_for(mv)
            hold_event  = hold_event_for(mv)
        else:  # BEEP_RECOVER
            onset_event = "RECOVERY_START"
            hold_event  = "UPRIGHT_HOLD_START"

        onset_time = t_beep - pre_s
        hold_time  = t_beep + post_s

        out_rows.append([onset_time, onset_event, f"type={mv}"])
        out_rows.append([hold_time,  hold_event,  f"type={mv}"])
        debug_rows.append([t_beep, ev, mv, onset_event, onset_time, hold_event, hold_time])

    # Ensure an initial upright at t0
    already_at_t0 = any((abs(r[0] - t0) <= 1e-9) and (r[1] == "UPRIGHT_HOLD_START") for r in out_rows)
    if not already_at_t0:
        out_rows.append([t0, "UPRIGHT_HOLD_START", f"type={mv_for_upright}"])

    # --- Write outputs ---
    os.makedirs(Path(out_csv).parent, exist_ok=True)

    def _sort_key(r):
        return (r[0], 0 if r[1] == "UPRIGHT_HOLD_START" else 1)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_sec","event","value"])
        for r in sorted(out_rows, key=_sort_key):
            w.writerow([f"{float(r[0]):.3f}", r[1], r[2]])

    with open(debug_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_beep","event","type","onset_event","onset_time","hold_event","hold_time"])
        for row in debug_rows:
            w.writerow([f"{row[0]:.6f}", row[1], row[2], row[3], f"{float(row[4]):.6f}", row[5], f"{float(row[6]):.6f}"])

    if verbose:
        print(f"[ok ] {Path(imu_csv).name}: events={len(out_rows)} beeps={len(beeps)} shift_used={shift:.3f} mode={mode}")

    return {"events": len(out_rows), "beeps": int(len(beeps)), "shift": shift}

# ---------- Batch driver ----------
def main():
    ap = argparse.ArgumentParser(description="Batch synthesize events from beep_schedules using manual offsets.")
    ap.add_argument("--root", type=str, default="data", help="Root folder to search (default: ./data).")
    ap.add_argument("--glob", type=str, default="airpods_motion_*.csv", help="Glob for IMU files (default: airpods_motion_*.csv).")
    ap.add_argument("--transition-pre", "--pre", dest="pre", type=float, default=0.2,
                    help="Seconds before a beep to place *onset* events (default: 0.2).")
    ap.add_argument("--transition-post", "--post", dest="post", type=float, default=1.0,
                    help="Seconds after a beep to place *hold* events (default: 1.0).")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="Seconds to SUBTRACT from every schedule t_sec before alignment (default: 0.0).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[error] Root folder does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    all_schedules = sorted(root.rglob("beep_schedule_*.csv"))
    imus = list_imus(root, args.glob)
    if not imus:
        print("[error] No IMU files found under", root, "matching pattern:", args.glob, file=sys.stderr)
        sys.exit(1)

    summary = []
    for imu_path in imus:
        imu_csv = str(imu_path)

        raw_core = _name_core(imu_path.stem, "airpods_motion_")
        session  = normalize_core(raw_core)

        out_csv   = str(imu_path.parent / f"events_inferred_{session}.csv")
        debug_csv = str(imu_path.parent / f"events_debug_{session}.csv")

        cleanup_variant_event_files(imu_path.parent, session)

        sch_path = find_schedule_for_imu(imu_path, all_schedules)
        if sch_path is None:
            print(f"[run] {imu_path.relative_to(root)}  SCHED=<none>  -> events_*_{session}.csv (upright@t0 only)")
            write_upright_only_outputs(imu_csv, out_csv, debug_csv, args.verbose)
            summary.append((str(imu_path.relative_to(root)), 1, 0.0, True))
            continue

        schedule_csv = str(sch_path)
        print(f"[run] {imu_path.relative_to(root)}  SCHED={sch_path.relative_to(root)}  -> events_*_{session}.csv")

        try:
            stats = synthesize_events_from_schedule(
                imu_csv, schedule_csv, out_csv, debug_csv,
                pre_s=args.pre, post_s=args.post, offset_s=args.offset, verbose=args.verbose
            )
            ok = True
        except Exception as e:
            ok = False
            print(f"[fail] {imu_path.name}: {e}", file=sys.stderr)
            write_upright_only_outputs(imu_csv, out_csv, debug_csv, args.verbose)
            stats = {"events":1,"beeps":0,"shift":0.0}

        cleanup_variant_event_files(imu_path.parent, session)

        summary.append((str(imu_path.relative_to(root)), stats["events"], stats["shift"], ok))

    if summary:
        print("\n=== Summary ===")
        for relpath, ev, sh, ok in summary:
            mark = "OK" if ok else "ERR"
            print(f"{mark:>3}  {relpath}: events={ev:4d} shift_used={sh:7.3f}")
    else:
        print("[warn] No files processed.")

if __name__ == "__main__":
    main()