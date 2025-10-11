#!/usr/bin/env python3
import argparse, csv, math, random
from pathlib import Path

# Optional WAV export (requires numpy)
try:
    import numpy as np, wave  # type: ignore
except Exception:
    np = None
    wave = None

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def build_movement_base(movements_csv):
    if movements_csv:
        base = [m.strip() for m in movements_csv.split(",") if m.strip()]
        if not base:
            base = None
        else:
            return base
    # Default plan ends with reach_right on purpose
    return [
        "normal_slouch","fast_slouch","micro_slouch",
        "lateral_left","lateral_right",
        "neck_only",
        "reach_left","reach_right"
    ]

def build_movement_plan(total_cycles, movements_csv, complete_plan):
    base = build_movement_base(movements_csv)
    L = len(base)
    if complete_plan:
        # Extend to the smallest multiple of L that is >= total_cycles
        k = (total_cycles + L - 1) // L  # ceil
        total_cycles = max(L, k * L)
    plan = [base[i % L] for i in range(total_cycles)]
    return plan, total_cycles, base

def generate_schedule(participant, minutes=5, cycle_len=15.0, jitter=1.0,
                      slouch_offset=None, recover_offset=None, seed=42,
                      movements_csv=None, make_wav=False, out_dir=None,
                      include_countin=True, initial_delay=2.0, complete_plan=False):
    random.seed(seed)
    total_seconds = int(minutes * 60)
    cycles_by_time = int(math.floor(total_seconds / cycle_len))
    if cycles_by_time < 1:
        raise ValueError("minutes too small for chosen cycle length")

    if slouch_offset  is None: slouch_offset  = 0.30 * cycle_len
    if recover_offset is None: recover_offset = 0.60 * cycle_len

    # Build plan and possibly extend cycles to complete the full movement list
    movement_plan, total_cycles, base = build_movement_plan(cycles_by_time, movements_csv, complete_plan)

    # Names ONLY include participant
    tag     = f"{participant}"
    outdir  = Path(out_dir) if out_dir else Path.cwd() / f"beep_schedules_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    schedule_csv = outdir / f"beep_schedule_{tag}.csv"
    inferred_csv = None  # intentionally not created per request
    wav_path     = outdir / f"beeps_{tag}.wav"

    rows = []
    for c in range(total_cycles):
        base_t = initial_delay + c * cycle_len  # shift whole cycle by initial delay
        j      = random.uniform(-jitter, jitter)

        t_go = base_t + slouch_offset + j
        # Leave room for 3-2-1 count-in if included
        min_go = base_t + (3.2 if include_countin else 0.2)
        max_go = base_t + (cycle_len - 6.0)
        t_go = clamp(t_go, min_go, max_go)

        t_rec = base_t + recover_offset + j
        t_rec = clamp(t_rec, t_go + 3.0, base_t + (cycle_len - 1.0))

        movement = movement_plan[c]

        if include_countin:
            rows.append((t_go - 3.0, "BEEP_COUNTIN", "3"))
            rows.append((t_go - 2.0, "BEEP_COUNTIN", "2"))
            rows.append((t_go - 1.0, "BEEP_COUNTIN", "1"))
        rows.append((t_go,  "BEEP_SLOUCH",  f"type={movement}"))
        rows.append((t_rec, "BEEP_RECOVER", f"type={movement}"))

    with open(schedule_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_sec","event","value"])
        for t, ev, val in rows: w.writerow([f"{t:.3f}", ev, val])

    # Intentionally do not create events_inferred_template CSV.

    if make_wav and (np is not None) and (wave is not None):
        duration   = initial_delay + total_cycles * cycle_len  # include initial silence
        sr = 44100
        n_samples  = int(round(duration * sr))
        audio      = np.zeros(n_samples, dtype=np.float32)

        def add_beep(t_center, dur_ms, freq_hz, amp=0.3):
            dur_s = dur_ms / 1000.0
            n = int(round(dur_s * sr))
            if n <= 2: return
            t0 = int(round(t_center * sr))
            start = max(0, t0 - n // 2); end = min(n_samples, start + n)
            start = end - n
            if start < 0: return
            t = np.arange(n) / sr
            s = np.sin(2*np.pi*freq_hz*t).astype(np.float32)
            win = 0.5 - 0.5*np.cos(2*np.pi*np.arange(n)/(n-1))
            s *= win.astype(np.float32); s *= amp
            audio[start:end] += s

        for t, ev, _ in rows:
            if ev == "BEEP_COUNTIN": add_beep(t, 150, 1000)
            elif ev == "BEEP_SLOUCH": add_beep(t, 320, 1200)
            elif ev == "BEEP_RECOVER": add_beep(t, 320, 900)

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0.99: audio /= (peak + 1e-6)

        wav_i16 = (audio * 32767.0).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(wav_i16.tobytes())

    # Also write a small README with derived info
    info_path = outdir / f"README_{tag}.txt"
    with open(info_path, "w") as f:
        f.write(f"Participant: {participant}\n")
        f.write(f"Cycle length: {cycle_len}s\n")
        f.write(f"Minutes requested: {minutes} -> cycles by time: {cycles_by_time}\n")
        f.write(f"Total cycles generated: {total_cycles} (complete_plan={complete_plan})\n")
        f.write(f"Initial delay: {initial_delay}s\n")
        f.write(f"Jitter: ±{jitter}s per cycle\n")
        f.write(f"Movements order (one pass): {', '.join(build_movement_base(movements_csv))}\n")
        f.write(f"Final movement in last cycle: {movement_plan[-1]}\n")
        f.write(f"Approx total duration: {initial_delay + total_cycles*cycle_len:.1f}s\n")

    # Keep return signature; inferred_csv is None since it's not created.
    return str(schedule_csv), None, (str(wav_path) if make_wav else None)

def main():
    p = argparse.ArgumentParser(description="Generate 5-min beep schedule (15 s cycles) with initial silence; names use only participant ID")
    p.add_argument("--participant", required=True)
    p.add_argument("--minutes", type=int, default=5)
    p.add_argument("--cycle-len", type=float, default=15.0, help="Seconds per cycle")
    p.add_argument("--jitter", type=float, default=1.0, help="± seconds applied once per cycle")
    p.add_argument("--slouch-offset", type=float, default=None)
    p.add_argument("--recover-offset", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--movements", type=str, default=None, help="Comma list rotated across cycles (last item is target end)")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--make-wav", action="store_true")
    p.add_argument("--no-countin", action="store_true")
    p.add_argument("--initial-delay", type=float, default=2.0, help="Seconds of initial silence (default 2.0)")
    p.add_argument("--complete-plan", action="store_true", help="Extend cycles so session ends on the LAST movement (completes full rotations)")
    args = p.parse_args()

    schedule_csv, inferred_csv, wav_path = generate_schedule(
        participant=args.participant,
        minutes=args.minutes,
        cycle_len=args.cycle_len,
        jitter=args.jitter,
        slouch_offset=args.slouch_offset,
        recover_offset=args.recover_offset,
        seed=args.seed,
        movements_csv=args.movements,
        make_wav=args.make_wav,
        out_dir=args.out_dir,
        include_countin=not args.no_countin,
        initial_delay=args.initial_delay,
        complete_plan=args.complete_plan,
    )
    print("Wrote:", schedule_csv)
    # No inferred template file created, so no print.
    if wav_path: print("Wrote:", wav_path)

if __name__ == "__main__":
    main()