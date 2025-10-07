import pandas as pd
import numpy as np

df_event = pd.read_csv(r"beep_schedules_Ivan0/events_inferred_template_Ivan0.csv")
df_imu = pd.read_csv(r"beep_schedules_Ivan0/airpods_motion_1759863949.csv")

# assume first column in IMU is time (t_sec). If not, rename accordingly.
t0 = float(df_imu.iloc[0, 0])
t1 = float(df_imu.iloc[-1, 0])

len_window = 1.5
stride = 0.5
m = 0.2  # margin around starts treated as transition

len_rec = t1 - t0  # duration in seconds of recording
windows_per_rec = int(np.floor((len_rec - len_window) / stride) + 1)
windows_per_rec = max(windows_per_rec, 0)                      # should be 718 for 6 min

labels_array = np.zeros((windows_per_rec, 1), dtype=int)  # 0=upright,1=transition,2=slouched

# sort events and ensure we have an initial upright at t0
events = df_event[['t_sec','event']].sort_values('t_sec').to_numpy().tolist()
if not events or events[0][1] != 'UPRIGHT_HOLD_START':
    events = [[t0, 'UPRIGHT_HOLD_START']] + events
times = np.array([float(t) for t, _ in events])
names = [e for _, e in events]

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

# label comes from the state at the end of the window (causal)
for i in range(windows_per_rec):
    current_time = t0 + len_window + i * stride
    labels_array[i, 0] = label_at_time(current_time)

## --------------------------------------------------------------------------------------------------
# Now create actual windows

rec_tot = 1             # und windows_per_rec = 718
total_amount_windows = rec_tot * windows_per_rec
X_tot = np.zeros((total_amount_windows, 75, 14)) 

#for i in range(total_amount_windows):

#    X_tot[i, :, :] = 



