import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import numpy as np
import tensorflow as tf
import platform
import pandas as pd
import matplotlib.pyplot as plt
#from events_and_windowing import X_and_y

def configure_tensorflow_gpu(prefer_gpu: bool = True):
    """
    - On macOS with tensorflow-metal installed: uses the Apple GPU.
    - On other systems with CUDA GPUs: enables memory growth.
    - Otherwise: disables GPU and runs on CPU.
    You can override with env var PREFER_GPU=0 to force CPU.
    """
    prefer_gpu = prefer_gpu and os.getenv("PREFER_GPU", "1") == "1"

    try:
        gpus = tf.config.list_physical_devices("GPU")
        if prefer_gpu and gpus:
            # CUDA-specific nicety (not needed on Apple Metal)
            if platform.system() != "Darwin":
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception:
                        pass
            print(f"[TF] Using GPU ({'Metal' if platform.system()=='Darwin' else 'CUDA'}), {len(gpus)} device(s) found.")
        else:
            # Explicitly hide GPUs (or none present)
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass
            print("[TF] Using CPU (GPU unavailable or disabled).")
    except Exception as e:
        # Any failure → fall back to CPU
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        print(f"[TF] Falling back to CPU due to config error: {e}")

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def configure_tensorflow():
    # For additional TensorFlow reproducibility
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

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
def main():
    #X_train, y_train = X_and_y("train", ['Ivan', 'Dario', 'David', 'Claire', 'Mohid'])


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
# --------------------------------------------------------------



## To generate 3 files: 
#               1. schedule.csv, beep sound (mav file), 
#               2. beep sound (mav file), 
#               3. empty csv (to be filled by detector)
''' (jitter is probabilistic)
python generate_beep_schedule_5m15s.py --participant Test0 \
  --minutes 5 --cycle-len 15 --jitter 1.0 --initial-delay 2.0 --make-wav \
  --movements "normal_slouch,fast_slouch,micro_slouch,lateral_left,lateral_right,neck_only,reach_left,reach_right" \
  --complete-plan
'''

'''
Windows laptop:
python generate_beep_schedule_5m15s.py `
  --participant Dario3 `
  --minutes 5 `
  --cycle-len 15 `
  --jitter 1.0 `
  --initial-delay 2.0 `
  --make-wav `
  --movements "normal_slouch,fast_slouch,micro_slouch,lateral_left,lateral_right,neck_only,reach_left,reach_right" `
  --complete-plan
'''

'''
# individually
(NICHT NUTZEN, BESSER IMMER ALLE GLEICH) python batch_infer_events.py --participants David0 --verbose
'''

'''
# Everyone
python batch_infer_events.py --verbose
'''


'''
Vom Laptop/Mac git pullen (nicht bloss git pull):
git fetch origin
git reset --hard "HEAD@{upstream}"   
git clean -fd
'''