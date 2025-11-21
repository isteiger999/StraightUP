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


''' (jitter is probabilistic)
python generate_beep_schedule_5m15s.py --participant ZDavC2 \
  --minutes 5 --cycle-len 15 --jitter 1.0 --initial-delay 2.0 --make-wav \
  --movements "normal_slouch,fast_slouch,micro_slouch,lateral_left,lateral_right,neck_only,reach_left,reach_right" \
  --complete-plan
'''

'''
Windows laptop:
python generate_beep_schedule_5m15s.py `
  --participant Abi1 `
  --minutes 5 `
  --cycle-len 15 `
  --jitter 1.0 `
  --initial-delay 2.0 `
  --make-wav `
  --movements "normal_slouch,fast_slouch,micro_slouch,lateral_left,lateral_right,neck_only,reach_left,reach_right" `
  --complete-plan
'''

'''
# Everyone
python batch_infer_events.py --verbose
python batch_infer_events.py --verbose --pre 0.05 --post 0.9
python batch_infer_events.py --verbose --pre 0.05 --post 0.9 --offset 2.0
'''
'''
# Single Person
python batch_infer_events.py \
  --root data/beep_schedules_MA0 \
  --glob "airpods_motion_*.csv" \
  --verbose --pre 0.05 --post 0.90 --offset 2.0

#-------- NEW VERSION (15 Nov)(no 2 sec subtraction)--------#
python batch_infer_events.py \
  --root data/beep_schedules_ZMohA0 \
  --glob "airpods_motion_*.csv" \
  --verbose --pre 0.05 --post 0.90 --offset 1.0
'''

'''
Vom Laptop/Mac git pullen (nicht bloss git pull):
git fetch origin
git reset --hard "HEAD@{upstream}"   
git clean -fd
'''