import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import numpy as np
import tensorflow as tf

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def configure_tensorflow():
    # For additional TensorFlow reproducibility
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Import this in EVERY script that needs reproducibility
set_seeds()
configure_tensorflow()

FEATURE_ORDER = [
    "quat_x","quat_y","quat_z","quat_w",
    "rot_x","rot_y","rot_z",
    "acc_x","acc_y","acc_z",
    "grav_x","grav_y","grav_z","pitch_rad"
]

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