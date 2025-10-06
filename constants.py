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
'''
python generate_beep_schedule_5m15s.py --participant Ivan0 \
  --minutes 5 --cycle-len 15 --jitter 1.0 --initial-delay 2.0 --make-wav \
  --movements "normal_slouch,fast_slouch,micro_slouch,lateral_left,lateral_right,neck_only,reach_left,reach_right"
'''

# To create detector
'''
python infer_events_from_beeps.py \
  --imu-csv /slouch_data/airpods_motion_1759773950.csv \
  --schedule-csv /beep_schedules_Ivan/beep_schedule_Ivan.csv \
  --out-csv /beep_schedules_Ivan/events_inferred_template_Ivan.csv
'''