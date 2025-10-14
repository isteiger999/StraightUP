from constants import set_seeds, configure_tensorflow, configure_tensorflow_gpu

set_seeds()
configure_tensorflow_gpu()   # decide GPU/CPU first
configure_tensorflow()       # then threads/repro settings

from CNN import CNN_model, export_coreml
from events_and_windowing import X_and_y, count_labels, edit_csv, count_all_zero_windows, verify_lengths, find_combinations
from TCN import train_eval_tcn
import warnings
from urllib3.exceptions import NotOpenSSLWarning
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


def main():
    print("Physical GPUs:", tf.config.list_physical_devices('GPU'))
    participants = ['Ivan', 'Dario', 'David', 'Mohid',]         # 'Ivaan',
    combinations, mean, std = find_combinations(participants, fraction = 1)  # fraction 0.1 means cut off  
    n = len(combinations)
    print(combinations)

    for index, (_, list_comb) in enumerate(sorted(combinations.items(), key=lambda kv: int(kv[0]))):

        # 1. Create non-overlapping datasets
        X_train, y_train = X_and_y("train", list_comb)
        X_val, y_val = X_and_y("val", list_comb)
        X_test, y_test = X_and_y("test", list_comb)

        #count_all_zero_windows(X_train)
        #count_all_zero_windows(X_val)
        #count_all_zero_windows(X_test)
        
        # 2. Train & Evaluate CNN
        #cnn = CNN_model(X_train, y_train, X_val, y_val, verbose = 1)
        TCN_model = train_eval_tcn(X_train, y_train, X_val, y_val, verbose=0)

        # 3. Testing the CNN
        #scores = cnn.evaluate(X_test, y_test, return_dict=True, verbose = 1)
        scores = TCN_model.evaluate(X_test, y_test, return_dict=True, verbose = 1)
        print(f"TCN {index + 1}/{len(combinations)} trained")

        for k, v in scores.items():
            mean[k] += v / n           # E[X]
            std[k]  += (v * v) / n     # E[X^2]
        

    # calculate std only now, after mean has already been calculated
    for k in std.keys():
        var = std[k] - mean[k] * mean[k]
        std[k] = float(np.sqrt(var if var > 0 else 0.0))
    print("mean stats")
    for k, v in mean.items():
        print(f"  {k}: {v:.6f}")
    print("std stats")
    for k, v in std.items():
        print(f"  {k}: {v:.6f}")


if __name__ == '__main__':
    main()
