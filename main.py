from constants import set_seeds, configure_tensorflow
set_seeds()
configure_tensorflow()
from CNN import CNN_model, export_coreml
from events_and_windowing import X_and_y, count_labels, edit_csv, count_all_zero_windows, verify_lengths
from TCN import train_eval_tcn
import warnings
from urllib3.exceptions import NotOpenSSLWarning
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


def main():
    # 1. (for each .csv file) Add additional columns, fix it to length 18'000
    edit_csv()

    # 2. create non-overlapping train, val & test sets
    X_train, y_train = X_and_y('train')
    print(count_labels(y_train))
    X_val, y_val = X_and_y('val')
    print(count_labels(y_val))
    X_test, y_test = X_and_y('test')
    print(count_labels(y_test))

    verify_lengths()
    #count_all_zero_windows(X_test)

    '''
    df0 = pd.read_csv(r"data/beep_schedules_Mohid2/airpods_motion_1760174060.csv")
    yaw = df0['yaw_rad']
    pitch = df0['pitch_rad']
    roll = df0['roll_rad']
    #plt.plot(np.arange(pitch.shape[0]), pitch, label="pitch")
    #plt.plot(np.arange(pitch.shape[0]), yaw, label="yaw")
    #plt.plot(np.arange(pitch.shape[0]), roll, label="roll")
    plt.xlabel("Time")
    plt.ylabel("[rad]")
    plt.legend()
    plt.show()
    '''
    
    plt.plot(np.arange(718), y_train[0*718:1*718])   # 1*718:2*718
    plt.ylabel("label")
    plt.xlabel("Time")
    plt.legend()
    #plt.show()
    
    
    # 3. Create & train CNN --> then evaluate
    cnn = CNN_model(X_train, y_train, X_val, y_val)
    print("CNN Eval Score:")
    cnn.evaluate(X_test, y_test)
    print("Done evaluating")
    # export_coreml(X_train, cnn)
    '''
    # 4. Try TCN
    TCN_model = train_eval_tcn(X_train, y_train, X_val, y_val, verbose=1)
    print("TCN Eval Score:")
    TCN_model.evaluate(X_test, y_test)
    print("Done evaluating")
    '''

if __name__ == '__main__':
    main()
