#from data_processing import plot_data, obtain_windows, drop_timestamp_inplace, train_test, count_all_zero_windows, count_labels
from CNN import CNN_model, export_coreml
from events_and_windowing import X_and_y, count_labels, edit_csv, count_all_zero_windows
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


def main():
    # 1. (for each .csv file) Add additional columns, fix it to length 18'000
    edit_csv()

    # 2. create non-overlapping train, val & test sets
    X_train, y_train = X_and_y('train')
    print(X_train[0,0,:])
    '''
    print(count_labels(y_train))
    X_val, y_val = X_and_y('val')
    print(count_labels(y_val))
    X_test, y_test = X_and_y('test')
    print(count_labels(y_test))
    
    #count_all_zero_windows(X_test)
    
    # 3. Create & train CNN --> then evaluate
    cnn = CNN_model(X_train, y_train, X_val, y_val)
    print("Eval Score:")
    cnn.evaluate(X_test, y_test)
    print("Done evaluating")
    # export_coreml(cnn)
    '''

if __name__ == '__main__':
    main()
