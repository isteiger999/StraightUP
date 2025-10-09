from data_processing import plot_data, obtain_windows, drop_timestamp_inplace, train_test, count_all_zero_windows, count_labels
from CNN import CNN_model, export_coreml
from events_and_windowing import X_and_y
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


def main():

    drop_timestamp_inplace(["slouch_data", "no_slouch_data"])
    X_train, y_train = X_and_y('train')
    print(count_labels(y_train))
    print(X_train.shape, y_train.shape)
    X_val, y_val = X_and_y('val')
    #X_test, y_test = X_and_y('test')
    #print(X_test.shape, y_test.shape)

    #count_all_zero_windows(X_test)
    '''
    cnn = CNN_model(X_train, y_train, X_val, y_val)
    print("Eval Score:")
    cnn.evaluate(X_test, y_test)
    print("Done evaluating")
    # export_coreml(cnn)
    '''

if __name__ == '__main__':
    main()
