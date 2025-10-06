from data_processing import plot_data, obtain_windows, drop_timestamp_inplace, train_test, count_all_zero_windows
from CNN import CNN_model, export_coreml
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


def main():

    drop_timestamp_inplace(["slouch_data", "no_slouch_data"])
    X_tot, y_tot = obtain_windows()
    print("hihi X_tot")
    count_all_zero_windows(X_tot)
    X_train, X_test, y_train, y_test = train_test(X_tot, y_tot)
    print("hihi X_train")
    #count_all_zero_windows(X_train)
    cnn = CNN_model(X_train, X_test, y_train, y_test)
    print(f"X_tot shape: {X_tot.shape}; y_tot shape: {y_tot.shape}")
    print(f"X_train shape: {X_train.shape}; y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}; y_test shape: {y_test.shape}")
    print("Eval Score:")
    cnn.evaluate(X_test, y_test)
    print("Done evaluating")
    export_coreml(cnn)

if __name__ == '__main__':
    main()
