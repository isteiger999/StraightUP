from data_processing import plot_data, obtain_windows, drop_timestamp_inplace, normalize
from CNN import CNN_model

def main():
    #plot_data()
    drop_timestamp_inplace(["slouch_data", "no_slouch_data"])
    X_tot, y_tot = obtain_windows()
    X_train, X_test, y_train, y_test = normalize(X_tot, y_tot)
    CNN_model(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
