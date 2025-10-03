from data_processing import plot_data, obtain_windows, drop_timestamp_inplace
from CNN import CNN

def main():
    #plot_data()
    drop_timestamp_inplace(["slouch_data", "no_slouch_data"])
    X_tot, y_tot = obtain_windows()
    CNN(X_tot, y_tot)


if __name__ == '__main__':
    main()
