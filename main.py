from constants import set_seeds, configure_tensorflow, configure_tensorflow_gpu

set_seeds()
configure_tensorflow_gpu()   # decide GPU/CPU first
configure_tensorflow()       # then threads/repro settings

from events_and_windowing import X_and_y, count_labels, edit_csv, count_all_zero_windows, verify_lengths, find_combinations, std_mean, individual_accuracy, ConfusionMatrixAverager, X_and_y_features
import warnings
from urllib3.exceptions import NotOpenSSLWarning

from models.TCN import train_eval_tcn
from models.CNN import CNN_model, export_coreml
from models.LSTM_class import LSTM_angles
from models.RF_class import RF_cl
from models.SVM_class import SVM_cl
from models.TabPFN_class import TabPFN
from models.XGBoost_class import XGBoost_cl


warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


def main():

    participants = ['Abi', 'Ivan', 'Dario', 'Mohid', 'Claire', 'David',
                    'ZDavB', 'ZDavC'                     # 'ZDavA',         
                    ]                                    # 'ZMohA', 'Svetlana'
    combinations, mean, std = find_combinations(participants, fraction = 0.5)  # fraction 0.1 means cut off  
    n = len(combinations)
    print(combinations)
    
    len_window_sec = 1.5
    stride = 0.5

    edit_csv()
    
    cm_avg = ConfusionMatrixAverager(
        # optional, but nice, since your labels are known
        class_names=["upright", "transition", "slouch"],
        save_dir="confusion_matrix"
    )
    
    for index, (_, list_comb) in enumerate(sorted(combinations.items(), key=lambda kv: int(kv[0]))):

        # 1. Create non-overlapping datasets
        X_train, y_train = X_and_y("train", list_comb, len_window_sec, stride, label_anchor="center")
        X_val, y_val = X_and_y("val", list_comb, len_window_sec, stride, label_anchor="center")
        X_test, y_test = X_and_y("test", list_comb, len_window_sec, stride, label_anchor="center")

        # 2. DL - Classification
        #model, history, name = CNN_model(X_train, y_train, X_val, y_val, verbose = 1)
        model, history, name = train_eval_tcn(X_train, y_train, X_val, y_val, verbose=1)
        #model, history, name = LSTM_angles(X_train, y_train, X_val, y_val, verbose = 1)
        #model, history, name = TabPFN(X_and_y_features(X_train), y_train, X_and_y_features(X_val), y_val) 

        ## ML - Classification ##
        #model, history, name = RF_cl(X_and_y_features(X_train), y_train, X_and_y_features(X_val), y_val)  
        #model, history, name = SVM_cl(X_and_y_features(X_train), y_train, X_and_y_features(X_val), y_val) 
        #model, history, name = XGBoost_cl(X_and_y_features(X_train), y_train, X_and_y_features(X_val), y_val) 

        # 3. Testing the CNN
        if name not in ["TabPFN", "RF", "SVM (RBF)", "XGBoost"]:
            scores = model.evaluate(X_test, y_test, return_dict=True, verbose = 1)
            cm_avg.add(history, X_test, y_test, batch_size=128, verbose=0)    # Confusion Matrix
        else:
            scores = model.evaluate(X_and_y_features(X_test), y_test, return_dict=True, verbose=1)

        print(f"{name} {index + 1}/{len(combinations)} trained")

        for k, v in scores.items():
            mean[k] += v / n           # E[X]
            std[k]  += (v * v) / n     # E[X^2]
        

    # calculate std only now, after mean has already been calculated
    std_mean(mean, std) 

    # print Confusion matrix (0 = upright, 1 = transition, 2 = slouch) #
    #png_recall = cm_avg.save_figure(model_tag="cnn", normalize="true")  # recall view (for precision use normalize="pred")
    #print("✅ Saved averaged confusion matrix")
    
    #export_coreml(X_train, model, out_path="PostureCNN.mlpackage")
    

if __name__ == '__main__':
    main() 