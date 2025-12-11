import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"   # helpful on TF 2.10 + cuDNN
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"    # optional, trades speed for reproducibility
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"     # avoid TF32 variability on Ampere
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"       # to silence a warning

import random, numpy as np
random.seed(42)
np.random.seed(42)

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


class SVM_class:
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                C=200,
                gamma="scale",   
                probability=False
            ))
        ])

    def fit(self, X, y):
        y = np.asarray(y).reshape(-1)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, return_dict=True, verbose=0):
        y = np.asarray(y).reshape(-1)
        y_pred = np.asarray(self.predict(X)).reshape(-1)

        # --- balanced accuracy ONLY between classes 1 and 2 ---
        mask_12 = np.isin(y, [1, 2])        # keep only samples with true label 1 or 2
        y_12 = y[mask_12]
        y_pred_12 = y_pred[mask_12]

        bal_acc_12 = balanced_accuracy_score(y_true=y_12, y_pred=y_pred_12)

        # f1 per class (still for 0,1,2 if you want)
        f1_scores = f1_score(y_true=y, y_pred=y_pred, average=None, labels=[0, 1, 2])

        if return_dict:
            return {
                "BA_no_upr": bal_acc_12,    # what you asked for
                "f1_sl": f1_scores[2]         # still f1 for class 2
            }

        # if you only care about 1 & 2 when not returning dict:
        return bal_acc_12


def SVM_cl(X_train, y_train, X_val, y_val):

    model = SVM_class()
    model.fit(np.concatenate([X_train, X_val], axis=0), np.concatenate([y_train, y_val], axis=0))
    history = None

    name = "SVM (RBF)"
    return model, history, name