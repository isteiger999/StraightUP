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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


class RF:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=26,
            random_state=41,
            n_jobs=-1,
            class_weight={0: 0.8, 1: 1.0, 2: 1.15},
            criterion="log_loss"      # gini, entropy, log_loss
        )

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


def RF_cl(X_train, y_train, X_val, y_val):

    model = RF()
    model.fit(np.concatenate([X_train, X_val], axis=0), np.concatenate([y_train, y_val], axis=0))
    history = None

    name = "RF"
    return model, history, name

