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

import tensorflow as tf
tf.random.set_seed(42)

import pandas as pd
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Dense, LSTM, GlobalAveragePooling1D, Dropout, Normalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
import tensorflow_addons as tfa


# we rewrite the (weighted) sparse CCE loss (instead of using tf.keras.losses.sparsecce) to evade the GPU determinism error
def weighted_sparse_cce_det(num_classes: int, class_weights=None):

    if class_weights is not None:
        # store as constant so it's not a trainable parameter
        class_weights_tf = tf.constant(class_weights, dtype=tf.float32)
    else:
        class_weights_tf = None

    def loss_fn(y_true, y_pred):
        # y_true may be shape (batch,) or (batch, 1) → flatten to 1-D
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        # standard CE part
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)  # guard log()
        one_hot = tf.one_hot(y_true, num_classes, dtype=tf.float32)
        per_example = -tf.reduce_sum(one_hot * tf.math.log(y_pred), axis=-1)

        # apply class weights if provided
        if class_weights_tf is not None:
            # weight per sample based on its class
            w = tf.gather(class_weights_tf, y_true)  # shape (batch,)
            per_example = per_example * w

        return per_example  # Keras will average over batch for you

    return loss_fn

# ---------- F1 Score for upright class (combines recall and precision)
class F1ForClass(tf.keras.metrics.Metric):
    """
    F1 score for a single class (hard predictions via argmax).
    Use: F1ForClass(class_id=2, name="f1_sl")
    """
    def __init__(self, class_id, name=None, **kwargs):
        super().__init__(name or f"f1_c{class_id}", **kwargs)
        self.class_id = int(class_id)
        self.tp = self.add_weight("tp", initializer="zeros")
        self.fp = self.add_weight("fp", initializer="zeros")
        self.fn = self.add_weight("fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: integer labels [B] or [B,1]
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        # y_pred: class probabilities/logits [B, C] -> hard preds via argmax
        y_hat  = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        pos_true = tf.equal(y_true, self.class_id)
        pos_pred = tf.equal(y_hat,  self.class_id)

        tp = tf.cast(tf.logical_and(pos_true, pos_pred), self.dtype)
        fp = tf.cast(tf.logical_and(tf.logical_not(pos_true), pos_pred), self.dtype)
        fn = tf.cast(tf.logical_and(pos_true, tf.logical_not(pos_pred)), self.dtype)

        if sample_weight is not None:
            sw = tf.cast(tf.reshape(sample_weight, (-1,)), self.dtype)
            tp = sw * tp
            fp = sw * fp
            fn = sw * fn

        self.tp.assign_add(tf.reduce_sum(tp))
        self.fp.assign_add(tf.reduce_sum(fp))
        self.fn.assign_add(tf.reduce_sum(fn))

    def result(self):
        eps = tf.keras.backend.epsilon()
        return (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn + eps)

    def reset_state(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(0.0)

class BalancedAccuracySubset(tf.keras.metrics.Metric):
    def __init__(self, include=(1,2), n_classes=3, name="BA_no_upright", **kwargs):
        super().__init__(name=name, **kwargs)
        self.include = set(include)
        self.n_classes = n_classes
        self.tp  = self.add_weight("tp",  shape=(n_classes,), initializer="zeros")
        self.pos = self.add_weight("pos", shape=(n_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=self.n_classes, dtype=self.dtype)
        y_pred_oh = tf.one_hot(y_pred, depth=self.n_classes, dtype=self.dtype)
        tp_batch  = tf.reduce_sum(y_true_oh * y_pred_oh, axis=0)
        pos_batch = tf.reduce_sum(y_true_oh, axis=0)
        if sample_weight is not None:
            sw = tf.cast(tf.reshape(sample_weight, (-1, 1)), self.dtype)
            tp_batch  = tf.reduce_sum(sw * (y_true_oh * y_pred_oh), axis=0)
            pos_batch = tf.reduce_sum(sw * y_true_oh, axis=0)
        self.tp.assign_add(tp_batch); self.pos.assign_add(pos_batch)

    def result(self):
        mask_present = tf.cast(self.pos > 0, self.dtype)
        inc_mask = tf.constant([1.0 if i in self.include else 0.0 for i in range(self.n_classes)],
                               dtype=self.dtype)
        mask = mask_present * inc_mask
        recalls = tf.where(self.pos > 0, self.tp / tf.maximum(self.pos, 1.0),
                           tf.zeros_like(self.pos))
        denom = tf.reduce_sum(mask)
        return tf.where(denom > 0, tf.reduce_sum(recalls * mask) / denom, 0.0)

    def reset_state(self):
        self.tp.assign(tf.zeros_like(self.tp)); self.pos.assign(tf.zeros_like(self.pos))


def LSTM_angles(X_train, y_train, X_val, y_val, verbose):
    lr = 1e-3
    l2_reg = l2(1e-3)
    norm = Normalization(axis=-1)
    norm.adapt(X_train.astype("float32"))

    LSTM_model = Sequential([
        Input(shape=X_train.shape[1:]),
        norm,
        LSTM(
            units = 64,                  # units = dim of output (scalar)
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True,
            bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
        ),
        LSTM(
            units = 64,                  # units = dim of output (scalar)
            #return_sequences=True,      # deactivated in last LSTM layer
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True,
            bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
        ),
        #GlobalAveragePooling1D(),
        Dropout(0.20),
        Dense(128, activation='relu', kernel_regularizer=l2_reg),
        Dropout(0.20),
        Dense(3, activation="softmax")
    ])
    #loss = tf.keras.losses.SparseCategoricalCrossentropy() # Sparse CCE does not require a one-hot encoded vector, works with scalar integer
    loss = weighted_sparse_cce_det(num_classes=3, class_weights=[0.8, 1.0, 1.15])

    LSTM_model.compile(
        loss=loss,
        optimizer=Adam(learning_rate = lr),
        metrics=[
            BalancedAccuracySubset(include=(1,2), n_classes=3, name="BA_no_upr"),
            F1ForClass(class_id=2, name="f1_sl"),  
        ]
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_f1_sl", mode="max", patience=12, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=6, min_lr=1e-5
        ),
    ]
    
    # simple deterministic shuffle once per run is fine
    shuffle_idx = np.random.permutation(len(X_train))
    Xs, ys = X_train[shuffle_idx], y_train[shuffle_idx]

    history = LSTM_model.fit(
        Xs, ys,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        shuffle=False,
        callbacks=callbacks,
        verbose=verbose
    )

    name = "LSTM"
    return LSTM_model, history, name
