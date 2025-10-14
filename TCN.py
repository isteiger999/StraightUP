from constants import set_seeds, configure_tensorflow
set_seeds()
configure_tensorflow()
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam

# ---------- Balanced Accuracy (metric you can maximize) ----------
class BalancedAccuracy(tf.keras.metrics.Metric):
    """
    Mean per-class recall (balanced accuracy).
    Handles classes with zero support by averaging only over present classes.
    """
    def __init__(self, n_classes=3, name="balanced_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = int(n_classes)
        self.tp  = self.add_weight(name="tp",  shape=(self.n_classes,), initializer="zeros", dtype=self.dtype)
        self.pos = self.add_weight(name="pos", shape=(self.n_classes,), initializer="zeros", dtype=self.dtype)

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

        self.tp.assign_add(tp_batch)
        self.pos.assign_add(pos_batch)

    def result(self):
        # recall per class = tp / pos (safe), average over classes with pos>0
        mask = tf.cast(self.pos > 0, self.dtype)
        recalls = tf.where(self.pos > 0, self.tp / tf.maximum(self.pos, 1.0), tf.zeros_like(self.pos, dtype=self.dtype))
        denom = tf.reduce_sum(mask)
        return tf.where(denom > 0, tf.reduce_sum(recalls * mask) / denom, tf.constant(0.0, dtype=self.dtype))

    def reset_state(self):
        tf.keras.backend.batch_set_value([(self.tp, np.zeros(self.n_classes, dtype=np.float32)),
                                          (self.pos, np.zeros(self.n_classes, dtype=np.float32))])

# ---------- TCN building blocks ----------
def tcn_block(x, filters, k, dropout, l2, dilation):
    y = layers.Conv1D(filters, k, padding="causal", dilation_rate=dilation,
                      kernel_regularizer=regularizers.l2(l2))(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    #y = layers.Dropout(dropout)(y)
    y = layers.SpatialDropout1D(dropout)(y)

    y = layers.Conv1D(filters, k, padding="causal", dilation_rate=dilation,
                      kernel_regularizer=regularizers.l2(l2))(y)
    y = layers.BatchNormalization()(y)

    res = x
    if res.shape[-1] != filters:
        res = layers.Conv1D(filters, 1, padding="same",
                            kernel_regularizer=regularizers.l2(l2))(res)
    y = layers.add([res, y])
    return layers.Activation("relu")(y)

def blocks_for_full_rf(seq_len, k, max_blocks=12):
    # Two causal convs per block → RF grows by 2*(k-1)*d each block (with doubling dilation)
    rf = 1
    d = 1
    blocks = 0
    while rf < seq_len and blocks < max_blocks:
        rf += 2 * (k - 1) * d
        d *= 2
        blocks += 1
    return blocks

# ---------- Train/Eval wrapper ----------
def train_eval_tcn(X_train, y_train, X_val, y_val, verbose,
                   *, kernel_size=5, base_filters=96, dropout=0.10, l2=8e-4,
                   batch_size=64, max_epochs=1000, n_classes=3):
    tf.keras.utils.set_random_seed(42)

    # Shapes & dtypes
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val   = np.asarray(X_val,   dtype=np.float32)
    y_train = np.asarray(y_train).squeeze().astype(np.int32)   # <-- integers for sparse CE
    y_val   = np.asarray(y_val).squeeze().astype(np.int32)

    assert X_train.ndim == 3 and X_val.ndim == 3, "X must be [N, T, C]"
    T, C = X_train.shape[1], X_train.shape[2]
    n_blocks = blocks_for_full_rf(T, kernel_size)

    # Model
    x_in = layers.Input(shape=(T, C))
    norm = layers.Normalization(axis=-1)
    norm.adapt(X_train)
    x = norm(x_in)

    d = 1
    for _ in range(n_blocks):
        x = tcn_block(x, base_filters, kernel_size, dropout, l2, dilation=d)
        d *= 2

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(0.20)(x)
    y = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(x_in, y)
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=[BalancedAccuracy(n_classes=n_classes, name="balanced_acc")]    # "accuracy", 
    )

    monitor = "val_balanced_acc"   # we want to MAXIMIZE this
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode="max", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]

    # For class imbalances (we have much more 0 than 1 and 2)
    present = np.unique(y_train.ravel())
    valid = np.array([c for c in [0,1,2] if c in present])
    class_weight = None
    if valid.size:
        cw = compute_class_weight('balanced', classes=valid, y=y_train.ravel())
        class_weight = {int(c): float(w) for c, w in zip(valid, cw)}
    
    shuffle_idx = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[shuffle_idx]
    y_train_shuffled = y_train[shuffle_idx]

    model.fit(
        X_train_shuffled, y_train_shuffled,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=verbose,
        class_weight=class_weight
    )
    return model
