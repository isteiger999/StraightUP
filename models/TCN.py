from constants import set_seeds, configure_tensorflow
set_seeds()
configure_tensorflow()
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ---------- Balanced Accuracy (metric you can maximize) ----------
class BalancedAccuracy(tf.keras.metrics.Metric):
    '''Balanced Accuracy is a classification metric that measures a model's 
    performance by giving equal importance to every class, regardless of how 
    many samples each class has in the dataset.
    Imagine a dataset of 100 images: 90 of Class A and 10 of Class B.
    90% accuary is misleading. For 3 label problem as ours, we need balanced accuracy > 0.33,
    otherwise random guessing would be even better'''
    def __init__(self, n_classes=3, name="balanced_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        # Per-class true positives and total positives
        self.tp = self.add_weight(name="tp", shape=(n_classes,), initializer="zeros")
        self.pos = self.add_weight(name="pos", shape=(n_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        y_true_oh = tf.one_hot(y_true, depth=self.n_classes, dtype=self.dtype)
        y_pred_oh = tf.one_hot(y_pred, depth=self.n_classes, dtype=self.dtype)

        tp_batch  = tf.reduce_sum(y_true_oh * y_pred_oh, axis=0)   # diag of CM
        pos_batch = tf.reduce_sum(y_true_oh, axis=0)               # row sums of CM

        if sample_weight is not None:
            sw = tf.cast(tf.reshape(sample_weight, (-1, 1)), self.dtype)
            tp_batch  = tf.reduce_sum(sw * (y_true_oh * y_pred_oh), axis=0)
            pos_batch = tf.reduce_sum(sw * y_true_oh, axis=0)

        self.tp.assign_add(tp_batch)
        self.pos.assign_add(pos_batch)

    def result(self):
        per_class_recall = self.tp / (self.pos + tf.keras.backend.epsilon())
        return tf.reduce_mean(per_class_recall)

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))

class WeightedSparseCCE(tf.keras.losses.Loss):
    """
    Weighted Sparse Categorical Cross-Entropy.
    Works with integer labels and either logits or softmax probs.
    """
    def __init__(self, class_weights, from_logits=False, name="weighted_sparse_cce"):
        super().__init__(name=name)
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, reduction="none"
        )

    def call(self, y_true, y_pred):
        # y_true: [B] or [B,1] integer labels
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        per_ex_loss = self.ce(y_true, y_pred)          # [B]
        w = tf.gather(self.class_weights, y_true)      # [B]
        return per_ex_loss * w

class RecallForClass(tf.keras.metrics.Metric):
    def __init__(self, class_id, n_classes=3, name=None, **kwargs):
        super().__init__(name or f"recall_c{class_id}", **kwargs)
        self.class_id = int(class_id)
        self.tp  = self.add_weight("tp", initializer="zeros")
        self.pos = self.add_weight("pos", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        tp  = tf.reduce_sum(tf.cast(tf.logical_and(y_true==self.class_id,
                                                   y_pred==self.class_id), self.dtype))
        pos = tf.reduce_sum(tf.cast(y_true==self.class_id, self.dtype))
        if sample_weight is not None:
            sw  = tf.cast(tf.reshape(sample_weight, (-1,)), self.dtype)
            tp  = tf.reduce_sum(sw * tf.cast(tf.logical_and(y_true==self.class_id,
                                                            y_pred==self.class_id), self.dtype))
            pos = tf.reduce_sum(sw * tf.cast(y_true==self.class_id, self.dtype))
        self.tp.assign_add(tp); self.pos.assign_add(pos)

    def result(self):
        return tf.where(self.pos > 0, self.tp / tf.maximum(self.pos, 1.0), 0.0)

    def reset_state(self):
        self.tp.assign(0.0); self.pos.assign(0.0)

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

# ----------

# ---------- TCN building blocks ----------
def tcn_block(x, filters, k, dropout, l2, dilation):
    y = layers.Conv1D(filters, k, padding="causal", dilation_rate=dilation,
                      kernel_regularizer=regularizers.l2(l2))(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Dropout(dropout)(y)
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
                   *, kernel_size=5, base_filters=16, dropout=0.20, l2=2e-3,  # had 8 filters before
                   max_epochs=1000, n_classes=3):
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
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(0.30)(x)
    y = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(x_in, y)
    loss = WeightedSparseCCE(class_weights=[0.8, 1.0, 1.15], from_logits=False)
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.legacy.Adam(5e-4),
        metrics=[
            BalancedAccuracySubset(include=(1,2), n_classes=n_classes, name="BA_no_upr"),
            F1ForClass(class_id=2, name="f1_sl"),  
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_f1_sl", mode="max", patience=12, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=6, min_lr=1e-5
        ),
    ]

    shuffle_idx = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[shuffle_idx]
    y_train_shuffled = y_train[shuffle_idx]

    history = model.fit(
        X_train_shuffled, y_train_shuffled,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=64,
        shuffle=False,
        callbacks=callbacks,
        verbose=verbose,
    )

    name = "TCN"

    return model, history, name
