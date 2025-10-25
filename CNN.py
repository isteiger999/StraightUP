from constants import set_seeds, configure_tensorflow
set_seeds()
configure_tensorflow()
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import coremltools as ct
import numpy as np

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
# ----------

def CNN_model(X_train, y_train, X_val, y_val, verbose, n_classes=3):
    T = X_train.shape[1]
    n_ch = X_train.shape[2]

    # labels should be integer class IDs: 0,1,2
    y_train = y_train.squeeze().astype("int32")
    y_val   = y_val.squeeze().astype("int32")

    l2 = regularizers.l2(2e-3)

    # Per-feature normalization (fit on train only)
    norm = layers.Normalization(axis=-1)
    norm.adapt(X_train.astype("float32"))
    
    # used to be: 32, 64, 96
    cnn = models.Sequential([
        layers.Input(shape=(T, n_ch)),
        norm,
        layers.Conv1D(24, 9, padding="causal", activation="relu", kernel_regularizer=l2),
        layers.MaxPooling1D(2),

        layers.Conv1D(36, 7,  padding="causal", activation="relu", kernel_regularizer=l2),
        layers.MaxPooling1D(2),

        layers.Conv1D(64, 5,  padding="causal", activation="relu", kernel_regularizer=l2),
        layers.MaxPooling1D(2),
        
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.30),
        layers.Dense(64, activation="relu", kernel_regularizer=l2),
        layers.Dropout(0.20),
        layers.Dense(n_classes, activation="softmax")   # 3 logits -> probs
    ])
    ## These metrices are then shown in the cnn.eval on X_val and y_val
    loss = WeightedSparseCCE(class_weights=[0.5, 1.0, 1.15], from_logits=False)

    cnn.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.legacy.Adam(5e-4), 
        metrics=[
            #BalancedAccuracy(n_classes=n_classes, name="balanced_acc"),
            BalancedAccuracySubset(include=(1,2), n_classes=n_classes, name="BA_no_upr"),
            RecallForClass(class_id=2, n_classes=n_classes, name="rec_sl"),
            RecallForClass(class_id=1, n_classes=n_classes, name="rec_trans")
        ]
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_BA_no_upr", mode="max", patience=15, restore_best_weights=True  #val_recall_slouched, val_BA_no_upr
        ),
        tf.keras.callbacks.ModelCheckpoint("best_by_BA.weights.h5", save_weights_only=True, save_best_only=True,
            monitor="val_BA_no_upr", mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=6, min_lr=1e-5
        ),
    ]

    shuffle_idx = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[shuffle_idx]
    y_train_shuffled = y_train[shuffle_idx]

    history = cnn.fit(
        X_train_shuffled, y_train_shuffled,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=128,
        shuffle=False,
        callbacks=callbacks,
        verbose=verbose,
    )

    return cnn, history


def export_coreml(X_train, model, out_path="PostureCNN.mlpackage"):
    T = X_train.shape[1]
    n_ch = X_train.shape[2]
    mlmodel = ct.convert(
        model,
        source="tensorflow",
        convert_to="mlprogram",
        inputs=[ct.TensorType(name=model.inputs[0].name.split(":")[0],
                              shape=(1, T, n_ch), dtype=np.float32)]
    )
    mlmodel.save(out_path)   # <- .mlpackage
    print(f"✅ Saved {out_path}")