import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.utils.class_weight import compute_class_weight
import coremltools as ct
import numpy as np

def CNN_model(X_train, y_train, X_val, y_val, n_classes=3):
    assert X_train.ndim == 3 and X_train.shape[1:] == (75, 13)
    assert X_val.shape[1:]   == (75, 13)

    # labels should be integer class IDs: 0,1,2
    y_train = y_train.squeeze().astype("int32")
    y_val   = y_val.squeeze().astype("int32")

    l2 = regularizers.l2(1e-4)

    # Per-feature normalization (fit on train only)
    norm = layers.Normalization(axis=-1)
    norm.adapt(X_train.astype("float32"))

    cnn = models.Sequential([
        layers.Input(shape=(75, 13)),
        norm,
        layers.Conv1D(32, 11, padding="same", activation="relu", kernel_regularizer=l2),
        layers.MaxPooling1D(2),

        layers.Conv1D(64, 9,  padding="same", activation="relu", kernel_regularizer=l2),
        layers.MaxPooling1D(2),

        layers.Conv1D(96, 7,  padding="same", activation="relu", kernel_regularizer=l2),
        layers.MaxPooling1D(2),

        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.30),
        layers.Dense(64, activation="relu", kernel_regularizer=l2),
        layers.Dropout(0.20),
        layers.Dense(n_classes, activation="softmax")   # 3 logits -> probs
    ])

    cnn.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]   # add Precision/Recall if you like
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3
        ),
    ]

    # For class imbalances (we have much more 0 than 1 and 2)
    present = np.unique(y_train.ravel())
    valid = np.array([c for c in [0,1,2] if c in present])
    class_weight = None
    if valid.size:
        cw = compute_class_weight('balanced', classes=valid, y=y_train.ravel())
        class_weight = {int(c): float(w) for c, w in zip(valid, cw)}

    cnn.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weight
    )

    return cnn


def export_coreml(model, out_path="PostureCNN.mlpackage"):
    mlmodel = ct.convert(
        model,
        source="tensorflow",
        convert_to="mlprogram",
        inputs=[ct.TensorType(name=model.inputs[0].name.split(":")[0],
                              shape=(1, 400, 14), dtype=np.float32)]
    )
    mlmodel.save(out_path)   # <- .mlpackage
    print(f"✅ Saved {out_path}")