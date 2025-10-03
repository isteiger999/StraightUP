import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def CNN_model(X_train, X_test, y_train, y_test):

    l2 = regularizers.l2(1e-4)

    # The new input shape is (400, 13)
    cnn = models.Sequential([
        # Input shape is (Timesteps, Channels) -> (400, 13)
        layers.Conv1D(filters=32, kernel_size=11, padding="same", activation="relu", kernel_regularizer=l2),
        layers.MaxPooling1D(2),

        layers.Conv1D(64, 9, padding="same", activation="relu", kernel_regularizer=l2),
        layers.MaxPooling1D(2),

        layers.Conv1D(96, 7, padding="same", activation="relu", kernel_regularizer=l2),
        layers.MaxPooling1D(2),

        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.30),
        layers.Dense(64, activation="relu", kernel_regularizer=l2),
        layers.Dropout(0.20),
        layers.Dense(1, activation="sigmoid")   # predicts P(class=1) (thats why output only 1D)
    ])

    cnn.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                tf.keras.metrics.AUC(name="roc_auc"),
                tf.keras.metrics.AUC(curve="PR", name="pr_auc")]
    )

    print(f"X_train shape: {X_train.shape}; y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}; y_test shape: {y_test.shape}")

    ## find optimal amount of epochs with 'patience'          
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                        patience=10, restore_best_weights=True), # patience 10: means even though validation error might not decrease anymore, we still go 10 epochs further to check if it really increases or just local minimum
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max",
                                            factor=0.5, patience=3)              # multiplies learning rate by 0.5 if after 3 epochs validation error does not decrease 
    ]

    history = cnn.fit(
        X_train, y_train,
        validation_split=0.2,      # or validation_data=(X_val, y_val)
        epochs=200,                # goes maximally up to 200 epochs
        batch_size=32,
        shuffle=True,              # Keras shuffles each epoch for arrays by default
        callbacks=callbacks
    )
    print("Eval Score:")
    cnn.evaluate(X_test, y_test)