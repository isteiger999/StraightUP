import torch
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
from sklearn.preprocessing import LabelEncoder

X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('StandWalkJump')

cnn2 = models.Sequential([
    # cnn
    layers.Conv1D(filters = 32, kernel_size = 20, activation = 'relu', input_shape=(2500, 4)),
    layers.MaxPooling1D(pool_size=(2)), # Changed to MaxPooling1D

    # fully connected
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(3,activation='sigmoid')
])

cnn2.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',    #later use 'sparse_categorical_crossentropy' --> gives most proable label, isntead of probabilities for each
            metrics=['accuracy'])

le = LabelEncoder()
le.fit(y_train)

y_train_encoded = le.transform(y_train)
y_test_encoded = le.transform(y_test)

print("Original Labels:", le.classes_)
print("Encoded Labels (Train Sample):", y_train_encoded[:5])
print(y_train_encoded)

cnn2.fit(X_train, y_train_encoded, epochs=10)
cnn2.evaluate(X_test, y_test_encoded)