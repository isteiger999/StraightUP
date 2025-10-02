import torch
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
from sklearn.preprocessing import LabelEncoder

X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset('StandWalkJump')

# normalize
for i in range(12):
  for j in range(4):
    X_train[i,:,j] = X_train[i,:,j] / max(X_train[i,:,j])

cnn2 = models.Sequential([
    # cnn
    l# Layer 1
    layers.Conv1D(filters=32, kernel_size=20, activation='relu', input_shape=(2500, 4)),
    layers.MaxPooling1D(pool_size=2), # Output length: approx 1240

    # Layer 2: Deeper features, more filters, smaller kernel
    layers.Conv1D(filters=64, kernel_size=10, activation='relu'),
    layers.MaxPooling1D(pool_size=2), # Output length: approx 615

    # fully connected
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3,activation='softmax')
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