import tensorflow as tf
from tensorflow import keras
import streamlit


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(path="boston_housing.npz", test_split=0.3, seed=85)

mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std
x_test -= mean
x_test /= std


inputs = keras.layers.Input((13,))
features = keras.layers.Dense(64, activation="relu")(inputs)
features = keras.layers.Dense(32, activation="relu")(features)
features = keras.layers.Dense(16, activation="relu")(features)

outputs = keras.layers.Dense(1)(features)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="mse", metrics=["mae", "accuracy"])



cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint/check",
                                                 save_weights_only=True,
                                                 verbose=1)



model.fit(x_train, y_train, epochs=150, batch_size=64, callbacks=[cp_callback])
test_mse_score, test_mae_score, accuracy = model.evaluate(x_test, y_test)




print(test_mae_score)
print(test_mse_score)
print(accuracy)