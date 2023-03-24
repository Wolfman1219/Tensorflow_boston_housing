import tensorflow as tf
from tensorflow import keras
import streamlit as st
import numpy as np
latest = tf.train.latest_checkpoint("checkpoint")

inputs = keras.layers.Input((13,))
features = keras.layers.Dense(64, activation="relu")(inputs)
features = keras.layers.Dense(32, activation="relu")(features)
features = keras.layers.Dense(16, activation="relu")(features)

outputs = keras.layers.Dense(1)(features)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="mse", metrics=["mae", "accuracy"])

model.load_weights(latest)


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

datas = [st.number_input(i) for i in column_names]
# print(datas)

st.write('prediction is ', model.predict(np.array([datas])))

# st.write('The current number is ', number)

# print(model.summary())