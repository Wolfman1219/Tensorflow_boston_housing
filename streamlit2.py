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


column_names = ['CRIM - per capita crime rate by town', 'ZN - proportion of residential land zoned for lots over 25,000 sq.ft.', 'INDUS - proportion of non-retail business acres per town.', 'CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)', 'NOX - nitric oxides concentration (parts per 10 million)', 'RM - average number of rooms per dwelling', 'AGE - proportion of owner-occupied units built prior to 1940', 'DIS - weighted distances to five Boston employment centres', 'RAD - index of accessibility to radial highways', 'TAX - full-value property-tax rate per $10,000', 'PTRATIO - pupil-teacher ratio by town', 'B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town', 'LSTAT - % \lower status of the population']

datas = [st.number_input(i) for i in column_names]
# print(datas)

st.write('prediction is ', model.predict(np.array([datas])))

# st.write('The current number is ', number)

# print(model.summary())