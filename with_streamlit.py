import tensorflow as tf
from tensorflow import keras
import streamlit as st

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(path="boston_housing.npz", test_split=0.3, seed=85)

mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std
x_test -= mean
x_test /= std

inputs = keras.layers.Input((13,))
features = keras.layers.Dense(64, activation="relu")(inputs)
outputs = keras.layers.Dense(1)(features)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Initialize empty lists to store the training history
epochs = []
train_loss = []
train_mae = []

# Create the chart outside the for loop
st.title('Har bir epochsda o\'zgaruvchi chart')
chart = st.line_chart(width=0, height=0, use_container_width=True)
# Train the model and update the training history after each epoch
for epoch in range(150):
    model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=0)
    train_metrics = model.evaluate(x_train, y_train, verbose=0)
    epochs.append(epoch+1)
    train_loss.append(train_metrics[0])
    train_mae.append(train_metrics[1])

    # Update the chart after each epoch
    chart_data = {"Epochs": epochs, "Training Loss": train_loss, "Training MAE": train_mae}
    chart_title = "Training History"
    chart_x_axis_label = "Epochs"
    chart_y_axis_label = "Loss and MAE"
    chart.add_rows(chart_data)

# Display the final chart
# st.write(chart)
