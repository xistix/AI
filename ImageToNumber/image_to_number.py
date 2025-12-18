# this code is based on the keras mnist example found in chapter 2 of the book "Deep Learning with Python" by Francois Chollet


# keras AI framework for building and training neural networks
import keras

# layers module from keras for building neural network layers
from keras import layers

# import the minst dataset (handwritten digits)
from keras.datasets import mnist

# import numpy for storing and manipulating tensor data
import numpy as np


#returns the training and testing data as tuples of numpy arrays
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")


# create the model with two dense (fully connected) layers   
model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)


# compile the model with adam optimizer and sparse categorical crossentropy loss function
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # -Sum(P_true * log(P_pred)); where P_true is one-hot encoded vector that is generated internally by keras from integer labels
    metrics=["accuracy"],
)

print("reshaping and normalizing the data...")

# preprocess the data by reshaping and normalizing
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

print(f"Train images shape: {train_images.shape}")
print(f"Test images shape: {test_images.shape}")


print("training the model...")
# train the model with training data for 5 epochs and batch size of 128
model.fit(train_images, train_labels, epochs=5, batch_size=128)


test_digits = test_images[0:10]
predictions = model.predict(test_digits)

print("shape of predictions:", predictions.shape)

print("Predictions for the first 10 test images:")
for i, prediction in enumerate(predictions):
    predicted_label = np.argmax(prediction)
    print(f"Image {i}: Predicted label: {predicted_label}, True label: {test_labels[i]}")


print("Evaluating the model on train data...")
# loss is the average value of the loss function across all test samples
# acc is the fraction of test samples that were classified correctly
test_loss, test_acc = model.evaluate(train_images, train_labels)
print("Test loss:", test_loss)
print(f"Test accuracy: {test_acc}")


print("Evaluating the model on test data...")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test loss:", test_loss)
print(f"Test accuracy: {test_acc}")