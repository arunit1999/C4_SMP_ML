import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

digits_mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = digits_mnist.load_data()
# View First 25 images in the dataset
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()

# Define CNN Model (Convolutional Neural Network)
inputs = keras.Input(shape = (28,28,1))
x = layers.experimental.preprocessing.Rescaling(scale = 1.0 / 255)(inputs)
x = layers.Conv2D(filters = 32, kernel_size = (3,3) , activation = "relu")(x)
x = layers.MaxPool2D(pool_size=(2,2))(x)
x = layers.Conv2D(filters = 64, kernel_size = (3,3), activation = "relu")(x)
x = layers.Conv2D(filters = 64, kernel_size = (3,3), activation = "relu")(x)
x = layers.MaxPool2D(pool_size=(2,2),padding = 'Valid')(x)
x = layers.Flatten()(x)
x = layers.Dense(units = 128, activation = 'relu')(x)
outputs = layers.Dense(10,activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics =[keras.metrics.Accuracy()])
model.fit(train_images, tf.one_hot(train_labels,10), epochs = 200)
test_loss, test_acc = model.evaluate(test_images, tf.one_hot(test_labels, 10), verbose =1)
print('\nTest Accuracy:', test_acc)
model.save('model.h5')