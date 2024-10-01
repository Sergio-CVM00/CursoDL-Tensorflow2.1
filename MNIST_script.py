# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:31:58 2024

@author: Sergio Cabeza de Vaca Montero
"""
###### Define dataset ######
import keras 
from keras.datasets import mnist
from keras import models
from keras import layers

(train_images, train_labels), (test_images, test_labels), = mnist.load_data()

# fix dimensions
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# Normalize values: 0-1
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

# transform labels to one-hot labels
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)


###### Design neural network model (NN) -> CNN ######

# model created with a list of layers. (Conv, Pooling, Core, ...)
model = models.Sequential()

# adding conv. 2D layer into the model
model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), input_shape=(28, 28, 1)))

# adding Activation layer
model.add(layers.Activation('relu'))

# adding Pooling layer
model.add(layers.MaxPooling2D(pool_size=2))

# adding Flatten layer
model.add(layers.Flatten())

# adding Dense layer (no. of neurons)
model.add(layers.Dense(units=84))
model.add(layers.Activation('relu'))
model.add(layers.Dense(units=10))
model.add(layers.Activation('softmax'))


###### TIME TO COMPILE IT! ######

# try diferent optimizers, Adam for example.
model.compile(optimizer = 'sgd', 
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# fit the model
model.fit(train_images,
          train_labels,
          epochs=5,
          batch_size=32,
          validation_data=(test_images, test_labels))

# evaluate the model
score = model.evaluate(test_images, test_labels)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])