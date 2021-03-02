# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:28:10 2021

@author: vigne
"""
#%%
"""
Importing libraries:

Matplotlib for data visualization.
PIL for support for opening, manipulating, and saving many different image file formats.
os provides functions for interacting with the operating system.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
#%%
"""
The dataset contains about 3,700 photos of flowers. The dataset contains 5 sub-directories, one per class:

flower_photo/
  daisy/
  dandelion/
  roses/
  sunflowers/
  tulips/
"""

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
#%%
"""
Loading the images off disk using image_dataset_from_directory utility.
It will convert to tf.data.Dataset for us. 
"""

batch_size = 32
img_height = 160
img_width = 160

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size = batch_size)

class_names = train_ds.class_names
print(class_names)
#%%
"""
Loading the images from dataset
"""

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
for image_batch, labels_batch in train_ds:
  print(image_batch.shape) #(32, 180, 180, 3)
  print(labels_batch.shape) #(32,)
  break
#%%
"""
Data Augmentation:
"""
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

for image, _ in train_ds.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')
#%%
"""
We will use tf.keras.applications.MobileNetV2 as our base model. 
This model expects pixel vaues in [-1,1], but our pixel values in  images are in [0-255]. 
To rescale them, use the preprocess_input included with the model.
"""

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
#%%
"""
About MobileNet V2:
Developed at Google. This is pre-trained on the ImageNet dataset, which contains 1.4M images and 1000 classes. 
It is a research training dataset with variety of categories like jackfruit and syringe. 
This base of knowledge will help us classify flowers from our specific dataset.
"""

IMG_SHAPE = (img_height, img_width) + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
#%%
image_batch, label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)
print(feature_batch.shape)
#%%
"""
Freeze the layers of the base model
"""

base_model.trainable = False
#%%
base_model.summary()
#%%
"""
To generate predictions from the block of features, we need to average over 5x5 locations.
Use tf.keras.layers.GlobalAveragePooling2D layer to convert the features to a single 1280-element vector per image.
"""

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
#%%
"""
Apply a tf.keras.layers.Dense layer to convert these features into a single prediction per image.
"""

prediction_layer = tf.keras.layers.Dense(5)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)
#%%
"""
Build a model:
"""

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
#%%
"""
Compile a model:
"""

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

'''
The 2.5M parameters in MobileNet are frozen, but there are 6.4K trainable parameters in the Dense layer. 
'''

len(model.trainable_variables)
#%%
"""
Train the model:
"""
'Firstly we will see the initial loss and accuracy of the model'

first_epochs = 10
losss, accuracyy = model.evaluate(val_ds)

print("initial loss: {:.2f}".format(losss))
print("initial accuracy: {:.2f}".format(accuracyy))

history = model.fit(train_ds,
                    epochs=first_epochs,
                    validation_data=val_ds)
#%%
"""
Plotting the curves:
"""

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
#%%
"""
Fine tune the metrics:

One way to increase performance even further is to fine-tune the weights of the top layers of the pre-trained model...
 alongside the training of the classifier we added.
Also, try to fine-tune a small number of top layers rather than the whole MobileNet model.
Higher up a layer is, the more specialized it is.
The goal of fine-tuning is to adapt these specialized features to work with the new dataset, rather than overwrite the generic learning.
"""
"""
Unfreeze the top layers of model:
"""
base_model.trainable = True
#%%
"""
Un-freeze the top layers of the model
"""

#how many layers are in the base model?
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
#%%
"""
Compile the new model:
    
As we are training a much larger model and want to readapt the pretrained weights... 
It is important to use a lower learning rate at this stage. Otherwise, your model could overfit very quickly.
"""
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()
len(model.trainable_variables)
#%%
"""
train the new model:
"""

epochs =  20

history_fine = model.fit(train_ds,
                         epochs=epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=val_ds)
#%%
"""
Plotting the curves:
"""

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([epochs-1,epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([epochs-1,epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
#%%
"""
Evaluation and prediction:
"""

loss, accuracy = model.evaluate(val_ds)
print('Test accuracy :', accuracy)

#Retrieve a batch of images from the test set
image_batch, label_batch = val_ds.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.relu(predictions)
#predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")