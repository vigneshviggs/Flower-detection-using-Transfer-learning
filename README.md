# Flower detection using Transfer learning steps:

## About MobileNetV2: 

Developed at Google. This is pre-trained on the ImageNet dataset, which contains 1.4M images and 1000 classes. 

It is a research training dataset with variety of categories like jackfruit and syringe. 

This base of knowledge will help us classify flowers from our specific dataset.

## Importing libraries:

    Matplotlib for data visualization.
    PIL for support for opening, manipulating, and saving many different image file formats.
    os provides functions for interacting with the operating system.

### The dataset contains about 3,700 photos of flowers. The dataset contains 5 sub-directories, one per class:

flower_photo/

    daisy/
  
    dandelion/
  
    roses/
  
    sunflowers/
  
    tulips/
  
## Loading the images off disk using image_dataset_from_directory utility. It will convert to tf.data.Dataset for us. 

## Data Augmentation:

Here, these methods are applied:

    RandomFlip
    
    RandomRotation with scale factor of 0.1
    
    RandomZoom with scale factor of 0.1
    
## Preprocess input:

We will use tf.keras.applications.MobileNetV2 as our base model. 

This model expects pixel vaues in [-1,1], but our pixel values in  images are in [0-255]. 

To rescale them, use the preprocess_input included with the model.

## Freeze the layers of the base model:
    
As the image imput values are between [0,255]; we need to standardize the values between [0,1]. Hence this can be done by using the Rescaling layer.

### To generate predictions from the block of features, we need to average over 5x5 locations.

Use tf.keras.layers.GlobalAveragePooling2D layer to convert the features to a single 1280-element vector per image.

### Apply a tf.keras.layers.Dense layer to convert these features into a single prediction per image.


## Build a model:

## Compile a model:

Here we have used adam optimizer and sparsecategoricalcrossentropy loss function.

## Train the model:

Epochs: 10

## plotting the curves using matplotlib:

## Fine tune the metrics:

One way to increase performance even further is to fine-tune the weights of the top layers of the pre-trained model alongside the training of the classifier we added.

Also, try to fine-tune a small number of top layers rather than the whole MobileNet model. Higher up a layer is, the more specialized it is.

The goal of fine-tuning is to adapt these specialized features to work with the new dataset, rather than overwrite the generic learning.

## Un-freeze the top layers of the model

## Compile the new model:
    
As we are training a much larger model and want to readapt the pretrained weights.

It is important to use a lower learning rate at this stage. Otherwise, your model could overfit very quickly.

## train the new model:

## Plotting the curves:


## Evaluation and prediction:
