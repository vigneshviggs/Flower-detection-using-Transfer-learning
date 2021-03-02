# Flower detection using Transfer learning steps:
 
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
  
## Normalization:
    
As the image imput values are between [0,255]; we need to standardize the values between [0,1]. Hence this can be done by using the Rescaling layer.

## Create a model:

## Compile:

Here we have used adam optimizer and sparsecategoricalcrossentropy loss function.

## Train the model:

Epochs: 10

## plotting the curves using matplotlib:

## Overfitting:

As we can see, the training accuracy is increasing linearly over time, whereas validation accuracy remains around 60% in the training process.
This is called overfitting. It can be avoided by doing data augmentation and dropout methods.

Here, these methods are applied:

    RandomFlip
    
    RandomRotation with scale factor of 0.1
    
    RandomZoom with scale factor of 0.1

## Adding dropout in the model, compile and train the model:
    
Here we will use layers.Dropout to add a dropout with factor of 0.2 and train using augmented images. We will use 15 epochs in this case.

## Visualizing training and testing loss and accuracy:

## Predict on new data:
