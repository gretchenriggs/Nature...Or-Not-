# Nature...Or Not?
I've trained a convolutional neural network to discern between natural & and man-made structures in satellite images from around the globe (and extra-terrestrial globes).

## Table of Contents
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Pre-Processing](#pre-processing)
- [Architecture](#architecture)
- [Scores](#scores)
- [Fun Tests](#fun-tests)
- [Next Steps](#next-steps)


## Motivation
Viewing Earth from above can give us an amazing and very different perspective on the changes going on in the world. We can monitor land use, reconnoiter areas of conflict, and survey natural disaster areas to find the best ways to bring in assistance...and, in the future, explore new planets for signs of life.

## Dataset
I gleaned 6,700 images from Googlemaps, manually and using the Google API with some scripting.  The images in my training and test sets come from around the world and contain a variety of natural settings, including forests, rivers, mountains, deserts, beaches, and ice and snow.

The original collected images were quite large, 1152x864x3, which requires more memory than most GPUs can handle, so they were cut down to 124x124x3 using python's PIL library.

The labeling for these images was done by hand, labeled 0 for images containing only nature and 1 for images containing any man-made object. I've included plowed fields as man-made objects, as the patterns rendered by the plow are more closely related to other man-made objects than to that of nature. And technically, these shapes are man-made.

I was able to augment the 6,700 labeled images by taking the 90, 180, and 270 degree rotations of the image, as well as creating mirror images of these 4 rotations.  This created 8 total sets of usable data, for a total of 53,600 images.

## Pre-Processing
The feature (X) values in my dataset are the RGB values of the pixels.  Thus, X is a 124x124x3 array of RGB pixel values.

The data was broken into a Training and a Test set, consisting of 80% and 20% of the data, respectively.

Next, the RGB pixel values were normalized from 1-255 to 0-1 by dividing by 255, and centered around 0, by subtracting the mean of the X training data pixel values.  The mean pixel values for the R, G, & B components were consistent with each other, so the decision was made to use the mean of all 3 channels together to subtract from the data.

The labels (y) values also need some preperation before being fed into the neural net.  The labeled y data is read into the Python script as 1D array (vector) of 0's and 1's.  Keras, the neural networks library used, running on top of Theano for this network, required the labels to be converted to binary class matrices. The vector entry [0] becomes [1, 0] and the entry [1] becomes [0, 1]. 

Lastly, for ease in dealing with the order of my input matrices, though I was using Theano as my backend for the neural network, I used the tf (TensorFlow) option for dimension order, which expects the data to be in num_of_images x num_rows x num_cols x num_channels order.  Since this is the usual ordering of my image data, 53600 x 124 x 124 x 3, this worked out nicely.  The default Theano dimension ordering is num_of_images x num_channels x num_rows x num_cols, which would have been 53600 x 3 x 124 x 124.
    
## Architecture
The convolutional neural network (CNN) can learn that hard fixed lines and shapes, discerned from greater pixel differentiation, are more likely to be man-made objects, whereas images with less pixel differentiation, hence softer edges, are more likely to come from a natural setting. Brighter and more uniform colors are generally more likely to come from man-made objects. And very regular patterns, some that may be indiscernable to the human eye, are more likely to be manufactured by mankind.

Here's the CNN I've built to detect these features in my extracted images:

--> Will input image of the CNN architecture I built and go over how CNN's work and what each step is doing.

CONV2D - 32 filters, 3x3<br>
Activation - RELU (Rectified Linear Unit)<br>
CONV2D - 32 filters, 3x3<br>
RELU<br>
MaxPooling - 2x2 (Desampling)<br>
DropOut -> 0.5<br>
CONV2D - 64 filters, 3x3<br>
RELU<br>
MaxPooling - 2x2 (Desampling)<br>
Flatten<br>
Dense Layer - 512 neurons<br>
RELU<br>
DropOut -> 0.5<br>
Dense Layer - 2 neurons (# of classes)<br>
SoftMax - classification based on prob, probs add up to 1<br>

Computed using Adadelta Gradient Descent optimizer.

Due to size of this dataset and the architecture of the neural network, this project was brought up onto the AWS Cloud to run on a GPU instance with 32 virtual CPU instances and 60 GB of Memory.

## Scores
The neural network I trained in the past few weeks was able to achieve an 83% level of accuracy on the test images it predicted on, compared to a 93% level of accuracy for the training data.  The model is able to generalize fairly well to unseen data.

--> Mention Precision, Recall, F1-Score, as well.  Perhaps an ROC plot and AUC value.
--> Show some images it didn't predict correctly to see if can reason why the net had a problem with them

## Fun Tests!
As I hinted at previously, this technology could be used in predicting if there is intelligent life outside of Earth by observing large volumes of satellite imagery and finding patterns it recognizes as being unnatural.

To test this idea, I captured some images from Google Mars and Google Moon to see if the CNN model could correctly predict if the images only contained natural objects.  Would the learning the model did on desert regions of the Earth be enough for it to make accurate predictions about the otherworldly images from the near cosmos?  Or perhaps find where E.T. and his buddies live?

--> Post results...dun dun duuunnnnn

More close to home, I also ran a subset of images from Rocky Mountain National Park through the prediction to get an estimate of how much of the Park contained man-made structure.

--> Post results...

## Next Steps
Now that I've created a model to predict a binary classification for satellite images, I'd like to go deeper and try to classify portions of the images as natural vs. man-made.

Also, bringing in more training data and further tuning of the neural nets hyperparameters could improve the model results, as could going deeper, creating more hidden layers to learn an even greater number of features.


