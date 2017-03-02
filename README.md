# Nature...Or Not?
I've trained a convolutional neural network to discern between natural & and man-made structures in satellite images from around the globe (and extra-terrestrial globes).

## Table of Contents
[Motivation](#motivation)
[Dataset](#dataset)
[Process](#process)
[Architecture](#architecture)
[Scores](#scores)
[Fun Tests!](#fun tests!)
[Next Steps](#next steps)


## Motivation
Viewing the world from above can give us an amazing and very different perspective on the changes going on in the world. We can monitor land use, reconnoiter areas of conflict, and survey natural disaster areas to find the best ways to bring in assistance...and, in the future, explore new planets for signs of life.

## Datasetd
I gleaned 6,700 images from Googlemaps, manually and using the Google API with some scripting.  The images in my training and test sets come from around the world and contain a variety of natural settings, including forests, rivers, mountains, deserts, beaches, and ice and snow.

The original saved out images were quite large, , so they were cut down to 124x124x3 using python's PIL library.

Each color image is 124x124x3 pixels. The labeling for these images was done by hand, 0 for images containing only nature and 1 for images containing any man-made object. I've included plowed fields as man-made objects, as the patterns rendered by the plow are more closely related to other man-made objects than to that of nature. And technically, these shapes man-made.


The neural network should be able to learn that hard fixed lines and shapes, discerned from greater pixel differentiation are more likely to be man-made objects, whereas images with less pixel differentiation, softer edges, are more likely to be from a natural setting. Also, brighter colors, along with the pixel differences, are more likely to come from man-made objects. 

Each color image is 250x250x3 pixels. I'm labeling these images by hand as 0 for natural and 1 for images containing man-made objects. I'm including plowed fields as man-made objects, as the shapes rendered by the plow are more closely related to other man-made objects than to that of nature. And technically, these shapes, and the plowed fields, are man-made.

Check Out Galvanize -->
