''' Running CNN prediction to classify satellite images as containing only
    natural objects (0) or containing some man-made objects (1).
    Using Theano with Tensorflow image_dim_ordering :
    (# images, # rows, # cols, # channels)
    (3200, 124, 124, 3) for the X_train images below
'''

import cPickle
import numpy as np
from keras.models import load_model


def preproc(X):
    ''' Set pixel values to be between 0 and 1 and center them around zero.
        Input: X feature array
        Output: X feature array, standardized and centered around zero.
    '''
    # Standardizing pixel values to be between 0 and 1
    X = X.astype("float32")
    X /= 255.0
    # Zero-center the data (important)
    X = X - np.mean(X)
    return X


if __name__ == '__main__':
    # Read in Moon dataset pickle file, 3200x124x124x3
    features = "X_arr_3200_moon.pkl"
    X = cPickle.load(open(features))
    X = np.asarray(X)

    # Setting pixel values between 0-1 and centering around zero
    X = preproc(X)

    # Load in saved Keras model
    model = load_model('finalized_model_all.h5')

    # Predicting if image contains man-made or only natural images using CNN
    #   Model
    y_pred = model.predict_classes(X)

    # Computing percent of man-made objects on Moon from model prediction
    percent_manmade = float(np.sum(y_pred))/len(y_pred)
