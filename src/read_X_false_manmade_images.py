import cPickle
import numpy as np
from skimage import img_as_float
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Reading false positive test images in from pickle file
    X_test_false_manmade = cPickle.load(open('X_test_false_manmade_images.pkl'))
    X_test_false_manmade = np.asarray(X_test_false_manmade)

    # Save images to disk for review
    for i in xrange(len(X_test_false_manmade)):
        plt.imshow(X_test_false_manmade[i])
        plt.savefig('X_test_false_manmade_' + str(i) + '_.jpg')
