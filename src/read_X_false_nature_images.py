import cPickle
import numpy as np
from skimage import img_as_float
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Reading false negative test images in from pickle file
    X_test_false_nature = cPickle.load(open('X_test_false_nature_images.pkl'))
    X_test_false_nature = np.asarray(X_test_false_nature)

    # Save images to disk for review
    for i in xrange(len(X_test_false_nature)):
        plt.imshow(X_test_false_nature[i])
        plt.savefig('X_test_false_nature_' + str(i) + '_.jpg')
