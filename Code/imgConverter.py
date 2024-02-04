# Convert the test and train images to a list
import os
import numpy as np
import pandas as pd

import sklearn
from sklearn.datasets import fetch_openml 

from PIL import Image

import utils

def load_images(save_path):
    # Dataset
    mnist = fetch_openml("mnist_784", data_home=save_path)

    X, y = mnist["data"].values, mnist["target"].values

    #Train dataset with first 60K numbers
    global X_train 
    global y_train 
    global X_test
    global y_test

    y_label = np.zeros((600000, 10))
    for i in range(0, X.shape[0]):
        y_label[i, int(y[i])] = 1

    X_train, y_train = X[:60000], y_label[:60000].astype(int)
    # Test dataset with 10K numbers
    X_test, y_test = X[60000:], y_label[60000:].astype(int)

def get_batch(size, train=True):
    if train:
        X, y = X_train, y_train
    else:
        X, y = X_test, y_test

    idx = np.random.choice(np.arange(X.shape[0]), size, replace=False)

    return X[idx] / 255.0, y[idx]

def show_image(index):
    img = X_train[index].reshape((28, 28)).astype(np.uint8)
    img = Image.fromarray(img)
    img.show()

    return X_train[index] / 255.0

def convert_image(image_path):
    img = Image.open(image_path).convert("L")
    
    if img.size != (28, 28):
        img = img.reshape((28,28))
    return np.array(img).flatten()



