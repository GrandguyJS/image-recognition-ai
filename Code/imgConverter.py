# Convert the test and train images to a list
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import Image

output_dir = "./Dataset/" # Where will the converted photo_arrays get saved
trainpath = "./Dataset/mnist_train.csv"
testpath = "./Dataset/mnist_test.csv"

# Only use once
def load_csv(train = True):
    if train:
        df = pd.read_csv(trainpath)
    else:
        df = pd.read_csv(testpath)

    # Convert labels to one-hot encoding using pandas' get_dummies
    labels = pd.get_dummies(df.iloc[:, 0], columns=np.arange(10), prefix='label', dtype=int).values

    # Normalize pixel values and convert to NumPy array
    pixels = (df.iloc[:, 1:].values / 255.0).astype(np.float32)

    # Save as NumPy array
    if train:
        np.save(output_dir + "train_labels.npy", labels)
        np.save(output_dir + "train_pixels.npy", pixels)
    else:
        np.save(output_dir + "test_labels.npy", labels)
        np.save(output_dir + "test_pixels.npy", pixels)

def load_data(train=True):
    if train:
        filename1 = "train_labels.npy"
        filename2 = "train_pixels.npy"
    else:
        filename1 = "test_labels.npy"
        filename2 = "test_pixels.npy"

    if not os.path.isfile(os.path.join(output_dir, filename1)):
        load_csv(train)

    global labels
    global numbers

    # Load labels and numbers directly
    labels = np.load(os.path.join(output_dir, filename1), allow_pickle=True)
    numbers = np.load(os.path.join(output_dir, filename2), allow_pickle=True)

    return True

def get_batch(size):
    idx = np.random.choice(np.arange(len(labels)), size, replace=False)
    return numbers[idx], labels[idx]

def show_image(index):
    image_array = numbers[index].reshape((28, 28)) * 255.0
    image = Image.fromarray(image_array)
    image.show()
    
def get_image_array(index):
    return numbers[index]