# Convert the test and train images to a list
import os
import numpy as np

# Loading the train and test directory
root_dir = "/Users/grandguymc/Downloads/cat:dog/"
train_prefix = "train/"
test_prefix = "test/"

output_dir = "../Dataset/"

from PIL import Image  

def resize_image(path):
    image = Image.open(path) # Open the image

    img = image.convert("L").resize((200, 200))

    image_array = np.array(img).flatten() / 255.0
    
    return image_array # The largest value will be 1, while all the others get scaled down

import random

def get_all_images(path, train = True):
    photos, labels = list(), list()
    # Load al the pics in the train dataset
    for i, file in enumerate(os.listdir(path + train_prefix)):
        if file.startswith("."):
            print("ds_store")
            continue

        output = [1, 0]
        if file.startswith("dog"):
            output = [0, 1]
        photo = resize_image(os.path.join(path, train_prefix, file))

        photos.append(photo)
        labels.append(output)

    photos = np.asarray(photos)
    labels = np.asarray(labels)

    np.save(output_dir + "photos.npy", photos)
    np.save(output_dir + "labels.npy", labels)



def get_image_batch(size, train=True):
    if not os.path.isfile("photos.npy"):
        get_all_images(root_dir, True)
    photos = np.load(output_dir + 'photos.npy')
    labels = np.load(output_dir + 'labels.npy')

    idx = np.random.choice(np.arange(len(photos)), size, replace=False)
    return photos[idx], labels[idx]

