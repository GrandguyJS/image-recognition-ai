# Convert the test and train images to a list
import os
import numpy as np
from tqdm import tqdm

# Loading the train and test directory
root_dir = "/Users/grandguymc/Downloads/cat:dog/"
train_prefix = "train/"
test_prefix = "test/"

output_dir = "../Dataset/"

from PIL import Image  

#Take your own images
import cv2

def resize_image(path):
    image = Image.open(path) # Open the image

    img = image.convert("L").resize((200, 200))

    image_array = np.array(img).flatten() / 255.0
    
    return image_array # The largest value will be 1, while all the others get scaled down

import random

def get_all_images(path, train = True):
    # Initialize empty photos and labels
    photos, labels = list(), list()
    # Load al the pics in the train dataset
    files = os.listdir(path + train_prefix)
    # Add progress bar
    for i in tqdm(range(0, len(files)), desc="Iterating trough image files"):
        if files[i].startswith("."):
            continue
        # if it is a cat we want to have the label set to 1
        output = [1]
        if files[i].startswith("dog"):
            # If it is a dog we set it to 0
            output = [0]
        photo = resize_image(os.path.join(path, train_prefix, files[i]))

        photos.append(photo)
        labels.append(output)

    photos = np.asarray(photos)
    labels = np.asarray(labels)

    np.save(output_dir + "photos.npy", photos)
    np.save(output_dir + "labels.npy", labels)
    return True

def load_all_images():
    if not os.path.isfile("../Dataset/photos.npy"):
        print("Creating npy files")
        get_all_images(root_dir, True)
    print("loading npy files")
    global images
    global labels
    images, labels = np.load(output_dir + "photos.npy")[:100], np.load(output_dir+"labels.npy")[:100]
    print("Loaded all images successfully!")
    return True

def get_image_batch(size, train=True):
    if not train:
        testimages, testlabels = np.load(output_dir + "photos.npy")[:size], np.load(output_dir+"labels.npy")[:size]
        return testimages, testlabels
    idx = np.random.choice(np.arange(len(images)), size, replace=False)
    return images[idx], labels[idx]

def take_picture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    ret, frame = cap.read()
    cv2.imwrite(output_dir + 'captured_image.jpg', frame)
    cap.release()
    img = resize_image(output_dir + "captured_image.jpg")
    return img

