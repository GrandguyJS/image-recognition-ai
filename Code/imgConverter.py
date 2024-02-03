# Convert the test and train images to a list
import os
import numpy as np
from tqdm import tqdm

# Input values
root_dir = "/Users/grandguymc/Downloads/cat:dog/" # Where is the dataset with the images
train_prefix = "train/" # The folder inside the dataset with the train_pictures
test_prefix = "test/" # The folder inside thr dataset with the test_values
output_dir = "../Dataset/" # Where will the converted photo_arrays get saved

from PIL import Image  

#Take your own images
import cv2

# Take a file and return an np.array with 40 000 values that represent the photo in black-white format with 1.0 being white, 0.0 being black
def resize_image(path):
    # Open the path
    image = Image.open(path) # Open the image
    # Take the file, turn it into grayscale, and squish it into the size 200x200
    img = image.convert("L").resize((200, 200))
    # The array itself would be of the shape 200x200 but we want a 1D array with 40000 values, so we flatten the array
    # We also divide by 255, so each value is under 1. After that 1 corresponds to 255 and 0 to 0
    image_array = np.array(img).flatten() / 255.0
    # Return the array
    return image_array # The largest value will be 1, while all the others get scaled down

import random

# Save all images in the dataset to an npy file
# Only has to get run once
def get_all_images(path, train = True):
    # This function goes into the dataset, iterates trough every file, turns into 200x200 grayscale and appends it into a list
    # Initialize empty photos and labels
    photos, labels = list(), list()
    # Load al the pics in the train dataset
    files = os.listdir(path + train_prefix)
    # Add progress bar with tqdm
    for i in tqdm(range(0, len(files)), desc="Iterating trough image files"):
        # If the file is .DS_Store or something else unwanted we just skip the iteration
        if files[i].startswith("."):
            continue
        # if it is a cat we want to have the label set to 1
        output = [1]
        if files[i].startswith("dog"):
            # If it is a dog we set it to 0
            output = [0]
        # Dog = 0 Cat = 1
        # Call the resize image on the file, which will return the 200x200 grayscale array of the file
        photo = resize_image(os.path.join(path, train_prefix, files[i]))
        # Append the photo array into photos and append the label to labels
        photos.append(photo)
        labels.append(output)
    # After iterating trough everything
    # Convert photos and labels to np.arrays
    # The shape should be (amount of photos, 40000)
    # Shape of labels (amount of labels, 1)
    photos = np.asarray(photos)
    labels = np.asarray(labels)

    # Save both nparrays as an npy file in the output directory
    np.save(output_dir + "photos.npy", photos)
    np.save(output_dir + "labels.npy", labels)
    # Return true
    return True

# To reduce training_time, load all images into a var (RAM instead of storage)
def load_all_images():
    # Instead of having to load the files each time we get a new batch, we can also load the entire file into a var so basically into the RAM
    # Note this requires a bit of RAM
    if not os.path.isfile("../Dataset/photos.npy"):
        print("Creating npy files")
        get_all_images(root_dir, True)
    print("loading npy files")
    # Set the 2 global variables for images and labels
    global images
    global labels
    # Set images to be every image np.array in the file, do the same for the labels
    images, labels = np.load(output_dir + "photos.npy"), np.load(output_dir+"labels.npy")
    print("Loaded all images successfully!")
    return True

# Get n random images from the images var and their corresponding labels
def get_image_batch(size, train=True):
    # This function returns n photos and their corresponding labels
    if not train:
        # If we don't want to train, we also didn't load all photos into a var, so we have to do that now, with the test_size
        testimages, testlabels = np.load(output_dir + "photos.npy")[:size], np.load(output_dir+"labels.npy")[:size]
        return testimages, testlabels
    # Get size-random indices of the images list
    idx = np.random.choice(np.arange(len(images)), size, replace=False)
    # Return all images and labels with these indices
    return images[idx], labels[idx]

# Take a picture with the camera and convert it into the desired np.array
def take_picture():
    # This function takes a picture, resizes it and returns the img as an np.array

    # Open the camera feed
    cap = cv2.VideoCapture(0)
    # Check if wehave camera feed
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    # Take a photo
    ret, frame = cap.read()
    # Write it to a file in the same directory as the dataset
    cv2.imwrite(output_dir + 'captured_image.jpg', frame)
    # Close the camera
    cap.release()
    # Resize the taken image with resize_image()
    img = resize_image(output_dir + "captured_image.jpg")
    # return the resized image as an np.array
    return img

