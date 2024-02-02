# Convert the test and train images to a list
import os
import numpy as np

# Loading the train and test directory
root_dir = "/Users/timo/Downloads/CatsAndDogs"
train_prefix = "train/"
test_prefix = "test/"

from PIL import Image  

def resize_image(path):
    image = Image.open(path) # Open the image

    image = image.convert("L") # Convert it to black and white

    new_image = image.resize((200, 200)) # Resize it to (200, 200)

    image_data = new_image.getdata()

    image_data_scaled = image_data/np.amax(image_data, axis=0)

    return image_data_scaled # The largest value will be 1, while all the others get scaled down

import random

def get_image_batch(size, train=True):
    prefix = test_prefix
    if train:
        prefix = train_prefix
        
    directory_list = os.listdir(root_dir + prefix) # If we want to train, get the list of all files in train
    
    filtered_directory_list = [item for item in directory_list if not item.startswith('.')]

    image_name_batch = random.sample(filtered_directory_list, size)

    image_batch = [None] * size
    
    for i,image_name in enumerate(image_name_batch): # Iterate trough all random image names
        image_batch[i] = resize_image(root_dir + prefix + image_name)
            
    if train: # If we want to train, we also have to give the right result, so cat or dog

        image_results = [None] * size #Array of 0 and 1 (0 = cat, 1 = dog)
        
        for i,image_name in enumerate(image_name_batch): # Iterate trough all image_names
            
            if image_name[:3] == "cat":
                image_results[i] = 0 # Cat
            else:
                image_results[i] = 1 # Dog
        
        return np.array(image_batch), np.array(image_results) # Return the image lists and results
    
    return np.array(image_batch) # Only return the image list as we don't train

