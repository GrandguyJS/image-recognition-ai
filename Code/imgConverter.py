# Convert the test and train images to a list
import os

# Loading the train and test directory
root_dir = "/Users/grandguymc/Downloads/Cat:Dog/"
train_dir = f"{root_dir}train/"
test_dir = f"{root_dir}test/"

os.chdir(root_dir) # Set working folder to

from PIL import Image  

def resize_image(path):
    image = Image.open(path) # Open the image

    image = image.convert("L") # Convert it to black and white

    new_image = image.resize((200, 200)) # Resize it to (200, 200)

    image_list = list(new_image.getdata()) # Turn it into a list

    return image_list

import random

def get_image_batch(size, train=True):
    if train:
        prefix = "train/"
        directory_list = os.listdir(train_dir) # If we want to train, get the list of all files in train
    else:
        prefix = "test/"
        directory_list = os.listdir(test_dir) # Else get all files in test

    image_name_batch = random.sample(directory_list, size)

    image_batch = [None] * size
    
    for i,image_name in enumerate(image_name_batch): # Iterate trough all random image names
        image_batch[i] = resize_image(prefix + image_name) # Get the image_list of the file and put it in the list image_batch
            
    if train: # If we want to train, we also have to give the right result, so cat or dog
        image_results = [None] * size
        for i,image_name in enumerate(image_name_batch): # Iterate trough all image_names
            image_results[i] = image_name[:3] # Set the image_result to the first three letters of the image file (dog/cat)
        return image_batch, image_results # Return the image lists and results
    
    return image_batch # Only return the image list as we don't train

lists = get_image_batch(10, False)

print(len(lists[0]))



