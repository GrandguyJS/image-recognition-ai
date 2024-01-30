# Convert the test and train images to a list
import os

# Loading the train and test directory
root_dir = "/Users/grandguymc/Downloads/Cat:Dog/"
train_prefix = f"train/"
test_prefix = f"test/"

from PIL import Image  

def resize_image(path):
    image = Image.open(path) # Open the image

    image = image.convert("L") # Convert it to black and white

    new_image = image.resize((200, 200)) # Resize it to (200, 200)

    image_list = list(new_image.getdata()) # Turn it into a list

    return image_list

import random

def get_image_batch(size, train=True):
    prefix = test_prefix
    if train:
        prefix = train_prefix
        
    directory_list = os.listdir(root_dir + prefix) # If we want to train, get the list of all files in train
    
    image_name_batch = random.sample(directory_list, size)

    image_batch = [None] * size
    
    for i,image_name in enumerate(image_name_batch): # Iterate trough all random image names
        image_batch[i] = resize_image(root_dir + prefix + image_name) # Get the image_list of the file and put it in the list image_batch
            
    if train: # If we want to train, we also have to give the right result, so cat or dog
        image_results = [None] * size
        for i,image_name in enumerate(image_name_batch): # Iterate trough all image_names
            image_results[i] = image_name[:3] # Set the image_result to the first three letters of the image file (dog/cat)
        return image_batch, image_results # Return the image lists and results
    
    return image_batch # Only return the image list as we don't train



