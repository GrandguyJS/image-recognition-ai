# Run, train and test the photos in the neural network

from neuralNetwork import NeuralNetwork, Level
from imgConverter import get_image_batch, resize_image, load_all_images, take_picture
import utils
import numpy as np
import pickle
import os

# Input Values
layers = [40000, 128, 1] # Set the neuralnetwork structure
batch_size = 32 # Set the amounts of photos the network will train with each iteration
network_path = "../Network/network.pkl" # Specify where the network will get saved after done training
train = True # Specify what you want to do. Train, Test or take your own picture
test = False
diy = False


# Code
# Load the neuralnetwork from the filename
nn = utils.loadFormerNetwork(network_path)

# Check if we want to train
if train:
    if nn is not None:
        # If the neralnetwork exists do nothing
        pass
    else:
        # Else create a new one
        nn = NeuralNetwork(layers)

    # Set loss to 1 and run the neuralnetwork as long as oss is bigger than x, in this case 0.003
    loss = 1
    # Set the best loss this will get better by time
    best = 1
    # Load all images so you can save yourself a lot of time
    load_all_images()
    try:
        generation = 0
        # Iterate the network 10000 times. That means run, backpropagate and repeat that 10 000 times
        for i in range(0, 10000):
            # Keep track of generation
            generation += 1
            # Get new random images and labels
            images, labels = get_image_batch(batch_size, True)
            # Get the loss
            loss = NeuralNetwork.train(nn, images, labels, 0.1)
            # Print the generation and the loss
            print(f"Generation: {str(generation)}, Loss: {str(loss)}")
            # We will keep track of the best loss soo far. For now it hasn't got any use
            if loss < best:
                best = loss
    # We added try and finnaly so you can abort the code and it will still save
    finally:
        # After all, save the neural network
        utils.save_object(nn, network_path)
        print("Saved NeuralNetwork under " + network_path)
    
    
if test:
    # Test the neural network
    # Get n random images from the dataset
    # Note you don't need to call load_all_images() as here it will load the files once again, because usually you don't test the neural network that often
    images, labels = get_image_batch(batch_size, False)
    # Get the accuracy with NeuralNetwork.test()
    accuracy = NeuralNetwork.test(nn, images, labels)
    # print the accuracy
    print("Accuracy: " + str(accuracy))

# Get your own photo
if diy:
    # Call the function that takes the photo which returns the np.array of the photo
    img = take_picture()
    # Run the input trough the neuralnetwork and get the prediction
    pred = NeuralNetwork.feedForward(nn, img)
    # Print the prediction
    print(pred)
