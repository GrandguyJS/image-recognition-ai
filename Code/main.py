# Run, train and test the photos in the neural network

from neuralNetwork import NeuralNetwork, Level
from imgConverter import get_image_batch, resize_image, load_all_images
import utils
import numpy as np
import pickle
import os

# Set Neural Network layers
layers = [40000, 128, 1]
# How many photos to input at once into the neural network
batch_size = 100
# Set the neuralnetwork path
network_path = "../Network/network.pkl"
# Load the neuralnetwork from the filename
nn = utils.loadFormerNetwork(network_path)
# Set to False if you want to test, elso to True
train = False
generation = 0
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
        for i in range(0, 10000):
            # Keep track of generation
            generation += 1
            # Get new random images and labels
            images, labels = get_image_batch(batch_size)
            # Get the loss
            loss = NeuralNetwork.train(nn, images, labels, 0.1)
            # Print new best
            print(f"Generation: {str(generation)}, Loss: {str(loss)}")

            if loss < best:
                best = loss
    finally:
        utils.save_object(nn, network_path)
        print("Saved NeuralNetwork under " + network_path)
    # After all, save the neural network
    
else:
    # Test code
    images, labels = get_image_batch(batch_size, False)
    accuracy = NeuralNetwork.test(nn, images, labels)
    print("Accuracy: " + str(accuracy))
