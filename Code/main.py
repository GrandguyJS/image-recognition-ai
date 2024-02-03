# Run, train and test the photos in the neural network

from neuralNetwork import NeuralNetwork, Level
from imgConverter import get_image_batch, resize_image
import utils
import numpy as np
import pickle
import os

# Set Neural Network layers
layers = [40000, 100, 100, 2]
# How many photos to input at once into the neural network
batch_size = 32

# Load the neuralnetwork from the filename
nn = utils.loadFormerNetwork("network.pkl")

train = True
if train:
    if nn is not None:
        # If the neralnetwork exists do nothing
        pass
    else:
        # Else create a new one
        nn = NeuralNetwork(layers)

    # Set loss to 1 and run the neuralnetwork as long as oss is bigger than x, in this case 0.003
    loss = 1
    generation = 1
    ngenerations = 20
    best = 1

    while loss > 0.003:
        # Get new random images and labels
        images, labels = get_image_batch(batch_size)
        # Get the loss
        loss = NeuralNetwork.train(nn, images, labels, 0.1)
        # Print new best
        try:
            if loss < best:
                print("-- NEW BEST --")
                print("Loss: " + str(loss))
                best = loss


        

    # After all, save the neural network
    utils.save_object(nn, "network.pkl")
else:
    path = "/Users/grandguymc/Downloads/Cat:Dog/train/dog.4277.jpg"
    img = resize_image(path)
    labelcars = path.split("/")[-1][:3]
    label = np.array([[0, 1]]) 
    if labelcars == "cat":
        label = np.array([[1, 0]]) 
    output = NeuralNetwork.feedForward(nn, img)
    print(np.mean(output - label))
