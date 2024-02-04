# Run, train and test the photos in the neural network

from neuralNetwork import NeuralNetwork, Level
from imgConverter import load_images, get_batch, show_image
import utils
import numpy as np
import os

# Input Values
layers = [784, 16, 16, 10] # Set the neuralnetwork structure
batch_size = 32 # Set the amounts of photos the network will train with each iteration
network_path = "./Network/network.pkl" # Specify where the network will get saved after done training
save_path = "./Dataset/" # Dataset path

train = False
test = False

# Download the images if not already
load_images(save_path)

# Loading the neural network
nn = utils.loadFormerNetwork(network_path)
def train(nn, gens):
    if nn is not None:
            pass
    else:
        nn = NeuralNetwork(layers)
    try:
        best_loss = 1
        for i in range(0, gens):
            X, y = get_batch(32, True)
            loss = NeuralNetwork.train(nn, X, y)
            if loss < best_loss:
                best_loss = loss
                print("New best loss: " + str(loss))
    finally:
        utils.save_object(nn, network_path)

def single_test(network, index):
    # Shows and retuns the image
    img = show_image(index)
    prediction = np.round(NeuralNetwork.feedForward(network, img))
    number_predicted = utils.getnum(prediction[0])
    print(number_predicted)

single_test(nn, 483)