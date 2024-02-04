# Run, train and test the photos in the neural network

from neuralNetwork import NeuralNetwork, Level
from imgConverter import load_images, get_batch, show_image, convert_image
import utils
import numpy as np
import os

# Input Values
layers = [784, 512, 10] # Set the neuralnetwork structure
batch_size = 128 # Set the amounts of photos the network will train with each iteration
network_name = utils.getName(layers)
network_path = f"./Network/{network_name}.pkl" # Specify where the network will get saved after done training
save_path = "./Dataset/" # Dataset path

train = False
test = False

# Download the images if not already
# Load the images into RAM for time-saving
load_images(save_path)


def train(network, epochs, gensperepoch):
    
    # Input the neural network, the amount of epochs and the generations per epoch. It will train it and print the average loss of each epoch.
    # It will also save the network if the training is done, if you cancel, or if an error occurs
    if network is not None:
            pass
    else:
        network = NeuralNetwork(layers)
    try:
        best_loss = network.loss
        for i in range(0, epochs):
            losses = []
            for j in range(0, gensperepoch):
                X, y = get_batch(32, True)
                loss = NeuralNetwork.train(network, X, y, 0.0001)
                losses.append(loss)
            avg_loss = np.mean(losses)
            if avg_loss < best_loss:
                best_loss = avg_loss
            print(f"Epoch {str(i+1)}: Average Loss: {str(np.mean(avg_loss))}")
                
    finally:
        utils.save_object(network, network_path)
    return True

def single_test(network, index):
    # Select a random index, it returns which number was guessed and also opens the image
    # Shows and retuns the image
    img = show_image(index)
    prediction = np.round(NeuralNetwork.feedForward(network, img))
    number_predicted = utils.getnum(prediction[0])
    return number_predicted

def test(network, size, X=None, y=None):
    # Test a neural network with a random test batch size and return the accuracy
    if not X and not y:
        X, y = get_batch(size, False)
    accuracy = NeuralNetwork.test(network, X, y)
    return accuracy

def compare_networks(directory, size):
    # Gets all neural networks in a directory, tests all of them with the same inputs and returns the best ones classifier
    X, y = get_batch(size, False)

    neuralnetworks = []
    for i in os.listdir(directory):
        neuralnetworks.append(utils.loadNetwork(directory+i))
    
    accuracies = []

    for i in neuralnetworks:
        accuracies.append(test(i, size))

    best_accuracy_network = neuralnetworks[accuracies.index(max(accuracies))]

    print(f"The best neural network is {best_accuracy_network.classifier} with an accuracy of {best_accuracy_network.accuracy}")
    # Return the best networks name
    return best_accuracy_network.classifier, best_accuracy_network.accuracy

def input_image(network, image_path):
    img = convert_image(image_path)
    prediction = np.round(NeuralNetwork.feedForward(network, img))
    number_predicted = utils.getnum(prediction[0])
    print("The image is: " + str(number_predicted))
    return number_predicted


nn = utils.loadNetwork(network_path)

input_image(nn, "./Dataset/Number1.jpg")


