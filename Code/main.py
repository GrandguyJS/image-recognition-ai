# Run, train and test the photos in the neural network

from neuralNetwork import NeuralNetwork, Level
from imgConverter import load_csv, load_data, get_batch, show_image, get_image_array
import utils
import numpy as np
import pickle
import os

# Input Values
layers = [784, 16, 16, 10] # Set the neuralnetwork structure
batch_size = 32 # Set the amounts of photos the network will train with each iteration
network_path = "./Network/network.pkl" # Specify where the network will get saved after done training

train = False
test = False

dataset_path = "./Dataset/mnist_train.csv"

load_data(train=True) # Load the labels and pictures

nn = utils.loadFormerNetwork(network_path)

if train:
    if nn is not None:
        pass
    else:
        nn = NeuralNetwork(layers)

    try:
        best_loss = 1
        generation = 0
        for i in range(0, 100000):
            generation += 1
            nums, labels = get_batch(batch_size)
            loss = NeuralNetwork.train(nn, nums, labels)

            if loss < best_loss:
                best_loss = loss
                print("New best: " + "Generation: " + str(generation) + ", Loss: " + str(loss))
    finally:
        utils.save_object(nn, network_path)
        print("Saved NeuralNetwork under " + network_path)

if test:
    nums, labels = get_batch(batch_size)

    NeuralNetwork.test(nn, nums, labels)

index = 1000

randomnum = get_image_array(index)
pred = utils.getnum(np.round(NeuralNetwork.feedForward(nn, randomnum)))

show_image(index)
print(pred)