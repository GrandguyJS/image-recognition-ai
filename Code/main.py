# Run, train and test the photos in the neural network

from neuralNetwork import NeuralNetwork, Level
from imgConverter import get_image_batch
import utils
import numpy as np
import pickle
import os


layers = [40000, 100, 100, 1]

nOfGenerations = 20
nOfNetworks = 10
nOfImgsPerNetwork = 5

# Loads the former best network from file, returns None if file doesnt exist
def loadFormerNetwork():
    # If a best Network is saved, load it and return it
    if(os.path.isfile("bestNetwork.pkl")):
        #Open the best network file
        with open('bestNetwork.pkl', 'rb') as inp:
            return pickle.load(inp) # Load the former best network object
    else: 
        return None


# Handles the learning process of one generation
# Uses the saved neural network and mutates it n times and checks if the accuracy improved
# If that is the case the more accurate neural network will be saved
def processGeneration():

    bestNetwork = loadFormerNetwork() # Best network of the generation
    bestAccuracy = 0 # Best accuracy of the generation

    imgs, res = get_image_batch(nOfImgsPerNetwork, True) # Loads the images and results

    #Loops through the number of networks per generation and calculates the accuracy by testing the network
    for i in range(0, nOfNetworks):

        nn = None # The current Neural Network
        accuracySum = 0 # The sum of each accuracy that is calculated

        #Sets the network to a new network or takes the former best network if it exists
        if bestNetwork != None and os.path.isfile("bestNetwork.pkl"):
            nn = loadFormerNetwork()
            if(i != 0): NeuralNetwork.mutate(nn, 0.01) # Mutate it by a value (0.01 seems efficient). First network wont be mutated to prevent unlearning
        else:
            # Create a new NeuralNetwork object
            nn = NeuralNetwork(layers)

        # Calculates an output for all test imgs
        for i, img in enumerate(imgs):
            output = NeuralNetwork.feedForward(nn, img) # Get output
            acc = NeuralNetwork.get_accuracy(output, res[i]) # Get accuracy
            accuracySum += acc # Add accuracy to the accuracy Sum
        
        accuracy = accuracySum / nOfImgsPerNetwork # Gets the average accuracy for the current network
        nn.accuracy = accuracy # Sets the networks accuracy
        print("Accuracy: " + str(accuracy))

        #If it is the best network of its generation
        if(accuracy > bestAccuracy):
            bestAccuracy = accuracy # Set the best Accuracy to this networks accuracy
            bestNetwork = nn # Set this network to the bestNetwork
            print("---New best Network---")
    
    utils.save_object(bestNetwork, 'bestNetwork.pkl') # Save the best network of the generation
    print("BEST NETWORK: ")
    print(bestNetwork.accuracy)
    print("*____GENERATION_END____*")


for i in range(0, nOfGenerations):
    processGeneration()
    
#Code for testing the network on one image
"""
n = loadFormerNetwork()
imgs, res = get_image_batch(1, True)
print(NeuralNetwork.feedForward(n, imgs[0]))
"""
