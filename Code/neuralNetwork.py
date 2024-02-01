# Neural Network Code
import random
import utils

import numpy as np
#---Class for the neural network---
class NeuralNetwork:
    
    # Neuron count is an array that specifies the hidden layers and neurons per layer
    # Eg.: [50, 70, 100, 2] has 50 inputs, 2 hidden layers with 70 an 100 neurons and 2 outputs
    def __init__(self, neuronCounts):
        self.levels = [] # Stores the levels

        # Creates the levels in the levels array
        for i in range(0, len(neuronCounts) - 1):
            # Creates a new level with the current neuron count as input and the following as output
            newLevel = Level(neuronCounts[i], neuronCounts[i+1])
            self.levels.append(newLevel)

    # Calculates the ouptut for the whole network, using the given inputs
    # Will return an array of 0 and 1
    # givenInputs is an array of numbers
    @staticmethod
    def feedForward(network, givenInputs): # Input correct result
        # Calculates the first level
        
        outputs = Level.feedForward(givenInputs, network.levels[0])

        # Loops through all following levels and updates outputs
        for i in range(1, len(network.levels)):
            # The level outputs are always the inputs of the new level
            outputs = Level.feedForward(outputs, network.levels[i])

        # Return the output of the last level
        return outputs
    

    # Randomly mutates the network by a certain amount
    # amount=1: every value is random
    # amount=0: no change at all
    @staticmethod
    def mutate(network, amount):

        for level in network.levels:

            # Mutates the biases by amount
            random_values_biases = np.random.uniform(-1, 1, level.biases.shape) # Random biases values
            level.biases = utils.lerp(level.biases, random_values_biases, amount) # Change the biases with the random biases by specific amount

            # Mutates the weights by amount
            random_values_weights = np.random.uniform(-1, 1, level.weights.shape) # Random weight values
            level.weights = level.weights + (random_values_weights - level.weights) * amount # Change the weights with the random weights by specific amount

    #Returns the accuracy of the network
    # 1 = 100% accurate
    @staticmethod
    def get_error(prediction, target):
        # get the average of the loss because the neural network has two outputs

        return np.mean(np.square(target - prediction)) # MSE loss function 

    # Feedforward and mutate the neuralnetwork
    # Output the new accuracy

    @staticmethod
    def train(network, X, y):
        # Input the network, the input and desired output
        prediction = NeuralNetwork.feedForward(network, X)
        loss = NeuralNetwork.get_error(prediction, y)

        print("Loss: " + str(loss))
        # Mutate the neural network by the loss
        NeuralNetwork.mutate(network, loss)

    
#---Class for one layer of the neural network---
class Level:

    def __init__(self, inputCount, outputCount):
        self.inputs = np.zeros(inputCount) #Array of inputs with the length of inputCount
        self.outputs = np.zeros(outputCount) #Array of outputs with the length of outputCount
        self.weights = np.random.randn(inputCount, outputCount) * 0.1 #2D Array of the weights and randomize it here
        self.biases = np.random.randn(outputCount) * 0.1 #2D Array of the biases


    # Calculates the output of one level, using the given inputs
    # Note that the level can only return 0 or 1 for each output
    # givenInputs is an array of numbers
    @staticmethod
    def feedForward(givenInputs, level): # as numpy array
        # Sets the given inputs to the level inputs
        level.inputs = givenInputs

        # Calculates the output with a sum product multiplication
        #Sigmoid ensures every value is between 0 and 1
        level.outputs = utils.sigmoid(np.dot(level.inputs, level.weights) + level.biases)

        #Return outputs
        return level.outputs
