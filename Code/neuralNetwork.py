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
    def feedForward(givenInputs, network):
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
            for i in range(0, len(level.biases)):
                level.biases[i] = utils.lerp(
                    level.biases[i], # Current value
                    random.random() * 2 - 1, # Random number between -1 and 1
                    amount # Amount it should change in that direction
                )

            # Mutates the weights by amount
            for i in range(0, len(level.weights)):
                for j in range(0, len(level.weights[i])):
                    level.weights[i][j] = utils.lerp(
                        level.weights[i][j], # Current value
                        random.random() * 2 - 1, # Random number between -1 and 1
                        amount # Amount it should change in that direction
                    )


#---Class for one layer of the neural network---
class Level:

    def __init__(self, inputCount, outputCount):
        self.inputs = np.zeros(inputCount) #Array of inputs with the length of inputCount
        self.outputs = np.zeros(outputCount) #Array of outputs with the length of outputCount
        self.weights = np.zeros((outputCount, inputCount)) #2D Array of the weights
        self.biases = np.zeros((outputCount, 1)) #2D Array of the biases

        Level.randomize(self) #Randomizes the level on initialization
    
    #Randomize the values of the level
    @staticmethod
    def randomize(level):

        # Sets every weight of the level to a random value between -1 and 1
        level.weights = np.random.randn(len(level.outputs), len(level.inputs)) * 0.1

        # Sets every bias of the level to a random value between -1 and 1
        level.biases = np.zeros((len(level.outputs), 1))


    # Calculates the output of one level, using the given inputs
    # Note that the level can only return 0 or 1 for each output
    # givenInputs is an array of numbers
    @staticmethod
    def feedForward(givenInputs, level): # as numpy array

        # Sets the given inputs to the level inputs
        level.inputs = givenInputs

        #Calculates the output with a matrix multiplication
        level.outputs = np.dot(level.weights, level.inputs) + level.biases

        #Return outputs
        #Sigmoid ensures every value is between 0 and 1
        return utils.sigmoid(level.outputs)