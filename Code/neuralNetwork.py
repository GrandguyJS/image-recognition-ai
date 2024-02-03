# Neural Network Code
import random
import utils
import json

import numpy as np

#---Class for the neural network---
class NeuralNetwork:
    
    # Neuron count is an array that specifies the hidden layers and neurons per layer
    # Eg.: [50, 70, 100, 2] has 50 inputs, 2 hidden layers with 70 an 100 neurons and 2 outputs
    def __init__(self, neuronCounts):
        self.levels = [] # Stores the levels

        self.accuracy = 0 # Stores the accuracy of this network

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
    @staticmethod
    def backward(network, X, y, prediction, learning_rate = 0.2):
        # Just don't ask me how this works. I don't understand this myself, but basically you look what wights are causing the output the most, and you change them in a specific direction relative to your error
        # How many training examples in the input
        dataset_size = X.shape[0] # Amount of inputs

        levels_inverted = network.levels[::-1] # Turns the levels around

        # Stores updated weights and biases
        updated_z = []
        updated_w = []
        updated_b = []

        # Calculate and append the loss gradient from the ouput layer to the last hidden layer
        updated_z.append(levels_inverted[0].outputs - y)
        updated_w.append(np.dot(levels_inverted[1].outputs.T, updated_z[-1]) / dataset_size)
        updated_b.append(np.sum(updated_z[-1], axis=0, keepdims=True) / dataset_size)

        # Do the same for all other layers iteratingly
        for i in range(1, len(levels_inverted)):
            # Skip output layer as we have already calculated everything from there

            updated_z.append(np.dot(updated_z[-1], levels_inverted[i-1].weights) * utils.deriv(levels_inverted[i].outputs))
            # If we are at the last layer we don't use i+1 but i.inputs what is the original inputs
            if i+1 != len(levels_inverted):
                updated_w.append(np.dot(levels_inverted[i+1].outputs.T, updated_z[-1]) / dataset_size)
            else:
                updated_w.append(np.dot(levels_inverted[i].inputs.T, updated_z[-1]) / dataset_size)

            # Append the calculated weights and biases

            updated_b.append(np.sum(updated_z[-1], axis=0, keepdims=True) / dataset_size)
        # Iterate trough the lists
        for i in range(0, len(updated_z)):
            index = len(updated_z) - i - 1

            # Change the weights and biases with the new calculated weights and biases by amount
            network.levels[i].weights -= float(learning_rate) * updated_w[index].T

            network.levels[i].biases -= float(learning_rate) * updated_b[index][0]
    

    #Returns the accuracy of the network
    # 1 = 100% accurate
    @staticmethod
    def get_loss(prediction, y):
        # The vaerage, of the absolute difference of all outputs
        return np.mean(abs(y - prediction))

    # In testing period, get how many images where classified correctly
    @staticmethod
    def get_accuracy(prediction, y):
        # Round the prediction, subtract the true solution, sum it, and print the errors of n predictions
        print(str(int(sum(np.round(prediction)-y))) + f" wrong out of {str(len(prediction))}")
        return (len(prediction)-int(sum(np.round(prediction)-y))) / len(prediction)
    
    @staticmethod
    def train(network, X, y, learning_rate = 0.1):
        # Run the neuralnetwork, calculate the loss and do backpropagation to the neural network and return the loss
        prediction = NeuralNetwork.feedForward(network, X)
        loss = NeuralNetwork.get_loss(prediction, y)
        NeuralNetwork.backward(network, X, y, prediction, learning_rate)
        return loss

    def test(network, X, y):

        prediction = NeuralNetwork.feedForward(network, X)
        
        accuracy = NeuralNetwork.get_accuracy(prediction, y)
        return accuracy
    
#---Class for one layer of the neural network---
class Level:

    def __init__(self, inputCount, outputCount):
        self.inputs = np.zeros(inputCount) #Array of inputs with the length of inputCount
        self.outputs = np.zeros(outputCount) #Array of outputs with the length of outputCount
        self.weights = np.random.randn(outputCount, inputCount) * np.sqrt(2.0 / (inputCount + outputCount)) #2D Array of the weights and randomize it here. Multiplied by a vlue, so that they aren't too large nor too small
        self.biases = np.zeros((1, outputCount)) #2D Array of the biases


    # Calculates the output of one level, using the given inputs
    # Note that the level can only return 0 or 1 for each output
    # givenInputs is an array of numbers
    @staticmethod
    def feedForward(givenInputs, level): # as numpy array
        # Sets the given inputs to the level inputs
        level.inputs = givenInputs

        # Calculates the output with a sum product multiplication
        # Sigmoid ensures every value is between 0 and 1
        
        level.outputs = utils.sigmoid(np.dot(level.inputs, level.weights.T) + level.biases)

        #Return outputs
        return level.outputs
