# Neural Network Class
import random

class NeuralNetwork:
    pass


#Class for one layer of the neural network
#Functions:
#   -randomize(level)
#       Randomizes the weights and biases of the level
#   -feedForward(givenInputs, level)
#       Calculates and returns the output of the level with the givenInputs
class Level:

    def __init__(self, inputCount, outputCount):
        self.inputs = [0] * inputCount #Array of inputs with the length of inputCount
        self.outputs = [0] * outputCount #Array of outputs with the length of outputCount

        self.biases = [0] * outputCount #Array of biasese with the lenght of outputCount
        # Weights will be a 2D array containing a value between -1 and 1 for every possible connection -
        # between inputs and outputs
        # To find a weight between input a and output b use: weights[a][b]
        self.weights = [None] * inputCount

        for i in range(0, inputCount-1): #Adds the second dimension
            self.weights[i] = [0] * outputCount #Creates a new Array with the lenght of ouput count

        Level.randomize(self) #Randomizes the level on initialization
    
    #Randomize the values of the level
    @staticmethod
    def randomize(level):

        #Sets every weight of the level to a random value between -1 and 1
        for i in range(0, len(level.inputs)-1):
            for j in range(0, len(level.outputs)-1):
                level.weights[i][j] = random.randint(-1, 1)
        
        #Sets every bias of the level to a random value between -1 and 1
        for i in range(0, len(level.biases)-1):
            level.biases[i] = random.randint(-1, 1)
    
    #Calculates the output of one level
    #note that the level can only return 0 or 1 for each output
    @staticmethod
    def feedForward(givenInputs, level):

        #Sets the given inputs to the level inputs
        for i in range(0, len(givenInputs)-1):
            level.inputs[i] = givenInputs[i]

        #Calculates output value for each output
        for i in range(0, len(level.outputs)-1):
            sum = 0 #Sum of all input values times their weights

            #Calculates sum
            for j in range(0, len(level.inputs)-1):
                #For every input -> multiply it with the weight of the input and the current output node
                sum += level.inputs[j] * level.weights[j][i]

            #If the sum is bigger than the bias of an output node it is active (1)
            if sum > level.biases[i]:
                level.outputs[i] = 1
            else: 
                level.outputs[i] = 0

        return level.outputs