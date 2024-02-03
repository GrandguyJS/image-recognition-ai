# Utility function
import numpy as np
import pickle
import os


# Neural Network functions
# Takes a number A and B and returns the middle value of both by the factor t
# For example if A is 10, B is 20 and t is 0.5 it would return 15
def lerp(A, B, t):
    return A+(B-A)*t

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def deriv(x):
    # This gets the preactivated value
    return sigmoid(x) * (1 - sigmoid(x))

# Returns the value the networks prediction and the target are apart
def get_loss(prediction, target):
    return np.abs(target-prediction)



# Saving neuralnetwork
#Function to save a network
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def loadFormerNetwork(filename):
    # If a best Network is saved, load it and return it
    if(os.path.isfile(filename)):
        #Open the best network file
        with open(filename, 'rb') as inp:
            return pickle.load(inp) # Load the former best network object
    else: 
        return None

def getnum(array):
    return np.argmax(array == 1)