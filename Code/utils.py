# Utility function
import numpy as np
import pickle

# Takes a number A and B and returns the middle value of both by the factor t
# For example if A is 10, B is 20 and t is 0.5 it would return 15
def lerp(A, B, t):
    return A+(B-A)*t

def sigmoid(x):
    return 1 / (1+np.exp(-x))

# Returns the value the networks prediction and the target are apart
def get_loss(prediction, target):
    return np.abs(target-prediction)

#Function to save a network
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)