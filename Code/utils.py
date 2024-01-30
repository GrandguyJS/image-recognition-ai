# Utility function
import numpy as np

# Takes a number A and B and returns the middle value of both by the factor t
# For example if A is 10, B is 20 and t is 0.5 it would return 15
def lerp(A, B, t):
    return A+(B-A)*t

def sigmoid(x):
    return 1 / (1+np.exp(-x))



def get_loss(predictions, targets):
    losses = []
    for i in range(0, len(predictions)):
        prediction = np.round(predictions[i])
        
        if np.array_equal(prediction, targets[i]):
            losses.append(1)
        else:
            losses.append(0)

    loss = sum(losses) / len(losses)
    return loss

    
    