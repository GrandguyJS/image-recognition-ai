# Run, train and test the photos in the neural network

from neuralNetwork import NeuralNetwork, Level
from imgConverter import get_image_batch
import utils

import numpy as np




layers = [40000, 100, 100, 2]
nn1 = NeuralNetwork(layers)
accuracy = 0
while accuracy < 0.9:
    imgs, res = get_image_batch(100, True)

    results = [None] * len(imgs)  
        
    for i, v in enumerate(imgs):
        results[i] = NeuralNetwork.feedForward(np.array(v), nn1)

    accuracy = NeuralNetwork.get_accuracy(results, res)

    NeuralNetwork.mutate(nn1, (1.0-accuracy))
    print(accuracy)

"""
epochs = 20

for n in range(0, epochs):
    imgs, res = get_image_batch(10, True)

    results = [None] * len(imgs)  
    
    for i, v in enumerate(imgs):

        results[i] = NeuralNetwork.feedForward(np.array(v), nn1)

    NeuralNetwork.mutate(nn1, 0.1)
    print(f"Epoch {n+1}: ", results)"""