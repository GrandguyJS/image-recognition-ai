# Run, train and test the photos in the neural network

from neuralNetwork import NeuralNetwork, Level
from imgConverter import get_image_batch
import utils

import numpy as np




layers = [40000, 100, 100, 2]
nn1 = NeuralNetwork(layers)
accuracy = 0



for i in range(0,1000):
    # Get 100 images with the y_label
    imgs, res = get_image_batch(100, True)
    # Run train() in the neuralnetwork, which will feedforward the inputs and mutate by the loss
    NeuralNetwork.train(nn1, imgs, res)
