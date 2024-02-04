> For the reader
1. Use `pip install -r requirements.txt` to download all needed libraries
2. Go into `main.py` and change the `save_dir` to the directory of your dataset.
3. Also change `network_path` to the path where your network will get saved
4. Next go to `main.py` and specify the layers and batch size
5. Specify if you want to train or test your neural network. 
6. Run `python main.py`
7. It will train the neural network and in the end if you want to test it, it will go trough the first n pictures in the converted dataset, get the output round it and see how many were correct.
8. Enjoy!

> Contributions

This was a school project me and [TimoI44](https://github.com/TimoI44) did. This wouldn't be possible without him.

# Neural Network structure
# 1. File Initializing
-   ### 1.1 Install dependencies
-   ### 1.2 Download all train and test images from Kaggle
-   ### 1.3 Convert images into 200x200 format
-   ### 1.4 Convert the formatted images into numpy arrays and specify the label
-   ### 1.5 Append and save all image and labels

# 2. Building the Neural Network
-   ### 2.1 Specify constructor inputs
-   ### 2.2 Set the Neural Network shape
-   ### 2.3 Initialize random weights and biases
-   ### 2.4 Activation Function (Sigmoid, Softmax)
-   ### 2.5 Forward-Propagation function
-   ### 2.6 Loss-Calculation
-   ### 2.7 Backward-Propagation and Gradient-descent

# 3. Training the Neural Network
-   ### 3.1 Initialize Training Data Set
-   ### 3.2 Setting the hyper-parameter (learnrate)
-   ### 3.3 Iterrate trough the data set
-   ### 3.4 Calculate loss and run trough Backwards-Propagation
-   ### 3.5 Change weights accordingly
-   ### 3.6 Save the weights and biases
-   ### 3.5 Repeat for multiple generations

# 4. Testing the Neural Network
-   ### 4.1 Initialize Testing Data Set
-   ### 4.2 Initialize trained weights and biases
-   ### 4.3 Run trough the Data Set
-   ### 4.4 Calculate accuracy
