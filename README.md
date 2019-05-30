# Deep Neural Network For Identifying Hand Written Digits

## This is an deep neural network for identifying hand written digits. 

### Here are the steps followed for building this deep neural network:
    
1. Build a Computation graph:</br>
   Below is a Computational graph for the feed-forward deep neural network that we will build </br>
    
   input >> unique weights >> hidden layer 1 (activation function) >> </br>
         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; >> unique weights >> hidden layer 2 (activation function) >> </br>
         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; >> unique weights >> hidden layer 3 (activation function) >> </br>
         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; >> unique weights >> output layer </br>
    
2.  Cost Function: </br>
    Compare the prediction to the intended output to find out how close is the prediction, by using some Cost or Loss function (as a Cross Entropy)</br>
    
3.  Backpropogation:</br>
    Then we are going to use a optimization function to optimize Cost 
    (minimize the Cost). We can use a optimization function such as 
    Adam Optimizer or SGD or AdaGrad. This optimization function goes backwords
    and manipulates the weights. This is called "Backpropogation"</br>
    
4. Epoch:</br>
   (Feed-forward) + (Backpropogation) = Epoch
   We will have to define the number of Epochs appropritately 
   
5. Number of class in out problem:</br>
   10 Classes (0 to 9)

### Architecture of this deep neural network:</br>

1. Number of hidden layers: 3 </br>
2. Number of neurons in hidden layer: 500 </br>
3. Number of classes (number of unique output) in the output layer: 10 (Each class represents a digit between 0 to 9) </br>
4. Number of weights in first hidden layer = number of inputs * number of neurons in first hidden layer </br>
5. Number of weights in second hidden layer = number of neurons in first hidden layer * number of neurons in second hidden layer </br>
6. Number of weights in third hidden layer = number of neurons in second hidden layer * number of neurons in third hidden layer </br>
7. Number of weights in output layer = number of neurons in third hidden layer * number of classes of output </br>
8. Train the neural network using the training data and get the predictions </br>
9. Compute the cost using the "SOFTMAX_CROSS_ENTROPY_WITH_LOGITS" </br>
10. Optimizes (minimize) the cost using an AdamOptimizer (The default learning rate defined in AdamOptimizer if 0.001) </br>
11. The entire dataSet, all the examples in the dataSet can not be feed into a NN at once. So we define Batch Size. The dataSet is divided into number of batches. Each batch will have 'n' number of examples, that is batch size. </br>
Note: Number of times we pass the entire dataSet to the neural network. One epoch means exactly only once the entire dataSet is passed forward and backward through the neural network. </br>



