# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:18:42 2019

@author: rajui
"""

"""
Dataset: 
    We're going to be working with the MNIST dataset, 
    which is a dataset that contains 60000 training sample images and 
    10000 testing sample images of hand-written and labeled digits, 0 through 9.
    So ten total "classes."

    The MNIST dataset images, are purely black and white, thresholded, images, 
    of size 28 x 28, or 784 pixels total.

Features:   
    Our features will be the pixel values for each pixel, thresholded. 
    Either the pixel is "blank" (nothing there, a 0), or 
    there is something there (1). Those are our features. 
    We're going to attempt to just use this extremely rudimentary data, and 
    predict the number we're looking at (0,1,2,3,4,5,6,7,8, or 9).
    
Output:
    We're hoping that our neural network will somehow create an inner-model 
    of the relationships between pixels, and be able to look at new examples 
    of digits and predict them to a high degree
"""

"""
Steps for building a deep neural network for this task:
    
1. Build a Computation graph:
   Below is a Computational graph for the feed-forward deep neural network 
   that we will build 
    
   input >> unique weights >> hidden layer 1 (activation function) >> 
         >> unique weights >> hidden layer 2 (activation function) >>
         >> unique weights >> hidden layer 3 (activation function) >>
         >> unique weights >> output layer
    
2.  Cost Function: 
    Compare the prediction to the intended output to find out how close is the
    prediction, by using some Cost or Loss function (as a Cross Entropy)
    
3.  Backpropogation:
    Then we are going to use a optimization function to optimize Cost 
    (minimize the Cost). We can use a optimization function such as 
    Adam Optimizer or SGD or AdaGrad. This optimization function goes backwords
    and manipulates the weights. This is called "Backpropogation"

4. Epoch:
   (Feed-forward) + (Backpropogation) = Epoch
   >> We will have to define the number of Epochs appropritately 
   
5. Number of class in out problem:
   10 Classes (0 to 9)
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets \
        ("D:\\Learning\\TensorFlow\\2.DNN\\DNNForIdentifyingHandwrittenDigits\\data", 
         one_hot=True)

"""
Number of inputs
Each image is 28 * 28 size, that is 784 pixels 
"""
numberOfInputs = 784

"""
The entire dataSet, all the examples in the dataSet can not be feed into a NN
tt once. So we define Batch Size.
The dataSet is divided into number of batches. Each batch will have 'n' 
number of examples, that is batch size.
"""
batchSize = 100

"""
Number of times we pass the entire dataSet to the neural network
One epoch means exactly only once the entire dataSet is passed forward and 
backward through the neural network
"""
numberOfEpochs = 10

#Matrix is height x width
#height - None
x = tf.placeholder('float', [None, numberOfInputs])
y = tf.placeholder('float')

"""
This function defines a model for our neural network.
By model, we mean, 
    >> defining the variables that hold, the weights and biases in each layer
    >> defining the computation of ((data * weights) + biases) for each layer
    >> defining the activation function for each layer
In other words, we define a computation graph for our neural network
"""
def neuralNetwork (data):
    #Number of hidden layers
    numberOfHiddenLayers = 3
    
    #Number of neurons in hidden layer 1, 2 and 3
    numberOfNeuronsInHiddenLayer1 = 500
    numberOfNeuronsInHiddenLayer2 = 500
    numberOfNeuronsInHiddenLayer3 = 500
    
    #Number of classes, number of unique outputs
    numberOfClassesOfOutput = 10
        
    """
    Number of weights in first hidden layer = 
                                    number of inputs * 
                                    number of neurons in first hidden layer
    """
    configForHiddenLayer1 = {
         'weights':tf.Variable(
                     tf.random_normal([numberOfInputs, 
                                       numberOfNeuronsInHiddenLayer1])),
         'biases' :tf.Variable(
                     tf.random_normal([numberOfNeuronsInHiddenLayer1]))
        }
    
    """
    Number of weights in second hidden layer = 
                                    number of neurons in first hidden layer * 
                                    number of neurons in second hidden layer
    """
    configForHiddenLayer2 = {
         'weights':tf.Variable(
                     tf.random_normal([numberOfNeuronsInHiddenLayer1, 
                                       numberOfNeuronsInHiddenLayer2])),
         'biases' :tf.Variable(
                     tf.random_normal([numberOfNeuronsInHiddenLayer2]))
        }

    """
    Number of weights in third hidden layer = 
                                    number of neurons in second hidden layer * 
                                    number of neurons in third hidden layer
    """
    configForHiddenLayer3 = {
         'weights':tf.Variable(
                     tf.random_normal([numberOfNeuronsInHiddenLayer2, 
                                       numberOfNeuronsInHiddenLayer3])),
         'biases' :tf.Variable(
                     tf.random_normal([numberOfNeuronsInHiddenLayer3]))
        }

    """
    Number of weights in output layer = 
                                    number of neurons in third hidden layer * 
                                    number of classes of output
    """
    configForOutputLayer = {
         'weights':tf.Variable(
                     tf.random_normal([numberOfNeuronsInHiddenLayer3, 
                                   numberOfClassesOfOutput])),
         'biases' :tf.Variable(
                     tf.random_normal([numberOfClassesOfOutput]))
        }

    #For each layer, we compute (input data * weights) + biases
    hiddenLayer1 = tf.add(tf.matmul(
                            data, configForHiddenLayer1['weights']) ,
                          configForHiddenLayer1['biases'])
    hiddenLayer1 = tf.nn.relu(hiddenLayer1) 

    hiddenLayer2 = tf.add(tf.matmul(hiddenLayer1, 
                                    configForHiddenLayer2['weights']) , 
                          configForHiddenLayer2['biases'])
    hiddenLayer2 = tf.nn.relu(hiddenLayer2) 

    hiddenLayer3 = tf.add(tf.matmul(hiddenLayer2, 
                                    configForHiddenLayer3['weights']) , 
                          configForHiddenLayer3['biases'])
    hiddenLayer3 = tf.nn.relu(hiddenLayer3) 
    
    output = tf.add(tf.matmul(hiddenLayer3, 
                                    configForOutputLayer['weights']) , 
                          configForOutputLayer['biases'])

    return output

"""
1. This function trains the neural network using the training data
   and outputs a prediction
2. Computes the cost using the "softmax_cross_entropy_with_logits"
3. Optimizes (minimize) the cost using an optimizer

"""
def trainNeuralNetwork (trainData):
    #Train the neural network using the training data and get the prediction
    prediction = neuralNetwork(trainData)
    """
    Compare the prediction to the intended output to find out how close is the
    prediction, by using some Cost or Loss function 
    """
    cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    
    #Next step is to optimize the cost function
    #The default learning rate defined in AdamOptimizer if 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True    
    with tf.Session (config=config) as session:
        session.run(tf.global_variables_initializer())
        
        for epoch in range(numberOfEpochs):
            epochLoss = 0
            for _ in range(int(mnist.train.num_examples/batchSize)):
                epochX, epochY = mnist.train.next_batch(batchSize)
                _, c = session.run([optimizer, cost], feed_dict={x: epochX, y: epochY})
                epochLoss += c

            print('Epoch ', epoch, 'completed out of ',numberOfEpochs,'loss: ', epochLoss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

trainNeuralNetwork(x)
       