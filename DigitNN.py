import numpy as np
import pandas as pd
from math import e

#structure of the neural network
#first layer is the firstLayer that is not the input
firstLayer = 50
secondLayer = 10

#learning parameters for gradient descent
epochs = 30
miniBatchSize = 100
iterations = int(60000/miniBatchSize)

#how big of a step you want to take during gradient descent
learningRate = 3

def sigmoid(val):
    return 1/(1+np.exp(-val))

def sigmoidPrime(val):
    return sigmoid(val)*(1-sigmoid(val))

#takes a label such as y=1 and outputs a vector v = [0,1,0,0,0,0,0,0,0,0]
#where the yth index is 1 and the rest is 0
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

#initialize the weights and biases of the neural network
def initialize():
    W1 =  np.random.uniform(-0.5,0.5,(firstLayer,784))
    W2 =  np.random.uniform(-0.5,0.5, (secondLayer, firstLayer))

    B1 = np.zeros((firstLayer,1))
    B2 = np.zeros((secondLayer,1))
    return W1, W2, B1, B2


def forwardPropogation(X, W1, B1, W2, B2):
            z1 = np.dot(W1,X) + B1
            a1 = sigmoid(z1)
            z2 = np.dot(W2,a1) + B2
            a2 = sigmoid(z2)   
            return z1,a1,z2,a2

def updateParameters(DW1, DW2, DB1, DB2, W1, W2, B1, B2, learningRate, miniBatchSize):
        W1 = W1 - learningRate * DW1/miniBatchSize
        W2 = W2 - learningRate * DW2/miniBatchSize
        B1 = B1 - learningRate * DB1/miniBatchSize
        B2 = B2 - learningRate * DB2/miniBatchSize
        return W1, W2, B1, B2

#after we update our parameters, we check to see how much better our
#neural can classify the digits in the testData set
def testAccuracy(W1, W2, B1, B2):
    cnt = 0
    for m in range(dataTest[0].size):
        image = X_test[:,m].reshape(784,1)

        #forward propogation
        z1,a1,z2,a2 = forwardPropogation(image, W1, B1, W2, B2)

        #check to see if output of neural network is equal to
        #the desired output
        if(np.argmax(a2) == Y_test[m]):
            cnt = cnt +1
    percent = cnt/10000*100
    print("Accuracy: " + str(percent) + "%")


#contains training data, 60,000 images each with 784 pixels
dataTrain = np.transpose(np.array(pd.read_csv('data/mnist_train.csv')))
#dimension of dataTrain is (784,60000), each image is a column

#contians testing data, 10,000 images
dataTest = np.transpose(np.array(pd.read_csv('data/mnist_test.csv')))
#dimension is (784,10000) each image is a column



#Y_train and Y_test gives the handwritten digit for each image
#one_hot_Y if Y=1 then one_hot_Y = [0,1,0,0,0,0,0,0,0,0]
#or if Y=9 then one_hot_Y = [0,0,0,0,0,0,0,0,0,1]
#shape of Y_train is 60,000x1
#shape of one_hot_Y is 10x60,000 each column is a one_hot representation of a label
Y_train = (dataTrain[0])
one_hot_Y = one_hot(Y_train)

#divide all pixel values by 255 so each pixel value is between 0 and 1
X_train = dataTrain[1:]/255



Y_test = (dataTest[0])
X_test = dataTest[1:]/255



#initialize weights
W1, W2, B1, B2 = initialize()


#training
#epochs is the number of times we go through the entire dataset
for e in range(epochs):
    for j in range(iterations):
        
        '''DW1 is the the partial derivative of the cost function with respect to Weight 1, 
           DW2 is the partial derivative of the cost function with respect to Weight 2,
           DB2 is the partial derivative of the cost function with respect to Bias 2,
           DB1 is the partial derivative of the cost function with respect to Bias 1,
        '''
        DW1 = np.zeros((firstLayer,784))
        DW2 = np.zeros((secondLayer,firstLayer))
        DB1 = np.zeros((firstLayer,1))
        DB2 = np.zeros((secondLayer,1))
        
        #iterate over each image and run it through the neural network
        #once we go over miniBatchSize number of images, we will update the paramters of the neural network
        #we can actually feed in multiple images into the neural network, but to keep it simple we will 
        #just stick with going one at a time even though it slows the program down
        for i in range(miniBatchSize):
            img = j*miniBatchSize+i
            #b/c numpy is weird the dimension of X_train[:,img] is (784,) so we have to reshape it
            #same with one-hot_Y
            image = (X_train[:,img]).reshape((784,1))
            label = one_hot_Y[:,img].reshape(10,1)

            #forward propogation
            z1,a1,z2,a2 = forwardPropogation(image, W1, B1, W2, B2)

            #backpropogation
            #this is dL/dz2
            error2 = (a2 - label) * sigmoidPrime(z2)

            DB2 = DB2 + error2
            DW2 = DW2 + np.dot(error2,a1.T)

            #this is dL/dz1
            error1 = np.dot(W2.T,error2) * sigmoidPrime(z1)
            
            DB1 = DB1 + error1
            DW1 = DW1 + np.dot(error1,image.T)
            
        W1, W2, B1, B2 = updateParameters(DW1, DW2, DB1, DB2, W1, W2, B1, B2, learningRate, miniBatchSize)
        testAccuracy(W1, W2, B1, B2)
