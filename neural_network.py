import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat=loadmat("ex3data1.mat")
X=mat["X"]
y=mat["y"]


mat2=loadmat("ex3weights.mat")
Theta1=mat2["Theta1"] # Theta1 has size 25 x 401
Theta2=mat2["Theta2"] # Theta2 has size 10 x 26

#print(Theta1)
def sigmoid(z):
	#return the sigmoid of z
    
    return 1/ (1 + np.exp(-z))

def predict(Theta1, Theta2, X):
    """
    Predict the label of an input given a trained neural network
    """
    m= X.shape[0]
    X = np.hstack((np.ones((m,1)),X))
    
    a1 = sigmoid(np.dot(X,Theta1.T))
    a1 = np.hstack((np.ones((m,1)), a1)) # hidden layer
    a2 = sigmoid(np.dot(a1,Theta2.T)) # output layer
    
    return np.argmax(a2,axis=1)+1

pred2 = predict(Theta1, Theta2, X)

#print(pred2) 
#print (y)
#print (pred2[:,np.newaxis])
#print(sum(pred2[:,np.newaxis]==y)[0])

print("Training Set Accuracy:",sum(pred2[:,np.newaxis]==y)[0]/5000*100,"%")