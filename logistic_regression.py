#%% Libraries
import pandas as pd
#it offers data structures and operations for manipulating numerical tables and time series. 
import numpy as np
#NumPy is a python library used for working with arrays.
#It also has functions for working in domain of linear algebra, fourier transform, 
#and matrices.
import matplotlib.pyplot as plt
#Matplotlib is a comprehensive library for creating static, animated, and 
#interactive visualizations in Python.

#%%Load the dataset
df=pd.read_csv("data.csv",sep=",")
# What kind of tumor is that? 
#A tumor develops when cells reproduce too quickly.
#Tumors can vary in size from a tiny nodule to a large mass, depending on the type, 
#and they can appear almost anywhere on the body.
#There are two main types of tumor:
#Benign: These are not cancerous. They either cannot spread or grow, or they do so very slowly. = B
#Malignant: Malignant tumors are cancerous. The cells can grow and spread to other parts of the body.     = M

#%%Clean the dataset
df.drop(["Unnamed: 32", "id"],axis=1,inplace=True) # axis : all column, inplace=save in data
# For classfication, I cannot use B or M, we should convert to 1 or 0, and can be easier. ( integer value )
df.diagnosis = [1 if each == "M " else 0 for each in df.diagnosis ]

#%% X and Y
y= df.diagnosis.values
x_data = df.drop(["diagnosis"],axis=1) #the rest of them are my features 

#%% Normalization
# High values of data may override other properties and disrupt this model.
# We will scale all features to 0 - 1 
x = (x_data - np.min(x_data)) / (np.max(x_data) -  np.min(x_data)).values

#%%  Dataset train - test split
# I have my data >> Used logistic regression for training >> I have my mathmetical model
# >> But how can I test this data?
# %80 of my data for training  ---- %20 of my data for testing
# I already know my %20, it is my dataset.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42) # divide randomly but I will run this code many time
# To keep accuracy, I set to 42, same random way it will divide

# row to column 
# [  ]  4096*348 pixel
x_train = x_train.T 
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

#%% Implementing Initializing Parameters and Sigmoid Function
def initialize_weights_and_bias(dimension):    
    # I have 30 features = dimension = weight
    w = np.full((dimension,1),0.01)
    #weight value 0.01 >> 30,1 matrix     
    b = 0.0 #float bias value
    return w,b
# w,b = initialize_weights_and_bias(30)

def sigmoid(z):
    y_head = 1 / ( 1 + np.exp(-z) )
    #formula
    return y_head
#  sigmoid(0)


#%% Implementing Forward and Backward Propagation

def forward_backward_propagation(w,b,x_train,y_train):
    #x_train my features 455
    #y_train for comparing at the end for to create my loss function and then update
    #forward propagation
    # feauter * weight and sum with bias = z 
    # (30,455) * (30,1) you cannot multiply, so change weight to (1,30)
    #z = b + px1w1 + px2w2 + ... + px4096*w4096
    z = np.dot(w.T,x_train) + b
    
    y_head = sigmoid(z)
    
    # Then we calculate loss(error) function. (It says that if you make wrong prediction, loss(error) becomes big.)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    
    # Cost function is summation of all loss(error).   sum / sample = normalize
    cost = (np.sum(loss)/x_train.shape[1])  #x_train.shape[1] for scaling
    
    #backward propagation
    # we need to decrease cost because if cost is high it means that we make wrong prediction.
    # every thing starts with initializing weights and bias. Therefore cost is dependent with them.
    # In order to decrease cost, we need to update weights and bias.
    derivative_weight = (np.dot(x_train,((y_head-y_train).T))) / x_train.shape[1] 
    derivative_bias = np.sum(y_head-y_train) / x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost,gradients
    
#%% Implementing Update Parameters
# WE WILL UPDATE WEIGHT AND BIAS
# X TIMES I WILL DO FORWARD AND BACKWARD FOR UPDATE (X_TRAIN - Y_TRAIN)
# HOW FAST YOU LEARN = LEARNING RATE
# HOW MANY TIMES I WILL GO AND COME = NUMBER OF ITERATION

def update(w,b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    #derivative of weight and bias >> gradients
    
    #updating = learning paramters is number of iteration times
    for i in range(number_of_iteration):
        cost, gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        #UPDATE >> formula
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"] 
        
        #number of iteration / 10 == 0, each 10 step ( can be any number ), view
        #save all costs, and each 10 step
        
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i,cost))
    
    # how can we find the number of iteration ?
    # close to derivative of weight/bias = 0, visulazation - control
    # update ( learn ) paramaters weights and bias
    parameters = {"weight":w, "bias":b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
        
        
#%% Implementing Prediction        

def predict(w,b,x_test):
    # x_test is a input for forward propagation
    
    z= sigmoid(np.dot(w.T,x_test)+b)
    #  z = b + px1w1 + px2w2 + ... 
    
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # Y prediction matrix >> 1,114 
    # if z is bigger than 0.5, our prediction is sign one(y_head = 1 )
    # if z is smaller than 0.5, our prediction is sign zero (y_head = 0 )
    
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1
    
    return Y_prediction

#%% Implementing Logistic Regression
        
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    # initialize
    dimension = x_train.shape[0]
    w,b = initialize_weights_and_bias(dimension)
    
    parameters, gradients, cost_list = update(w,b,x_train,y_train, learning_rate, num_iterations)

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    # print test errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) *100))
    
logistic_regression(x_train,y_train,x_test,y_test,learning_rate=1,num_iterations=100)
     

#%% sklearn with LR
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))

# score > predict and then show me the accuracy





