
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt

import requests
# get_ipython().run_line_magic('matplotlib', 'inline')



df = pd.read_csv("insurance_data.csv")
df.head()




from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df[['age','affordibility']],df.bought_insurance,test_size=0.2, random_state=25)





# SCALING
x_train_scaled = x_train.copy()
x_train_scaled['age'] = x_train_scaled['age'] / 100

x_test_scaled = x_test.copy()
x_test_scaled['age'] = x_test_scaled['age'] / 100



x_test_scaled['age']



model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_scaled ,y_train,epochs=5000)



model.evaluate(x_test_scaled,y_test)



# prediction
model.predict(x_test_scaled)




y_test


# now get the value of bias and weight

coef, intercept = model.get_weights()


coef , intercept


# Here , w1=5.060867, w2=1.4086502, bias =-2.9137027

# Instead of model.predict, write our own prediction function that uses w1,w2 and bias


import math
def sigmoid(x):
    return 1 / (1+ math.exp(-x))

sigmoid(5)

# FUNCTIONS
def pred_func(age,aff ):
    sums= coef[0]*age + coef[1]*aff +intercept
    return sigmoid(sums)


pred_func(.47 , 1)


# ### the goal is to come up with same w1, w2 and bias that keras model calculated. We want to show how keras/tensorflow would have computed these values internally using gradient descent

# log loss functions and sigmoid functions
# 



def sigmoid_numpy(x):
    return 1/(1+np.exp(-x))


sigmoid_numpy(np.array([12,0,1]))



def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i,epsilon) for i in y_predicted]
    y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))





def gradient_descent(age ,aff, y_true ,epochs ,loss_thresold):
    w1=w2=1
    bias=0
    rate=0.5
    n=len(age)
    
    for i in range(epochs):
        sums = w1*age +w2 *aff + bias
        y_predicted=sigmoid_numpy(sums)
        loss= log_loss(y_true,y_predicted)
        
        
        #finding derivates
        w1d= (1/n)*np.dot(np.transpose(age),(y_predicted - y_true))
        w2d= (1/n)*np.dot(np.transpose(aff),(y_predicted - y_true))
        biasd=np.mean(y_predicted-y_true)
        bias=bias-rate*biasd
        
        print(f'epoch: {i} , w1: {w1}, w2: {w2}, bias: {bias}, loss:{loss}')

        if loss>=loss_thresold:
            break
    
    
    return w1,w2,bias




gradient_descent(x_train_scaled['age'],x_train_scaled['affordibility'],y_train,1000, 0.4631)



coef , intercept


# # custom functions 


class myNN:
    def __init__(self):
        self.w1 = 1 
        self.w2 = 1
        self.bias = 0
        
    def fit(self, x, y, epochs, loss_thresold):
        self.w1, self.w2, self.bias = self.gradient_descent(x['age'],x['affordibility'],y, epochs, loss_thresold)
        print(f"Final weights and bias: w1: {self.w1}, w2: {self.w2}, bias: {self.bias}")
        
    def predict(self, x_test):
        weighted_sum = self.w1*x_test['age'] + self.w2*x_test['affordibility'] + self.bias
        return sigmoid_numpy(weighted_sum)

    def gradient_descent(self, age,affordability, y_true, epochs, loss_thresold):
        w1 = w2 = 1
        bias = 0
        rate = 0.5
        n = len(age)
        for i in range(epochs):
            weighted_sum = w1 * age + w2 * affordability + bias
            y_predicted = sigmoid_numpy(weighted_sum)
            loss = log_loss(y_true, y_predicted)
            
            w1d = (1/n)*np.dot(np.transpose(age),(y_predicted-y_true)) 
            w2d = (1/n)*np.dot(np.transpose(affordability),(y_predicted-y_true)) 

            bias_d = np.mean(y_predicted-y_true)
            w1 = w1 - rate * w1d
            w2 = w2 - rate * w2d
            bias = bias - rate * bias_d
            
            if i%50==0:
                print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
            
            if loss<=loss_thresold:
                print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
                break

        return w1, w2, bias




custommodel = myNN()
custommodel.fit(x_train_scaled,y_train ,epochs=500 ,loss_thresold=0.4631)




coef , intercept


custommodel.predict(x_test_scaled)


model.predict(x_test_scaled)

