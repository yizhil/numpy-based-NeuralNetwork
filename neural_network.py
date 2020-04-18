# -*- coding: utf-8 -*-
#block 1

import pandas as pd
from sklearn import svm
import time
import numpy as np
import arff

def sigmoid(h):
    return 1/(1+np.exp(-h))
#sizes: # of neurons in each layer, layer1 is input layer
#weights, biases: weights and b vector for each layer
#initialize weights with a standard normal distribution
def initialWeights(sizes):
    b0 = [np.random.randn(y,1) for y in sizes[1:]]
    w0 = [np.random.randn(y,x)
               for x,y in zip(sizes[:-1],sizes[1:])]
#    b0 = [np.zeros((y,1)) for y in sizes[1:]]
#    w0 = [np.zeros((y,x)) 
#               for x,y in zip(sizes[:-1],sizes[1:])]
    return w0, b0

def feedforward(x,w0,b0):
    #rtemp is a (n,1) column np-array
    rtemp = np.c_[x.copy()]
    #r stores output value in each layer
    r = [rtemp]
    #loop over each layer
    for w,b in zip(w0,b0):
        rtemp = sigmoid(np.matmul(w,rtemp)+b)
        r.append(rtemp)
    return r

def compute_error(y,r,w0):
    error = [np.zeros_like(rr) for rr in r[1:]]
    error[-1] = (1-r[-1])*(y-r[-1])*r[-1]
    #for i in range(len(error)-1, -1, -1):
    for i in range(2,len(error)+1):
        error[-i] = (1-r[-i])*np.dot(w0[-i+1].T,error[-i+1])*r[-i]
    return error

def backward_propagation(wUpdate,bUpdate,r,error):
     for i in range(len(bUpdate)):
         wUpdate[i] += error[i]*(r[i].T)
         bUpdate[i] += error[i]
     return wUpdate, bUpdate

def compute_accuracy(y_pred,y):
    return (y_pred == y).mean()


def predict_y(testData,w0,b0,activation):
    if activation == 0:
        func = sigmoid
    elif activation == 1:
        func = step_function
    elif activation == 2:
        func = rectify_linear
    elif activation == 3:
        func = hyperbolic_tangent
        
    r = testData[:,:-1].copy().T
    #loop over layers to calculate final output
    for w,b in zip(w0,b0):
        r = func(np.matmul(w,r)+b)
    
    if activation == 0:
        predict_y = (r>0.5).astype(int)
    else:
        predict_y = (r>0).astype(int)
    return predict_y

#sigmoid output
def neural_network(trainData,testData,sizes,K,epislon):
    w0, b0 = initialWeights(sizes)
    accuracy_hist = np.zeros(K)
    numData = len(trainData)
    for k in range(K):
        #y_pred = np.zeros(numData)-1
        w_update = [np.zeros((y,x)) for x,y in zip(sizes[:-1],sizes[1:])]
        b_update = [np.zeros((y,1)) for y in sizes[1:]]
        for i in range(numData):
            r = feedforward(trainData[i,:-1],w0,b0)
            #y_pred[i] = int(r[-1][0,0]>0.5)  
            error = compute_error(trainData[i,-1],r,w0)            
            w_update,b_update = backward_propagation(w_update,b_update,r,error)
        for j in range(len(sizes)-1):
            w0[j] += epislon*w_update[j]
            b0[j] += epislon*b_update[j]
        #print(y_pred)
        #if k%30 == 0:
        #    print(k,b0[0][0])
        #    print(y_pred[-5:])]
        y_pred = predict_y(testData,w0,b0,activation=0)
        accuracy_hist[k] = compute_accuracy(y_pred,testData[:,-1])
        print("Epoch {} finished".format(k))
        #print(y_pred[:10])
    return w0,b0,accuracy_hist

def neural_network_with_momentum(trainData,testData,sizes,K,epislon1,epislon2):
    # 
    w0, b0 = initialWeights(sizes)
    accuracy_hist = np.zeros(K)
    numData = len(trainData)
    w_update_curr = [np.zeros_like(w) for w in w0]
    w_update_prev = [np.zeros_like(w) for w in w0]
    b_update_curr = [np.zeros_like(b) for b in b0]
    b_update_prev = [np.zeros_like(b) for b in b0]
    for k in range(K):
       # y_pred = np.zeros(numData)-1
        
        w_update_prev = w_update_curr
        b_update_prev = b_update_curr
        
        w_update_curr = [np.zeros_like(w) for w in w0]
        b_update_curr = [np.zeros_like(b) for b in b0]
        for i in range(numData):
            r = feedforward(trainData[i,:-1],w0,b0)
            #y_pred[i] = int(r[-1][0,0]>0.5)  
            error = compute_error(trainData[i,-1],r,w0)            
            w_update_curr,b_update_curr = backward_propagation(w_update_curr,b_update_curr,r,error)
        for j in range(len(sizes)-1):
            w0[j] += epislon1*w_update_curr[j] + epislon2*w_update_prev[j]
            b0[j] += epislon1*b_update_curr[j] + epislon2*b_update_prev[j]
        print("Epoch {} finished".format(k))
        y_pred = predict_y(testData,w0,b0,activation=0)
        accuracy_hist[k] = compute_accuracy(y_pred,testData[:,-1])
    return w0,b0,accuracy_hist

#concatenate data from 5 files
def concate_data(file,numFile):
    dataAll = []
    for i in range(1,numFile+1):
        filename = str(i) + file
        data = arff.load(open(filename, 'r'))
        #dataIn = data['data']
        dataIn = data['data']
        dataAll.append(dataIn)
    return np.concatenate(dataAll)

#fill na with 0 and normalize data
#dataIn is a (n,65) list with last column as y
def clean_data(data):
    dataIn = np.array(data).astype(float)
    dataIn[np.isnan(dataIn)] = 0
    #normalize data
    dataIn[:,:-1] = (dataIn[:,:-1] - np.mean(dataIn[:,:-1],axis=0))/ \
                    np.std(dataIn[:,:-1],axis=0)
    return dataIn

def split_dataset(dataIn):
    num_ones = int(np.sum(dataIn[:,-1]))
    sortedData = np.array(sorted(dataIn.tolist(),key=lambda x:x[-1]))[-2*num_ones:]
    np.random.shuffle(sortedData)
    return sortedData[:int(1.5*num_ones)], sortedData[int(1.5*num_ones):]

def test_step_size(train,test,sizes=[64,200,300,150,1],K=500):
    epislons = [0.0001,0.001,0.01,0.1,1,10]
    acc_list = []
    df = pd.DataFrame(columns=['Last Accuracy','Max','Min','Avg','Time(min)'])
    df.index.name = 'step size'
    for epislon in epislons:
        start = time.time()
        w, b, acc = neural_network(train,test,sizes,K,epislon)
        used_time = round((time.time() - start)/60,2)
        df.loc[epislon] = [acc[-1],max(acc),min(acc),np.mean(acc),used_time]
        acc_list.append(acc)
        print(epislon," epislon finished!")
    return df, acc_list

def test_neurons(train,test,epislon=0.0001,K=500):
    sizes=[[64,100,1],[64,200,1],[64,200,300,1],
            [64,200,300,150,1],[64,300,200,150,1],[64,300,400,200,1]]
    acc_list = []
    df = pd.DataFrame(columns=['Last Accuracy','Max','Min','Avg','Time(min)'])
    df.index.name = 'Hidden layer # of neurons'
    for size in sizes:
        start = time.time()
        w, b, acc = neural_network(train,test,size,K,epislon)
        used_time = round((time.time() - start)/60,2)
        df.loc[str(size)] = [acc[-1],max(acc),min(acc),np.mean(acc),used_time]
        acc_list.append(acc)
        print(str(size)," finished!")
    return df, acc_list

#%%
#block 2
#Other improvements include neural pruning, neural competition, 
#step size, rectify linear and hyperbolic functioon 
def neural_network_with_pruning(trainData,testData,sizes,K,P,epislon):
    #prune after P literations
    w0, b0 = initialWeights(sizes)
    accuracy_hist = np.zeros(K)
    numData = len(trainData)
    for k in range(K):
        if k==P:
            removed_index = [np.argmin(np.sum(np.abs(w),axis=1)) for w in w0[:-1]]
            for i in range(len(sizes)-2):
                b0[i] = np.delete(b0[i],removed_index[i],axis=0)
                w0[i] = np.delete(w0[i],removed_index[i],axis=0)
                w0[i+1] = np.delete(w0[i+1],removed_index[i],axis=1)
        #y_pred = np.zeros(numData)-1
        w_update = [np.zeros_like(w) for w in w0]
        b_update = [np.zeros_like(b) for b in b0]
        for i in range(numData):
            r = feedforward(trainData[i,:-1],w0,b0)
            #y_pred[i] = int(r[-1][0,0]>0.5)  
            error = compute_error(trainData[i,-1],r,w0)            
            w_update,b_update = backward_propagation(w_update,b_update,r,error)
        for j in range(len(sizes)-1):
            w0[j] += epislon*w_update[j]
            b0[j] += epislon*b_update[j]
            
        y_pred = predict_y(testData,w0,b0,activation=0)
        accuracy_hist[k] = compute_accuracy(y_pred,testData[:,-1])
    return w0,b0,accuracy_hist

def neural_network_with_competition(trainData,testData,sizes,K,q,epislon):
    # incorporate q times previous output, 0<q<1
    w0, b0 = initialWeights(sizes)
    accuracy_hist = np.zeros(K)
    numData = len(trainData)
    r0 = [np.zeros((y,1)) for y in sizes]
    r_curr = [r0 for i in range(numData)]
    r_prev = [r0 for i in range(numData)]
    for k in range(K):
        #y_pred = np.zeros(numData)-1
        w_update = [np.zeros_like(w) for w in w0]
        b_update = [np.zeros_like(b) for b in b0]
        for i in range(numData):
            #r_prev[i] = [r_curr[i][x] for x in range(len(sizes))]
            r_prev[i] = r_curr[i]
            r_curr[i] = [feedforward(trainData[i,:-1],w0,b0)[x] + q*r_prev[i][x] for x in range(len(sizes))]
            #y_pred[i] = int(r_curr[i][-1][0,0]>0.5)  
            error = compute_error(trainData[i,-1],r_curr[i],w0)            
            w_update,b_update = backward_propagation(w_update,b_update,r_curr[i],error)
        for j in range(len(sizes)-1):
            w0[j] += epislon*w_update[j]
            b0[j] += epislon*b_update[j]
            
        y_pred = predict_y(testData,w0,b0,activation=0)
        accuracy_hist[k] = compute_accuracy(y_pred,testData[:,-1])
    return w0,b0,accuracy_hist

def step_function(h):
    return (h>=0).astype(int)
def rectify_linear(h):
    return np.maximum(0,h)
def hyperbolic_tangent(h):
    return (np.exp(2*h)-1) / (np.exp(2*h)+1) 
def step_function_deriv(h):
    return np.zeros_like(h)
def rectify_linear_deriv(h):
    return (h>0).astype(int)
def hyperbolic_tangent_deriv(h):
    return (1-h) * (1+h)
def feedforward_activation(x,w0,b0,activation):
    if activation == 1:
        func = step_function
    elif activation == 2:
        func = rectify_linear
    elif activation == 3:
        func = hyperbolic_tangent
    #rtemp is a (n,1) column np-array
    rtemp = np.c_[x.copy()]
    #r stores output value in each layer
    r = [rtemp]
    #loop over each layer
    for w,b in zip(w0,b0):
        rtemp = func(np.matmul(w,rtemp)+b)
        r.append(rtemp)
    return r

def compute_error_with_activation(y,r,w0,activation):
    if activation == 1:
        func_deriv = step_function_deriv
    elif activation == 2:
        func_deriv = rectify_linear_deriv
    elif activation == 3:
        func_deriv = hyperbolic_tangent_deriv
    error = [np.zeros_like(rr) for rr in r[1:]]
    #error[-1] = (1-r[-1])*(y-r[-1])*r[-1]
    error[-1] = (y-r[-1])*func_deriv(r[-1])
    #for i in range(len(error)-1, -1, -1):
    for i in range(2,len(error)+1):
        error[-i] = np.dot(w0[-i+1].T,error[-i+1])*func_deriv(r[-i])
    return error

def neural_network_with_activation(trainData,testData,sizes,K,epislon,activation):
    w0, b0 = initialWeights(sizes)
    accuracy_hist = np.zeros(K)
    numData = len(trainData)
    for k in range(K):
        #y_pred = np.zeros(numData)-1
        w_update = [np.zeros_like(w) for w in w0]
        b_update = [np.zeros_like(b) for b in b0]
        for i in range(numData):
            r = feedforward_activation(trainData[i,:-1],w0,b0,activation)
            # change threshold to 0 for these 3 functions
            #y_pred[i] = int(r[-1][0,0]>0)  
            error = compute_error_with_activation(trainData[i,-1],r,w0,activation)            
            w_update,b_update = backward_propagation(w_update,b_update,r,error)
        for j in range(len(sizes)-1):
            w0[j] += epislon*w_update[j]
            b0[j] += epislon*b_update[j]
        
        y_pred = predict_y(testData,w0,b0,activation)
        accuracy_hist[k] = compute_accuracy(y_pred,testData[:,-1])
    return w0,b0,accuracy_hist

#%%
#block 3
#load data, clean data, and split data
data = concate_data('year.arff',5)
dataIn = clean_data(data)
train,test = split_dataset(dataIn)
#np.random.shuffle(dataIn)
#sizes = np.array([64,200,300,150,1])
#sizes = np.array([64,200,300,200,1])
#ww,bb,acc = neural_network(train,test,sizes,20,0.001)
#w1,b1,acc1 = neural_network_with_pruning(train,test,sizes,50,10,0.0001)
#w2,b2,acc2 = neural_network_with_competition(dataIn[:500,:],sizes,100,0.1,0.001)
#w3,b3,acc3 = neural_network_with_momentum(train,test,sizes,10,0.001,0.0005)
#w4,b4,acc4 = neural_network_with_activation(train,test,sizes,10,0.0001,3)

#%%
#block 4
#test step size effect
df,acc_list = test_step_size(train,test)
#test neuron effect
df1,acc_list1 = test_neurons(train,test)
#test improved neural network
sizes = np.array([64,200,300,150,1])
start = time.time()
w1,b1,acc1 = neural_network_with_momentum(train,test,sizes,500,0.0001,0.00005)
used_time = round((time.time() - start)/60,2)

#%%
#block 5
#test SVM
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
error_tolerence = [0.001,0.01,0.1,1,10,100,150,200,300]
df2 = pd.DataFrame(columns = kernels)
for c in error_tolerence:
    for k in kernels:
        clf = svm.SVC(C=c,kernel=k)
        clf.fit(train[:,:-1],train[:,-1])
        pred_y = clf.predict(test[:,:-1])
        acc = (pred_y==test[:,-1]).mean()
        df2.loc[str(c),k] = acc
