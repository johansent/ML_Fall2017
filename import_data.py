# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:57:52 2017

@author: joh10
"""
import pandas
import numpy as np
import os

#'X.dat', 'Xtest.dat', 'Y.dat', 'Ytest.dat'
def import_data(path, train_data, test_data = None, train_labels = None, test_labels = None, removeCol = False, head = 0, norm = False, sep = ' ', testIndex = 0):
    #filepath = 'C:/Users/joh10/Desktop/FSU/FA17/5635/git/Data/' + path + '/'
    filepath = os.pardir + '\\Data\\' + path + '\\'
    
### NOTE: add header, otherwise we miss first row    
    X = pandas.read_table(filepath + train_data, sep = sep, header = head)
    if train_labels != None:
        Y = pandas.read_table(filepath + train_labels, sep = sep, header = head)
    else:
        # separate training data, if not already
        Y = X.loc[:,len(X.columns)-1]
        X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
        
        

    if test_data != None:
        Xtest = pandas.read_table(filepath + test_data, sep = sep, header = head)
        if test_labels != None:
            Ytest = pandas.read_table(filepath + test_labels, sep = sep, header = head)
        else:
            # separate test data, if not already
            Ytest = Xtest.loc[:,len(Xtest.columns)-1]
            Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)
    else:
        #take part of the training and put it in the testing
        Xtest = X[testIndex:X.shape[0]]
        X = X[0:testIndex]
        Ytest = Y[testIndex:Y.shape[0]]
        Y = Y[0:testIndex]
        
        
        
    
        
    if removeCol:
        #conner.xyz  at https://stackoverflow.com/questions/20517650/how-to-delete-the-last-column-of-data-of-a-pandas-dataframe
        X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
        Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)
        
    if normalize:
        X, Xtest = normalize(X, Xtest)
        

    return X, Y, Xtest, Ytest
    
def normalize(X, Xtest):
    npX = np.array(X)
    npXt = np.array(Xtest)
    std = np.std(npX, axis = 0)
    zeros = np.array(np.where(std == 0))
#    print('zeros', np.shape(zeros), zeros)
    zeros.sort()
    zeros = zeros[::-1]
#    print('zeros', np.shape(zeros), zeros)
#    print(std[zeros[0][0]])
    N = len(zeros[0])
    for i in range(N):
        #print(zeros[0][i])
#        print(i)
        npX = np.delete(npX,zeros[0][N-i-1] , 1)
        npXt = np.delete(npXt,zeros[0][N-i-1],1)
        std = np.delete(std, zeros[0][N-i-1])
    mns = np.mean(npX, axis = 0)
    Xnew = (npX - mns)/std
    Xtest_new = (npXt - mns)/std
    Xnew = pandas.core.frame.DataFrame(Xnew)
    Xtest_new = pandas.core.frame.DataFrame(Xtest_new)
    #print(Xnew)
    
    #print(np.shape(npX), np.shape(npXt))
    
    return Xnew, Xtest_new    
    
    
    
    
    