# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:57:52 2017

@author: joh10
"""
import pandas
import numpy as np
import os

def import_data(path, train1, test1, train2 = None, test2 = None, removeCol = False, head = 0, norm = False):
    #filepath = 'C:/Users/joh10/Desktop/FSU/FA17/5635/git/Data/' + path + '/'
    filepath = os.pardir + '\\Data\\' + path + '\\'
    
### NOTE: add header, otherwise we miss first row    
    train_data = pandas.read_table(filepath + train1, sep = ' ', header = head)
    if train2 != None:
        train2_data = pandas.read_table(filepath + train2, sep = ' ', header = head)
    #else:
        # separate training data, if not already
        
        

    
    test_data = pandas.read_table(filepath + test1, sep = ' ', header = head)
    if test2 != None:
        test2_data = pandas.read_table(filepath + test2, sep = ' ', header = head)
    #else:
        ## separate test data, if not already
        
    if removeCol:
        train_data.drop(train_data.columns[len(train_data.columns)-1], axis=1, inplace=True)
        test_data.drop(test_data.columns[len(test_data.columns)-1], axis=1, inplace=True)
        
    if normalize:
        train_data, test_data = normalize(train_data, test_data)
        

    return train_data, train2_data, test_data, test2_data
    
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
    
    
    
    
    