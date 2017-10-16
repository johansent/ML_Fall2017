# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:57:52 2017

@author: joh10
"""
import pandas
import numpy as np

def import_data(path, train1, test1, train2 = None, test2 = None, col = -1, head = 0, norm = False):
    filepath = 'C:/Users/joh10/Desktop/FSU/FA17/5635/git/Data/' + path + '/'
    #filepath = os.pardir + '\\Data\\' + path + '\\'
    
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
        
    if normalize:
        train_data, test_data = normalize(train_data, test_data)
        

    return train_data, train2_data, test_data, test2_data
    
def normalize(X, Xtest):
    npX = np.array(X)
    mns = np.mean(npX, axis = 0)
    std = np.std(npX, axis = 0)
    Xnew = (npX - mns)/std
    Xtest_new = (np.array(Xtest) - mns)/std
    Xnew = pandas.core.frame.DataFrame(Xnew)
    Xtest_new = pandas.core.frame.DataFrame(Xtest_new)
    
    return Xnew, Xtest_new    
    
    
    
    
    