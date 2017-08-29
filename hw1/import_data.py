# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:57:52 2017

@author: joh10
"""
import pandas

def import_data(train1, test1, train2 = None, test2 = None):
    filepath = 'C:/Users/joh10/Desktop/FSU/FA17/5635/hw1/'
    
### NOTE: add header, otherwise we miss first row    
    
    train_data = pandas.read_table(filepath + train1, sep = ' ')
    if train2 != None:
        train2_data = pandas.read_table(filepath + train2, sep = ' ')
    #else:
        # separate training data, if not already
        

    
    test_data = pandas.read_table(filepath + test1, sep = ' ')
    if test2 != None:
        test2_data = pandas.read_table(filepath + test2, sep = ' ')
    #else:
        ## separate test data, if not already

    return train_data, train2_data, test_data, test2_data