# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:21:11 2017

@author: Kyle
"""

from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import FSAmult

sys.path.append(os.path.abspath('../'))
from import_data import import_data


def Analysis(X,Y,Xtest,Ytest,learnRate, name, min_table):
    niter = 100
    testErrors = []
    trainErrors = []
    K = [5,9,18]#,27,36]
    for k in K:
        print(k)
        y1, y2, loss = FSAmult.TrainWeights(X,Y,Xtest,Ytest,niter,k,learnRate)
        testErrors.append(y1[-1])
        trainErrors.append(y2[-1])
        
    
        if k == 9:
            FSAmult.Plot(range(niter), loss, None, title = name + ' Loss At k = ' + str(k), labels = ['iteration count', 'Loss'])
        min_table[name + ' ' + str(k)] = [y1[-1], y2[-1]]
    
    FSAmult.Plot(K, testErrors, trainErrors, name + ' min Errors at k', labels = ['k','Misclassification Error'])
    
    return min_table


### Script
import timeit
start = timeit.default_timer()
min_table = {}

#Satimage Data
X, Y, Xtest, Ytest = import_data('satimage', 'X.dat', 'Xtest.dat', 'Y.dat', 'Ytest.dat', head = None, norm = True)
#print(type(Y))
min_Table = Analysis(X,Y,Xtest,Ytest,.001, 'Satimage', min_table)

#covType Data
#X, Y, Xtest, Ytest = import_data('covtype', 'covtype.data', head = None, norm = True, sep = ',', testIndex = 15120)




##Gisette Data
#    
##Read in the Data
#X, Y, Xtest, Ytest = import_data('gisette', 'gisette_train.data', 'gisette_valid.data','gisette_train.labels', 'gisette_valid.labels', head = None, norm = True, removeCol = True)
#min_table = Analysis(X,Y,Xtest,Ytest,.0001, 'Gisette', min_table)
#

##Arcene Test
#X, Y, Xtest, Ytest = import_data('arcene', 'arcene_train.data', 'arcene_valid.data','arcene_train.labels', 'arcene_valid.labels', head = None, norm = True, removeCol = True)
#min_table = Analysis(X,Y,Xtest,Ytest,.01, 'Arcene', min_table)


## Madelon
#X, Y, Xtest, Ytest = import_data('madelon', 'madelon_train.data', 'madelon_valid.data','madelon_train.labels', 'madelon_valid.labels', head = None, norm = True, removeCol = True)
#min_table = Analysis(X,Y,Xtest,Ytest,.0001, 'Madelon', min_table)


## Hill/Valley
#X, Y, Xtest, Ytest = import_data('hill_valley', 'X.dat', 'Xtest.dat','Y.dat', 'Ytest.dat', head = None, norm = True)

#Summary Of Results
print('{0:17s} {1:12s} {2:12s}'.format('Data name', 'Test Error', 'Training Error'))
print('---------------------------------------')
for k in min_table.keys():
    print('{0:12s} {1:13f} {2:12f}'.format(k, min_table[k][0], min_table[k][1]))
    
stop = timeit.default_timer()
print(stop - start)

     
     

    
    

