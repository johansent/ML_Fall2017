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
import FSA

sys.path.append(os.path.abspath('../'))
from import_data import import_data


def Analysis(X,Y,Xtest,Ytest,learnRate, name, min_table):
    niter = 500
    testErrors = []
    trainErrors = []
    K = [10,30,100,300]
    for k in K:
        y1, y2, loss = FSA.TrainWeights(X,Y,Xtest,Ytest,niter,k,learnRate)
        testErrors.append(y1[-1])
        trainErrors.append(y2[-1])
    
        if k == 10:
            FSA.Plot(range(niter), loss, None, title = name + ' Loss At k = 10', labels = ['iteration count', 'Loss'])
        min_table[name + ' ' + str(k)] = [y1[-1], y2[-1]]
    
    FSA.Plot(K, testErrors, trainErrors, name + ' min Errors at k', labels = ['k','Misclassification Error'])
    
    return min_table


### Script
import timeit
start = timeit.default_timer()
min_table = {}
#
##Gisette Data
#    
##Read in the Data
X, Y, Xtest, Ytest = import_data('gisette', 'gisette_train.data', 'gisette_valid.data','gisette_train.labels', 'gisette_valid.labels', head = None, norm = True, removeCol = True)
min_table = Analysis(X,Y,Xtest,Ytest,.0001, 'Gisette', min_table)
#
#niter = 500
#testErrors = []
#trainErrors = []
#K = [10,30,100,300]
#for k in K:
#    y1, y2 = FSA.TrainWeights(X,Y,Xtest,Ytest,niter,100,.0001)
#    testErrors.append(min(y1))
#    trainErrors.append(min(y2))
#    
#    if k == 10:
#        FSA.Plot(range(niter), y1, y2, 'Gisette Errors At k = 10')
#    min_table['Gisette' + str(k)] = [min(y1), min(y2)]
#    
#FSA.Plot(K, testErrors, trainErrors, 'Gisette min Errors at k')


#Arcene Test
#X, Y, Xtest, Ytest = import_data('arcene', 'arcene_train.data', 'arcene_valid.data','arcene_train.labels', 'arcene_valid.labels', head = None, norm = True, removeCol = True)
#
#min_table = Analysis(X,Y,Xtest,Ytest,.01, 'Arcene', min_table)


#niter = 500
#y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter,20, .001)    
#
#Plot(range(niter), y1, y2, 'Arcene Errors')
##Plot(K,eTest,eTrain,'Arcene')
#min_table['Arcene'] = [min(y1), min(y2)]

## Madelon
#X, Y, Xtest, Ytest = import_data('madelon', 'madelon_train.data', 'madelon_valid.data','madelon_train.labels', 'madelon_valid.labels', head = None, norm = True, removeCol = True)
#min_table = Analysis(X,Y,Xtest,Ytest,.0001, 'Madelon', min_table)
#
#niter = 500
#y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter, 10, .001)
#
#Plot(range(niter), y1, y2, 'Madelon Errors', 3)
#min_table['Madelon'] = [min(y1), min(y2)]

#
## Hill/Valley
#X, Y, Xtest, Ytest = import_data('hill_valley', 'X.dat', 'Xtest.dat','Y.dat', 'Ytest.dat', head = None, norm = True)
#
#niter = 50
#y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter, 10, .01)
#
#Plot(range(niter), y1, y2, 'Hill/Valley Errors', 3)
#min_table['Hill/Valley'] = [min(y1), min(y2)]


#Summary Of Results
print('{0:17s} {1:12s} {2:12s}'.format('Data name', 'Test Error', 'Training Error'))
print('---------------------------------------')
for k in min_table.keys():
    print('{0:12s} {1:13f} {2:12f}'.format(k, min_table[k][0], min_table[k][1]))
    
stop = timeit.default_timer()
print(stop - start)

     
     

    
    

