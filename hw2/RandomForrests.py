# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 10:02:36 2017

@author: Kyle
"""


from import_data import import_data
from sklearn.ensemble import RandomForestClassifier as RFC
import matplotlib.pyplot as plt
import sklearn
import numpy as np

def GrowForrest(x,y,xtest,ytest,K):
    errorsTrain = []
    errorsTest = []
    for k in K:
        clf = RFC(n_estimators = k)
        clf = clf.fit(X, np.ravel(Y))
    
        error = 1 - clf.score(X, Y)
        errorsTrain.append(error)
    
        error = 1-clf.score(Xtest, Ytest)
        errorsTest.append(error)
        
    return errorsTest , errorsTrain

def Plot(x,y1,y2,title,legendLoc = 1):
    plt.title(title)
    plt.plot(x,y1,label = 'Test Error')
    plt.plot(x,y2,label = 'Training Error')
    plt.legend(loc = legendLoc)
    plt.show()
    
    
min_table = {}
K = [3, 10, 30, 100, 300]

#MADELON TEST

X, Y, Xtest, Ytest = import_data('madelon', 'madelon_train.data', 'madelon_valid.data','madelon_train.labels', 'madelon_valid.labels', head = None)
X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)  #conner.xyz  at https://stackoverflow.com/questions/20517650/how-to-delete-the-last-column-of-data-of-a-pandas-dataframe

eTest, eTrain = GrowForrest(X, Y, Xtest, Ytest,K)
Plot(K,eTest,eTrain,'Madelon')

min_table['Madelon     '] = [min(eTest), K[np.argmin(eTest)]]

#SATIMAGE TEST
X, Y, Xtest, Ytest = import_data('satimage', 'X.dat', 'Xtest.dat','Y.dat', 'Ytest.dat', head = None)
eTest, eTrain = GrowForrest(X, Y, Xtest, Ytest,K)
Plot(K,eTest,eTrain,'Satimage')

min_table['Satimage    '] = [min(eTest), K[np.argmin(eTest)]]

# POKER TEST
X, Y, Xtest, Ytest = import_data('poker', 'X.dat', 'Xtest.dat','Y.dat', 'Ytest.dat', head = None)
eTest, eTrain = GrowForrest(X, Y, Xtest, Ytest,K)
Plot(K,eTest,eTrain,'Poker')

min_table['Poker       '] = [min(eTest), K[np.argmin(eTest)]]

#HILL/VALLEY Test
X, Y, Xtest, Ytest = import_data('hill_valley', 'X.dat', 'Xtest.dat','Y.dat', 'Ytest.dat', head = None)
eTest, eTrain = GrowForrest(X, Y, Xtest, Ytest,K)
Plot(K,eTest,eTrain,'Hill Valley')

min_table['Hill/Valley '] = [min(eTest), K[np.argmin(eTest)]]

#Arcene Test
X, Y, Xtest, Ytest = import_data('arcene', 'arcene_train.data', 'arcene_valid.data','arcene_train.labels', 'arcene_valid.labels', head = None)
X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)

eTest, eTrain = GrowForrest(X, Y, Xtest, Ytest,K)
Plot(K,eTest,eTrain,'Arcene')

min_table['Arcene      '] = [min(eTest), K[np.argmin(eTest)]]

#Summary Of Results
print('{0:12s} {1:15s} {2:12s}'.format('Data name', 'Minimum Value', '# of Trees'))
print('---------------------------------------')
for k in min_table.keys():
    print('{0:12s} {1:13f} {2:12d}'.format(k, min_table[k][0], min_table[k][1]))
