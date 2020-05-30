import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

from utils import *


def ValCruzada_Knn_sklearn(X, y, k_Fold, cantVecinos = 5):
    clf = KNeighborsClassifier(n_neighbors = cantVecinos)
    kf = StratifiedKFold(n_splits = k_Fold)
    
    cr = []

    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #print(X_train, X_test)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cr.append(classification_report(y_test, y_pred,output_dict=True))
        
    return promedioClassification(cr)

def testHyperparameters_Knn_sklearn(X_train,y_train,k_Fold,cantVecinos_HPArr):
    res = {}
    for cantVecinos in cantVecinos_HPArr:
        res[cantVecinos] = ValCruzada_Knn_sklearn(X_train,y_train,k_Fold,cantVecinos = cantVecinos)
    for key in res.keys():
        print('Cantidad de vecinos ',key)
        print('Resultado promedio de Validacion Cruzada')
        print(res[key])
        print('\n')
        
def ValCruzada_NB_sklearn(X, y, k_Fold):
    clf = MultinomialNB()
    kf = StratifiedKFold(n_splits = k_Fold)
    
    cr = []

    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #print(X_train, X_test)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cr.append(classification_report(y_test, y_pred,output_dict=True))
        
    return promedioClassification(cr)

def testHyperparameters_NB_sklearn(X_train,y_train,k_Fold):

    res = ValCruzada_NB_sklearn(X_train,y_train,k_Fold)

    print('Resultado promedio de Validacion Cruzada')
    print(res)
    print('\n')
        