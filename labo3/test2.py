import numpy as np
import pandas as pd

import sklearn
import sklearn.preprocessing
import sklearn.feature_selection

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from knn import run_Knn
from naiveBayes import *

from knn import *
from utils import *
from chi2 import *
if __name__ == '__main__':
    ds = pd.read_csv('adult.data', header=None, names=dataColumns, delimiter=",")

    # pre-procesamiento

    # 1. atributos numericos faltantes
    # print( dataColumns )

    # for col in columnas_numericas:
    #     # Contamos cuántos NaN son
    #     cuantasSonNan = ds[col].isna().sum()
    #     if cuantasSonNan > 0:
    #         str = "Para la columna {} Cantidad de instancias sin valor: {}".format( col, cuantasSonNan )
    #         print(str)

    #train,test = train_test_split(ds, ds.income,test_size=0.20,stratify=True)

    X = ds.drop(['income'], axis=1)
    y = ds['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    print(y_train.value_counts())
    #print(y_test.value_counts())


    ros = RandomOverSampler(random_state=42)
    X_resO, y_resO = ros.fit_resample(X_train, y_train)
    print(y_resO.value_counts())

    rus = RandomUnderSampler(random_state=42)
    X_resU, y_resU = rus.fit_resample(X_train, y_train)
    print(y_resU.value_counts())

