import randomForest as rf
import sys
import time
from datetime import datetime
from id3 import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

class Cronometro():
    def __init__(self, size):
        self.ini = time.time()
        delta = self.estimarTiempo(size)
        fin = self.ini + delta
        deltaMin = round(delta / 60)
        finStr = datetime.fromtimestamp(fin).strftime("%H:%M")
        print('Termina a las:', finStr, '({} min aprox)'.format(deltaMin))

    # retorna en segundos el tiempo estimado restante
    def estimarTiempo(self, lenEjemplos):
        # suponiendo linealidad
        # sabiendo que 619 s tardo con 99% ~ 899
        # regla de 3
        return round(lenEjemplos * 619 / 899)

    def detener(self):
        elapsed_time = time.time() - self.ini
        print("Total time in seconds =", round(elapsed_time / 60), 'minutos')
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)

def get_atrs(arbol):
    if type(arbol) != Arbol:
        return [str(type(arbol))]

    izq = get_atrs(arbol.ramas[0])
    der = get_atrs(arbol.ramas[1])

    return list(set().union(izq, der))

def cargaDatos():
    # datos
    input_file = "qsar_oral_toxicity.csv"
    dataset = pd.read_csv(input_file, header=None, delimiter=";")
    l = len(dataset.columns) - 1
    y = dataset[l]  # la columna con c(x) ~ la target function

    # 1 = 0.9998
    # 8  = 0.999
    X, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.20)  # 20% test y 80% training
    X.reset_index(drop=True, inplace=True)
    return X, X_test, y_train, y_test

def id3_test(ejemplos):
    l = len(ejemplos.columns) - 1  # 1024
    attrIdxPredecir = l
    atributos = list(range(l))  # [0, 1, 2, .. 1023])
    res = crearArbol(ejemplos, attrIdxPredecir, atributos)
    return res

def kfoldCV_randomForest(X, y):
    scores = []
    tamanioMuestra = [len(X.index) // 2, len(X.index) // 3]
    numArboles = [100]
    numAtributos = [10, 20]
    a = [tamanioMuestra, numArboles, numAtributos]
    hyperparametros = list(itertools.product(*a))

    for hyperpar in hyperparametros:
        score = cross_val_score_random_forest(X, y, hyperpar[0], hyperpar[1], hyperpar[2], n_splits=2,
                                              estado_aleatorio=42)
        scores.append(score)
        print(hyperpar, " - ", score)

    i = np.argmax(score)
    return hyperparametros[i]

def cross_val_score_random_forest(X, y, proporcionMuestra, numArboles, numAtributos, n_splits, estado_aleatorio):
    scores = []
    cv = KFold(n_splits, random_state=estado_aleatorio, shuffle=False)
    for train_index, test_index in cv.split(X):
        # print("Train Index: ", train_index, "\n")
        # print("Test Index: ", test_index)

        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[
            test_index]
        arboles = rf.crearRandomForest_Wrapper(X, proporcionMuestra, numArboles, numAtributos, conPrints=True, numOfThreads=4)
        y_pred = rf.predecirRandomForest(arboles, X_test)
        scores.append(f1_score(y_test, y_pred, average='macro'))
    return np.mean(scores)

def prueba_randomForest_conCV(X, X_test, y_train, y_test):
    tamanioMuestra, numArboles, numAtributos = kfoldCV_randomForest(X, y_train)
    arboles = rf.crearRandomForest_Wrapper(X, tamanioMuestra, numArboles, numAtributos, conPrints=True, numOfThreads=4)

    y_pred = rf.predecirRandomForest(arboles, X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def id3_setup_and_run(X, X_test, y_train, y_test):
    n = len(X.index)
    c = Cronometro(n)

    arbol = id3_test(X)

    c.detener()

    # X_test = X_test.drop(len(X.index), axis=1)
    y_pred = predecir(arbol, X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def randomForest_setup_and_run(X, X_test, y_train, y_test):
    tamanioMuestra = len(X.index)

    c = Cronometro(200)
    arboles = rf.crearRandomForest_Wrapper(X, tamanioMuestra, numArboles=100, numAtributos=10, conPrints=True, numOfThreads=4)

    # for i, arbol in enumerate(arboles):
    #     atrs = get_atrs(arbol)
    #     if len(atrs) > 1:
    #         print_arbol(arbol, 0)
    #         print('para el arbol:', i, atrs)
    #         break

    y_pred = rf.predecirRandomForest(arboles, X_test)
    c.detener()

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def runID3():
    sys.setrecursionlimit(1500)

    X, X_test, y_train, y_test = cargaDatos()


    id3_setup_and_run(X, X_test, y_train, y_test)

def runRandomForest():
    sys.setrecursionlimit(1500)

    X, X_test, y_train, y_test = cargaDatos()

    randomForest_setup_and_run(X, X_test, y_train, y_test)


def val_score_random_forest(X_train, X_test, y_train, y_test, proporcionMuestra, numArboles, numAtributos):
    arboles = rf.crearRandomForest_Wrapper(X_train, proporcionMuestra, numArboles, numAtributos, conPrints=True, numOfThreads=4)
    y_pred = rf.predecirRandomForest(arboles, X_test)
    return f1_score(y_test, y_pred, average='macro')
