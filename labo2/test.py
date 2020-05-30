import random
import time
import randomForest as rf
from datetime import datetime
from id3 import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import itertools


def test_ganancia():
    # ganancia por dedicacion deberia ser = 0.5
    g = ganancia(
        ejempos_ppts,
        attrIdxPredecir=5,  # Salva
        attr=0,  # Dedicacion
        dominioTargetFunction={'Alta', 'Baja', 'Media'}
    )
    print(g)

    # ganancia por dedicacion deberia ser = 0.0
    g = ganancia(
        ejempos_ppts,
        attrIdxPredecir=5,  # Salva
        attr=4,  # Humor
        dominioTargetFunction={'Bueno', 'Malo'}
    )
    print(g)


def id3_test(ejemplos):
    l = len(ejemplos.columns) - 1  # 1024
    attrIdxPredecir = l
    atributos = list(range(l))  # [0, 1, 2, .. 1023])
    res = crearArbol(ejemplos, attrIdxPredecir, atributos)
    return res


def unique_test(ejemplos, attrIdxPredecir):
    # viejo algoritmo de obtenerValores:
    resViejo = []
    for i, row in ejemplos.iterrows():
        val = row[attrIdxPredecir]
        if val not in resViejo:
            resViejo.append(val)
    # fin viejo

    # viejo deberia ser igual a pandas:
    resNuevo = obtenerValores(ejemplos, attrIdxPredecir)
    print(resViejo, resNuevo)
    assert resViejo == resNuevo


def entropia_VIEJO(S, attrIdxPredecir, dominioTargetFunction=['Negativo', 'Positivo']):
    res = 0
    for posibleValor in dominioTargetFunction:
        cant = count(S, attrIdxPredecir, posibleValor)
        p = cant / len(S)
        if p != 0:
            res -= p * math.log(p, 2)
    return res


def test_entropia():
    ejemplos = pd.DataFrame(mini_dataset)
    e_viejo = entropia_VIEJO(ejemplos, attrIdxPredecir=2, dominioTargetFunction={False, True})
    e_nuevo = entropia(ejemplos, attrIdxPredecir=2, dominioTargetFunction={False, True})
    print(e_viejo, e_nuevo)
    assert e_viejo == e_nuevo


mini_dataset = [
    #   0         1       2
    ['Soleado', 'Noche', True],
    ['Soleado', 'Dia', True],
    ['Nublado', 'Noche', False],
    ['Nublado', 'Noche', True],
    ['Lluvioso', 'Noche', True],
    ['Lluvioso', 'Dia', True]
]

ejempos_ppts = [
    # Dedicación Dificultad Horario Humedad Humor Doc Salva
    #  0        1          2         3      4       5
    ['Alta', 'Alta', 'Nocturno', 'Media', 'Bueno', 'Positivo'],
    ['Baja', 'Media', 'Matutino', 'Alta', 'Malo', 'Negativo'],
    ['Media', 'Alta', 'Nocturno', 'Media', 'Malo', 'Positivo'],
    ['Media', 'Alta', 'Matutino', 'Alta', 'Bueno', 'Negativo'],
]


def masComun_test(ejemplos, attrIdxPredecir):
    res_viejo = calcResultadoMasComun(ejemplos, attrIdxPredecir)
    res_nuevo = calcResultadoMasComun(ejemplos, attrIdxPredecir)
    print(res_viejo, res_nuevo)
    assert res_viejo == res_nuevo


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


def sample_test():
    data = [11, 22, 33, 44, 55, 66, 77, 88, 99]
    df = pd.DataFrame(data)
    print(df.sample(10, replace=True))


def mean_test():
    l = [[10, 100], [20, 200]]
    # l = [['positive', 'negative'],['positive', 'positive']]
    # res = pd.Series(l).mean(axis=0)
    # res = np.mean(l, axis=0) # axis 0 ~ por columnas
    print(res)


def max_test():
    votacion = {'positivos': 22, 'negativos': 11}
    res = max(votacion, key=lambda key: votacion[key])
    print(res)


def f(s):
    if s == "<class 'int'>":
        return ""
    if s == "<class 'numpy.int64'>":
        return "numpy"
    if s == "<class 'str'>":
        return ""


def get_atrs(arbol):
    if type(arbol) != Arbol:
        return [str(type(arbol))]

    izq = get_atrs(arbol.ramas[0])
    der = get_atrs(arbol.ramas[1])

    return list(set().union(izq, der))


def print_arbol(arbol, ident):
    t = type(arbol)
    if t != Arbol:
        print(' ' * ident + f(str(t)) + str(arbol))
    else:
        d = arbol.data
        print(' ' * ident + f(str(type(d))) + str(d) + '*')
        print_arbol(arbol.ramas[0], ident + 1)
        print_arbol(arbol.ramas[1], ident + 1)


def debugger():
    data = ''' {"columns":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],"index":[1,11,11,10,6,10,6,14,6,7,0,6],"data":[[0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,"negative"],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,"positive"],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,"positive"],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"],[0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,"negative"],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"]]}  '''
    muestra = pd.read_json(data, orient='split')
    arbol = rf.crearArbol(muestra, numAtributos=10)

    atrs = get_atrs(arbol)
    if len(atrs) > 1:
        print_arbol(arbol, 0)
        print(atrs)


def debugger2():
    data = ''' {"columns":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],"index":[1,11,11,10,6,10,6,14,6,7,0,6],"data":[[0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,"negative"],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,"positive"],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,"positive"],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"],[0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,"negative"],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"negative"]]}  '''
    muestra = pd.read_json(data, orient='split')

    l = len(muestra.columns) - 1
    attrIdxPredecir = l
    atributos = list(range(l))
    random.shuffle(atributos)
    print(attrIdxPredecir, atributos)
    arbol = crearArbol(muestra, attrIdxPredecir, atributos)

    atrs = get_atrs(arbol)
    if len(atrs) > 1:
        print_arbol(arbol, 0)
        print(atrs)

def cargaDatos():
    # datos
    input_file = "qsar_oral_toxicity_small.csv"
    dataset = pd.read_csv(input_file, header=None, delimiter=";")
    l = len(dataset.columns) - 1
    y = dataset[l]  # la columna con c(x) ~ la target function

    # 1 = 0.9998
    # 8  = 0.999
    X, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.20)  # 20% test y 80% training
    X.reset_index(drop=True, inplace=True)
    return X, X_test, y_train, y_test


def prueba_id3(X, X_test, y_train, y_test):
    n = len(X.index)
    c = Cronometro(n)

    arbol = id3_test(X)

    c.detener()

    #X_test = X_test.drop(len(X.index), axis=1)
    y_pred = predecir(arbol, X_test)        
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def prueba_randomForest(X, X_test, y_train, y_test):
    tamanioMuestra = len(X.index)   
    
    c = Cronometro(200)
    arboles = rf.crearRandomForest_Wrapper(X, tamanioMuestra, numArboles=100, numAtributos=10, conPrints=True, numOfThreads=4)

    for i, arbol in enumerate(arboles):
        atrs = get_atrs(arbol)
        if len(atrs) > 1:
            print_arbol(arbol, 0)
            print('para el arbol:', i, atrs)
            break

    y_pred = rf.predecirRandomForest(arboles, X_test)
    c.detener()

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    

    
def cross_val_score_random_forest(X, y, tamanioMuestra, numArboles, numAtributos, n_splits, estado_aleatorio):
    scores = []
    cv = KFold(n_splits, random_state=estado_aleatorio, shuffle=False)
    for train_index, test_index in cv.split(X):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)

        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        arboles = rf.crearRandomForest_Wrapper(X, tamanioMuestra, numArboles, numAtributos, conPrints=True, numOfThreads=4)
        y_pred = rf.predecirRandomForest(arboles, X_test)
        scores.append(accuracy_score(y_test, y_pred))
    return np.mean(scores)
    

def prueba_randomForest_conCV(X, X_test, y_train, y_test):

    # corto el dataset
    mini_X, _ = np.split(X, [int(0.2 * len(X))])
    mini_Y, _ = np.split(y_train, [int(0.2 * len(X))])

    tamanioMuestra, numArboles, numAtributos = kfoldCV_randomForest(X, y_train)
    arboles = rf.crearRandomForest_Wrapper(X, tamanioMuestra, numArboles, numAtributos, conPrints=True, numOfThreads=4)

    y_pred = rf.predecirRandomForest(arboles, X_test)
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def prueba_grl():
    import sys
    sys.setrecursionlimit(1500)

    X, X_test, y_train, y_test = cargaDatos()

    # ################ parte a: id3 (descomentar para activar)

    # prueba_id3(X, X_test, y_train, y_test)

    # ################ parte b: random forest (descomentar para activar)

    # prueba_randomForest(X, X_test, y_train, y_test)

    prueba_randomForest_conCV(X, X_test, y_train, y_test)


def kfoldCV_randomForest(X, y):
    scores = []
    tamanioMuestra = [len(X.index) // 2, len(X.index) // 3]
    numArboles = [50]
    numAtributos = [10]
    a = [tamanioMuestra, numArboles, numAtributos]
    hyperparametros = list(itertools.product(*a))

    print(hyperparametros)

    for hyperpar in hyperparametros:
        score = cross_val_score_random_forest(X, y, hyperpar[0], hyperpar[1], hyperpar[2], n_splits=2,
                                              estado_aleatorio=42)
        scores.append(score)
        print(hyperpar, " - ", score)

    i = np.argmax(score)

    print('La mejor combinacion de hypers fue:', hyperparametros[i])

    return hyperparametros[i]


if __name__ == "__main__":
    # debugger()
    # debugger2()

    start_time = time.time()
    prueba_grl()
    print("TIEMPO DE EJECUCION --- %s seconds ---" % (time.time() - start_time))


