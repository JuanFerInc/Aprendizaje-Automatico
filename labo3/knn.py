import numpy as np
from multiprocessing import Process
import multiprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils import *
def distnacia_Euclidiana(df,instancia):
    #np.subtrace hace Matrix - Vector, resta el vector en cada fila de la matriz
    #np.square y np.sqrt realiza la operacion por cada posicion ij
    #np.sum(matrix,axis=1) da la duma de cada fila como arreglo
    return np.sqrt(np.sum(np.square( np.subtract(df.values, instancia.values)) ,axis=1))

#df es panda DataFrame con todos los datos
#K es la cantidad de vecinos a tomar en cuenta, Mejor usar impar, Valoeres comunes 1,3,5
#instancia es un Dataframe row que se utiliza para calcular el resultado
def Knn(df,evaluacion,k,instancia,prioCercano = True):
    e = (10**-10)
    posiblesValores = list(df[df.columns.values[df.shape[1] - 1]].unique())
    dist = distnacia_Euclidiana(df,instancia)
    ksmallest = dict(zip(range(0,len(dist)),dist))
    ksmallestIndex = np.argpartition(dist, k)[:k]

    if prioCercano:
        res = [0]*len(posiblesValores)
        for key in ksmallestIndex:
            eval = evaluacion.iloc[key]
            res[posiblesValores.index(eval)] += 1/((dist[key]**2)+e)
        return posiblesValores[res.index(max(res))]
    else:
        res = evaluacion.iloc[ksmallestIndex].value_counts().index[0]
        return res

def run_Knn(X_train, y_train,X_test,cantVecinos,prioCercano,sharedDic = None,pos = 0):
    res = []
    for iter, instancia in X_test.iterrows():
        res.append(Knn(X_train, y_train, cantVecinos, instancia, prioCercano))
    if sharedDic != None:
        sharedDic[pos] = res
    return res

def ValCruzada_Knn(X_train,y_train,k_Fold,cantVecinos = 4,priCercano = True):
    particion = np.array_split(X_train,k_Fold)
    manager = multiprocessing.Manager()
    sharedDic = manager.dict()
    processes = []
    pos = 0
    testsY = []
    for testx in particion:
       trainX = X_train.drop(list(testx.index))
       trainY = y_train.drop(list(testx.index))
       testY = y_train[testx.index]
       testsY.append(testY)
       p = Process(target=run_Knn, args=(trainX, trainY, testx, cantVecinos, priCercano, sharedDic,pos))
       p.start()
       processes.append(p)
       pos+=1
    for p in processes:
        p.join()
    classifications = []
    for x in sharedDic.keys():
        classifications.append(classification_report(sharedDic[x], testsY[x],output_dict=True))

    return promedioClassification(classifications)



def testHyperparameters_Knn(X_train,y_train,k_Fold,cantVecinos_HPArr,consDistancia=False):
    res = {}
    for cantVecinos in cantVecinos_HPArr:
        res['Sin Considerar distancia con ' + str(cantVecinos) + ' vecinos '] = ValCruzada_Knn(X_train,y_train,k_Fold,cantVecinos = cantVecinos,priCercano =False)
        if consDistancia:
            res['Considerar distancia con ' + str(cantVecinos) + ' vecinos '] = ValCruzada_Knn(X_train, y_train, k_Fold, cantVecinos = cantVecinos,priCercano =True)
    for key in res.keys():
        print(key)
        print('Resultado promedio de Validacion Cruzada')
        print(res[key])
        print('\n')