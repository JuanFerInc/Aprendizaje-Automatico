import pandas as pd
import numpy as np
from multiprocessing import Process
import multiprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils import *
#df tiene que ser panda dataframe
#atributos es un  arreglo con los nombres de las columnas en df sin la columna final

def prob_Eval(df, evaluacion):
    resTable = pd.DataFrame(columns=evaluacion,index=[0])
    for Vj in evaluacion:
        resTable[Vj] =  len(df[df['Evaluacion']==Vj]) / df.shape[0]#df.shape[0] es la cantidad de filas
    return resTable

#calcula bien
def probAi_dado_Vj(df,columnA,Ai,Vj,m):
    evaluacion = df.columns.values[df.shape[1] - 1]
    p = 1 / len(df.columns.unique()) # 1/ cant de posibles valores que toma el atributo
    return (len(df.loc[(df[columnA] == Ai) & (df[evaluacion] == Vj)])+p*m) / (len(df[df[evaluacion]==Vj])+m)

def buildTableAtributo(df,atributo,valoresAtributo,m):
    columnas = valoresAtributo
    filas = list(df['Evaluacion'].unique())
    tableDeAtributo = pd.DataFrame(columns=columnas,index=filas)
    for valorAtributo in columnas:
        for evaluaciones in filas:
            tableDeAtributo[valorAtributo][evaluaciones] = probAi_dado_Vj(df,atributo,valorAtributo,evaluaciones,m)
    return tableDeAtributo

def buildTableRes(df,evaluacion,atribuots,valoresDeAtribuots,m):
    df.insert(df.shape[1], 'Evaluacion', evaluacion.values, True)
    res = {}
    for atr in atribuots:
        res[atr] = buildTableAtributo(df,atr,valoresDeAtribuots[atr],m)

    evaluacioensFinales = evaluacion.unique()
    res['Evaluacion'] = prob_Eval(df,evaluacioensFinales)


    df = df.drop(['Evaluacion'], axis=1)
    return res
#instancia tiene que ser un row de panda.iterrows()
def naive_bayes(tablasProb,evaluaciones,atributos,instancia):
    posiblesValores = evaluaciones.unique()
    res = []
    e = 10**-10
    for val in posiblesValores:
        prob = tablasProb['Evaluacion'][val][0]
        for atr in atributos:
            prob*= tablasProb[atr][instancia[atr]][val]
        if(prob == 0):
            prob+=e
        res.append(prob)
    total = 1/sum(res)
    res = list(np.dot(res, total))

    return posiblesValores[ res.index(max(res))]

def valoresPosibles(X_train,X_test):
    df = pd.concat([X_train,X_test])

    valoresPosibles = {}
    for columna in list(df.columns):
        valoresPosibles[columna] = list(df[columna].unique())
    return valoresPosibles

def run_Naive_Bayes(X_train, y_train,X_test,sharedDic = None,pos = 0,m=1):
    atributos = list(X_train.columns.values)
    valoresDeAtribuots = valoresPosibles(X_train,X_test)
    tablaDeProb = buildTableRes(X_train,y_train,atributos,valoresDeAtribuots,m)

    res = []
    for iter, instancia in X_test.iterrows():
        res.append(naive_bayes(tablaDeProb,y_train,atributos, instancia))

    if sharedDic != None:
        sharedDic[pos] = res
    return res



def ValCruzada_NaiveBayes(X_train,y_train,k_Fold,m):
    particion = np.array_split(X_train,k_Fold)
    priCercano = True
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
       p = Process(target=run_Naive_Bayes, args=(trainX, trainY, testx, sharedDic,pos,m))
       p.start()
       processes.append(p)
       pos+=1
    for p in processes:
        p.join()
    classifications = []

    for x in sharedDic.keys():
        classifications.append(classification_report(sharedDic[x], testsY[x],output_dict=True))

    return promedioClassification(classifications)

def testHyperparameters_NB(X_train,y_train,k_Fold,valoresM_HPArr):
    res = {}

    for m in valoresM_HPArr:
        res[m] = ValCruzada_NaiveBayes(X_train, y_train, k_Fold,m=m)
    for key in res.keys():
        print('Valor de M-Estimador ',key)
        print('Resultado promedio de Validacion Cruzada')
        print(res[key])
        print('\n')