from numpy import log2 as log
import math
import numpy as np
import time
"""
ID3(Examples, Targetattribute, Attributes)

    # Examples are the training examples. Targetattribute is the attribute whose value is to be
    # predicted by the tree. Attributes is a list of other attributes that may be tested by the learned
    # decision tree. Returns a decision tree that correctly classifies the given Examples.

    - Create a Root node for the tree
    - If all Examples are positive, Return the single-node tree Root, with label = +
    - If all Examples are negative, Return the single-node tree Root, with label = -
    - If Attributes is empty, Return the single-node tree Root, with label = most common value of Targetattribute in Examples

    - Otherwise Begin
        - A <- the attribute from Attributes that best* classifies Examples
        - The decision attribute for Root <- A
        - For each possible value, vi, of A,
            - Add a new tree branch below Root, corresponding to the test A = vi
            - Let Examples_vi be the subset of Examples that have value vi for A
            - If Examples_vi is empty 
                Then below this new branch add a leaf node with label = most common
                value of Target_attribute in Examples
            - Else below this new branch add the subtree
                ID3(Examples_vi, Target_attribute, Attributes - {A}))
    Return Root
"""


class Arbol:
    def __init__(self):
        self.ramas = {}
        # ramas = {'0': arbol, '1':'positivo'}
        self.data = None


def count(S, attrIdxPredecir, valor):
    res = 0
    for i, row in S.iterrows():
        if row[attrIdxPredecir] == valor:
            res += 1
    return res

def obtenerValores(ejemplos, attrIdxPredecir):
    return ejemplos[attrIdxPredecir].unique().tolist()


def calcResultadoMasComun(ejemplos, attrIdxPredecir):
    return ejemplos[attrIdxPredecir].value_counts().index.values[0]
###################################################################################################################

def entropiaAux(S, attrIdxPredecir, dominioTargetFunction=['Negativo', 'Positivo']):
    labels = S.to_numpy()[:, attrIdxPredecir]

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute standard entropy.
    for i in probs:
        ent -= i * math.log(i, 2)

    return ent


def ganancia(ejemplos, attrIdxPredecir, attr, dominioTargetFunction=[0, 1]):
    aux = 0.0

    for posibleValor in dominioTargetFunction:
        ejemplosValor = entradasConEseValor(ejemplos, attr, posibleValor)
        if len(ejemplosValor.index) != 0:
            aux += len(ejemplosValor) / len(ejemplos) * entropiaAux(ejemplosValor, attrIdxPredecir)

    return entropiaAux(ejemplos, attrIdxPredecir) - aux
def obtenerMejorAttrAux(ejemplos, attrIdxPredecir, atributos):
    mejor = atributos[0]
    maxGanancia = 0

    for attr in atributos:
        nuevaGanancia = ganancia(ejemplos, attrIdxPredecir, attr)
        if nuevaGanancia > maxGanancia:
            maxGanancia = nuevaGanancia
            mejor = attr

    return mejor


###################################################################################################################
def entropia(df, attribute, isBinary=True):
    colConRes = df.keys()[-1]
    if isBinary:

        cantPositivo0 = len(df[attribute][df[attribute] == 0][df[colConRes] == 'positive'])
        cantPositivo1 = len(df[attribute][df[attribute] == 1][df[colConRes] == 'positive'])
        cantNegativo0 = len(df[attribute][df[attribute] == 0][df[colConRes] == 'negative'])
        cantNegativo1 = len(df[attribute][df[attribute] == 1][df[colConRes] == 'negative'])

        total0 = cantPositivo0 + cantNegativo0
        total1 = cantPositivo1 + cantNegativo1

        try:
            if cantPositivo0 == 0:
                res0 = (cantNegativo0 / total0) * log(cantNegativo0 / total0)
            elif cantNegativo0 == 0:
                res0 = (cantPositivo0 / total0) * log(cantPositivo0 / total0)
            else:
                res0 = (cantPositivo0 / total0) * log(cantPositivo0 / total0) + (cantNegativo0 / total0) * log(
                    cantNegativo0 / total0)
        except ZeroDivisionError as e:
            res0 = 0
        try:
            if cantPositivo1 == 0:
                res1 = (cantNegativo1 / total1) * log(cantNegativo1 / total1)
            elif cantNegativo1 == 0:
                res1 = (cantPositivo1 / total1) * log(cantPositivo1 / total1)
            else:
                res1 = (cantPositivo1 / total1) * log(cantPositivo1 / total1) + (cantNegativo1 / total1) * log(
                    cantNegativo1 / total1)
        except ZeroDivisionError as e:
            res1 = 0
        res = abs((total0 / (total0 + total1)) * res0 + (total1 / (total0 + total1)) * res1)
    else:
        entropyTotal = 0
        # Busca en la ultima columna que este el resultado
        target_variables = df[colConRes].unique()  # obtenemos las posibles evaluaciones pos,neg etc
        variables = df[attribute].unique()  # posibles valores que puede tomar el atributo
        for variable in variables:
            entropy = 0
            for target_variable in target_variables:
                num = len(df[attribute][df[attribute] == variable][
                              df[colConRes] == target_variable])  # cuenta la cantidad de tuplas con valor
                den = len(df[attribute][df[attribute] == variable])  # cuenta la cantidad de filas en total
                fraction = num / den
                if fraction != 0:
                    entropy += -fraction * log(fraction)
            entropyTotal += entropy
        res = abs(entropyTotal)

    # arrayOfEntropy[attribute] = res
    return res


# main
def obtenerMejorAttr(df, attributes,attrIdxPredecir):
    # if (len(attributes) < 150):
    #     return obtenerMejorAttrAux(df, attrIdxPredecir, attributes)
    entropiaCalculada = {}
    for key in attributes:
        entropyAtribute = entropia(df, key)
        if entropyAtribute == 0:
            return key
        entropiaCalculada[key] = entropyAtribute
    resKey = min(entropiaCalculada, key=entropiaCalculada.get)
    return resKey

###################################################################################################################
def entradasConEseValor(ejemplos, attr, posibleValor):
    return ejemplos[ejemplos[attr] == posibleValor]


def ID3(ejemplos, attrIdxPredecir, atributos, maxNivel):
    arbol = Arbol()

    valores = obtenerValores(ejemplos, attrIdxPredecir)

    # - If all Examples are positive, Return the single-node tree Root, with label = +
    # - If all Examples are negative, Return the single-node tree Root, with label = -
    if len(valores) == 1:
        return valores[0]  # si todos los ejemplos tiene el mismo valor etiqueto con ese valor

    # - If Attributes is empty, Return the single-node tree Root, with label = most common value of Targetattribute in Examples
    if len(atributos) == 0 or maxNivel == 0:
        return calcResultadoMasComun(ejemplos, attrIdxPredecir)
    # Caso con varias tuplas mismo valor atributo distinto resultado
    duplicate = ejemplos.drop_duplicates(inplace=False)

    if duplicate.shape[0] <= 2:
        return calcResultadoMasComun(ejemplos, attrIdxPredecir)

    # start_time = time.time()
    mejorAtributo = obtenerMejorAttr(ejemplos, atributos,attrIdxPredecir)
    # print("TIEMPO DE EJECUCION --- %s seconds ---" % (time.time() - start_time))

    arbol.data = mejorAtributo
    for posibleValor in [0, 1]:
        nuevosAtributos = atributos.copy()
        # nuevosAtributos = copy.deepcopy(atributos)
        nuevosAtributos.remove(mejorAtributo)

        ejemplosConEseValor = entradasConEseValor(ejemplos, mejorAtributo, posibleValor)

        if len(ejemplosConEseValor.index) == 0:
            arbol.ramas[posibleValor] = calcResultadoMasComun(ejemplos, attrIdxPredecir)
        else:
            subArbol = ID3(ejemplosConEseValor, attrIdxPredecir, nuevosAtributos, maxNivel - 1)
            arbol.ramas[posibleValor] = subArbol

    return arbol


def crearArbol(ejemplos, attrIdxPredecir, atributos):
    res = ID3(ejemplos, attrIdxPredecir, atributos,10)

    return res


def predecir(arbol, X_test):
    # self.ramas = {}
    # ramas = {'0': arbol, '1':'positivo'}
    # self.data = None

    y_pred = []
    for _, fila in X_test.iterrows():
        arbolIter = arbol
        while (not isinstance(arbolIter, str)):  # mientras que no haya llegado a un string

            attrIdx = arbolIter.data
            valorAttr = fila[attrIdx]
            arbolIter = arbolIter.ramas[valorAttr]

        # es un string -> lo retorno
        y_pred.append(arbolIter)

    return y_pred
