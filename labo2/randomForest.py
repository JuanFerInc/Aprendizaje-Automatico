import random
from id3 import crearArbol as crarArbolId3
from id3 import predecir as predecirId3

# genera una sub-muestra del dataset `entrenamiento`
def subMuestra(entrenamiento, proporcionMuestra):
    tamanioMuestra = int(len(entrenamiento.index) * proporcionMuestra)
    muestra = entrenamiento.sample(tamanioMuestra, replace = True)
    muestra.reset_index(drop=True, inplace=True)
    return muestra


def crearArbol(muestra, numAtributos):

    totalAtributos = len(muestra.columns) - 1 #1024
    
    atributos = []
    while len(atributos) < numAtributos:
        index = random.randrange(totalAtributos) # valores entre 0 y 1023
        if index not in atributos:
            atributos.append(index)
    arbol = crarArbolId3(muestra, totalAtributos, atributos)

    # atrs = np.random.choice(totalAtributos, size=numAtributos, replace=False).tolist()
    # arbol = crarArbolId3(muestra, totalAtributos, atrs)
    
    return arbol


def crearRandomForest_Wrapper(entrenamiento, tamanioMuestra, numArboles, numAtributos, conPrints=False, numOfThreads=4):
    import itertools
    
    # lo hacemos con procesos, no con threads
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(processes=numOfThreads)

    async_results = []

    d = round(numArboles / numOfThreads)    
    desde = 0
    hasta = d
    for i in range(numOfThreads):
        # Proceso `i` le corresponde generar `desde` `hasta`

        # params = (entrenamiento, tamanioMuestra, d, numAtributos, conPrints, i, resultados)
        
        async_result = pool.apply_async(crearRandomForest, (entrenamiento, tamanioMuestra, d, numAtributos, conPrints, i,))
        async_results.append(async_result)

        # siguiente rango
        desde, hasta = hasta, hasta + d

    # espero a que terminen todos
    pool.close()
    pool.join()

    # terminaron todos, los mergeo en 1 sola estructura
    resultados = map(lambda x: x.get(), async_results)
    # ahora que tengo una lista de arboles, los uno en una sola lista
    return list(itertools.chain.from_iterable(resultados))

# crea el forest
def crearRandomForest(entrenamiento, tamanioMuestra, numArboles, numAtributos, conPrints=True, subThread=0):

    arboles = []

    for i in range(numArboles):
        if conPrints:
            print('[Thread %d] haciendo arbol %d' % (subThread, i))
        muestra = subMuestra(entrenamiento, tamanioMuestra)
        arbol = crearArbol(muestra, numAtributos)
        arboles.append(arbol)

    return arboles

def predecirRandomForest(forest, X_test):
    y_pred = []

    predicciones = [predecirId3(arbol, X_test) for arbol in forest]
    for i in range(len(predicciones[0])):
        # i = el test num
        # predic = la prediccion de todos los tests hecha por el arbol predic
        # predic[i] = la prediccion para el test i segun el arbol `predic`
        votacion = {}
        for predic in predicciones:
            if predic[i] in votacion:
                votacion[predic[i]]+=1
            else:
                votacion[predic[i]]=1
        prediccion = max(votacion, key=lambda key: votacion[key])
        y_pred.append(prediccion)

    return y_pred