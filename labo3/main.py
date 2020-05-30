import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import sklearn.preprocessing
import sklearn.feature_selection

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB


from knn import run_Knn
from naiveBayes import *

from knn import *
from utils import *
from chi2 import *
from diff import *
from validacion_cruzada_sklearn import *
from estadisticas_target import *
from estadisticas_valores_faltantes import *
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

 
if __name__ == '__main__':
    ds1 = pd.read_csv('adult.data', header=None, names=dataColumns, delimiter=", ", engine='python')
    ds2 = pd.read_csv('adult.test', header=None, names=dataColumns, delimiter=", ", engine='python')
    frames = [ds1, ds2]
    ds = pd.concat(frames)
    ds.reset_index(drop=True,inplace=True)

    test = patron_ni_w_ni_o()

    # contamos cuantos hay
    # print(ds.income.value_counts())
    # <=50K    24720
    #  >50K      7841


    # fusionar data + test y fijarse si tienen la misma distribucion de
    # datos faltantes, .value_counts() etc..

    # pre-procesamiento

    # 1. atributos numericos faltantes

    for col in columnas_numericas:
        # Contamos cuántos NaN son
        str = "Para la columna {} Cantidad de instancias sin valor: {}".format( col, ds[col].isna().sum() )
        print(str)


    # 2. Preprocesamiento - Atributos categóricos

    # Transformamos el dominio de la columna income a binario
    #  <=50K ~ 0
    #  >50K  ~ 1


    # df["column1"].replace({"a": "x", "b": "y"}, inplace=True)


    # dominios de las columnas categoricas
    for cc in columnas_categoricas:
        columna = cc['columna']
        dominio_esperado = cc['dominio']
        values = ds[columna].value_counts() # retorna un pandas.series
        dominio = values.index.tolist()
        valoresNoUsados, valoresIncompatibles = diff(dominio_esperado, dominio)
        
        if len(valoresIncompatibles) > 0: 
            print('La columna "{}" tiene valores fuera de dominio: {}'.format(columna, valoresIncompatibles))
        
        if len(valoresNoUsados) > 0: 
            print('La columna "{}" tiene valores que nunca ha usado: {}'.format(columna, valoresNoUsados))


    # La columna "workclass" tiene valores fuera de dominio
    # La columna "occupation" tiene valores fuera de dominio
    # La columna "native-country" tiene valores fuera de dominio


    # One-hot-encoding para los multiclase,
    # binario si no.
    for cc in columnas_categoricas:
        esMulticlase = len(cc['dominio']) > 2
        columna = cc['columna']

        if esMulticlase:

            # Utilizamos scikit-learn para crear un one-hot-encoder
            ohe=sklearn.preprocessing.OneHotEncoder(sparse=False)
            # Obtenemos las categorías a partir de los datos de entrenamiento
            ohe.fit(ds[columna].to_numpy().reshape(-1,1))
            # Obtenemos los nuevos valores a partir del valor original
            new=ohe.transform(ds[columna].to_numpy().reshape(-1,1))

            for i, valor_posible in enumerate(ohe.categories_[0]):
                nomnbre_nueva_columna = columna + '_' + valor_posible
                ds[nomnbre_nueva_columna]=new[:,i]

            # la dropeamos
            ds = ds.drop([columna], axis=1)

        else:
            # si no es muliclase -> es binario
            # luego, si es binario lo codeamos con 0's y 1's

            # Creamos un labelEncoder utilizando scikit-learn
            le=sklearn.preprocessing.LabelEncoder()
            # Obtenemos las clases a partir de los valores del conjunto de entrenamiento
            le.fit(ds[columna])
            # Mostramos las clases obtenidas
            le.classes_
            # Ajustamos el campo sex, transformándolo
            ds.loc[:,columna] = le.transform(ds.loc[:,columna])




    # print(ds.columns)
    # print(ds[['income']].head)
    # print(ds[['workclass_Private','workclass_Federal-gov','workclass_?']].head)

    X = ds.drop(['income'], axis=1)
    y = ds['income']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    # Estratificacion

    # Validacion cruzada

    # Estandarizacion de atributos

    media_x = X_train[columnas_numericas].mean()
    std_x = X_train[columnas_numericas].std()
    X_train[columnas_numericas] = (X_train[columnas_numericas] - media_x) / std_x
    X_test[columnas_numericas] = (X_test[columnas_numericas] - media_x) / std_x


    X_train_bayes = X_train.copy()
    X_test_bayes = X_test.copy()

    for col in columnas_numericas:
        percentile_25 = X_train_bayes[col].quantile(0.25)
        percentile_50 = X_train_bayes[col].quantile(0.50)
        percentile_75 = X_train_bayes[col].quantile(0.75)

        nueva_columna = 'bin_' + col # <- columnas no son binarias, toma valores [0,3]
        # agregar una sola columna con distintos valores
        X_train_bayes.loc[X_train_bayes[col] <= percentile_25, nueva_columna] = 0
        X_train_bayes.loc[(X_train_bayes[col] > percentile_25) & (X_train_bayes[col] <= percentile_50), nueva_columna] = 1
        X_train_bayes.loc[(X_train_bayes[col] > percentile_50) & (X_train_bayes[col] <= percentile_75), nueva_columna] = 2
        X_train_bayes.loc[X_train_bayes[col] > percentile_75, nueva_columna] = 3

        X_test_bayes.loc[X_test_bayes[col] <= percentile_25, nueva_columna] = 0
        X_test_bayes.loc[(X_test_bayes[col] > percentile_25) & (X_test_bayes[col] <= percentile_50), nueva_columna] = 1
        X_test_bayes.loc[(X_test_bayes[col] > percentile_50) & (X_test_bayes[col] <= percentile_75), nueva_columna] = 2
        X_test_bayes.loc[X_test_bayes[col] > percentile_75, nueva_columna] = 3

    X_train_bayes = X_train_bayes.drop(columnas_numericas, axis=1)
    X_test_bayes = X_test_bayes.drop(columnas_numericas, axis=1)


    # Seleccion de atributos

    #METODO 1: Test de chi 2
    
    # seleccion de los top n=10 atributos segun el test de chi-2

    chi2_scores_knn = test_chi2(X_train, y_train).sort_values("Chi2-score", ascending=False)
    ax_chi2 = chi2_scores_knn.head(30).plot.bar(x='Columna', y='Chi2-score', rot=0)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    #print(chi2_scores_knn)

    columnas_seleccionadas_knn = chi2_scores_knn['Columna'].head(20)
    X_train_seleccion_atributos_knn = X_train[columnas_seleccionadas_knn.tolist()]
    print(X_train_seleccion_atributos_knn.head())
    
    
    testHyperparameters_Knn_sklearn(X_train_seleccion_atributos_knn,y_train,k_Fold=3,cantVecinos_HPArr=[3,5])

    
    chi2_scores_NB = test_chi2(X_train_bayes, y_train).sort_values("Chi2-score", ascending=False)
    columnas_seleccionadas_NB = chi2_scores_NB['Columna'].head(20)
    X_train_seleccion_atributos_NB = X_train_bayes[columnas_seleccionadas_NB.tolist()]
    
    testHyperparameters_NB_sklearn(X_train_seleccion_atributos_NB, y_train,k_Fold=3)
    
    
    
    
    
    """
    #METODO 2: Ordenar por varianza 
    #Idea: Los atributos que valen siempre lo mismo seguramente no sean buenos predictores
    
    # seleccion de los top n=10 atributos segun varianza
    #n = 10
    #x = df[['A', 'B','C']] # las columnas que se van a testear
    #sigmma_scores = x.std().sort_values("Chi2-score", ascending=False).head(2)
    varianza = X_train_bayes.std().sort_values(ascending=False).head(30)
    varianza.plot.bar()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    """

    """
    METODO 3: 
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    """
    """
    #Validacion Cruzada KNN
    import time
    start_time = time.time()
    print("STARTIN NB Test Hyper Parametros")
    k_Fold = 4
    valoresM_HPArr = [0.2, 0.4, 1, 1.5, 2]
    testHyperparameters_NB(X_train_bayes, y_train, k_Fold, valoresM_HPArr)
    print("--- %s seconds ---" % (time.time() - start_time))
    k_Fold = 2
    cantVecinos_HPArr = [1]
    start_time = time.time()
    print("STARTIN KNN Test Hyper parametros")
    testHyperparameters_Knn(X_train,y_train,k_Fold,cantVecinos_HPArr)
    print("--- %s seconds ---" % (time.time() - start_time))
    # Evaluacion Naive Bayes
    print("STARTIN NAIVE BAYES")
    start_time = time.time()
    predNB = run_Naive_Bayes(X_train_bayes, y_train,X_test_bayes)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(confusion_matrix(y_test, predNB))
    print(classification_report(y_test, predNB))
    # Evaluacion KNN
    k=5
    prioCercano = True
    predKnn = run_Knn(X_train, y_train,X_test,k,prioCercano)
    print(confusion_matrix(y_test, predKnn))
    print(classification_report(y_test, predKnn))
    """

    """
    # knn de scikit learn
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    #Naive Bayes
    # [[3959 1033]
    #  [ 338 1183]]
    #               precision    recall  f1-score   support
    #
    #            0       0.92      0.79      0.85      4992
    #            1       0.53      0.78      0.63      1521
    #
    #     accuracy                           0.79      6513
    #    macro avg       0.73      0.79      0.74      6513
    # weighted avg       0.83      0.79      0.80      6513

    # KNN implementdo
    # [[4487  505]
    #  [ 611  910]]
    #               precision    recall  f1-score   support
    #
    #            0       0.88      0.90      0.89      4992
    #            1       0.64      0.60      0.62      1521
    #
    #     accuracy                           0.83      6513
    #    macro avg       0.76      0.75      0.75      6513
    # weighted avg       0.82      0.83      0.83      6513

    # knn de scikit learn
    # [[4497  495]
    #  [ 609  912]]
    #               precision    recall  f1-score   support
    #
    #            0       0.88      0.90      0.89      4992
    #            1       0.65      0.60      0.62      1521
    #
    #     accuracy                           0.83      6513
    #    macro avg       0.76      0.75      0.76      6513
    # weighted avg       0.83      0.83      0.83      6513
    """
    """
    #Naive Bayes de scikit learn
    clf = MultinomialNB()
    clf.fit(X_train_bayes, y_train)
    y_pred_bayes = clf.predict(X_test_bayes)
    
    print(confusion_matrix(y_test, y_pred_bayes))
    print(classification_report(y_test, y_pred_bayes))
    """

