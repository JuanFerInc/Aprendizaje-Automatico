import numpy as np
import pandas as pd

import sklearn
import sklearn.preprocessing
import sklearn.feature_selection

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

import json

import matplotlib.pyplot as plt

from utils import *


df = pd.read_csv('sub-data.txt', delimiter=",").drop(['CASEID'], axis=1)


class Stats():
    def __init__(self):

        # me quedo solo con las filas que tienen self_placement > 0 
        ok = df[ (df['LIBCPRE_SELF']>0)]

        # todas las columnas menos la target
        cols_restantes = [col for col in columnas if col not in ['LIBCPRE_SELF']]

        # obtengo las distribuciones absolutas para cada valor posible
        # del dominio *de la columna target* (LIBCPRE_SELF)
        # considero solo dominio positivo (~valido)
        dominio_target = dominio_positivo('LIBCPRE_SELF')
        dists_absolutas = {}
        for val in dominio_target:
            filas_con_ese_valor = ok[ (ok['LIBCPRE_SELF']==val)]
            dists_absolutas[val] = obtener_distribuciones(filas_con_ese_valor, cols_restantes)
        
        
        # dumpeo
        print(json.dumps( dists_absolutas, sort_keys=True, indent=4))

        """
        # ploteo
        # plotear_estadisticas(dist_g, pie_chart)
        # plotear_estadisticas(dist_l, pie_chart)

        df = df.drop(['income'], axis=1)
        dist_originales = obtener_distribuciones(df, cols=cols_restantes)

        # print(json.dumps( dist_originales, sort_keys=True, indent=4))

        self.dist_relativas_g = comparar_distribuciones(self.dist_g, dist_originales)
        self.dist_relativas_l = comparar_distribuciones(self.dist_l, dist_originales)

        # las ordeno
        for dist_relativas in [self.dist_relativas_g, self.dist_relativas_l]:
            for i, x in dist_relativas.items():

                if esCategorica(i):
                    # si es categorica -> las ordeno segun value
                    dist_relativas[i] = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
                else:
                    # si es numerica -> las ordeno segun key
                    # pero antes la key la paso de `str` a `int`
                    # dist_relativas[i] = {int(k): v for k, v in x.items()}
                    dist_relativas[i] = {int(k): v for k, v in sorted(x.items(), key=lambda item: int(item[0]))}

    # dumpeo
    # print(json.dumps( dist_relativas_g, sort_keys=True, indent=4))
    # print(json.dumps( dist_relativas_l, sort_keys=True, indent=4))

    #
    # # ploteo edad
    def plotear_edad(self):
        plotear_duo( "age", self.dist_relativas_g, self.dist_relativas_l)

    # # ploteo education
    def ploteo_education(self):
        plotear_duo( "education", self.dist_relativas_g, self.dist_relativas_l)
    #
    # # ploteo education-num
    def ploteo_education_num(self):
        plotear_duo( "education-num", self.dist_relativas_g, self.dist_relativas_l)
    #
    # # ploteo hours-per-week
    def ploteo_houres_per_week(self):
        plotear_duo( "hours-per-week", self.dist_relativas_g, self.dist_relativas_l)
    #
    # # ploteo marital-status
    def ploteo_martial_status(self):
        plotear_duo( "marital-status", self.dist_relativas_g, self.dist_relativas_l)
    #
    # # ploteo native-country
    def ploteo_native_country(self):
        plotear_duo( "native-country", self.dist_relativas_g, self.dist_relativas_l)
    #
    # # ploteo occupation
    def ploteo_occupation(self):
        plotear_duo( "occupation", self.dist_relativas_g, self.dist_relativas_l)
    #
    # # ploteo race
    def ploteo_race(self):
        plotear_duo( "race", self.dist_relativas_g, self.dist_relativas_l)
    #
    # # ploteo relationship
    def ploteo_relationship(self):
        plotear_duo( "relationship", self.dist_relativas_g, self.dist_relativas_l)
    #
    # # ploteo workclass
    def ploteo_workclass(self):
        plotear_duo( "workclass", self.dist_relativas_g, self.dist_relativas_l)
    #
    # # ploteo sex
    def ploteo_sex(self):
        plotear_duo( "sex", self.dist_relativas_g, self.dist_relativas_l)
    # ploteo los porcentajes que representan cada uno de los valores posibles
    def ploteo_porcentaje(self):
        pie_chart(
            labels = ['g','l'],
            sizes = [len(self.g), len(self.l)],
            title= 'Porcentajes que representan',
            plotearConCifras=True
        )

    # plotear_estadisticas(dist_relativas_g, bars_chart)
    # plotear_estadisticas(dist_relativas_l, bars_chart)

    """

def obtener_distribuciones(df, cols):

    # armo el dic que va a contener toda la data
    # son simples contadores
    estadisticas = {}
    for col in cols:
        estadisticas[col] = {}

    for col in cols:
        _dominio = dominio(col) if esCategorica(col) else df[col].unique()
        
        dominioMuyGrande = len(_dominio) > 200
        if dominioMuyGrande:
            continue # si es muy grande que ni lo procese
        
        for valor in _dominio:
            estadisticas[col][str(valor)] = len( df[ df[col]==valor ] )

    # los ordeno
    for i, x in estadisticas.items():
        estadisticas[i] = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}

    return estadisticas


"""
Dados dos diccionarios del tipo estadistica a y b,
retorna los ratios, de forma tal que:
si los datos de a son representativos, al ser divididos entre b
deberian de dar ~ 1.
"""
def comparar_distribuciones(a, b):
    import copy 
    
    ratios = copy.deepcopy(a)

    for col, x in b.items():
        if len(x) == 0:
            continue # si no tiene data, siguiente
        for valor, counted in x.items():
            if counted == 0:
                ratios[col][valor] = 0 # para evitar la div por 0
            else:
                if valor not in ratios[col].keys():
                    ratios[col][valor] = 0 # si es que niguno poblacion con esas caracteristicas (col = valor) cumplia con las propiedades que definen a los atributos de a (w=? y o=?)
                ratios[col][valor] = ratios[col][valor] / counted
    
    return ratios


def plotear_estadisticas(estadisticas, plotter):
    # ploteo
    for col, x in estadisticas.items():
        if len(x.keys()):
            labels = x.keys()
            sizes = x.values()
            plotearConCifras = True # len(labels) < 10
            plotter(labels, sizes, col, plotearConCifras)


def bars_chart(labels, sizes, title=False, plotearConCifras=True):
    fig, ax = plt.subplots()
    if title:
        ax.title.set_text(title)
    ax.bar(labels,sizes)
    plt.xticks(rotation=90, ha="right")
    plt.show()


def pie_chart(labels, sizes, title=False, plotearConCifras=True):

    # Data to plot
    # colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

    fig1, ax1 = plt.subplots()
    if title:
        ax1.title.set_text(title)

    args = {
        'x': sizes,
        'labels': labels,
        # 'autopct':'%1.1f%%', 
        # explode=(0.1, 0, 0, 0), explode 1st slice, 
        # colors=colors,
        # shadow=True, 
        # startangle=140
    }

    if plotearConCifras:
        args['autopct'] = '%1.1f%%'

    plt.pie(**args)

    plt.axis('equal')
    plt.show()


# donde a y b son daframes
def ratios(a, b):
    x = round(len(a)/len(b), 2)
    return x, 1 - x


# donde a y b son ratios
def percentage(a, b):
    return str(a * 100) + "%", str(b * 100) + "%"

if __name__ == '__main__':
    stats = Stats()