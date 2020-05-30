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

from knn import run_Knn
from naiveBayes import *

from knn import *
from utils import *
from chi2 import *
from diff import *

import matplotlib.pyplot as plt


ds = pd.read_csv('adult.data', header=None, names=dataColumns, delimiter=", ", engine='python')

# graficas para SETs (conjuntos) a b y c que se pueden interlapear
def graficas_circulares(a, b, c, ab, ac, bc, abc):

    from matplotlib_venn import venn3

    fig1, ax1 = plt.subplots()

    ax1.title.set_text('Cantidad de filas con valores inválidos según columna')

    v3 = venn3(subsets = {
            '100': a,  # tamanio de workclass
            '010': b,  # tamanio de occupation
            '110': ab, # interseccion workclass ^ occupation
            '001': c,  # tamanio de native-country
            '101': ac,  # interseccion workclass ^ native-country
            '011': bc, # interseccion occupation ^ native-country
            '111': abc}, # intersecciond de todos
            set_labels = ('workclass', 'occupation', 'native-country')
        )

    v3.get_patch_by_id('100').set_color('red')
    v3.get_patch_by_id('010').set_color('yellow')
    v3.get_patch_by_id('001').set_color('blue')
    v3.get_patch_by_id('110').set_color('orange')
    v3.get_patch_by_id('101').set_color('purple')
    v3.get_patch_by_id('011').set_color('green')
    v3.get_patch_by_id('111').set_color('grey')

    v3.get_label_by_id('100').set_text( str(a) )
    v3.get_label_by_id('010').set_text( str(b) )
    v3.get_label_by_id('001').set_text( str(c) )
    v3.get_label_by_id('110').set_text( str(ab) )
    v3.get_label_by_id('101').set_text( str(ac) )
    v3.get_label_by_id('011').set_text( str(bc) )
    v3.get_label_by_id('111').set_text( str(abc) )

    for text in v3.subset_labels:
        text.set_fontsize(13)

    plt.show()


def estadisticas_faltantes():

    # cuantas son las rows que tienen atributos faltantes y que
    # procentaje representan?

    # solo las invalidas
    w = ds[ ds['workclass']=='?']
    o = ds[ ds['occupation']=='?']
    n = ds[ ds['native-country']=='?']

    wo = ds[ (ds['workclass']=='?') & (ds['occupation']=='?')]
    wn = ds[ (ds['workclass']=='?') & (ds['native-country']=='?')]
    on = ds[ (ds['occupation']=='?') & (ds['native-country']=='?')]

    
    won = ds[ (ds['workclass']=='?') & (ds['occupation']=='?') & (ds['native-country']=='?')]

    total = len(ds)


    # w_o = ds[ (ds['workclass']=='?') | (ds['occupation']=='?')]
    # print('total:', total)
    # print('w OR o:', len(w_o))
    # print('-----------')

    print('W:', len(w))
    print('o:', len(o))
    print('n:', len(n))
    print('w & o:', len(wo))
    print('w & n:', len(wn))
    print('o & n:', len(on))
    print('w & o & n:', len(won))

    return len(w), len(o), len(n), len(wo), len(wn), len(on), len(won)
    

def relacion_entre_workclass_y_occupation():
    """
    existe alguna distribucion entre wrokclass y occupations ???
    (solo para aquellas tal que workclass=?)
    """
    
    """
    CONCLUSION:
    De todos los tipos de "workclass", los unicos que no registran "occupation" son los que "never-worked"
    todos ellos (los unicos 7) no registran "occupation"
    """


    # armo el dic que va a contener toda la data
    estadisticas = {}
    for col in categoricas:
        estadisticas[col] = {}

    """ 
    ejemplo:

    estadisticas = {
        'workclass' = {
            'Federal-gov' = {
                'ok': 2000,
                '?': 158
            }
            ...
        },
        ...
    }

    """

    total = len(ds)

    # for col in dataColumns:
    for col in ['workclass']:
        if col in categoricas:
            idx = categoricas.index(col)
            dominio = columnas_categoricas[idx]['dominio']
            for valor in dominio:
                estadisticas[col][valor] = {}
                cant_faltante = len( ds[ (ds[col]==valor) & (ds['occupation']=='?') ] )
                estadisticas[col][valor]['?'] = cant_faltante
                estadisticas[col][valor]['ok'] = total - cant_faltante    

    # print(json.dumps( estadisticas, sort_keys=True, indent=4))



class patron_ni_w_ni_o():
    def __init__(self):
        # "nwo" = "NI Workclass, NI Occupation"
        nwo = ds[ (ds['workclass']=='?') & (ds['occupation']=='?')]
        # que onda con estos ^^^ ? cual es el patron?

        # como voy a encontrar un patron?
        # para ello voy a hacer para cada valor de las otras columnas:
        #   una sumatoria, tal que:
        # los valores con una sumatoria mayor indican una mayor ocurrencia
        # de valores falantes en las columnas workclass y occupation
        # y espero asi poder inferir una distribucion no uniforme
        # eg. la mayoria de la gente <18 no tiene el attr workclass etc...

        # dropeo los datos que ya no me aporan nada
        nwo = nwo.drop(['workclass', 'occupation'], axis=1)

        cols_restantes = [col for col in dataColumns if col not in ['workclass', 'occupation']]

        self.dist_nwo = obtener_distribuciones(nwo, cols=cols_restantes)

        # dumpeo
        # print(json.dumps( dist_nwo, sort_keys=True, indent=4))

        # ploteo todas
        # plotear_estadisticas(dist_nwo, pie_chart)

        
        
        """
        GRAFICAS CIRCULARES
        ploteo por separado:
        """








        """
        CONCLUSIONES - se destacan solo las distribuciones poco uniformes y outliers que llaman la atencion
        ============
    
        De todas aquellas filas que no tienen ni Workclass ni Occupation:
    
            raza:
              - el 82.2% es White, pero es representativo de la muestra original con un x% whites -> 82/90 ~ 1
              - el 11.6% es Black, pero no es representativo de la muestra original con un x% blacks -> 11.6 ~ 
              - el 3.5% es asian-pac-islander
    
            capital-gain:
              - el 93.3% tiene capital-gain=0
                sera representativo?
            
            capital-loss:
              - el 96.84% tiene capital-loss=0 (muy correlacionado con el anterior)
                sera representativo?
    
            hours-per-week:
              - hay un inusual outlier con hours-per-week=40 con el 37.52% de los casos
                sera representativo de la muestra original ese hours-per-week=40?
    
            native-country:
              - de forma analoga con la raza, United-State acapara el 90.35% de los casos
                pero es representativo con la muestra original
            
            income:
              - <50k 89.6%, pero es representativo ?
        """

        ###########################################

        """
            Ahora las quiero comparar a las distribuciones de la muestra original
        ya que por ejemplo:
        de todas las filas que tienen workclass=?, el 82% eran blancos
        a priori puede parecer como un outlier, pero si la muestra original
        tenia un 95% de gente blanca -> entonces no eran tantos y respetaba la dist original.
            Analogamente, el procentaje de negros en workclass=? era de 11.6%,
        pero si representan una poblacion del ~5% en la muestra original -> entonces
        si estamos ante un outlier.
        """

        df = ds.drop(['workclass', 'occupation'], axis=1)
        dist_originales = obtener_distribuciones(df, cols=cols_restantes)

        # print(json.dumps( dist_originales, sort_keys=True, indent=4))


        self.dist_relativas = comparar_distribuciones(self.dist_nwo, dist_originales)

        # como interpretar:
        # si dist_relativas['race']['white'] = 1 -> el 100% de los blancos cumple con los atributos que definen a dist_nwo
        # si dist_relativas['sex']['Female'] = 0.5 -> el 50% de los negros cumple con los atributos que definen a dist_nwo
        # parece obvio, pero es facil olvidar que
        # se esta comparando whites contra whites
        # y females contra females

        # las ordeno
        for i, x in self.dist_relativas.items():

            if esCategorica(i):
                # si es categorica -> las ordeno segun value
                self.dist_relativas[i] = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
            else:
                # si es numerica -> las ordeno segun key
                # pero antes la key la paso de `str` a `int`
                # dist_relativas[i] = {int(k): v for k, v in x.items()}
                self.dist_relativas[i] = {int(k): v for k, v in sorted(x.items(), key=lambda item: int(item[0]))}

    # dumpeo
    # print(json.dumps( dist_relativas, sort_keys=True, indent=4))
    #
    # print("para ver las graficas hay que descomentar las lineas de 'ploteo'")

    # ploteo
    def ploteo_age(self):
        plotear_estadisticas(self.dist_relativas['age'], 'age',bars_chart)
    def ploteo_educacion_num(self):
        plotear_estadisticas(self.dist_relativas['education-num'], 'education-num', bars_chart)
    def ploteo_educacion(self):
        plotear_estadisticas(self.dist_relativas['education'], 'education', bars_chart)
    def ploteo_hours_per_week(self):
        plotear_estadisticas(self.dist_relativas['hours-per-week'], 'hours-per-week', bars_chart)
    def ploteo_native_country(self):
        plotear_estadisticas(self.dist_relativas['native-country'], 'native-country', bars_chart)
    def ploteo_race(self):
        plotear_estadisticas(self.dist_relativas['race'], 'race', bars_chart)
    def ploteo_marital_status(self):
        plotear_estadisticas(self.dist_relativas['marital-status'], 'marital-status', bars_chart)
    def ploteo_sex(self):
        plotear_estadisticas(self.dist_relativas['sex'], 'sex', bars_chart)

    def graficaCirculares(self):
        # plotear "capital-gain" y "capital-loss"
        plotear_2_pies(self.dist_nwo, "capital-gain", "capital-loss", plotearConCifras=False)

        # ploteo hours-per-week
        data = sort_by_value(self.dist_nwo["hours-per-week"])
        pie_chart(
            labels=data.keys(),
            sizes=data.values(),
            title="hours-per-week",
            plotearConCifras=False
        )

        # income
        data = self.dist_nwo["income"]
        pie_chart(
            labels=data.keys(),
            sizes=data.values(),
            title="income",
            plotearConCifras=True
        )

        # native-country
        data = sort_by_value(self.dist_nwo["native-country"])
        pie_chart(
            labels=data.keys(),
            sizes=data.values(),
            title="native-country",
            plotearConCifras=False
        )

        # race
        data = self.dist_nwo["race"]
        pie_chart(
            labels=data.keys(),
            sizes=data.values(),
            title="race",
            plotearConCifras=True
        )

        # sex
        data = self.dist_nwo["sex"]
        pie_chart(
            labels=data.keys(),
            sizes=data.values(),
            title="sex",
            plotearConCifras=True
        )


def patron_no_native_country():

    # "nnc" = "No Native-Country"
    nnc = ds[ (ds['native-country']=='?')]

    # dropeo los datos que ya no me aporan nada
    nnc = nnc.drop(['native-country'], axis=1)

    cols_restantes = [col for col in dataColumns if col not in ['native-country']]

    dist_nwo = obtener_distribuciones(nnc, cols=cols_restantes)

    # dumpeo
    # print(json.dumps( dist_nwo, sort_keys=True, indent=4))

    # ploteo
    plotear_estadisticas(dist_nwo, pie_chart)

    df = ds.drop(['native-country'], axis=1)
    dist_originales = obtener_distribuciones(df, cols=cols_restantes)

    # print(json.dumps( dist_originales, sort_keys=True, indent=4))

    dist_relativas = comparar_distribuciones(dist_nwo, dist_originales)

    # las ordeno
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
    print(json.dumps( dist_relativas, sort_keys=True, indent=4))

    # ploteo
    # plotear_estadisticas(dist_relativas, bars_chart)


def obtener_distribuciones(df, cols):

    # armo el dic que va a contener toda la data
    # son simples contadores
    estadisticas = {}
    for col in cols:
        estadisticas[col] = {}

    for col in cols:
        _dominio = dominio(col) if esCategorica(col) else df[col].unique()
        if len(_dominio) > 200:
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


def plotear_estadisticas(estadisticas,key, plotter):
    # ploteo

    if len(estadisticas.keys()):
        labels = estadisticas.keys()
        sizes = estadisticas.values()
        plotearConCifras = True # len(labels) < 10
        plotter(labels, sizes, key, plotearConCifras)


def bars_chart(labels, sizes, title=False, plotearConCifras=True):
    plt.rcParams['figure.figsize'] = [20, 5]
    fig, ax = plt.subplots()
    if title:
        ax.title.set_text(title)
    ax.bar(labels,sizes)
    plt.xticks(rotation=90, ha="right")
    plt.show()


def sort_by_value(dic):
    return {k: v for k, v in sorted(dic.items(), reverse=True, key=lambda item: item[1])}


def plotear_2_pies(data, col_izq, col_der, plotearConCifras):
    fig, ax = plt.subplots(1, 2)

    izq = sort_by_value(data[col_izq])
    der = sort_by_value(data[col_der])

    # grafica de la izq
    fill_pie_chart( ax[0], list(izq.keys()), list(izq.values()), title=col_izq, plotearConCifras=False)

    # grafica de la der
    fill_pie_chart( ax[1], list(der.keys()), list(der.values()), title=col_der, plotearConCifras=False)

    # fig.savefig( columna + ".png")
    plt.show()


def fill_pie_chart(ax, labels, sizes, title=False, plotearConCifras=True):

    if title:
        ax.title.set_text(title)

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
        patches = ax.pie(**args)

    else:
        patches, texts = ax.pie(sizes, startangle=90)
        # The slices will be ordered and plotted counter-clockwise.
        ax.legend(patches, list(labels)[:10], loc="best")

    ax.axis('equal')
    # ax.tight_layout()


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
        patches = plt.pie(**args)

    else:
        patches, texts = plt.pie(sizes, startangle=90)
        # The slices will be ordered and plotted counter-clockwise.
        plt.legend(patches, list(labels)[:10], loc="best")

    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# donde a y b son daframes
def ratios(a, b):
    x = round(len(a)/len(b), 2)
    return x, 1 - x


# donde a y b son ratios
def percentage(a, b):
    return str(a * 100) + "%", str(b * 100) + "%"


def viudos_vs_divorciados():
    nwo = ds[ (ds['workclass']=='?') & (ds['occupation']=='?')]

    v = nwo[ (nwo['marital-status']=='Widowed')]
    d = nwo[ (nwo['marital-status']=='Divorced')]

    mv = v[ (v['sex']=='Male')]
    md = d[ (d['sex']=='Male')]

    print("Viudos y NWO: {Hombres: %s Mujeres: %s} " % percentage(*ratios(mv, v)))
    print("Divorciados y NWO: {Hombres: %s Mujeres: %s} " % percentage(*ratios(md, d)))


def avg_edad(x):
    return round(x["age"].mean(), 1)

def edades_promedio_viudos_vs_divorciados():
    nwo = ds[ (ds['workclass']=='?') & (ds['occupation']=='?')]

    v = nwo[ (nwo['marital-status']=='Widowed')]
    d = nwo[ (nwo['marital-status']=='Divorced')]

    mv = v[ (v['sex']=='Male')]
    fv = v[ (v['sex']=='Female')]
    md = d[ (d['sex']=='Male')]
    fd = d[ (d['sex']=='Female')]

    print("Promedio edad {NWO, viudo, hombre} : %d" % avg_edad(mv) )
    print("Promedio edad {NWO, viuda, mujer} : %d" % avg_edad(fv) )
    print("Promedio edad {NWO, divorciado, hombre} : %d" % avg_edad(md) )
    print("Promedio edad {NWO, divorciada, mujer} : %d" % avg_edad(fd) )
    

if __name__ == '__main__':
    # w, o, n, wo, wn, on, won = estadisticas_faltantes()
    # graficas_circulares(a=w, b=o, c=n, ab=wo, ac=wn, bc=on, abc=won)
    
    # relacion_entre_workclass_y_occupation()
    patron_ni_w_ni_o()
    # bars_chart( None, [11,22,33])
    # viudos_vs_divorciados()
    # edades_promedio_viudos_vs_divorciados()

    # patron_no_native_country()

    """
    dudas:
     - que significa el valor "own-child" en la columna relationship (entre own-child y not-in-family suman 53.2% de los casos de nwo)
    """