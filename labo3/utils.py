import copy
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np

def promedioClassification(arregloClas=[]):
    res = copy.deepcopy(arregloClas[0])
    total = len(arregloClas)
    primerPaso = True
    for mat in arregloClas:
        for evaluaciones in mat.keys():
            if evaluaciones == 'accuracy':
                if not primerPaso:
                    res[evaluaciones] += mat[evaluaciones]/total
                else:
                    res[evaluaciones] = mat[evaluaciones] / total
            else:
                for tipoDeMedida in mat[evaluaciones].keys():
                    if not primerPaso:
                        res[evaluaciones][tipoDeMedida] += (mat[evaluaciones][tipoDeMedida] / total)
                    else:
                        res[evaluaciones][tipoDeMedida] = (mat[evaluaciones][tipoDeMedida] / total)
        primerPaso = False
    resDF = pd.DataFrame.from_dict(res)
    return resDF

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



# 15 las columnas

# age,  workclass,  fnlwgt,  education,  education-num,  marital-status, occupation,  relationship,  race,  sex,  capital-gain,  capital-loss, hours-per-week,  native-country, income

# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# income

dataColumns = ['age',  'workclass',  'fnlwgt',  'education',  'education-num',  'marital-status', 'occupation',  'relationship',  'race',  'sex',  'capital-gain',  'capital-loss', 'hours-per-week',  'native-country', 'income']

columnas_numericas = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

categoricas = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income' ]

columnas_categoricas = [
    {
     'columna': 'workclass',
     'dominio': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
    }, 
    
    {
     'columna': 'education',
     'dominio': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
    }, 
    
    {
     'columna': 'marital-status',
     'dominio': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
    }, 
    
    {
     'columna': 'occupation',
     'dominio': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
    }, 
    
    {
     'columna': 'relationship',
     'dominio': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    }, 
    
    {
     'columna': 'race',
     'dominio': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
    }, 
    
    {
     'columna': 'sex',
     'dominio': ['Female', 'Male'],
    }, 
    
    {
     'columna': 'native-country',
     'dominio': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'],
    },
    {
     'columna': 'income',
     'dominio': ['<=50K', '>50K']
    }
]

# retorna true si col es categorica 
def esCategorica(col):
    return col in categoricas

# retorna el dominio del col:categorica 
def dominio(col):
    if esCategorica(col):
        idx = categoricas.index(col)
        return columnas_categoricas[idx]['dominio']


# Name: income, dtype: int64
#  Private             22696
#  Self-emp-not-inc     2541
#  Local-gov            2093
#  ?                    1836
#  State-gov            1298
#  Self-emp-inc         1116
#  Federal-gov           960
#  Without-pay            14
#  Never-worked            7
# Name: workclass, dtype: int64
# ---------
#  HS-grad         10501
#  Some-college     7291
#  Bachelors        5355
#  Masters          1723
#  Assoc-voc        1382
#  11th             1175
#  Assoc-acdm       1067
#  10th              933
#  7th-8th           646
#  Prof-school       576
#  9th               514
#  12th              433
#  Doctorate         413
#  5th-6th           333
#  1st-4th           168
#  Preschool          51
# Name: education, dtype: int64
# ---------
#  Married-civ-spouse       14976
#  Never-married            10683
#  Divorced                  4443
#  Separated                 1025
#  Widowed                    993
#  Married-spouse-absent      418
#  Married-AF-spouse           23
# Name: marital-status, dtype: int64
# ---------
#  Prof-specialty       4140
#  Craft-repair         4099
#  Exec-managerial      4066
#  Adm-clerical         3770
#  Sales                3650
#  Other-service        3295
#  Machine-op-inspct    2002
#  ?                    1843
#  Transport-moving     1597
#  Handlers-cleaners    1370
#  Farming-fishing       994
#  Tech-support          928
#  Protective-serv       649
#  Priv-house-serv       149
#  Armed-Forces            9
# Name: occupation, dtype: int64
# ---------
#  Husband           13193
#  Not-in-family      8305
#  Own-child          5068
#  Unmarried          3446
#  Wife               1568
#  Other-relative      981
# Name: relationship, dtype: int64
# ---------
#  White                 27816
#  Black                  3124
#  Asian-Pac-Islander     1039
#  Amer-Indian-Eskimo      311
#  Other                   271
# Name: race, dtype: int64
# ---------
#  Male      21790
#  Female    10771
# Name: sex, dtype: int64
# ---------
#  United-States                 29170
#  Mexico                          643
#  ?                               583
#  Philippines                     198
#  Germany                         137
#  Canada                          121
#  Puerto-Rico                     114
#  El-Salvador                     106
#  India                           100
#  Cuba                             95
#  England                          90
#  Jamaica                          81
#  South                            80
#  China                            75
#  Italy                            73
#  Dominican-Republic               70
#  Vietnam                          67
#  Guatemala                        64
#  Japan                            62
#  Poland                           60
#  Columbia                         59
#  Taiwan                           51
#  Haiti                            44
#  Iran                             43
#  Portugal                         37
#  Nicaragua                        34
#  Peru                             31
#  Greece                           29
#  France                           29
#  Ecuador                          28
#  Ireland                          24
#  Hong                             20
#  Trinadad&Tobago                  19
#  Cambodia                         19
#  Laos                             18
#  Thailand                         18
#  Yugoslavia                       16
#  Outlying-US(Guam-USVI-etc)       14
#  Honduras                         13
#  Hungary                          13
#  Scotland                         12
#  Holand-Netherlands                1
# Name: native-country, dtype: int64
# ---------






##################################
# para generar las transformaciones
# for cc in columnas_categoricas:
#     columna = cc['columna']
#     dominio_esperado = cc['dominio']
#     l = len(dominio_esperado)
#     trans = list(range(l))
#     dictionary = dict(zip(dominio_esperado, trans))
#     print(columna, '\n', dictionary, '\n\n')


