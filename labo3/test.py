import numpy as np
import pandas as pd


from chi2 import *

from sklearn.metrics import make_scorer, roc_auc_score


def test_normalizacion():
    # df = pd.DataFrame(np.arange(12).reshape(3, 4),
    # columns=['A', 'B', 'C', 'D'])

    df = pd.DataFrame([[2, 2, 6, 0],
                    [2, 4, 7, 0],
                    [3, 7, 8, 1]], columns=['A', 'B', 'C', 'target'])

    # df = df.drop(['B', 'C'], axis=1)


    # m = df[['A', 'C']].mean()
    # std = df[['A', 'C']].std()

    # df[['A', 'C']] = (df[['A', 'C']] - m) / std

    return df


def binning():
    df = pd.DataFrame([[2.2, 0],
                    [2, 1],
                    [0.5, 0],
                    [0.2, 0],
                    [3.8, 0],
                    [-1.6, 0],
                    [-2.56, 1]], columns=['A', 'B'])
    percentile_25 = df.A.quantile(0.25)
    percentile_50 = df.A.quantile(0.50)
    percentile_75 = df.A.quantile(0.75)

    # agregar una columna por cada percentil con un booleano
    df['A_25'] = (df['A'] < percentile_25).astype(int)

    # agregar una sola columna con distintos valores
    df.loc[df['A'] <= percentile_25, 'C'] = 0
    df.loc[(df['A'] > percentile_25) & (df['A'] <= percentile_50), 'C'] = 1
    df.loc[(df['A'] > percentile_50) & (df['A'] <= percentile_75), 'C'] = 2
    df.loc[df['A'] > percentile_75, 'C'] = 3

    print(df)


def sigmma_test(df, topN=10):
    x = df[['A', 'B','C']]
    sigmmas = x.std().to_frame()
    sigmmas.columns = ['sigmma']
    sigmmas.index.name = 'Columna' 
    return sigmmas.sort_values("sigmma", ascending=False).head(topN)


def test_correlacion_chi2(df):
    x = df[['A', 'B','C']]
    y = df['target']

    # selecciono solo las top 2 columnas segun el chi2 test
    chi2_scores = test_chi2(x, y).sort_values("Chi2-score", ascending=False).head(2)

    print(chi2_scores['Columna'].values)


if __name__ == "__main__":

    df = test_normalizacion()
    x = df[['A', 'B','C']]
    print( sigmma_test(x) )

    # binning()

