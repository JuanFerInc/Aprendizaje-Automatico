import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

def test_chi2(x, target):
    
    # chi2 solo anda con valores positivos
    x = x + abs(x.min())
    target = target + abs(target.min())

    # The null hypothesis for chi2 test is that "two categorical variables are independent
    # So a higher value of chi2 statistic means "two categorical variables are dependent"
    # and MORE USEFUL for classification.
    
    chi2_selector = SelectKBest(chi2, k=2)
    chi2_selector.fit(x, target)

    res = list(zip(x.columns.values, chi2_selector.scores_, chi2_selector.pvalues_))

    return pd.DataFrame(res, columns=['Columna', 'Chi2-score', 'p-valor'])