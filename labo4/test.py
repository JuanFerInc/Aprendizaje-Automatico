import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils import *

df = pd.read_csv('sub-data.txt', delimiter=",").drop(['CASEID'], axis=1)

# me quedo solo con los primeros 100 porque estoy desde la notebook
df = df[:1000]


# https://datascience.stackexchange.com/questions/29840/how-to-count-grouped-occurrences
# https://pstblog.com/2016/10/04/stacked-charts
#grouped = df.groupby(['DEM_AGEGRP_IWDATE'])[['LIBCPRE_SELF']].sum()
grouped = df.groupby(['DEM_AGEGRP_IWDATE','LIBCPRE_SELF']).size().to_frame('count').reset_index()
#print(grouped.head())
pivot_df = grouped.pivot(index='DEM_AGEGRP_IWDATE', columns='LIBCPRE_SELF', values='count')
pivot_df.plot.bar(stacked=True)
plt.show()
#grouped.plot(x=grouped.index.year, kind='bar', stacked=True)








# agePos = {
#     1: "17-20",
#      2: "21-24",
#      3: "25-29",
#      4: "30-34",
#      5: "35-39",
#      6: "40-44",
#      7: "45-49",
#      8: "50-54",
#      9: "55-59",
#     10: "60-64",
#     11: "65-69",
#     12: "70-74",
#     13: "75+",
# }



"""
res = {
    1: [20, 35, 30, 35, 27],
    2: [25, 32, 34, 20, 25]
}

ind = np.arange(len(res.keys()))    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

plots = []

for k,v in res.items():
    if k == 
    plots.append(
        
    )
    
    
p1 = plt.bar(ind, menMeans, width)
p2 = plt.bar(ind, womenMeans, width, bottom=menMeans)


plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()
"""