import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils import *

df = pd.read_csv('sub-data.txt', delimiter=",").drop(['CASEID'], axis=1)

# me quedo solo con los primeros 100 porque estoy desde la notebook
df = df[:100]

# imprime info basica tipo mu, siggma^2, quintiles, min/max
# print(df.describe())

kmeans = KMeans(n_clusters=3, random_state=0).fit(df)
Z = kmeans.predict(df)


# PCA Projection to 2D
# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['LIBCPRE_SELF']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

# targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
targets = list(self_placement.keys())
targets = [0,1,2]

colors = ['blue','red','green','blue','cyan','yellow','orange','black','pink','brown','purple']
for target, color in zip(targets,colors):
    indicesToKeep = Z == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

plt.show()



# intento de plote

# from mpl_toolkits.mplot3d import Axes3D
# plt.rcParams['figure.figsize'] = (16, 9)
# plt.style.use('ggplot')

# X = np.array(df[["DEM_RACEETH", "RELIG_IMPORT", "INCGROUP_PREPOST"]])
# y = np.array(df['LIBCPRE_SELF'])
# X.shape
# fig = plt.figure()
# ax = Axes3D(fig)
# colores=['blue','red','green','blue','cyan','yellow','orange','black','pink','brown','purple']
# asignar=[]
# for row in y:
#     asignar.append(colores[row])
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar,s=60)