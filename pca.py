import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('white')

data = pd.read_csv('ShinySeshatLatLong.csv',index_col=0)

cols = ['Supernatural_enforcement_of_fairness',
        'Supernatural_enforcement_of_human_reciprocity',
        'Supernatural_enforcement_of_ingroup_loyalty']

data = data[cols]

def truncate(x):
    if x < 0:
        return 0
    if x > 1:
        return 1
    return x

for col in cols:
    data[col] = data[col].apply(truncate)

pca = PCA()
pca.fit(data)
components = pca.transform(data)

print(pca.explained_variance_)
#print(pca.explained_variance_ratio_)

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(loadings, columns=['PC1','PC2','PC3'], index=cols)
print(loading_matrix)

pcs = (pd.DataFrame(components,columns=['PC1','PC2','PC3']))


x = pcs['PC1']
y = pcs['PC2']
sns.kdeplot(x,y,cmap='plasma_r',shade=True,shade_lowest=False,bw=0.15, alpha = 1)
plt.show()

## Notes:
## PC1 and PC2 seem to both explain a lot of variance. Unknown if the PCs are actually
## significant; refer to procedure outline in https://link.springer.com/article/10.1007/s13253-019-00355-5 to determine this
