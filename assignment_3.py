# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import numpy as np
import pandas as pd
import seaborn as sns

from helpers import load_data

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as EM

from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection as RP
from sklearn.feature_selection import VarianceThreshold

# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
from sklearn.metrics import silhouette_score, mean_squared_error

# -

def run_clustering(X, y=None, algo=KMeans, k=range(2, 11)):
    out = []
    for i in k:
        algo_fit = algo(i, random_state=0).fit(X)
        ss = silhouette_score(X, algo_fit.predict(X))
        out.append({
            'k': i,
            'silhouette_score': ss,
            'algorithm': str(algo.__name__),
            'fit': algo_fit,
        })
        
    best_index = np.argmax([elem['silhouette_score'] for elem in out])
    for i, elem in enumerate(out):
        elem['best'] = i == best_index
    
    return out


def apply_pca2(X, y=None):
    pca = PCA(n_components=2, random_state=0)
    return pca.fit_transform(X)



def plot_and_save(title, kind=sns.lineplot, **kwargs):
    plot = kind(**kwargs)
    plot.set_title(title.upper())
    fig = plot.get_figure()
    fig.savefig(f"plots/{title.replace(' ', '_')}.png")



def rmse_compared_to_original(X, algo):
    """compare the original data to the tranformed, then projected into the original space"""
    Xtransform = algo.transform(X)
    return mean_squared_error(algo.inverse_transform(Xtransform), X, squared=False)



def run_dimensionality_reduction(X, y=None, algo=PCA, k=None):
    if k is None:
        k = range(1, X.shape[1])
        
    out = []
    for i in k:
        try:
            algo_fit = algo(i, random_state=0).fit(X)
        except:
            algo_fit = algo(i).fit(X)
            
        transform = algo_fit.transform(X)
        x = {
            'k': i,
            'algorithm': str(algo.__name__),
            'fit': algo_fit,
            'transform': transform,
            'rmse': rmse_compared_to_original(X, algo_fit),
            'features': transform.shape[1],
        }
        
        if 'explained_variance_ratio_' in dir(algo_fit):
            x['explained_variance'] = sum(algo_fit.explained_variance_ratio_)
        else:
            x['explained_variance'] = None
        
        out.append(x)
        
    return out


# +
X1, y1 = load_data()
print(X1.shape)

# X2, y2, _ = load_data(ebert=True)
# print(X2.shape)

# -

df = pd.DataFrame(run_clustering(X1, k=range(2, 11)))
title = 'kmeans dataset 1'
plot_and_save(title, data=df, x="k", y="silhouette_score")


# +
pca1 = apply_pca2(X1)
temp = pd.DataFrame(pca1)
temp['cluster'] = df[df.best==True].fit.iloc[0].predict(X1)

title = 'kmeans dataset 1 clusters pca 1 and 2'
plot_and_save(title, kind=sns.scatterplot, data=temp, x=0, y=1, hue='cluster', style='cluster')
# -



df = pd.DataFrame(run_clustering(X1, algo=EM, k=range(2, 11)))
title = 'em dataset 1'
plot_and_save(title, data=df, x="k", y="silhouette_score")


# +
pca1 = apply_pca2(X1)
temp = pd.DataFrame(pca1)
temp['cluster'] = df[df.best==True].fit.iloc[0].predict(X1)

title = 'em dataset 1 clusters pca 1 and 2'
plot_and_save(title, kind=sns.scatterplot, data=temp, x=0, y=1, hue='cluster', style='cluster')
# -

run_dimensionality_reduction(X1, k=range(2, 3))[0]['fit'].get_covariance()

# +
########## dimensionality reduction

df = pd.DataFrame(run_dimensionality_reduction(X1))

# -

dr = []
for i, algo in enumerate((PCA, FastICA, RP, VarianceThreshold)):
    k = np.arange(0, 1, 0.025) if i == 3 else None
    dr.extend(run_dimensionality_reduction(X1, algo=algo, k=k))


df = pd.DataFrame(dr)

title='dimensionality reduction dataset 1'
plot_and_save(title, data=df, x="features", y="rmse", hue='algorithm', kind=sns.lineplot, alpha=0.5)


title='dimensionality reduction dataset 1 pca variance'
plot_and_save(title, data=df[df.algorithm=='PCA'], x="features", y="explained_variance", hue='algorithm', kind=sns.lineplot, alpha=0.5)







