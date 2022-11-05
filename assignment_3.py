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
from pprint import pprint

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
X, y = load_data()
print(X.shape)

# X2, y2, _ = load_data(ebert=True)
# print(X2.shape)

# -

df = pd.DataFrame(run_clustering(X, k=range(2, 11)))
title = 'kmeans dataset 1'
plot_and_save(title, data=df, x="k", y="silhouette_score")


# +
pca1 = apply_pca2(X)
temp = pd.DataFrame(pca1)
temp['cluster'] = df[df.best==True].fit.iloc[0].predict(X)

title = 'kmeans dataset 1 clusters pca 1 and 2'
plot_and_save(title, kind=sns.scatterplot, data=temp, x=0, y=1, hue='cluster', style='cluster')
# -



df = pd.DataFrame(run_clustering(X, algo=EM, k=range(2, 11)))
title = 'em dataset 1'
plot_and_save(title, data=df, x="k", y="silhouette_score")


# +
pca1 = apply_pca2(X)
temp = pd.DataFrame(pca1)
temp['cluster'] = df[df.best==True].fit.iloc[0].predict(X)

title = 'em dataset 1 clusters pca 1 and 2'
plot_and_save(title, kind=sns.scatterplot, data=temp, x=0, y=1, hue='cluster', style='cluster')
# -

run_dimensionality_reduction(X, k=range(2, 3))[0]['fit'].get_covariance()

# +
########## dimensionality reduction

df = pd.DataFrame(run_dimensionality_reduction(X))

# -

dr = []
for i, algo in enumerate((PCA, FastICA, RP, VarianceThreshold)):
    k = np.arange(0, 1, 0.025) if i == 3 else None
    dr.extend(run_dimensionality_reduction(X, algo=algo, k=k))


df = pd.DataFrame(dr)

title='dimensionality reduction dataset 1'
plot_and_save(title, data=df, x="features", y="rmse", hue='algorithm', kind=sns.lineplot, alpha=0.5)



title='dimensionality reduction dataset 1 pca variance'
plot_and_save(title, data=df[df.algorithm=='PCA'], x="features", y="explained_variance", hue='algorithm', kind=sns.lineplot, alpha=0.5)



# +
N_FEATURES = 17  # half of "optimal" number of dimensions

D = df[df.features==N_FEATURES].drop_duplicates(subset=['algorithm'])

part3 = []
for i, row in D.iterrows():
    x = row['transform']
    dr_algo = row['algorithm']
    for c_algo in (KMeans, EM):
        out = run_clustering(x, algo=c_algo)
        for o in out:
            o['dimension_reduction'] = dr_algo
            if o['algorithm'] == 'GaussianMixture':
                o['algorithm'] = 'EM'
            
        part3.extend(out)

# -

part3 = pd.DataFrame(part3)

# +

title = f'clustering with {N_FEATURES} features of dataset 1'
plot_and_save(title, data=part3, x="k", y="silhouette_score", hue='dimension_reduction', style='algorithm')
# -

D[D.algorithm == 'PCA']

# +
x = D[D.algorithm == 'PCA']['transform'].iloc[0] ### PCA
pca1 = apply_pca2(x)
temp = pd.DataFrame(pca1)
temp['cluster'] = KMeans(5, random_state=0).fit(x).predict(x)

title = 'kmeans dataset 1 with 17 features clusters pca 1 and 2'
plot_and_save(title, kind=sns.scatterplot, data=temp, x=0, y=1, hue='cluster', style='cluster')

# +
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler

# original dataset
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)
X_train, y_train = RandomOverSampler(random_state=0).fit_resample(X_train, y_train)

parameters = {
    'hidden_layer_sizes': [(x,) for x in 2 ** np.arange(1, 5)],
    'activation': ['logistic'],
    'solver': ['sgd'],
    'alpha': [0],
    'batch_size': 2 ** np.arange(7, 11),
    'learning_rate_init': [0.001, 0.002, 0.003],
    'max_iter': [500],
    'momentum': [0, 0.25, 0.5, 0.75],
    'early_stopping': [True],
    'n_iter_no_change': [20],
    'random_state': [0],
}

N=200
cv = RandomizedSearchCV(
    MLPClassifier(),
    parameters,
    n_iter=N, scoring='f1', n_jobs=-1,
    random_state=0, verbose=1,
)
cv.fit(X_train, y_train)

# store all results for exploration
frame = pd.DataFrame(cv.cv_results_)

pprint({
    'best_params': cv.best_params_,
    'best_score': cv.best_score_,
    'best_time': frame[frame.rank_test_score==1].mean_fit_time.iloc[0],
})

# get the best model and run over the entire train and test set for comparison
best_estimator = cv.best_estimator_
pprint({
    'full_train_f1': f1_score(best_estimator.predict(X_train), y_train),
    'full_test_f1': f1_score(best_estimator.predict(X_test), y_test),
    'full_train_accuracy': best_estimator.score(X_train, y_train),
    'full_test_accuracy': best_estimator.score(X_test, y_test),
})
# -



# +
# dimenionality reduced dataset
print(x.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2, stratify=y)
X_train, y_train = RandomOverSampler(random_state=0).fit_resample(X_train, y_train)

parameters = {
    'hidden_layer_sizes': [(x,) for x in 2 ** np.arange(1, 5)],
    'activation': ['logistic'],
    'solver': ['sgd'],
    'alpha': [0],
    'batch_size': 2 ** np.arange(7, 11),
    'learning_rate_init': [0.001, 0.002, 0.003],
    'max_iter': [500],
    'momentum': [0, 0.25, 0.5, 0.75],
    'early_stopping': [True],
    'n_iter_no_change': [20],
    'random_state': [0],
}

N=200
cv = RandomizedSearchCV(
    MLPClassifier(),
    parameters,
    n_iter=N, scoring='f1', n_jobs=-1,
    random_state=0, verbose=1,
)
cv.fit(X_train, y_train)

# store all results for exploration
frame = pd.DataFrame(cv.cv_results_)

pprint({
    'best_params': cv.best_params_,
    'best_score': cv.best_score_,
    'best_time': frame[frame.rank_test_score==1].mean_fit_time.iloc[0],
})

# get the best model and run over the entire train and test set for comparison
best_estimator = cv.best_estimator_
pprint({
    'full_train_f1': f1_score(best_estimator.predict(X_train), y_train),
    'full_test_f1': f1_score(best_estimator.predict(X_test), y_test),
    'full_train_accuracy': best_estimator.score(X_train, y_train),
    'full_test_accuracy': best_estimator.score(X_test, y_test),
})

# +
# dimenionality reduced dataset, predicting clusters
print(x.shape)

# two means since our original data is two means
clusters = KMeans(2, random_state=0).fit(x).predict(x)

X_train, X_test, y_train, y_test = train_test_split(x, clusters, random_state=0, test_size=0.2, stratify=clusters)
X_train, y_train = RandomOverSampler(random_state=0).fit_resample(X_train, y_train)

parameters = {
    'hidden_layer_sizes': [(x,) for x in 2 ** np.arange(1, 5)],
    'activation': ['logistic'],
    'solver': ['sgd'],
    'alpha': [0],
    'batch_size': 2 ** np.arange(7, 11),
    'learning_rate_init': [0.001, 0.002, 0.003],
    'max_iter': [500],
    'momentum': [0, 0.25, 0.5, 0.75],
    'early_stopping': [True],
    'n_iter_no_change': [20],
    'random_state': [0],
}

N=200
cv = RandomizedSearchCV(
    MLPClassifier(),
    parameters,
    n_iter=N, scoring='f1', n_jobs=-1,
    random_state=0, verbose=1,
)
cv.fit(X_train, y_train)

# store all results for exploration
frame = pd.DataFrame(cv.cv_results_)

pprint({
    'best_params': cv.best_params_,
    'best_score': cv.best_score_,
    'best_time': frame[frame.rank_test_score==1].mean_fit_time.iloc[0],
})

# get the best model and run over the entire train and test set for comparison
best_estimator = cv.best_estimator_
pprint({
    'full_train_f1': f1_score(best_estimator.predict(X_train), y_train),
    'full_test_f1': f1_score(best_estimator.predict(X_test), y_test),
    'full_train_accuracy': best_estimator.score(X_train, y_train),
    'full_test_accuracy': best_estimator.score(X_test, y_test),
})
# -

(max( # f1 score
    f1_score(y, clusters),
    f1_score(y, 1 - clusters),
),
max( # accuracy
    np.mean(y == clusters),
    np.mean(y == 1 - clusters),
))




