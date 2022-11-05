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
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

FILE = '20220918_data.csv'
FILE_EBERT = 'roger-ebert-great-movies.csv'


# -

def load_data(ebert=False):
    if ebert:
        return load_ebert()
    
    df = pd.read_csv(FILE)
    df.drop(columns=['hire_date'], inplace=True)

    # create dummy variables out of the categorical demographic features
    for var in ('gender', 'ethnicity'):
        temp = pd.get_dummies(df[var], prefix=var, drop_first=True)
        df.drop(columns=[var], inplace=True)
        df = df.join(temp)

    y = df['terminated_in_first_6_months']
    X = df.drop(columns=['terminated_in_first_6_months'])        
        
    # scale all inputs between -1 and 1
    X = MinMaxScaler((-1, 1)).fit_transform(X)

    return X, y


def load_ebert():
    """the third object are all non Ebert reviews to label as being great movies or not"""
    df = pd.read_csv(FILE_EBERT)
    
    features = [
        'sentiment_sentiment_neg',
        'sentiment_sentiment_neu',
        'sentiment_sentiment_pos',
        'sentiment_sentiment_compound',
        'review_length',
    ]
    
    ebert_index = df.reviewer == 'Roger Ebert'
    
    # scale all inputs between -1 and 1
    scaler = MinMaxScaler((-1, 1))
    scaler.fit(df[features])
    
    X = scaler.transform(df[ebert_index][features])
    y = df[ebert_index]['gm']
    
    # keep titles of non ebert films for funsies
    non_ebert_reviews = df[~ebert_index]
    non_ebert_reviews.loc[:, features] = scaler.transform(non_ebert_reviews[features])

    return X, y, non_ebert_reviews


