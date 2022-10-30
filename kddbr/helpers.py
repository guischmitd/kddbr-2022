from matplotlib import pyplot as plt
import cv2
import numpy as np
import pandas as pd
from context import DATADIR
from lightgbm import LGBMRegressor
from sklearn.multioutput import RegressorChain
from sklearn.base import BaseEstimator, clone

import joblib

def read_before_and_after(img_path, switch_rb=False, normalize=False):
    out = cv2.imread(str(img_path))
    out = out[:, :, ::-1] if switch_rb else out
    out = out / 255. if normalize else out
    
    h, w, c = out.shape
    
    return out[:, :h, :], out[:, h:, :]


def plot_row(row, datadir = DATADIR):
    print(row)
    
    path = datadir / 'raw' / 'train' / 'train' / row['Filename']
    if not path.exists():
        path = datadir / 'raw' / 'test' / 'test' / row['Filename']
    
    img1, img2 = read_before_and_after(path)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(img1[:, :, ::-1])
    plt.subplot(1, 2, 2)
    plt.imshow(img2[:, :, ::-1])

    return img1, img2


class EnsembledRegressorChains(BaseEstimator):
    def __init__(self, estimator, weights=None) -> None:
        super().__init__()
        self.estimator = estimator
        self.weights = weights or [0.5, 0.5]
        self.chains = [RegressorChain(clone(self.estimator), order=[1, 0]), RegressorChain(clone(self.estimator), order=[0, 1])]
        self._feature_names = []

    def fit(self, X, y):
        self._feature_names = X.columns

        for chain in self.chains:
            chain.fit(X, y)

        return self

    def predict(self, X):
        y_pred = np.zeros((len(X), 2))
        for chain, weight in zip(self.chains, self.weights):
            y_pred = y_pred + (weight * chain.predict(X))
        
        return y_pred

    def get_feature_names(self):
        return self._feature_names


def build_model():
    estimator = LGBMRegressor(n_estimators=3000, random_state=51, n_jobs=-1)
    model = EnsembledRegressorChains(estimator, weights=[0.5, 0.5])

    return model


def get_Xy_cols(df):
    targets = ['North', 'East']
    predictors = [c for c in df.columns if (c not in targets + ['sequence', 'Filename'])]

    return predictors, targets


def generate_submission_files(model_path, test_feats):
    model = joblib.load(model_path)

    predictors = model.get_feature_names()

    y_pred = model.predict(test_feats[predictors])
    
    df = test_feats.copy()
    df[['North', 'East']] = y_pred

    north_preds = df['North'].reset_index()
    north_preds['Id'] = north_preds['Filename'] + ':North'
    north_preds = north_preds.rename({'North': 'Predicted'}, axis=1)
    north_preds = north_preds.set_index('Id')['Predicted']

    east_preds = df['East'].reset_index()
    east_preds['Id'] = east_preds['Filename'] + ':East'
    east_preds = east_preds.rename({'East': 'Predicted'}, axis=1)
    east_preds = east_preds.set_index('Id')['Predicted']

    return pd.concat([east_preds, north_preds]).sort_index()