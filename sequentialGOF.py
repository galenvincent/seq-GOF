import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors as nn
import sklearn.ensemble as ens
import sklearn.neural_network as nnet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from tqdm import tqdm

# Use Knn with this loss
def prob_class_loss(Y, Y_pred):
    pos_cases = np.where(Y == 1)
    return (Y_pred**2).mean() - 2/len(Y)*Y_pred[pos_cases].sum()

class LongSequence:
    # This is a general class that holds a long sequence (or maybe a number of long sequences)
    # which can then be chopped up into smaller sequences using the methods within.
    def __init__(self, data):
        self.data = data
        self.N = data.shape[1]
        self.num_seq = data.shape[0]

    def extract_overlap():
        pass

class NormalSequence:
    # Class for creating a LongSequence of simple normal randoms.
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def draw(self, size):
        # Size is a tuple of (num_rows, num_cols), or alternatively (numer of sequences, length of sequences).
        return LongSequence(np.random.normal(loc = self.mu, scale = self.sigma, size = size))

class MarkovChain:
    # Class that you can give an order / transition matrix that will initialize 
    # the chain for being drawn from in the future (with a "draw" method or something
    # similar). Draw method should return instance of LongSequence Class.
    def __init__(self):
        pass

    def draw():
        pass

class MCTrain(MarkovChain):
    # Inhererted class from the MarkovChain class. The initializer on this class 
    # will instead take in a MC order and some data and then fit the transition 
    # matrix (see Trey's code for how to do this). It will then call the MarkovChain
    # initializer and create an instance of that class, all other methods stay
    # the same.
    def __init__(self):
        
        # Train MC
        self.train_data = ...
        super().__init__(...)
        pass

class KnnRegressor:
    '''
    K-nearest neighbor regression.

    Note: For >1 dimension, all dimensions should be on the same scale.
    '''
    def __init__(self, variables=["x"], k=None):
        self.regression = None
        self.variables = variables
        self.k = k

    def fit(self, data):
        n = len(data)
        if self.k == 'heuristic':
            self.k = int(np.floor(np.sqrt(n)))
            self.regression = nn.KNeighborsClassifier(n_neighbors=self.k)
            self.regression.fit(
                data[self.variables].values.reshape(-1, len(self.variables)),
                data['Y'].values
            )
        elif self.k is None:
            ks = [2**ii+1 for ii in range(3, int(np.log2(n)+1))]
            loss = np.zeros(int(np.log2(n)-2))
            ii = 0
            for kk in ks:
                self.regression = nn.KNeighborsClassifier(n_neighbors=kk)
                loss[ii] = cross_val_score(self.regression,
                                           data[self.variables].values.reshape(-1, len(self.variables)),
                                           data['Y'].values,
                                           cv=10,
                                           scoring=make_scorer(
                                               prob_class_loss,
                                               needs_proba=True
                                           )
                                           ).mean()
                ii += 1
            self.k = ks[np.where(loss == loss.min())[0][0]]
            self.regression = nn.KNeighborsClassifier(n_neighbors=self.k)
            self.regression.fit(
                data[self.variables].values.reshape(-1, len(self.variables)), 
                data['Y'].values
            )
        else:
            self.regression = nn.KNeighborsClassifier(k=self.k)
            self.regression.fit(
                data[self.variables].values.reshape(-1, len(self.variables)), 
                data['Y'].values
            )

    def predict(self, data):
        return self.regression.predict_proba(
            data[self.variables].values.reshape(-1, len(self.variables))
        )[:, 1]





