import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors as nn
import sklearn.ensemble as ens
import sklearn.neural_network as nnet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from tqdm import tqdm
import copy

class LongSequence:
    # This is a general class that holds a long sequence (or maybe a number of long sequences)
    # which can then be chopped up into smaller sequences using the methods within.
    def __init__(self, data):
        self.data = data
        self.N = data.size

    def extract_overlap(self, L):
        # Convert a long sequence into a collection of short sequences of length 
        # L, where 1 < L < N. Assume that later entries in self.data are more
        # recent in time (the typical convention). 
        s_set = []
        data_rev = self.data[::-1]
        for ii in range(self.N - L + 1):
            s_set.append(data_rev[ii : ii + L])
        s_set = np.array(s_set)

        # name columns appropriately based on lag
        cols = ['x']
        for ii in range(1, s_set.shape[1]):
            cols.append('x-' + str(ii))

        s_set_pd = pd.DataFrame(s_set, columns=cols)

        return s_set_pd

class TrainTestData:
    def __init__(self, real_data, emulated_data, n1, n0, m1 = None, m0 = None):
        '''
        This class is for creating and holding testing and training data in the 
        correct form.

        real_data and emulated_data are DataFrames as returned from extract_overlap.
        n1: training size from real data
        n0: training size from emulated data
        m1: evaluation size from real data (not neccesary to fill in if n1 + m1 = total real)
        m0: evaluation size from emulated data (not neccesary to fill in if n0 + m0 = total emulated)
        '''
        if m1 is None:
            m1 = real_data.shape[0] - n1
        if m0 is None:
            m0 = emulated_data.shape[0] - n0


        # Check that n1, n0, m1, m0 entered make sense
        assert n1 + m1 <= real_data.shape[0], "n1 + m1 is larger than number of rows in real_data."
        assert n0 + m0 <= emulated_data.shape[0], "n1 + m1 is larger than number of rows in real_data."

        assert real_data.shape[1] == emulated_data.shape[1], "Real and emulated data must have same number of columns."

        self.n1 = n1
        self.n0 = n0
        self.m1 = m1
        self.m0 = m0

        self.real = real_data
        self.emulated = emulated_data

        # Append real and emulated data frames with Y = 1 and 0 and organize into 
        # training and evaluation sets
        real_data['Y'] = 1
        emulated_data['Y'] = 0

        real_train = real_data.head(n1)
        real_eval = real_data.tail(m1).reset_index(drop = True)
        emulated_train = emulated_data.head(n0)
        emulated_eval = emulated_data.tail(m0).reset_index(drop = True)

        self.training = pd.concat([real_train, emulated_train], ignore_index=True)
        self.evaluation = pd.concat([real_eval, emulated_eval], ignore_index=True)

        
class NormalSequence:
    # Class for creating a LongSequence of simple normal randoms.
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def draw(self, size):
        # Size is an integer giving the length of the sequence to be drawn
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

# Use Knn with this loss
def prob_class_loss(Y, Y_pred):
    pos_cases = np.where(Y == 1)
    return (Y_pred**2).mean() - 2/len(Y)*Y_pred[pos_cases].sum()

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
            n = int(n*0.9) # adjust sample size for 10 fold cross validation
            ks = [2**ii+1 for ii in range(3, int(np.log2(n)+1))]
            loss = np.zeros(len(ks))
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


class Simulation:
    def __init__(self, real_dist, emulated_dist, n1, n0, m1, m0, L = 1):
        
        self.real_dist = real_dist
        self.emulated_dist = emulated_dist
        self.N_real = n1 + m1
        self.N_emulated = n0 + m0
        self.n1 = n1
        self.n0 = n0
        self.m1 = m1
        self.m0 = m0
        self.L = L

        # Data generation
        #self.real_dist = NormalSequence(mu1, sigma1)
        real_Z = self.real_dist.draw(self.N_real)
        real_S_set = real_Z.extract_overlap(L)

        #self.emulated_dist = NormalSequence(mu0, sigma0)
        emulated_Z = self.emulated_dist.draw(self.N_emulated)
        emulated_S_set = emulated_Z.extract_overlap(L)

        self.data = TrainTestData(real_S_set, emulated_S_set, n1, n0, m1, m0)

        self.tested = False
        self.B = None

    def test(self, regression, B = 200, progress_bar = False):
        self.B = B
        self.r0 = copy.copy(regression)
        
        # Fit regression and get/save local scores
        self.r0.fit(self.data.training)
        self.pi_hat = self.data.training['Y'].mean()
        self.p0 = self.r0.predict(self.data.evaluation) - self.pi_hat
        self.data.evaluation['local_stat'] = self.p0
        self.data.evaluation['prob_est'] = self.r0.predict(self.data.evaluation)

        self.P = np.zeros((len(self.data.evaluation), B))
        if progress_bar:
            for bb in tqdm(range(B), desc = 'Computing null distribution', leave=False):
                real_Z_b = self.emulated_dist.draw(self.n1)
                real_S_set_b = real_Z_b.extract_overlap(self.L)

                emulated_Z_b = self.emulated_dist.draw(self.n0)
                emulated_S_set_b = emulated_Z_b.extract_overlap(self.L)

                data_b = TrainTestData(real_S_set_b, emulated_S_set_b, self.n1, self.n0, 0, 0)

                r_b = copy.copy(regression)
                r_b.fit(data_b.training)

                self.P[:, bb] = r_b.predict(self.data.evaluation) - self.pi_hat
        else:
            for bb in range(B):
                real_Z_b = self.emulated_dist.draw(self.n1)
                real_S_set_b = real_Z_b.extract_overlap(self.L)

                emulated_Z_b = self.emulated_dist.draw(self.n0)
                emulated_S_set_b = emulated_Z_b.extract_overlap(self.L)

                data_b = TrainTestData(real_S_set_b, emulated_S_set_b, self.n1, self.n0, 0, 0)

                r_b = copy.copy(regression)
                r_b.fit(data_b.training)

                self.P[:, bb] = r_b.predict(self.data.evaluation) - self.pi_hat
        
        pvals = np.zeros(len(self.data.evaluation))
        for ii in range(len(pvals)):
            pvals[ii] = (np.sum(np.abs(self.p0[ii]) < np.abs(self.P[ii, :])) + 1) / (self.B + 1)
        self.data.evaluation['local_pval'] = pvals
        self.data.evaluation['imp_score'] = 1/pvals
        self.data.evaluation['imp_score_sign'] = (self.data.evaluation['local_stat'] > self.data.evaluation['local_stat'].mean()).astype(int)*2 - 1
        self.tested = True

    def get_global(self):
        if not self.tested:
            raise Exception("Test statistics not computed. Run test() first.")

        glob_obs = np.power(self.p0, 2).sum()
        glob_null = np.power(self.P, 2).sum(axis = 0)
        
        return (len(np.where(glob_null > glob_obs)[0]) + 1) / (self.B + 1)




