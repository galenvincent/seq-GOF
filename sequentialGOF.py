import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats 
import pomegranate as pom
import sklearn.neighbors as nn
import sklearn.ensemble as ens
import sklearn.neural_network as nnet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, log_loss, mean_squared_error, mean_absolute_error, brier_score_loss
from tqdm import tqdm
import copy
import math

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
        assert n1 + m1 <= real_data.shape[0], f"n1 + m1 ({n1 + m1}) is larger than number of rows in real_data ({real_data.shape[0]})."
        assert n0 + m0 <= emulated_data.shape[0], f"n0 + m0 ({n0 + m0}) is larger than number of rows in emulated_data ({emulated_data.shape[0]})."

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

    def true_probabilities(self, data, emulator_dist, pi):
        """
        Calculate true values for m_post = P(Y = 1 | X).

        Arguments:
            - data (pandas.DataFrame): Result of calling extract_overlap from LongSequence class.
            - emulator_dist (NormalSequence): The NormalSequence object for the emulator distribution
            - pi (float): On [0, 1]. Value of marginal P(Y = 1) to use for calculation.
        """

        def m_post(row):
            p_x_given_1 = scipy.stats.norm.pdf(row, loc = self.mu, scale = self.sigma)
            p_x_given_0 = scipy.stats.norm.pdf(row, loc = emulator_dist.mu, scale = emulator_dist.sigma)
            return (p_x_given_1[0] * pi)/(p_x_given_1[0] * pi + p_x_given_0[0] * (1 - pi))
        
        return np.array(data.apply(m_post, axis = 1))


class MarkovChain:
    """
    Class that you can give an order / transition matrix that will initialize 
    the chain for being drawn from in the future (with a "draw" method or something
    similar). Draw method should return instance of LongSequence Class.

    Parameters:
       - d_0: Dictionary of zeroth order transition probabilities (i.e. marginals)
       - d_1: Dictionary of first order transition probabilities 
       - d_2: Dictionary of second order transition probabilities
    """
    
    def __init__(self,
                 d0 = {'0': 0.5, '1': 0.5}, 
                 d1 = {'0,0': 0.5, '0,1': 0.5,
                        '1,0': 0.5, '1,1': 0.5}, 
                 d2 = {'00,0': 0.125, '00,1': 0.875,
                        '10,0': 0.875, '10,1': 0.125,
                        '01,0': 0.5, '01,1': 0.5,
                        '11,0': 0.5, '11,1': 0.5},
                 order = 2
                 ):

        d0_pom = pom.DiscreteDistribution(d0)

        d1_pom = pom.ConditionalProbabilityTable([['1', '1', d1['1,1']],
                                            ['1', '0', d1['1,0']],
                                            ['0', '1', d1['0,1']],
                                            ['0', '0', d1['0,0']]], [d0_pom])

        d2_pom = pom.ConditionalProbabilityTable([['1', '1', '1', d2['11,1']],
                                            ['1', '1', '0', d2['11,0']],
                                            ['1', '0', '1', d2['10,1']],
                                            ['1', '0', '0', d2['10,0']],
                                            ['0', '1', '1', d2['01,1']],
                                            ['0', '1', '0', d2['01,0']],
                                            ['0', '0', '1', d2['00,1']],
                                            ['0', '0', '0', d2['00,0']]], [d0_pom, d1_pom])
        
        self.order = order

        if self.order == 1:
            self.chain = pom.MarkovChain([d0_pom, d1_pom])
        elif self.order == 2:
            self.chain = pom.MarkovChain([d0_pom, d1_pom, d2_pom])

    def draw(self, n, burnin = 50):
        samp = self.chain.sample(n + burnin)[burnin:]
        return LongSequence(np.array(list(map(int, samp))))

    def true_probabilities(self, data, emulator_dist, pi):
        """
        Calculate true values for m_post = P(Y = 1 | X).

        Arguments:
            - data (pandas.DataFrame): Result of calling extract_overlap from LongSequence class.
            - emulator_MC (MarkovChain): The MC object for the emulator distribution
            - pi (float): On [0, 1]. Value of marginal P(Y = 1) to use for calculation.
        """
        
        def m_post(row):
            row_formatted = [str(y) for y in list(row[::-1])]
            p_x_given_1 = np.exp(self.chain.log_probability(row_formatted))
            p_x_given_0 = np.exp(emulator_dist.chain.log_probability(row_formatted))
            return (p_x_given_1 * pi)/(p_x_given_1 * pi + p_x_given_0 * (1 - pi))
        
        return np.array(data.apply(m_post, axis = 1))

    def get_trans_matrix(self):
        return self.chain.distributions[self.order].to_dict()['table']

class MCTrain(MarkovChain):
    # Inhererted class from the MarkovChain class. The initializer on this class 
    # will instead take in a MC order and some data and then fit the transition 
    # matrix (see Trey's code for how to do this). It will then call the MarkovChain
    # initializer and create an instance of that class, all other methods stay
    # the same.
    def __init__(self, training_data, order = 2):
        
        # Train MC
        self.chain = pom.MarkovChain.from_samples(training_data, k = order)

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
            n = int(math.floor(n*0.85)) # adjust sample size for 10 fold cross validation
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
        self.N_real = n1 + m1 + L - 1
        self.N_emulated = n0 + m0 + L -1
        self.n1 = n1
        self.n0 = n0
        self.m1 = m1
        self.m0 = m0
        self.L = L

        # Data generation
        real_Z = self.real_dist.draw(self.N_real)
        real_S_set = real_Z.extract_overlap(L)

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
        self.data.evaluation['LPD'] = self.p0
        self.data.evaluation['prob_est'] = self.r0.predict(self.data.evaluation)

        self.P = np.zeros((len(self.data.evaluation), B))
                          
        if progress_bar:
            for bb in tqdm(range(B), desc = 'Computing null distribution', leave=False):
                real_Z_b = self.emulated_dist.draw(self.n1 + self.L - 1)
                real_S_set_b = real_Z_b.extract_overlap(self.L)

                emulated_Z_b = self.emulated_dist.draw(self.n0 + self.L - 1)
                emulated_S_set_b = emulated_Z_b.extract_overlap(self.L)

                data_b = TrainTestData(real_S_set_b, emulated_S_set_b, self.n1, self.n0, 0, 0)

                r_b = copy.copy(regression)
                r_b.fit(data_b.training)

                self.P[:, bb] = r_b.predict(self.data.evaluation) - self.pi_hat
        else:
            for bb in range(B):
                real_Z_b = self.emulated_dist.draw(self.n1 + self.L - 1)
                real_S_set_b = real_Z_b.extract_overlap(self.L)

                emulated_Z_b = self.emulated_dist.draw(self.n0 + self.L - 1)
                emulated_S_set_b = emulated_Z_b.extract_overlap(self.L)

                data_b = TrainTestData(real_S_set_b, emulated_S_set_b, self.n1, self.n0, 0, 0)

                r_b = copy.copy(regression)
                r_b.fit(data_b.training)

                self.P[:, bb] = r_b.predict(self.data.evaluation) - self.pi_hat
        
        # Calculate local p-values
        #pvals = np.zeros(len(self.data.evaluation))
        #for ii in range(len(pvals)):
        #    pvals[ii] = (np.sum(np.abs(self.p0[ii]) < np.abs(self.P[ii, :])) + 1) / (self.B + 1)
        #self.data.evaluation['local_pval'] = pvals
        #self.data.evaluation['imp_score'] = 1/pvals
        #self.data.evaluation['imp_score_sign'] = (self.data.evaluation['LPD'] > self.data.evaluation['LPD'].mean()).astype(int)*2 - 1
        self.tested = True

        self.adjust_probs()
        self.true_probs()

    def get_global(self):
        if not self.tested:
            raise Exception("Test statistics not computed. Run test() first.")

        glob_obs = np.power(self.p0, 2).sum()
        glob_null = np.power(self.P, 2).sum(axis = 0)
        
        return (len(np.where(glob_null > glob_obs)[0]) + 1) / (self.B + 1)

    def adjust_probs(self):
        """
        Adjust posterior pobability estimates P(Y = 1 | X) to the probabilities
        as they are calculated using the evaluation set proportions of Y = 1 and Y = 0.
        """

        if not self.tested:
            raise Exception("Orginal probabilities not computed. Run test() first.")

        pi_hat_1 = self.n1/(self.n1 + self.n0)
        pi_hat_0 = 1 - pi_hat_1

        pi_hat_eval_1 = self.m1/(self.m1 + self.m0)
        pi_hat_eval_0 = 1 - pi_hat_eval_1

        adjusted_probs = (pi_hat_eval_1/pi_hat_1)*self.data.evaluation['prob_est']/((pi_hat_eval_1/pi_hat_1)*self.data.evaluation['prob_est'] + (pi_hat_eval_0/pi_hat_0)*(1-self.data.evaluation['prob_est']))
        
        self.data.evaluation['adjusted_prob_est'] = adjusted_probs

    def true_probs(self):
        """
        Get the true posterior probability P(Y = 1 | X) w.r.t. the evaluation 
        set proportions of Y = 1 and Y = 0.
        """

        if not self.tested:
            raise Exception("Orginal probabilities not computed. Run test() first.")

        covars = [f'x-{j}' for j in range(self.L)] # Get lagged covariates to use e.g. ['x', 'x-1', 'x-2'] for L = 3
        covars[0] = 'x'

        true_probs = self.real_dist.true_probabilities(data = self.data.evaluation[covars], 
                                                       emulator_dist = self.emulated_dist, 
                                                       pi = self.m1/(self.m1 + self.m0))

        self.data.evaluation['true_prob'] = true_probs

    def cross_entropy(self, separate = False):
        if not self.tested:
            raise Exception("Test statistics not computed. Run test() first.")

        return log_loss(self.data.evaluation['Y'], self.data.evaluation['prob_est'])

    def prior_adjusted_cross_entropy(self):
        if not self.tested:
            raise Exception("Test statistics not computed. Run test() first.")
        if 'adjusted_prob_est' not in self.data.evaluation:
            raise Exception("Adjusted probabilities not computed. Run adjust_probs() first.")

        return log_loss(self.data.evaluation['Y'], self.data.evaluation['adjusted_prob_est'])

    def brier_score(self):
        if not self.tested:
            raise Exception("Test statistics not computed. Run test() first.")
        
        return brier_score_loss(self.data.evaluation['Y'], self.data.evaluation['prob_est'])
    
    def prior_adjusted_brier_score(self):
        if not self.tested:
            raise Exception("Test statistics not computed. Run test() first.")
        if 'adjusted_prob_est' not in self.data.evaluation:
            raise Exception("Adjusted probabilities not computed. Run adjust_probs() first.")

        return brier_score_loss(self.data.evaluation['Y'], self.data.evaluation['adjusted_prob_est'])

    def mse(self):
        if not self.tested:
            raise Exception("Test statistics not computed. Run test() first.")
        if 'adjusted_prob_est' not in self.data.evaluation:
            raise Exception("Adjusted probabilities not computed. Run adjust_probs() first.")
        if 'true_prob' not in self.data.evaluation:
            raise Exception("True probabilities not computed. Run true_probs() first.")

        return mean_squared_error(self.data.evaluation['adjusted_prob_est'], self.data.evaluation['true_prob'])

    def mae(self):
        if not self.tested:
            raise Exception("Test statistics not computed. Run test() first.")
        if 'adjusted_prob_est' not in self.data.evaluation:
            raise Exception("Adjusted probabilities not computed. Run adjust_probs() first.")
        if 'true_prob' not in self.data.evaluation:
            raise Exception("True probabilities not computed. Run true_probs() first.")

        return mean_absolute_error(self.data.evaluation['adjusted_prob_est'], self.data.evaluation['true_prob'])

        



