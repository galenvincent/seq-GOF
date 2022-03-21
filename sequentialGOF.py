import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error, mean_absolute_error, brier_score_loss
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.api import acf
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

        return s_set_pd.sort_index(ascending = False, ignore_index=True).iloc[:, ::-1] # Re-order rows and columns back to something that makes more logical sense

class CustomArProcess(ArmaProcess):
    """
    Custom version of the statsmodels ArmaProcess class with a few additional 
    functions that will be useful for our application.
    """
    def __init__(self, ar = None, scale = 1.0, nobs = 100):
        self.scale = scale
        super().__init__(ar=ar, ma=None, nobs=nobs)
    
    def generate_sample_custom(self, nsample=100, starters=None, scale=None):
        """
        Generate a sample based on some starting values given by 'starters'.
        Returns a seqeunce where the first elements are given by the values in 
        'starters'.
        """
        assert starters is not None, 'starters must be provided. Use generate_sample if no need for starters.'
        nstarters = len(starters)
        nlags = len(self.arcoefs)
        assert nstarters >= nlags, f'starters is length {nstarters} but must be at at least as long as order of AR process, order = {nlags}.'

        scale = scale if scale else self.scale

        eps = np.random.normal(loc = 0, scale = scale, size = nsample)
        y = np.zeros(nsample)
        y = np.concatenate((starters, y))
        
        for ii in range(nstarters, nstarters + nsample):
            y[ii] = np.dot(y[(ii-nlags):ii], self.arcoefs) + eps[ii - nstarters]
        
        return y
    
    def draw(self, n):
        nburn = 100
        nlags = len(self.arcoefs)
        eps = np.random.normal(loc = 0, scale = self.scale, size = n+nburn)
        y = np.zeros(n+nburn)
        for ii in range(nlags, n+nburn):
            y[ii] = np.dot(y[(ii-nlags):ii], self.arcoefs) + eps[ii]

        return y[nburn:]
    
    def evaluate_likelihood(self, data, scale=None):
        """
        Evaluate the likelihood of some sample for an AR(1) model.
        See chapter 5.2 of https://www.degruyter.com/document/doi/10.1515/9780691218632/html 
        data[0] is earier in time, data[n] is later in time
        """
        assert len(self.arcoefs) == 1, f'This method is currently only implemented for AR(1) models. This is an AR({len(self.arcoefs)}) model.'
        n = len(data)
        phi = self.arcoefs[0]
        scale = scale if scale else self.scale

        process_sd = scale/np.sqrt(1 - phi**2)
        f_1 = scipy.stats.norm.pdf(data[0], loc = 0, scale = process_sd)

        prod = 1
        for ii in range(1, n):
            prod = prod * scipy.stats.norm.pdf(data[ii], loc = phi*data[ii-1], scale = scale)
        
        return f_1 * prod

    def true_probabilities(self, data, generative_dist, pi, K):
        """
        Calculate true values for m_post = P(Y = 1 | X) based on known true and emulator distributions,
        and specified marginal pi = P(Y = 1).

        P(Y = 1 | X = x) = P(X = x| Y = 1) * P(Y = 1) / [ P(X = x| Y = 1) * P(Y = 1) + P(X = x| Y = 0) * P(Y = 0) ]

        Arguments:
            - data (pandas.DataFrame): Result of calling extract_overlap from LongSequence class. 
                Containing only and all columns for the sequence data (e.g. x, x-1, x-2, ..., x - L + 1).
            - generative_dist (CustomArProcess): The CustomArProcess object for the generative distribution.
            - pi (float): On [0, 1]. Value of marginal P(Y = 1) to use for calculation.
            - K (int): Number of datapoints in the generated sequence B.
        """

        def m_post(row):
            assert 'Y' not in row, 'Response Y should not be a column in data'
            row_B = row[-K:]
            p_x_given_1 = self.evaluate_likelihood(row_B)
            p_x_given_0 = generative_dist.evaluate_likelihood(row_B)
            return (p_x_given_1 * pi)/(p_x_given_1 * pi + p_x_given_0 * (1 - pi))
        
        return np.array(data.apply(m_post, axis = 1))

class VarProcess:
    def __init__(self, coefs, sigma):
        """
        Parameters:
            - coefs (np.ndarray): p x k x k array holding coefficients for the VAR model.
            - sigma (np.ndarray): k x k covariance matrix for gaussian noise on the VAR model.
        """
        self.coefs = coefs
        self.sigma = sigma
        self.p = coefs.shape[0]
        self.k = coefs.shape[1]
        
        assert coefs.shape[1] == coefs.shape[2], 'coefs has improper dimensions.'


    def generate_sample(self, nsample=100, burnin=100, starters=None):
        """
        Generate a simulated sample from the VAR model, with or without a starter sequence

        Parameters:
        - nsample (int):
        - burnin (int): If starters == None, how many data points at beginning of
            simulation to throw out?
        - starters (np.ndarray): If None, generate unconcitionally. Otherwise, 
            an n x k ndarray of observations to start off the party. Must be the
            case that n >= p
        - f (function): Takes starters as an argument, 
        """

        # Initialize with zeros if no starters provided
        if starters is None:
            starters = np.zeros((self.p, self.k))
            nstarters = self.p
            nsim = nsample + burnin 
        else:
            nstarters = starters.shape[0]
            assert nstarters >= self.p, 'Must provide at least p observations in starters.'
            nsim = nsample
            burnin = 0
        
        # Simulate errors
        W =  np.random.multivariate_normal(np.zeros(self.k), self.sigma, size=nsim)
        
        # Simulate process
        Y = np.zeros((nsim, self.k))
        Y[:nstarters,:] = starters
        
        for t in range(nstarters, nsim):
            Y[t] = W[t]
            for i in range(1, self.p+1):
                Y[t] += self.coefs[i-1] @ Y[t-i]
        
        Y = Y[burnin:]
        
        return Y
    

class DataConstructor:
    def __init__(self, real_data, generative_mod, ntrain, neval, mtrain, meval, L, J, null_hyp = False):
        """
             - f (function): Takes in sequence of length J, returns True if th
        """
        assert ntrain + neval <= len(real_data), f"ntrain + neval = {ntrain + neval}, len(real_data) = {len(real_data)}."

        train_data = real_data.iloc[:ntrain]
        test_data = real_data.iloc[ntrain:(ntrain+neval)].reset_index(drop = True)     

        cols = list(real_data.columns)
        self.seq_cols = cols

        def get_generated(row, m, add_label = True):
            S = row.to_numpy()
            A = S[:J]
            
            # If generating under the null, create list of all generated sequences, 
            # otherwise, use the real sequence as the first element of the list
            if null_hyp:
                seq_list = []
                m_rep = m + 1
            else:
                seq_list = [S]
                m_rep = m

            for jj in range(m_rep):
                seq_list.append(generative_mod.generate_sample_custom(nsample = L - J, starters = A))
            df = pd.DataFrame(seq_list, columns = cols)
            
            if add_label:
                labs = np.zeros(m+1, dtype = int)
                labs[0] = 1
                df['Y'] = labs
            
            return df

        # Construct training set
        T_list = []
        for row in train_data.iterrows():
            T_list.append(get_generated(row[1], mtrain))
        T = pd.concat(T_list, ignore_index = True)
        self.training = T

        # Construct evaluation set
        if neval > 0:
            V_list = []
            for row in test_data.iterrows():
                V_list.append(get_generated(row[1], meval))
            V = pd.concat(V_list, ignore_index = True)
            self.evaluation = V
        else:
            self.evaluation = None
        
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

class ARLogisticRegressor:
    """
    Logistic regression based on empirical autoregression coefficient of input sequence.

    Arguments:
        - columns (list of str): Column names for the sequence to use, where earlier 
            items in the list represent earlier in time: columns[0] is earlier in 
            time than columns[1], etc.
        - nlags (int): Number of autocorrelation lags to use in the regression.
            e.g. nlags = 1 will use only the first-order autocorrelation to predict class label.
        - **kwargs: Argumnets to be passed to sklearn.linear_model.LogisticRegression.
            Any hyperparameters for the regression.
    """
    def __init__(self, columns, nlags, **kwargs):
        self.regression = LogisticRegression(**kwargs)
        self.nlags = nlags
        self.columns = columns
    
    def fit(self, data, get_acf = False):
        """
        Arguments:
            - data (pd.DataFrame): Training data. Must have response column named 'Y'.
        """
        # Estimate autocorrelation
        data_cut = data[self.columns]
        acfs = data_cut.apply(lambda x: acf(x, nlags = self.nlags, fft = True)[1:], axis = 1, result_type = 'expand')
        
        # Fit regression
        assert 'Y' in data, 'Response column with name Y not found in provided training data.'
        self.regression.fit(acfs, data['Y'])
        if get_acf:
            return acfs
    
    def get_acf(self, data):
        data_cut = data[self.columns]
        acfs = data_cut.apply(lambda x: acf(x, nlags = self.nlags, fft = True)[1:], axis = 1, result_type = 'expand')
        return acfs

    def predict(self, data):
        """
        Return predicted probabilities P(Y = 1 | X) for provided data X.

         Arguments:
            - data (pd.DataFrame): Data to provide probabilistic estimates for.
        """
        data_cut = data[self.columns]
        acfs = data_cut.apply(lambda x: acf(x, nlags = self.nlags, fft = True)[1:], axis = 1, result_type = 'expand')
        return self.regression.predict_proba(acfs)[:, 1]


class Simulation:
    def __init__(self, real_dist, generative_mod, ntrain, neval, mtrain, meval, L, J):
        
        self.real_dist = real_dist
        self.generative_mod = generative_mod
        self.ntrain = ntrain
        self.neval = neval 
        self.ntot = ntrain + neval + L - 1
        self.mtrain = mtrain
        self.meval = meval
        self.L = L
        self.J = J
        self.K = L - J

        # Data generation
        real_Z = LongSequence(self.real_dist.draw(self.ntot))
        self.real_S_set = real_Z.extract_overlap(L)

        self.data = DataConstructor(self.real_S_set, self.generative_mod, self.ntrain, self.neval, self.mtrain, self.meval, self.L, self.J)

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
        self.data.training['acf'] = self.r0.get_acf(self.data.training)

        self.P = np.zeros((len(self.data.evaluation), B))
        self.ACF = np.zeros((len(self.data.evaluation), B))
                          
        if progress_bar:
            for bb in tqdm(range(B), desc = 'Computing null distribution', leave=False):
                data_b = DataConstructor(self.real_S_set, self.generative_mod, self.ntrain, 0, self.mtrain, self.meval, self.L, self.J, null_hyp = True)

                r_b = copy.copy(regression)
                r_b.fit(data_b.training)

                self.P[:, bb] = r_b.predict(self.data.evaluation) - self.pi_hat
                self.ACF[:, bb] = r_b.get_acf(data_b.training).to_numpy().flatten()

        else:
            for bb in range(B):
                data_b = DataConstructor(self.real_S_set, self.generative_mod, self.ntrain, 0, self.mtrain, self.meval, self.L, self.J, null_hyp = True)

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

        pi_hat_1 = self.pi_hat
        pi_hat_0 = 1 - pi_hat_1

        pi_hat_eval_1 = self.data.evaluation['Y'].mean()
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

        true_probs = self.real_dist.true_probabilities(data = self.data.evaluation[self.data.seq_cols], 
                                                       generative_dist = self.generative_mod, 
                                                       pi = self.data.evaluation['Y'].mean(),
                                                       K = self.K)

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

        



