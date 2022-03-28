#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Use the methods defined in sequentialGOF to run some experiments 
# for validity, power, etc.

import sequentialGOF as gof
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import argparse

import multiprocessing
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, action='store', default=200)
parser.add_argument("--ntrain", type=int, action='store', default=300)
parser.add_argument("--neval", type=int, action='store', default=300)
parser.add_argument("--mtrain", type=int, action='store', default=10)
parser.add_argument("--meval", type=int, action='store', default=1)
parser.add_argument("--L", type=int, action='store', default=16)
parser.add_argument("--J", type=int, action='store', default=8)
parser.add_argument("--Q", type=int, action='store', default=200)
parser.add_argument("--stride", type=int, action='store', default=1)

parser.add_argument("--alpha", type=float, action='store', default=0.6) # AR Coefficient
parser.add_argument("--delta", type=float, action='store', default=-0.6) # Emulator AR Coefficient

parser.add_argument("--folder", type=str, action='store', default='')
parsed = parser.parse_args()

N = parsed.N
ntrain = parsed.ntrain
neval = parsed.neval
mtrain = parsed.mtrain
meval = parsed.meval
Q = parsed.Q
L = parsed.L
J = parsed.J
alpha = parsed.alpha
delta = parsed.delta
stride = parsed.stride
save_folder = parsed.folder


def perform_test(ii, real_dist, emulator_dist, ntrain, neval, mtrain, meval, L, J, Q, stride):
    
    sim = gof.Simulation(real_dist, emulator_dist, ntrain, neval, mtrain, meval, L, J, stride)

    covars = [f'x-{j}' for j in range(L-J, -1, -1)] # Get coefficients corresponding to "future sequence" B
    covars[L-J] = 'x'
    
    reg = gof.ARLogisticRegressor(columns = covars, nlags = 1)

    sim.test(regression = reg, B = Q)

    sim.data.evaluation['replication'] = ii + 1

    return [sim.data.evaluation, sim.get_global(), 
            sim.cross_entropy(), sim.prior_adjusted_cross_entropy(),
            sim.brier_score(), sim.prior_adjusted_brier_score(),
            sim.mse(), sim.mae()]

num_cores = multiprocessing.cpu_count()
iterations = tqdm(range(N), desc = "Replications")
parallel_verbose = Parallel(n_jobs = num_cores, verbose = 5)

# Create MC distributions/estimation
real_dist = gof.CustomArProcess(ar = np.array([1, -alpha]), scale = np.sqrt(1 - alpha**2))
emulator_dist = gof.CustomArProcess(ar = np.array([1, -delta]), scale = np.sqrt(1 - delta**2))

raw_output = parallel_verbose(delayed(perform_test)(ii, real_dist, emulator_dist, ntrain, neval, mtrain, meval, L, J, Q, stride) for ii in iterations)

pvals_glob_list = []
pvals_loc_list = []
cross_entropy_list = []
adj_cross_entropy_list = []
brier_list = []
adj_brier_list = []
mse_list = []
mae_list = []

for x in raw_output:
    pvals_loc_list.append(x[0])
    pvals_glob_list.append(x[1])
    cross_entropy_list.append(x[2])
    adj_cross_entropy_list.append(x[3])
    brier_list.append(x[4])
    adj_brier_list.append(x[5])
    mse_list.append(x[6])
    mae_list.append(x[7])

pvals_loc = pd.concat(pvals_loc_list, ignore_index=True)
pvals_glob = pd.DataFrame(list(zip(pvals_glob_list, cross_entropy_list, adj_cross_entropy_list, brier_list, adj_brier_list, mse_list, mae_list)),
                          columns = ['pval', 'ce', 'adj_ce', 'bs', 'adj_bs', 'mse', 'mae'])



pvals_loc.to_csv(save_folder +
                 'reps_'+str(int(N))+
                 '-Q_'+str(int(Q))+
                 '-L_'+str(int(L))+
                 '-stride_'+str(int(stride))+
                 '-alpha_'+str(round(alpha, 1))+
                 '-delta_'+str(round(delta, 1))+
                 '-ntrain_'+str(int(ntrain))+
                 '-mtrain'+str(int(mtrain))+
                 '-neval_'+str(int(neval))+
                 '-meval_'+str(int(meval))+
                 '-local'+
                 '.csv', index = False)

pvals_glob.to_csv(save_folder +
                 'reps_'+str(int(N))+
                 '-Q_'+str(int(Q))+
                 '-L_'+str(int(L))+
                 '-stride_'+str(int(stride))+
                 '-alpha_'+str(round(alpha, 1))+
                 '-delta_'+str(round(delta, 1))+
                 '-ntrain_'+str(int(ntrain))+
                 '-mtrain'+str(int(mtrain))+
                 '-neval_'+str(int(neval))+
                 '-meval_'+str(int(meval))+
                 '-global'+
                 '.csv', index = False)