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
parser.add_argument("--L", type=int, action='store', default=1)
parser.add_argument("--n1", type=int, action='store', default=200)
parser.add_argument("--n0", type=int, action='store', default=200)
parser.add_argument("--m1", type=int, action='store', default=50)
parser.add_argument("--m0", type=int, action='store', default=50)
parser.add_argument("--B", type=int, action='store', default=200)

# Add any MC arguments down here. Alpha is a placeolder (not used)
parser.add_argument("--alpha", type=float, action='store', default=0.125)

parser.add_argument("--folder", type=str, action='store', default='')
parsed = parser.parse_args()

N = parsed.N
L = parsed.L
n1 = parsed.n1
n0 = parsed.n0
m1 = parsed.m1
m0 = parsed.m0
B = parsed.B
alpha = parsed.alpha
save_folder = parsed.folder


def perform_test(ii, real_dist, emulated_dist, n1, n0, m1, m0, L, B):
    
    sim = gof.Simulation(real_dist, emulated_dist, n1, n0, m1, m0, L)

    covars = [f'x-{j}' for j in range(L)] # Get lagged covariates to use e.g. ['x', 'x-1', 'x-2'] for L = 3
    covars[0] = 'x'
    
    # TODO: Change regressor?
    reg = gof.KnnRegressor(variables = covars)

    sim.test(reg, B)

    sim.data.evaluation['replication'] = ii + 1

    return [sim.data.evaluation, sim.get_global(), 
            sim.cross_entropy(), sim.prior_adjusted_cross_entropy(),
            sim.brier_score(), sim.prior_adjusted_brier_score(),
            sim.mse(), sim.mae()]

num_cores = multiprocessing.cpu_count()
iterations = tqdm(range(N), desc = "Replications")
parallel_verbose = Parallel(n_jobs = num_cores, verbose = 5)

# Create MC distributions/estimation
real_MC = gof.MarkovChain()
emulated_MC = gof.MarkovChain(order = 1)

raw_output = parallel_verbose(delayed(perform_test)(ii, real_MC, emulated_MC, n1, n0, m1, m0, L, B) for ii in iterations)

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
                 '-B_'+str(int(B))+
                 '-L_'+str(int(L))+
                 '-n1_'+str(int(n1))+
                 '-n0_'+str(int(n0))+
                 '-m1_'+str(int(m1))+
                 '-m0_'+str(int(m0))+
                 '-local'+
                 '.csv', index = False)
pvals_glob.to_csv(save_folder +
                    'reps_'+str(int(N))+
                    '-B_'+str(int(B))+
                    '-L_'+str(int(L))+
                    '-n1_'+str(int(n1))+
                    '-n0_'+str(int(n0))+
                    '-m1_'+str(int(m1))+
                    '-m0_'+str(int(m0))+
                    '-global'+
                    '.csv', index = False)