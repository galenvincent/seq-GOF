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
parser.add_argument("--n1", type=int, action='store', default=150)
parser.add_argument("--n0", type=int, action='store', default=150)
parser.add_argument("--m1", type=int, action='store', default=50)
parser.add_argument("--m0", type=int, action='store', default=50)
parser.add_argument("--B", type=int, action='store', default=200)
parser.add_argument("--mu1", type=float, action='store', default=0.0)
parser.add_argument("--mu0", type=float, action='store', default=2.0)
parser.add_argument("--sigma1", type=float, action='store', default=1.0)
parser.add_argument("--sigma0", type=float, action='store', default=1.0)
parser.add_argument("--folder", type=str, action='store', default='')
parsed = parser.parse_args()

N = parsed.N
L = parsed.L
n1 = parsed.n1
n0 = parsed.n0
m1 = parsed.m1
m0 = parsed.m0
B = parsed.B
mu1 = parsed.mu1
mu0 = parsed.mu0
sigma1 = parsed.sigma1
sigma0 = parsed.sigma0
save_folder = parsed.folder


def perform_test(ii, real_dist, emulated_dist, n1, n0, m1, m0, L, B):
    
    sim = gof.Simulation(real_dist, emulated_dist, n1, n0, m1, m0, L)

    reg = gof.KnnRegressor(variables = ['x'])

    sim.test(reg, B)

    sim.data.evaluation['replication'] = ii + 1

    return [sim.data.evaluation, sim.get_global()]

num_cores = multiprocessing.cpu_count()
iterations = tqdm(range(N), desc = "Replications")
parallel_verbose = Parallel(n_jobs = num_cores, verbose = 5)

real_normal_dist = gof.NormalSequence(mu1, sigma1)
emulated_normal_dist = gof.NormalSequence(mu0, sigma0)

raw_output = parallel_verbose(delayed(perform_test)(ii, real_normal_dist, emulated_normal_dist, n1, n0, m1, m0, L, B) for ii in iterations)

pvals_glob_list = []
pvals_loc_list = []

for x in raw_output:
    pvals_loc_list.append(x[0])
    pvals_glob_list.append(x[1])

pvals_glob = np.array(pvals_glob_list)
pvals_loc = pd.concat(pvals_loc_list, ignore_index=True)

pvals_loc.to_csv(save_folder +
                 'reps_'+str(int(N))+
                 '-B_'+str(int(B))+
                 '-L_'+str(int(L))+
                 '-n1_'+str(int(n1))+
                 '-n0_'+str(int(n0))+
                 '-m1_'+str(int(m1))+
                 '-m0_'+str(int(m0))+
                 '-mu1_'+str(mu1)+
                 '-mu0_'+str(mu0)+
                 '-sigma1_'+str(sigma1)+
                 '-sigma0_'+str(sigma0)+
                 '-local'+
                 '.csv')
np.savetxt(save_folder +
            'reps_'+str(int(N))+
            '-B_'+str(int(B))+
            '-L_'+str(int(L))+
            '-n1_'+str(int(n1))+
            '-n0_'+str(int(n0))+
            '-m1_'+str(int(m1))+
            '-m0_'+str(int(m0))+
            '-mu1_'+str(mu1)+
            '-mu0_'+str(mu0)+
            '-sigma1_'+str(sigma1)+
            '-sigma0_'+str(sigma0)+
           '-global'+
           '.csv', pvals_glob, delimiter=',')