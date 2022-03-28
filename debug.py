import sequentialGOF as gof
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

ii = 0
alpha = 0.0
delta = 0.0
ntrain = 300
neval = 300
mtrain = 3
meval = 1
L = 16
J = 8
Q = 10

# Create MC distributions/estimation
real_dist = gof.CustomArProcess(ar = np.array([1, -alpha]), scale = np.sqrt(1 - alpha**2))
emulator_dist = gof.CustomArProcess(ar = np.array([1, -delta]), scale = np.sqrt(1 - delta**2))


sim = gof.Simulation(real_dist, emulator_dist, ntrain, neval, mtrain, meval, L, J)

covars = [f'x-{j}' for j in range(L-1, -1, -1)] # Get lagged covariates to use e.g. ['x-2', 'x-1', 'x'] for L = 3
covars[L-1] = 'x'

reg = gof.ARLogisticRegressor(columns = covars, nlags = 1)

sim.test(regression = reg, B = Q, progress_bar = True)