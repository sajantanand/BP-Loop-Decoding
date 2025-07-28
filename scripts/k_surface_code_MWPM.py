import os
import sys
from copy import deepcopy
import pickle
import shutil

nc = sys.argv.pop(1)
os.environ['OMP_NUM_THREADS'] = nc
os.environ['OPENBLAS_NUM_THREADS'] = nc
os.environ['MKL_NUM_THREADS'] = nc
os.environ['VECLIB_MAXIMUM_THREADS'] = nc
os.environ['NUMEXPR_NUM_THREADS'] = nc

import numpy as np
import scipy as sp
import time
from tqdm import tqdm

from src.codes import toric
import fssa
from pymatching import Matching

L_str = sys.argv.pop(1)
Ls = [int(l) for l in L_str.split('-')]
k_str = sys.argv.pop(1)
ks = [int(k) for k in k_str.split('-')]
num_shots = int(sys.argv.pop(1))
seed = int(sys.argv.pop(-1))
if seed >= 0:
    np.random.seed(seed)
else:
    np.random.seed(int(time.time()))

def MWPM_num_decoding_failures(H, logicals, error_probability, num_shots):
    matching = Matching.from_check_matrix(H, weights=np.log((1-error_probability)/error_probability), faults_matrix=logicals)
    noise = (np.random.random((num_shots, H.shape[1])) < error_probability).astype(np.uint8)
    shots = (noise @ H.T) % 2
    actual_observables = (noise @ logicals.T) % 2
    predicted_observables = matching.decode_batch(shots)
    num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))
    return num_errors

def MWPM_simulation(Ls, ks, ps, num_shots):
    #ps = np.linspace(0.08, 0.12, 20)
    log_errors_all_k = []
    std_err_all_k = []
    pths = []
    for k in ks:
        log_errors_all_L = []
        std_err_all_L = []
        for L in Ls:   
            print("Simulating L={}, k={}...".format(L, k))
            H = toric.surface_code_z(L, k)
            log = toric.surface_code_logical_z(L, k)
            log_errors = []
            for p in tqdm(ps):
                num_errors = MWPM_num_decoding_failures(H, log, float(p), num_shots)
                log_errors.append(num_errors/num_shots)
            log_errors_all_L.append(np.array(log_errors))
        for L, logical_errors in zip(Ls, log_errors_all_L):
            std_err = (logical_errors*(1-logical_errors)/num_shots)**0.5
            std_err_all_L.append(std_err)
        log_errors_all_k.append(np.array(log_errors_all_L))
        std_err_all_k.append(np.array(std_err_all_L))
        
        try:
            res = fssa.autoscale(Ls, ps, np.array(log_errors_all_L), np.array(std_err_all_L), 0.1, 1, 1)
            pths.append((res['rho'], res['nu'], res['zeta'], res['errors']))
        except ValueError as e:
            print(f"No logical error encountered with distance {Ls}: {np.any(np.isclose(np.array(log_errors_all_L), 0.0), axis=1)}.")
            print(np.min(np.array(log_errors_all_L), axis=1))
            pths.append((np.nan, np.nan, np.nan, np.nan))
        print("k={} threshold p_th={}.".format(k, np.round(pths[-1][0], 5)))
    return np.array(log_errors_all_k), np.array(std_err_all_k), pths

if __name__ == '__main__':
    #ps = np.linspace(0.08, 0.12, 20)
    ps = np.logspace(np.log(5.e-2), np.log(0.075), 5, base=np.e)
    log_errors, std_err, ths = MWPM_simulation(Ls, ks, ps, num_shots)

    with open(f"MWPM_L{L_str}_k{k_str}_ns{num_shots}_seed{seed}.pkl", "wb") as f:
        pickle.dump({'ps': ps,
                     'Ls': Ls,
                     'ks': ks,
                     'num_shots': num_shots,
                     'seed': seed,
                     'logical_error': log_errors,
                     'std_err': std_err,
                     'ths': ths}, f)
