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
from ldpc import BpDecoder
from ldpc import BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder

L_str = sys.argv.pop(1)
Ls = [int(l) for l in L_str.split('-')]
k_str = sys.argv.pop(1)
ks = [int(k) for k in k_str.split('-')]
num_shots = int(sys.argv.pop(1))
BP_method = sys.argv.pop(1)                 # product_sum
max_iter = int(sys.argv.pop(1))             # 7
schedule = sys.argv.pop(1)                  # serial
OSD = int(sys.argv.pop(1))                  # 0 - Bare BP; 1 - OSD; 2 - LSD
OSD_method = sys.argv.pop(1)
OSD_order = int(sys.argv.pop(1))
seed = int(sys.argv.pop(1))
if seed >= 0:
    np.random.seed(seed)
else:
    np.random.seed(int(time.time()))

BP_params = {'bp_method': BP_method,
             'max_iter': max_iter,
             'schdule': schedule}
OSD_params = {'osd_method': OSD_method,
              'osd_order': OSD_order}

def build_decoder(H, error_probability, BP_params, OSD, OSD_params):
    if OSD == 1:
        bp =    BpOsdDecoder(
                H,
                error_rate = error_probability,
                bp_method = BP_params.get('bp_method', 'product_sum'),
                max_iter = BP_params.get('max_iter', 7),
                schedule = BP_params.get('schedule', 'serial'),
                osd_method = OSD_params.get('osd_method', 'osd_cs'), #set to OSD_0 for fast solve
                osd_order = OSD_params.get('osd_order', 2)
            )
    elif OSD == 2:
        bp =    BpLsdDecoder(
                H,
                error_rate = error_probability,
                bp_method = BP_params.get('bp_method', 'product_sum'),
                max_iter = BP_params.get('max_iter', 7),
                schedule = BP_params.get('schedule', 'serial'),
                osd_method = OSD_params.get('osd_method', 'lsd_cs'), #set to OSD_0 for fast solve
                osd_order = OSD_params.get('osd_order', 2)
            )
    elif OSD == 0:
        bp =    BpDecoder(
                H, #the parity check matrix
                error_rate = error_probability, # the error rate on each bit
                bp_method = BP_params.get('bp_method', 'product_sum'), #BP method. The other option is `minimum_sum'
                max_iter = BP_params.get('max_iter', 7), #the maximum iteration depth for BP
                schedule = BP_params.get('schedule', 'serial'),
            )
    else:
        raise ValueError()
    return bp

def BP_num_decoding_failures(H, logicals, error_probability, num_shots, BP_params, OSD, OSD_params):
    bp = build_decoder(H, error_probability, BP_params, OSD, OSD_params)
    noise = (np.random.random((num_shots, H.shape[1])) < error_probability).astype(np.uint8)
    shots = (noise @ H.T) % 2
    actual_observables = (noise @ logicals.T) % 2
    predicted_observables = []
    for sh in shots:
        # This is the recovery string.
        predicted_observables.append(bp.decode(sh))
    predicted_observables = (np.vstack(predicted_observables) @ logicals.T) % 2
    num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))
    return num_errors
    
def BP_simulation(Ls, ks, num_shots, BP_params, OSD, OSD_params):
    ps = np.linspace(0.08, 0.12, 20)
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
                num_errors = BP_num_decoding_failures(H, log, float(p), num_shots, BP_params, OSD, OSD_params)
                log_errors.append(num_errors/num_shots)
            log_errors_all_L.append(np.array(log_errors))
        for L, logical_errors in zip(Ls, log_errors_all_L):
            std_err = (logical_errors*(1-logical_errors)/num_shots)**0.5
            std_err_all_L.append(std_err)
        log_errors_all_k.append(np.array(log_errors_all_L))
        std_err_all_k.append(np.array(std_err_all_L))

        res = fssa.autoscale(Ls, ps, np.array(log_errors_all_L), np.array(std_err_all_L), 0.1, 1, 1)
        pths.append((res['rho'], res['nu'], res['zeta'], res['errors']))
        print("k={} threshold p_th={}.".format(k, np.round(pths[-1][0], 5)))
    return np.array(log_errors_all_k), np.array(std_err_all_k), pths

if __name__ == '__main__':
    log_errors, std_err, ths = BP_simulation(Ls, ks, num_shots, BP_params, OSD, OSD_params)
    
    if OSD != 0:
        with open(f"BP{OSD}_L{L_str}_k{k_str}_ns{num_shots}_seed{seed}_BPm{BP_method}_mi{max_iter}_sch{schedule}_OSDm{OSD_method}_OSDo{OSD_order}.pkl", "wb") as f:
            pickle.dump({'ps': np.linspace(0.08, 0.12, 20),
                         'Ls': Ls,
                         'ks': ks,
                         'num_shots': num_shots,
                         'seed': seed,
                         'OSD': OSD,
                         'BP_params': BP_params,
                         'OSD_params': OSD_params,
                         'logical_error': log_errors,
                         'std_err': std_err,
                         'ths': ths}, f)
    else:
        with open(f"BP{OSD}_L{L_str}_k{k_str}_ns{num_shots}_seed{seed}_BPm{BP_method}_mi{max_iter}_sch{schedule}.pkl", "wb") as f:
            pickle.dump({'ps': np.linspace(0.08, 0.12, 20),
                         'Ls': Ls,
                         'ks': ks,
                         'num_shots': num_shots,
                         'seed': seed,
                         'OSD': OSD,
                         'BP_params': BP_params,
                         'logical_error': log_errors,
                         'std_err': std_err,
                         'ths': ths}, f)

