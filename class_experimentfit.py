#%%
# standard library imports
import math
import os
import sys
from collections import defaultdict
import datetime
from dataclasses import dataclass, field
from typing import List, Optional
import logging
import importlib
import copy
from multiprocessing import Pool
#import time

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln
import pandas as pd
from tqdm import tqdm
import RNA
# assert RNA version is >= 2.7
if int(RNA.__version__.split('.')[0]) < 2 or (int(RNA.__version__.split('.')[0]) == 2 and int(RNA.__version__.split('.')[1]) < 7):
    raise ImportError(f"RNA version {RNA.__version__} is not supported. Please install RNA version >= 2.7.")
# USE ANDRONESCU 2007 PARAMETERS
#RNA.params_load_RNA_Andronescu2007()
# local imports
import class_experiment
from class_experiment import clocked, Experiment

kb=1.98717/1000 # grepped from vienna source code

def initialize_logger(name: str, log_file_path: str, debug: bool, print_to_std_out: bool) -> logging.Logger:
    # Always include process id to make logger names unique in multiprocessing
    name = f"{name}_{os.getpid()}"
    logger = logging.getLogger(name)
    # Always clear any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(log_file_path+"[%(levelname)s] %(message)s")
    
    if print_to_std_out:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def log_binomial(n, k):
    '''Compute the log of the binomial coefficient n choose k. Gammaln is an efficient way to compute the log of the factorial.'''
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def convert_p_bind_1d_to_dict(p_bind_1d, DMS_mode):
    '''Convert the 1D array of p_bind to a dictionary.
    If DMS_mode is True, input is a 4-element array and will put 0A, 0C to 0 and 1G, 1U to equal to  0G, 0U.'''
    if DMS_mode:
        p_bind_dict = {key: value for key, value in zip([(0, 'A'), (0, 'C'), (0, 'G'), (0, 'U')], p_bind_1d)}
        p_bind_dict[(1, 'A')] = 0
        p_bind_dict[(1, 'C')] = 0
        p_bind_dict[(1, 'G')] = p_bind_dict[(0, 'G')]
        p_bind_dict[(1, 'U')] = p_bind_dict[(0, 'U')]
    else:
        p_bind_dict = {key: value for key, value in zip([(0, 'A'), (0, 'C'), (0, 'G'), (0, 'U'), (1, 'A'), (1, 'C'), (1, 'G'), (1, 'U')], p_bind_1d)}
    return p_bind_dict

def convert_p_bind_dict_to_1d(p_bind_dict, DMS_mode):
    """
    Convert the dictionary of p_bind back to a 1D array.
    In DMS_mode (True), the dictionary is assumed to have keys:
      (0,'A'), (0,'C'), (0,'G'), (0,'U'),
    and paired keys (1,'G') and (1,'U') duplicate (0,'G') and (0,'U').
    Returns a 4-element array.
    When DMS_mode is False, the dictionary is assumed to have keys for both unpaired
    and paired bases in the order:
       (0,'A'), (0,'C'), (0,'G'), (0,'U'),
       (1,'A'), (1,'C'), (1,'G'), (1,'U').
    Returns an 8-element array.
    """
    if DMS_mode:
        return np.array([
            p_bind_dict[(0, 'A')],
            p_bind_dict[(0, 'C')],
            p_bind_dict[(0, 'G')],
            p_bind_dict[(0, 'U')]
        ])
    else:
        return np.array([
            p_bind_dict[(0, 'A')],
            p_bind_dict[(0, 'C')],
            p_bind_dict[(0, 'G')],
            p_bind_dict[(0, 'U')],
            p_bind_dict[(1, 'A')],
            p_bind_dict[(1, 'C')],
            p_bind_dict[(1, 'G')],
            p_bind_dict[(1, 'U')]
        ])
class ExperimentFit(Experiment):
    '''Takes in input an instance of Experiment (data at a given concentration) and computes the mutation profile and the loss, and their derivatives with respect to the parameters.'''
    def __init__(self, exp, *, infer_1D_sc, eps_b, is_training, debug=False, do_plots=False, logger = logging.getLogger('main'), custom_mask=None, use_interpolated_ps=False):
        # Initialize the Experiment attributes. These include reagent, concentration, sequence, df, etc.
        if hasattr(exp, 'path_to_info_txt'):    # this means we have all the information of the experiment in that file
            super().__init__(exp.path_to_info_txt)
        else:   # this is usually when we want to test on a simple sequence with no data
            super().__init__(seq=exp.seq)
            if hasattr(exp, 'df'):
                self.df = exp.df
            #if exp.reagent == 'DMS combined in vitro':
            self.temp_C = exp.temp_C
            self.temp_K = exp.temp_K
            self.reagent = exp.reagent
            self.system_name = exp.system_name
            self.conc_mM = exp.conc_mM

        # assert all attributes are the same
        for attr in exp.__dict__.keys():
            # do not consider ID
            if attr in ['ID', 'df', 'raw_df']:
                continue
            if hasattr(self, attr):
                assert getattr(self, attr) == getattr(exp, attr), f"Attribute {attr} does not match: {getattr(self, attr)} != {getattr(exp, attr)}"
            else:
                setattr(self, attr, getattr(exp, attr))
        self.debug = debug
        self.do_plots = do_plots
        self.infer_1D_sc = infer_1D_sc
        self.eps_b = eps_b
        RNA.cvar.temperature = self.temp_C  # set ViennaRNA temperature globally
        self.kBT = (self.temp_K * kb)
        self.beta = 1 / self.kBT
        self.is_training = is_training
        self.loss_history = []
        self.logger = logger
        self.use_interpolated_ps = use_interpolated_ps and infer_1D_sc
        self._cached_pairing_probs = None
        self._cached_gradients = None
        
        # Initialize position mask
        if custom_mask is not None:
            # Use the provided custom mask
            if len(custom_mask) != self.N_seq:
                raise ValueError(f"Custom mask length {len(custom_mask)} does not match sequence length {self.N_seq}")
            self.position_mask = np.array(custom_mask, dtype=bool)
            n_masked = np.sum(~self.position_mask)
            self.logger.info(f"ExperimentFit initialized with custom mask: {n_masked} positions excluded from loss calculation")
        else:
            # Default behavior: mask first and last 25 nucleotides
            self.position_mask = np.ones(self.N_seq, dtype=bool)
            if not self.is_synthetic:
                mask_start, mask_end = 25, 25
            else:
                mask_start, mask_end = 0, 0
            if mask_start > 0:
                self.position_mask[:mask_start] = False
            if mask_end > 0:
                self.position_mask[-mask_end:] = False
            
            self.logger.debug(f"ExperimentFit initialized with default edge masking: first {mask_start} and last {mask_end} positions excluded from loss calculation")
        
        self.logger.debug(f"Effective sequence length for fitting: {np.sum(self.position_mask)} / {self.N_seq}")

    # 0: SOFT CONSTRAINS APPLICATION
    # 0.1: sfot constrains from lambda_sc
    def apply_soft_constraints(self, lambda_sc, fold_compound):
        ''' Apply 1D array of soft constraints to the ViennaRNA fold compound. lambda must be passed is kbt units'''
        if lambda_sc is None:
            return
        # apply soft constraints from lambda_sc
        # CONVERT to kcal/mol
        #lambda_sc = [value * kb * self.temp_K for value in lambda_sc]  # Convert to kcal/mol
        for i in range(self.N_seq):
            if lambda_sc[i] == 0:
                continue
            elif lambda_sc[i] < 0:
                fold_compound.sc_add_up(i+1, -lambda_sc[i]) # add energy to unpaired bases
            elif lambda_sc[i] > 0: # the effect is the same but it more stable this way if lambda_sc is negative
                for j in range(len(lambda_sc)): # we go through all the couples twice, this is because penalty for pairing will be lambda_i + lambda_j
                    if j>i: # this is because vienna does not accept j<i
                        fold_compound.sc_add_bp(i+1,j+1,lambda_sc[i])
                    elif i>j:
                        fold_compound.sc_add_bp(j+1,i+1,lambda_sc[i])

    # soft constrains from p_b
    def apply_penalty_for_paired_bases(self, penalty, fold_compound):
        # if penalties is negative, alternative implementation is needed, but for p_b>0 you get m>0
        for i in range(self.N_seq):
            for j in range(i+1,self.N_seq):
                fold_compound.sc_add_bp(i+1,j+1,2*penalty) # penalty is already in kcal/mol

    # 1: compute p(c_b)
    # 1a: compute p(c_b|s_b,n_b)
    def compute_mu_dependent_quantities(self, mu, p_b, p_bind):
        '''Returns some key quantities that depend on mu: e^beta*mu, e^beta*mu', p(c_b|s_b,n_b)'''
        mu_prime = mu - p_b
        e_beta_mu = math.exp(self.beta*mu)
        e_beta_mu_prime = math.exp(self.beta*mu_prime)
        e_beta_mu_norm = e_beta_mu / (1 + e_beta_mu)
        e_beta_mu_prime_norm = e_beta_mu_prime / (1 + e_beta_mu_prime)
        # probability of chemical binding
        p_cb_given_sb_nb = dict() # p(cb=1|s_b,n_b) -> p_cb[(s_b,n_b)] , cb = 1 is implicit.
        for s_b in [0, 1]:
            physical_binding_probability = e_beta_mu_norm if s_b == 0 else e_beta_mu_prime_norm
            for n_b in ['A', 'C', 'G', 'U']:
                p_cb_given_sb_nb[(s_b, n_b)] = p_bind[s_b,n_b] * physical_binding_probability # p(cb=0|s_b,n_b) = 1 - p(cb=1|s_b,n_b)
        return e_beta_mu, e_beta_mu_prime, p_cb_given_sb_nb
    
    # 1b: compute p(s_b)
    def compute_pairing_probabilities(self, fold_compound, rescale=False):
        '''Compute the pairing probabilities. Note: this does not apply soft constrains'''
        RNA.cvar.temperature = self.temp_C
        if rescale:
            fold_compound.exp_params_rescale(fold_compound.mfe()[1]*0.01) # rescale the energy parameters to avoid pf overflow
        fold_compound.pf()
        bpp = np.array(fold_compound.bpp())[1:,1:]   # base pair probability matrix. Ask Giovanni for the [1:,1:]
        pairing = np.sum(bpp+bpp.T,axis=0)    # pairing probability for each base
        return pairing

    def dps_dlambda_sc(self, penalty, p_sb, lambda_sc):
        # compute dps_dlambda_j
        p_i_paired_given_j_unpaired = np.zeros((self.N_seq, self.N_seq))
        for j in range(self.N_seq):
            # basically what we do in compute_ps_from_scratch but we add an hard constraint on j
            fc = RNA.fold_compound(self.seq)
            RNA.cvar.temperature = self.temp_C
            fc.hc_add_up(j+1) # forbid pairing at position j
            self.apply_penalty_for_paired_bases(penalty, fc)
            self.apply_soft_constraints(lambda_sc, fc)
            p_i_paired_given_j_unpaired[:, j] = self.compute_pairing_probabilities(fc)
        # tested with explicit loop, vectorization should be ok
        first_term = (1-p_sb[np.newaxis,:]) * (p_sb[:,np.newaxis] - p_i_paired_given_j_unpaired)
        second_term = (1-p_sb[:,np.newaxis]) * (p_sb[np.newaxis,:] - p_i_paired_given_j_unpaired.T)
        if self.debug:
            assert np.max(np.abs(first_term - second_term)) < 1e-6
        dps_i_dlambda_j = -(first_term + second_term)/(2*self.kBT)
        dps_i_dlambda_j/= 2  # empirical factor of 2 to match finite differences
        return dps_i_dlambda_j
    
    def compute_and_store_ps_and_derivatives(self, penalty, lambda_sc, interpolate, compute_derivatives_anyway=False):
        lambda_sc_array = None if lambda_sc is None else np.asarray(lambda_sc, dtype=float)
        lambda_cache = None
        fold_compound = RNA.fold_compound(self.seq)
        RNA.cvar.temperature = self.temp_C
        # apply penalty for paired bases
        self.apply_penalty_for_paired_bases(penalty, fold_compound)
        # apply soft constraints from lambda_sc
        if self.use_interpolated_ps and lambda_sc_array is not None:
            lambda_rounded = np.round(lambda_sc_array, 2)
            self.apply_soft_constraints(lambda_rounded, fold_compound)
        else: # it's probably the same lol
            self.apply_soft_constraints(lambda_sc_array, fold_compound)
        # compute pairing probabilities
        pairing_probs = self.compute_pairing_probabilities(fold_compound)
        # interpolate
        if interpolate and lambda_sc_array is not None:
            # compute derivatives wrt lambda_sc
            dps_dlambda_sc = self.dps_dlambda_sc(penalty, pairing_probs, lambda_sc_array)
            epsilon=1e-10 # this is to avoid numerical issues when pairing is 0 or 1
            x=np.log(pairing_probs+epsilon)-np.log(1-pairing_probs+epsilon)
            dx=(1/(pairing_probs+epsilon)+1/(1-pairing_probs+epsilon))
            x+=dx*np.matmul(dps_dlambda_sc, lambda_sc_array-lambda_rounded)
            pairing_probs_interpolated=-epsilon+(1+2*epsilon)/(1+np.exp(-x))
            try:
                assert np.allclose(pairing_probs, pairing_probs_interpolated, atol=5e-2)
            except AssertionError:
                self.logger.warning("Warning: interpolated pairing probabilities differ significantly from computed ones.")
                self.logger.warning(f"Max difference: {np.max(np.abs(pairing_probs - pairing_probs_interpolated))}")
            # cap values to [0,1]
            pairing_probs_interpolated = np.clip(pairing_probs_interpolated, 0.0, 1.0)
            lambda_cache = lambda_sc_array.copy()
            self._cached_gradients = {'penalty': penalty, 'lambda_sc': lambda_cache, 'dps_dlambda_sc': dps_dlambda_sc, 'interpolated': interpolate}
            pairing_probs = pairing_probs_interpolated
        else:
            if compute_derivatives_anyway and lambda_sc_array is not None:
                dps_dlambda_sc = self.dps_dlambda_sc(penalty, pairing_probs, lambda_sc_array)
                lambda_cache = lambda_sc_array.copy()
                self._cached_gradients = {'penalty': penalty, 'lambda_sc': lambda_cache, 'dps_dlambda_sc': dps_dlambda_sc, 'interpolated': interpolate}
            else:
                lambda_cache = None if lambda_sc_array is None else lambda_sc_array.copy()
        if lambda_sc_array is None:
            lambda_cache = None
        self._cached_pairing_probs = {'penalty': penalty, 'lambda_sc': None if lambda_cache is None else lambda_cache.copy(), 'pairing_probs': pairing_probs, 'interpolated': interpolate}

    def compare_keys_of_cached_ps(self, penalty, lambda_sc, interpolated):
        '''Compare the keys of the cached pairing probabilities with the current ones.'''
        if self._cached_pairing_probs is None:
            return False
        if not np.isclose(self._cached_pairing_probs['penalty'], penalty):
            return False
        if not (self._cached_pairing_probs['interpolated'] == interpolated):
            return False
        cached_lambda = self._cached_pairing_probs['lambda_sc']
        if lambda_sc is None:
            if cached_lambda is not None:
                return False
        else:
            if cached_lambda is None or not np.allclose(cached_lambda, lambda_sc):
                return False
        return True
    
    def compare_keys_of_cached_gradients(self, penalty, lambda_sc, interpolated):
        '''Compare the keys of the cached gradients with the current ones.'''
        if self._cached_gradients is None:
            return False
        if not np.isclose(self._cached_gradients['penalty'], penalty):
            return False
        cached_lambda = self._cached_gradients['lambda_sc']
        if lambda_sc is None:
            if cached_lambda is not None:
                return False
        else:
            if cached_lambda is None or not np.allclose(cached_lambda, lambda_sc):
                return False
        if not (self._cached_gradients['interpolated'] == interpolated):
            return False
        return True

    def get_ps(self, penalty, lambda_sc, interpolated=False):
        '''Compute the pairing probabilities from scratch, i.e. for new fold compound after applying soft constraints
            for paired bases and soft constraints from lambda_sc'''
        # check if we can use cached pairing probabilities
        key_is_same = self.compare_keys_of_cached_ps(penalty, lambda_sc, interpolated)
        if self._cached_pairing_probs is None or not key_is_same:
            self.logger.debug("Computing pairing probabilities from scratch.")
            self.compute_and_store_ps_and_derivatives(penalty, lambda_sc, interpolated)
            return self._cached_pairing_probs['pairing_probs']
        else:
            return self._cached_pairing_probs['pairing_probs']

    def get_dps_dlambda_sc(self, penalty, lambda_sc, interpolated):
        if not self.infer_1D_sc:
            return None 
        # check for cached values
        key_is_same = self.compare_keys_of_cached_gradients(penalty, lambda_sc, interpolated)
        if self._cached_gradients is not None and key_is_same:
            return self._cached_gradients['dps_dlambda_sc']
        else:
            self.compute_and_store_ps_and_derivatives(penalty, lambda_sc, interpolated, compute_derivatives_anyway=True)
            return self._cached_gradients['dps_dlambda_sc']
            

    # 1c: compute p(c_b)
    def compute_p_cb(self, mu, p_b, p_bind, pairing_probs):
        '''Compute p(c_b) for the whole sequence'''
        # compute mu dependent quantities (and p(c_b|s_b,n_b))
        _, _, p_cb_given_sb_nb = self.compute_mu_dependent_quantities(mu, p_b, p_bind)
        # apply penalty for paired bases
        penalty = self.compute_penalty_m(mu, p_b)
        # compute p(s_b)
        try:
            assert 0 <= penalty <= 5
        except AssertionError:
            self.logger.warning('Warning, penalty is not in [0,5]')
            self.logger.warning(f'penalty: {penalty}, mu: {mu}, p_b: {p_b}')
        # compute p(c_b)
        p_cb = np.zeros(self.N_seq)
        for i,nb in enumerate(self.seq):    # weighted sum over s_b=0,1
            p_cb[i] = p_cb_given_sb_nb[(1, nb)]*pairing_probs[i] + p_cb_given_sb_nb[(0, nb)]*(1-pairing_probs[i])
        return p_cb

    # 2a: mutation profile model
    def compute_mutation_profile(self, m0, m1,eps_b, p_cb):
        '''Compute the mutation profile predicted by the model with the current parameters.
        Later: compute its derivatives'''
        # compute mutation profile
        Mb = 1 - np.exp(-eps_b) * (math.exp(-m0) +  p_cb*(math.exp(-m1) - math.exp(-m0)))
        return Mb
    
    # 2b: derivatives
    # 2b.1: derivatives of p(s_b)
    # 2b.1.3 derivatives of p(s_b) wrt mu and p_b
    def compute_penalty_m(self, mu, p_b):
        '''Compute the penalty for paired bases in kcal/mol'''
        mu_prime = mu-p_b
        penalty_in_kbt = -math.log((1+math.exp(self.beta*mu_prime))/(1+math.exp(self.beta*mu)))
        penalty_in_kcal_per_mol = penalty_in_kbt * self.kBT #TBC: to be checked
        return penalty_in_kcal_per_mol
    
    def numerical_derivatives_of_ps(self, mu, p_b, p_sb, lambda_sc):
        '''Compute the numerical derivatives of the pairing probabilities wrt mu and p_b.'''
        # df_dm with finite differences
        m = self.compute_penalty_m(mu, p_b)
        dm = .02
        # compute pp for forward difference
        pairing_probs_plus = self.get_ps(m+dm, lambda_sc, self.use_interpolated_ps)
        dps_dm = (pairing_probs_plus - p_sb)/dm # this should be the same as summing over the derivatives wrt lambda_j
        # dm_dmu
        dm_dmu = self.beta*math.exp(self.beta*mu)*(math.exp(self.beta*p_b)-1) #numerator
        dm_dmu /= (1+math.exp(self.beta*mu))*(math.exp(self.beta*mu)+math.exp(self.beta*p_b)) #denominator
        # dm_dpb
        mu_prime = mu-p_b
        dm_dpb = self.beta*math.exp(self.beta*mu_prime)
        dm_dpb /= (1+math.exp(self.beta*mu_prime))
        return dps_dm*dm_dmu, dps_dm*dm_dpb

    # 2b.2: derivatives of mutation profile
    def dMb_dmu(self, mu, p_b, dps_dmu, p_bind, m0, m1, eps_b, p_sb):
        # we need: dp(cb|s_b,n_b)/d(mu) -> dp(cb)/d(mu)
        # take p_bind(s_b,n_b) and multiply by dp(k=1|s_b)/d(mu) 
        #dpcb_given_sb_nb = {key: value* beta * math.exp(beta*mu)/((1+math.exp(beta*mu))**2) for key, value in p_bind.items()} 
        dpcb_given_sb_nb = {}
        for key, value in p_bind.items():
            if key[0] == 0: # unpaired
                dpcb_given_sb_nb[key] = value* self.beta * math.exp(self.beta*mu)/((1+math.exp(self.beta*mu))**2)
            else:
                dpcb_given_sb_nb[key] = value* self.beta * math.exp(self.beta*(mu-p_b))/((1+math.exp(self.beta*(mu-p_b)))**2)
        _, _, p_cb_given_sb_nb = self.compute_mu_dependent_quantities(mu, p_b, p_bind)
        # compute dp(cb)/d(mu) with chain rule
        dpcb_dmu = np.zeros(self.N_seq)
        for i,nb in enumerate(self.seq):
            dpcb_dmu[i] = p_cb_given_sb_nb[(1, nb)]*dps_dmu[i] + p_sb[i]*dpcb_given_sb_nb[(1, nb)] 
            dpcb_dmu[i] += -p_cb_given_sb_nb[(0, nb)]*dps_dmu[i] + (1-p_sb[i])*dpcb_given_sb_nb[(0, nb)]
        dMb_dmu = - np.exp(-eps_b)*(math.exp(-m1)-math.exp(-m0))*dpcb_dmu
        return dMb_dmu, dpcb_dmu, dpcb_given_sb_nb

    def dMb_dpb(self, mu, p_b, dps_dpb, p_bind, m0, m1, eps_b, p_sb):
        derivative_of_physical_binding = - self.beta*math.exp(self.beta*(mu-p_b))/((1+math.exp(self.beta*(mu-p_b)))**2)
        dpcb_given_sb_nb = {key: value*derivative_of_physical_binding for key, value in p_bind.items() if key[0] == 1}
        _, _, p_cb_given_sb_nb = self.compute_mu_dependent_quantities(mu, p_b, p_bind)
        # compute dp(cb)/d(p_b) with chain rule
        dpcb_dpb = np.zeros(self.N_seq)
        for i,nb in enumerate(self.seq):    # should vectorize this
            dpcb_dpb[i] = p_cb_given_sb_nb[(1, nb)]*dps_dpb[i] + p_sb[i]*dpcb_given_sb_nb[(1, nb)]
            dpcb_dpb[i] += -p_cb_given_sb_nb[(0, nb)]*dps_dpb[i] # last term is 0
        dMb_dpb = - np.exp(-eps_b)*(math.exp(-m1)-math.exp(-m0))*dpcb_dpb
        return dMb_dpb

    def dMb_dpbind(self, mu, p_b, p_bind, m0, m1, p_sb, eps_b):
        # Initialize the result array
        dMb_dpbind = np.zeros((self.N_seq, 8))
        # Precompute common values
        prob_physical_binding_unpaired = math.exp(self.beta * mu) / (1 + math.exp(self.beta * mu))
        prob_physical_binding_paired = math.exp(self.beta * (mu - p_b)) / (1 + math.exp(self.beta * (mu - p_b)))
        exp_neg_eps_b = np.exp(-eps_b)
        exp_diff = math.exp(-m1) - math.exp(-m0)
        # Create a dictionary to map nucleotides to their indices in p_bind
        nucleotide_indices = {nb: [] for nb in set(self.seq)}
        for j, key in enumerate(p_bind.keys()):
            nucleotide_indices[key[1]].append((key[0], j))
        # Vectorize the outer loop
        for nb, indices in nucleotide_indices.items():
            mask = np.array([n == nb for n in self.seq])
            for key0, j in indices:
                # compute dp(cb)/d(p_bind)
                if key0 == 0:  # if unpaired
                    dMb_dpbind[mask, j] = (1 - p_sb[mask]) * prob_physical_binding_unpaired
                else:
                    dMb_dpbind[mask, j] = p_sb[mask] * prob_physical_binding_paired
                # compute dM/dp_bind = -exp(-eps_b) * (exp(-m1) - exp(-m0)) * dp(cb)/d(p_bind)
                dMb_dpbind[mask, j] *= -exp_neg_eps_b[mask] * exp_diff
        return dMb_dpbind

    def dMb_dlambda_sc(self, mu, p_b, p_bind, m0, m1, eps_b, lambda_sc):
        if not self.infer_1D_sc:
            return np.full((self.N_seq, self.N_seq), np.nan)
        penalty = self.compute_penalty_m(mu, p_b)
        dps_i_dlambda_j =  self.get_dps_dlambda_sc(penalty, lambda_sc, self.use_interpolated_ps)
        # compute dpc_i_dlambda_j
        _, _, p_cb_given_sb_nb = self.compute_mu_dependent_quantities(mu, p_b, p_bind)
        dpc_i_dlambda_j = np.full((self.N_seq, self.N_seq), np.nan)
        for i, nb in enumerate(self.seq):
            dpc_i_dlambda_j[i, :] = (p_cb_given_sb_nb[(1, nb)]-p_cb_given_sb_nb[(0, nb)]) * dps_i_dlambda_j[i, :]
        # compute dMb_dlambda_j, also this was tested with explicit loop
        dMb_dlambda_j = -np.exp(-eps_b)[:,np.newaxis] * (math.exp(-m1) - math.exp(-m0)) * dpc_i_dlambda_j
        return dMb_dlambda_j

    def all_derivatives(self, mu, p_b, p_bind, m0, m1, p_cb, eps_b, p_sb, lambda_sc):
        '''Compute all derivatives of the mutation profile wrt the parameters.'''
        dps_dmu, dps_dpb = self.numerical_derivatives_of_ps(mu, p_b, p_sb, lambda_sc)
        dMb_dmu, *_ = self.dMb_dmu(mu, p_b, dps_dmu, p_bind, m0, m1, eps_b, p_sb)
        dMb_dpb = self.dMb_dpb(mu, p_b, dps_dpb, p_bind, m0, m1, eps_b, p_sb)
        dMb_dpbind = self.dMb_dpbind(mu, p_b, p_bind, m0, m1, p_sb, eps_b)
        dMb_dm0 = np.exp(-eps_b) * (1 - p_cb) * math.exp(-m0)
        dMb_dm1 = np.exp(-eps_b) * p_cb * math.exp(-m1)
        dMb_dlambda_sc = self.dMb_dlambda_sc(mu, p_b, p_bind, m0, m1, eps_b, lambda_sc)
        return dMb_dmu, dMb_dpb, dMb_dpbind, dMb_dm0, dMb_dm1, dMb_dlambda_sc


    # 3:Fit
    # 3.1: Mutation rate and its gradient
    #@clocked
    def mut_rate_and_its_grad(self, *, mu_r, p_b, p_bind, m0, m1, lambda_sc, compute_gradient):
        '''Compute loss for this system and the gradient of the mutation profile wrt the parameters.
        Params must be passed packed.
        Returns:
            mut_rate_model: np.array, predicted mutation profile (values in (0,1))
            grad_of_M: dict {param_name: grad_wrt_param} or None if compute_gradient is False'''
        mu_j = mu_r + self.kBT * np.log((self.conc_mM+.1)/1000) if self.conc_mM is not None else mu_r
        p_sb = self.get_ps(self.compute_penalty_m(mu_j, p_b), lambda_sc, interpolated=self.use_interpolated_ps)
        p_cb = self.compute_p_cb(mu_j, p_b, p_bind, p_sb)
        if compute_gradient:
            dM_dmu, dM_dpb, dM_dpbind, dM_dm0, dM_dm1, dM_dlambda_sc = self.all_derivatives(mu_j, p_b, p_bind, m0, m1, p_cb, self.eps_b, p_sb, lambda_sc)
            for i, dMi_dtheta_j_partial in enumerate([dM_dmu, dM_dpb, dM_dpbind, dM_dm0, dM_dm1, dM_dlambda_sc]):
                # concatenate partial dMi_dtheta_j to get the full matrix
                if i == 0:
                    dMi_dthetaj = dMi_dtheta_j_partial
                elif i == 1:
                    dMi_dthetaj = np.concatenate((dMi_dthetaj[:,None], dMi_dtheta_j_partial[:,None]), axis=1)
                elif len(dMi_dtheta_j_partial.shape) == 2:
                    dMi_dthetaj = np.concatenate((dMi_dthetaj, dMi_dtheta_j_partial), axis=1)
                else:
                    dMi_dthetaj = np.concatenate((dMi_dthetaj, dMi_dtheta_j_partial[:,None]), axis=1)
            grad_of_M = {'mu_r': dM_dmu, 'p_b': dM_dpb, 'p_bind': dM_dpbind, 'm0': dM_dm0, 'm1': dM_dm1, 'lambda_sc': dM_dlambda_sc}
            if not self.infer_1D_sc:
                grad_of_M['lambda_sc'] = None
        else:
            grad_of_M = None

        mut_rate_model = self.compute_mutation_profile(m0, m1, self.eps_b, p_cb)
        # assert mut_rate_model is in (0,1)
        try:
            assert np.all(mut_rate_model >= 0)
            assert np.all(mut_rate_model <= 1)
        except AssertionError:
            self.logger.warning('Warning, mut rate is not in (0,1) for system %s.', self.ID)
            self.logger.warning(f'mu_r: {mu_r}, p_b: {p_b}, p_bind: {p_bind}, m0: {m0}, m1: {m1}, eps_b: {self.eps_b}')
            self.logger.warning(f'lambda_sc: {lambda_sc}')
        # quantities that depend on the experiment data such as dL/dM will be computed by multisys_loss_and_grad
        return mut_rate_model, grad_of_M
    
    #3.1: Loss and its gradient
    def loss_and_grad(self, mut_rate_model, grad_of_mut_rate):
        """
        Compute the negative log-likelihood loss and its gradient with respect to the parameters.
        Experimental data is retrieved from Experiment.df.
        Inputs:
            mut_rate_model: np.array, predicted mutation profile (values in (0,1))
            grad_of_mut_rate:    np.array, derivative matrix of the mutation profile with respect to parameters.
            compute_gradient: bool, whether to compute the gradient.
        Returns:
            loss: scalar, the negative log-likelihood loss.
            grad:  np.array containing the gradient with respect to parameters (or None if grad_of_mut_rate is None).
        """
        mut_obs = self.df['mut_count'].values
        coverage = self.df['total_count'].values
        
        # Apply position mask to exclude masked positions from loss calculation
        mut_obs_masked = mut_obs[self.position_mask]
        coverage_masked = coverage[self.position_mask] 
        mut_rate_model_masked = mut_rate_model[self.position_mask]
        
        # Compute log of the binomial coefficient for each position
        log_binom = np.array([log_binomial(cov, mut) for cov, mut in zip(coverage_masked, mut_obs_masked)])
        # Compute the log-likelihood at each position
        log_prob = log_binom + (mut_obs_masked * np.log(mut_rate_model_masked) +
                                (coverage_masked - mut_obs_masked) * np.log(1 - mut_rate_model_masked))
        # Total negative log-likelihood loss (only from unmasked positions)
        loss = -np.sum(log_prob)
        
        if grad_of_mut_rate is None:
            return loss, None
            
        # Compute derivative of loss with respect to the mutation profile: dL/dM
        # Initialize full gradient array with zeros (masked positions will remain zero)
        dL_dM = np.zeros_like(mut_rate_model)
        # Only compute gradients for unmasked positions
        dL_dM[self.position_mask] = -(mut_obs_masked / mut_rate_model_masked - 
                                      (coverage_masked - mut_obs_masked) / (1 - mut_rate_model_masked))
        
        # Chain rule: dL/dtheta_j = sum_i (dL/dM_i * dM_i/dtheta_j)
        grad = {}
        for param, grad_wrt_param in grad_of_mut_rate.items():
            if grad_wrt_param is None:
                grad[param] = None
                continue
            if grad_wrt_param.ndim == 1:
                grad_wrt_param = grad_wrt_param[:, np.newaxis]
            grad[param] = np.sum(grad_wrt_param * dL_dM[:, np.newaxis], axis=0)
        # we have to pack the p_bind component into a dictionary
        grad['p_bind'] = convert_p_bind_1d_to_dict(grad['p_bind'], False)
        return loss, grad

def group_by_system(experiments):
    '''Takes in input a list of Experiment instances and
    returns a dictionary with the experiments grouped by system in a dictionary'''
    systems = defaultdict(list) # defaltdict is a dictionary that creates a new list if the key is not found
    for exp in experiments:
        systems[exp.system_name].append(exp)
    return dict(systems)    # {sys_i: [exp1, exp2, ...]}

def generate_lambdas_indices(systems, DMS_mode, infer_1D_sc):
    '''Returns a dictionary with the indices of the lambda_sc parameters
    for each system when parameters are in a 1D array (for scipy)'''
    if not infer_1D_sc:
        return None
    lambdas_indices = {}
    last_index = 11 if not DMS_mode else 7
    for system in systems:
        lambdas_indices[system.sys_name] = list(range(last_index + 1, last_index + 1 + system.N_seq))
        last_index = lambdas_indices[system.sys_name][-1]
    return lambdas_indices

class System:
    '''Collect multiple experiments from the same system (different concentrations or replicas) and initialize the ExperimentFit instances.
    Contains also plotting functions.'''
    def __init__(self, experiments, validation_exps, debug, do_plots, infer_1D_sc, logger, use_interpolated_ps, custom_mask=None):
        self.logger = logger
        self.exps_train = [exp for exp in experiments] if experiments else None
        self.exps_val = [exp for exp in validation_exps] if validation_exps else None
        if self.exps_train:
            ref_exp = self.exps_train[0]
        elif self.exps_val:
            ref_exp = self.exps_val[0]
        else:
            raise ValueError("No training or validation experiments provided.")
        self.sys_name = ref_exp.system_name
        self.reagent = ref_exp.reagent
        self.N_seq = ref_exp.N_seq
        self.seq = ref_exp.seq
        self.temp_C = ref_exp.temp_C
        self.temp_K = ref_exp.temp_K
        self.kBT = (self.temp_K * kb)
        self.debug = debug
        self.exps_all = [exp for exp in (self.exps_train if self.exps_train else []) + (self.exps_val or [])]
        self.use_interpolated_ps = use_interpolated_ps and infer_1D_sc
        
        # Store masking parameters
        self.custom_mask = custom_mask
        # verify if there is a 0 conc experiment
        self.concs_mM = set([exp.conc_mM for exp in self.exps_train]) if self.exps_train else set() | set([exp.conc_mM for exp in self.exps_val]) if self.exps_val else set()
        if not 0 in self.concs_mM:
            self.logger.warning('no 0 conc experiment found. eps_b will be set to zeros.')
            self.eps_b = np.zeros(self.exps_train[0].N_seq)
        else:
            self.eps_b = self.get_eps_b_from_zeroconc_profiles()
        # initialize ExperimentFit instances
        self.exp_fits_train = [ExperimentFit(exp, infer_1D_sc=infer_1D_sc, eps_b=self.eps_b, is_training=True, debug=debug, do_plots=do_plots, logger=self.logger, custom_mask=self.custom_mask, use_interpolated_ps=self.use_interpolated_ps) for exp in experiments] if experiments else None
        self.exp_fits_val = [ExperimentFit(exp, infer_1D_sc=infer_1D_sc, eps_b=self.eps_b, is_training=False, debug=debug, do_plots=do_plots, logger=self.logger, custom_mask=self.custom_mask, use_interpolated_ps=self.use_interpolated_ps) for exp in validation_exps] if validation_exps else None
        self.exp_fits_all = [exp_fit for exp_fit in (self.exp_fits_train or []) + (self.exp_fits_val or [])]
        self.reps = set([exp.rep_number for exp in self.exps_all])
        # assert that all experiments have the same system attributes
        for exp in self.exps_all:
            assert exp.system_name == self.sys_name
            assert exp.reagent == self.reagent
            assert exp.temp_C == self.temp_C
            assert exp.seq == self.seq, f'seq of {exp} is different from the first experiment in the system'
            assert exp.N_seq == self.N_seq
        for exp_fit in self.exp_fits_all:
            assert exp_fit.system_name == self.sys_name
            assert exp_fit.reagent == self.reagent
            assert exp_fit.temp_C == self.temp_C
            assert exp_fit.seq == self.seq, f'seq of {exp_fit} is different from the first experiment in the system'
            assert exp_fit.N_seq == self.N_seq
    
    def __str__(self):
        parts = [f"System: {self.sys_name}"]
        if self.exps_train:
            parts.append(f"Training Experiments: {self.exps_train}")
        if self.exps_val:
            parts.append(f"Validation Experiments: {self.exps_val}")
        return "\n".join(parts)

    def __repr__(self):
        return self.__str__()
    
    def get_eps_b_from_zeroconc_profiles(self):
        '''Get eps_b from the mutation profiles of the 0 conc experiments using the inverse formula.'''
        zeroconc_exps = [exp for exp in self.exps_all if exp.conc_mM == 0]
        average_mut_profile = np.zeros(self.N_seq)
        for exp in zeroconc_exps:
            average_mut_profile += exp.df['mut_count']/exp.df['total_count']
        average_mut_profile /= len(zeroconc_exps)
        eps_b = np.array(-np.log(1-average_mut_profile))
        self.logger.debug(f'computed eps_b from 0 conc experiments: {eps_b[:10]}...')
        return eps_b
    
    def plot_pairing_probs(self, mu_r, p_b, lambda_sc, save_fig_path=None, ax=None, fig=None, plot_no_lambda_case=True, plot_lambdas=True, xlim=None, **kwargs):
        '''Plot the pairing probabilities for this system along the sequence.'''
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        # create a dictionary of exp_fit with unique concentrations
        exp_fits_dict = {exp_fit.conc_mM: exp_fit for exp_fit in self.exp_fits_all}
        # map colors linearly
        colors = plt.cm.viridis(np.linspace(0, 1, len(exp_fits_dict)))
        # plot inferred pairing probabilities
        for exp_fit, color in zip(exp_fits_dict.values(), colors):
            mu_j = mu_r + self.kBT * np.log((exp_fit.conc_mM+.1)/1000) if exp_fit.conc_mM is not None else mu_r
            ls = '-' 
            pp = exp_fit.get_ps(exp_fit.compute_penalty_m(mu_j, p_b), lambda_sc, interpolated=self.use_interpolated_ps)
            indices = np.arange(self.N_seq)
            if hasattr(exp_fit, 'df'):
                if exp_fit.df is not None:
                    indices = exp_fit.df.index
            ax.plot(indices, pp, label=f'{exp_fit.conc_mM}mM', linestyle=ls, color=color)
            pp_no_penalty = exp_fit.get_ps(0.0, lambda_sc, interpolated=self.use_interpolated_ps)
            ax.plot(indices, pp_no_penalty, label=f'{exp_fit.conc_mM}mM no penalty no lambdas', linestyle='--', alpha=.5)
            if plot_no_lambda_case and lambda_sc is not None:
                pp_no_lambda = exp_fit.get_ps(exp_fit.compute_penalty_m(mu_j, p_b), np.zeros(self.N_seq), interpolated=self.use_interpolated_ps)
                ax.plot(indices, pp_no_lambda, label=f'{exp_fit.conc_mM}mM no lambda', linestyle='-.')
        ax.set_title(self.sys_name)
        ax.set_xlabel('Position')
        ax.set_ylabel('Pairing probability')
        ax.legend()
        #if xlim is None:
        #    xlim = (self.N_seq // 2 - 50 + indices[0], self.N_seq // 2 + 50 + indices[0])
        #ax.set_xlim(xlim)
        ax.set_ylim(0,1)
        # twin axis for lambda_sc in grey
        if plot_lambdas and lambda_sc is not None:
            ax2 = ax.twinx()
            ax2.plot(indices, lambda_sc, color='grey', linestyle='--', label='lambda_sc', alpha=.4)
            ax2.set_ylabel('lambda_sc', color='grey')
        if save_fig_path is not None:
            plt.savefig(save_fig_path, bbox_inches='tight')
        return fig, ax
    
    def plot_mut_profiles(self, mu_r, p_b, p_bind, m0, m1, lambda_sc, save_fig_path=None, ax=None, fig=None, xlim = None, ylim=None, seq_as_xticks=False, plot_lambdas=True):
        '''Plot the mutation profiles for this system.'''
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        # create a dictionary of exp_fit with unique concentrations
        exp_fits_dict = {exp_fit.conc_mM: exp_fit for exp_fit in self.exp_fits_all}
        # map lienarly colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(exp_fits_dict)))
        # plot inferred mutation profiles
        for i,exp_fit in enumerate(exp_fits_dict.values()):
            #ls = '-' if exp_fit.is_training else '--'
            mut_rate_model, _ = exp_fit.mut_rate_and_its_grad(mu_r=mu_r, p_b=p_b, p_bind=p_bind, m0=m0, m1=m1, lambda_sc=lambda_sc, compute_gradient=False)
            indices = np.arange(self.N_seq)
            if hasattr(exp_fit, 'df'):
                if exp_fit.df is not None:
                    indices = exp_fit.df.index
            ax.plot(indices, mut_rate_model, label=f'model {exp_fit.conc_mM}mM rep{exp_fit.rep_number}', linestyle='-', color = colors[i])
        # plot exp data
        # map concentrations to colors
        colors_dict = {exp_fit.conc_mM: color for exp_fit, color in zip(exp_fits_dict.values(), colors)}
        for exp in self.exps_all:
            color = colors_dict[exp.conc_mM]
            # Get the position mask from any exp_fit (they should all have the same mask)
            position_mask = exp_fits_dict[exp.conc_mM].position_mask if hasattr(exp_fits_dict[exp.conc_mM], 'position_mask') else np.ones(len(exp.df), dtype=bool)
            
            # Plot experimental data with connecting line and transparency for masked positions
            mutation_rates = exp.df['mut_count']/exp.df['total_count']
            
            # First plot the connecting line (with full opacity)
            ax.plot(exp.df.index, mutation_rates, color=color, linewidth=1, alpha=0.7, ls='--',
                   label=f'data {exp.conc_mM}mM rep{exp.rep_number}')
            
            # Then plot scatter points with varying transparency for masked positions
            alphas = np.where(position_mask, 1.0, 0.3)  # Full opacity for unmasked, 30% for masked
            for idx, rate, alpha in zip(exp.df.index, mutation_rates, alphas):
                ax.scatter(idx, rate, color=color, alpha=alpha, s=10, marker='o')
        ax.set_xlabel('Position')
        ax.set_ylabel('Mutation rate')
        ax.set_title(self.sys_name)
        ax.legend()
        if ylim is None:
            # Collect all experimental mutation rates
            exp_mut_rates = []
            for exp in self.exps_all:
                exp_mut_rates.extend(exp.df['mut_count']/exp.df['total_count'])
            # Calculate 95th percentile
            upper_limit = np.percentile(exp_mut_rates, 95)
            # Set y-limits from 0 to 90th percentile
            ax.set_ylim(0, upper_limit)
        else:
            ax.set_ylim(ylim)
        #if xlim is None:
        #    xlim = (self.N_seq // 2 - 50 + indices[0], self.N_seq // 2 + 50 + indices[0])
        #ax.set_xlim(xlim)
        if seq_as_xticks:
            # we need to use the indices of the sequence as xticks
            ax.set_xticks(indices)
            ax.set_xticklabels(exp_fit.seq)
        #twin axis for lambsa in grey
        if plot_lambdas and lambda_sc is not None:
            ax2 = ax.twinx()
            ax2.plot(exp.df.index, lambda_sc, color='grey', linestyle='--', label='lambda_sc', alpha=.4)
            ax2.set_ylabel('lambda_sc', color='grey')
        if save_fig_path is not None:
            plt.savefig(save_fig_path, bbox_inches='tight')
        return fig, ax
    
    def plot_averages_vs_conc(self, params, save_fig_path = None, ax = None, fig = None):
        '''Plot the average mutation rate for each nucleotide vs concentration for model and experiments.'''
        if ax is None:
            fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
        # iterate over all experiments and compute average mutation for each nucleotide
        data = pd.DataFrame()
        #cols of data will be: avg_exp_A, avg_exp_C, avg_exp_G, avg_exp_U, avg_model_A, avg_model_C, avg_model_G, avg_model_U
        for exp_fit in self.exp_fits_all:
            # copy df to modify it
            df_copy = exp_fit.df.copy()
            # add model predictions at correct indices
            mut_rate_model, _ = exp_fit.mut_rate_and_its_grad(**params, compute_gradient=False)
            df_copy['mut_rate_model'] = mut_rate_model
            # check that the sequence is the same
            df_copy['seq_check'] = list(exp_fit.seq)
            assert np.all(df_copy['seq_check'] == df_copy['ref_nt'])
            for nb in 'ACGU':
                # select the nucleotide and only consider unmasked positions
                nucleotide_mask = df_copy['ref_nt'] == nb
                # Apply position mask to only include unmasked positions
                combined_mask = nucleotide_mask & exp_fit.position_mask
                
                # compute average mutation rate for this nucleotide (only unmasked positions)
                if np.any(combined_mask):
                    avg_exp_nb = df_copy['mut_rate'][combined_mask].mean()
                    avg_model_nb = df_copy['mut_rate_model'][combined_mask].mean()
                else:
                    # If no unmasked positions for this nucleotide, use NaN
                    avg_exp_nb = np.nan
                    avg_model_nb = np.nan
                
                data.loc[exp_fit.ID, f'avg_exp_{nb}'] = avg_exp_nb
                data.loc[exp_fit.ID, f'avg_model_{nb}'] = avg_model_nb
            # create concentraiton col
            data.loc[exp_fit.ID, 'conc_mM'] = exp_fit.conc_mM
        data.sort_values('conc_mM', inplace=True)

        # sort rows by concentration when possible
        # If concentrations are None, we create a dummy x-axis coordinate.
        if data['conc_mM'].isnull().all():
            data['x_axis'] = 0
            xticks = [0]
            xtick_labels = ['None']
        else:
            data.sort_values('conc_mM', inplace=True)
            data['x_axis'] = data['conc_mM']
            xticks = None
            xtick_labels = None

        # plot: for each nucleotide
        for i, nb in enumerate('ACGU'):
            ax = axes[i // 2, i % 2]
            x = data['x_axis']
            y = data[f'avg_exp_{nb}']
            ax.scatter(x, y, label=f'exp {nb}', color='blue', marker='x')
            ax.plot(x, data[f'avg_model_{nb}'], label=f'model', color='red', marker='o')
            ax.text(0.5, 0.85, f'{nb}', horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes, fontsize=16)
            ax.set_ylim(0, None)
            if xticks is not None:
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtick_labels)
        axes[1, 0].set_xlabel('Concentration (mM)')
        axes[1, 1].set_xlabel('Concentration (mM)')
        axes[0, 0].set_ylabel('Average Mutation rate')
        axes[1, 0].set_ylabel('Average Mutation rate')
        fig.suptitle(f'{self.sys_name}')
        # Put legend outside
        axes[1, 1].legend(loc='upper left', bbox_to_anchor=(1, 1))
        if save_fig_path is not None:
            plt.savefig(save_fig_path, bbox_inches='tight')
        return fig, axes
            


    def plot_lambdas(self, lambdas, save_fig_path=None):
        if lambdas is None:
            return
        plt.figure()
        exp_fit = self.exp_fits_train[0]
        plt.plot(exp_fit.df.index, lambdas)
        plt.title(f'lambda_sc for system {self.sys_name}')
        plt.xlabel('Position')
        plt.ylabel('lambda_sc')
        if save_fig_path is not None:
            plt.savefig(save_fig_path, bbox_inches='tight')
        if plt.isinteractive():
            plt.show()
        else:
            plt.close()



# create a class that takes as input multiple experiments (different systems) and combines them into a single fitting process

@dataclass
class MultiSystemsFit:
    experiments: List   # list of Experiment instances
    validation_exps: Optional[List] = None  # list of Experiment instances
    output_suffix: str = 'test' # name of subfolder fi fits where results are stored 
    root_dir: str = 'fits'  # root directory where the output_suffix folder is created
    debug: bool = False # debug mode: additional checks and prints
    do_plots: bool = True
    description: Optional[str] = None  # this is added in the log file
    DMS_mode: bool = True  # if True, p_bind are only 4 (instead of 8)g461
    infer_1D_sc: bool = False  # inference of 1d soft constrains for each system
    max_iter: Optional[int] = None  # maximum number of iterations for the optimization
    overwrite: bool = False  # overwrite the log file if it exists
    guess: Optional[str] = None  # Initial guess: None, random, path or 'last'
    custom_mask: Optional[List] = None  # Custom mask for positions to include/exclude from fitting
    check_gradient: bool = False
    print_to_std_out: bool = True
    fix_physical_params: bool = False  # if True, the physical parameters are fixed
    fix_lambda_sc: bool = False  # if True, the lambda_sc parameters are fixed
    iteration_count: int = 0  # number of iterations
    use_interpolated_ps: bool = False  # whether to use interpolated pairing probabilities
    # These attributes are initialized in __post_init__
    output_dir: str = field(init=False)
    systems: List = field(init=False)
    lambdas_indices: Optional[dict] = field(init=False)
    log_file_path: str = field(init=False)
    logger: logging.Logger = field(init=False)  # logger used for printing
    last_plot_callback_time: Optional[float] = field(init=False)  # time of the last plot callback

    def __post_init__(self):
        # Check for custom_mask usage - not yet supported with MultiSystemsFit
        if self.custom_mask is not None:
            raise NotImplementedError(
                "Custom masking is not yet supported with MultiSystemsFit because different systems "
                "can have different sequence lengths, making a single custom_mask parameter ambiguous. "
                "This feature will be implemented in the future to support per-system custom masks. "
                "For now, please use the default masking behavior (first and last 25 nucleotides) "
                "or use individual System/ExperimentFit classes directly for custom masking."
            )
        
        # Create output directory
        self.output_dir = os.path.join(self.root_dir, self.output_suffix)
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.output_dir, f'{self.output_suffix}.log')
        if self.overwrite:
            os.remove(self.log_file_path) if os.path.exists(self.log_file_path) else None
        # Initialize logger using a unique name per instance
        unique_logger_name = f"{self.output_suffix}_{os.getpid()}"
        self.logger = initialize_logger(unique_logger_name, self.log_file_path, self.debug, self.print_to_std_out)
        self.logger.debug(f'Initialized logger with log file: {self.log_file_path}')
        # Initialize systems via grouping experiments
        self.systems = self._initialize_systems(self.experiments, self.validation_exps)
        # Generate lambda_sc indices for optimization if needed
        self.lambdas_indices = generate_lambdas_indices(self.systems, self.DMS_mode, self.infer_1D_sc)
        # Set parameter positions for each system
        self._generate_params_positions()
        # Write the log file header
        self.write_log_file_header()
        self.last_plot_callback_time = None
        self.evaluation_count = 0
        self.params_history = {'mu_r': [], 'p_b': [], 'p_bind': {key: [] for key in [(0, 'A'), (0, 'C'), (0, 'G'), (0, 'U'), (1, 'A'), (1, 'C'), (1, 'G'), (1, 'U')]}, 'm0': [], 'm1': []}

    def _initialize_systems(self, experiments, validation_exps):
        '''Generate useful dictionaries that map experiment IDs to Experiment and ExperimentFit instances.'''
        # Group exps by system and create System instances
        systems_dict_train = group_by_system(experiments)
        systems_dict_val = group_by_system(validation_exps) if validation_exps else None
        all_system_names = set(systems_dict_train.keys()) | (set(systems_dict_val.keys()) if systems_dict_val else set())
        # Create System instances
        systems = {}
        for system_name in all_system_names:
            train_exps = systems_dict_train[system_name] if system_name in systems_dict_train else None
            validation_exps = systems_dict_val[system_name] if systems_dict_val and system_name in systems_dict_val else None
            # Note: custom_mask is always None here since MultiSystemsFit doesn't support it yet
            systems[system_name] = System(train_exps, validation_exps, \
                                         self.debug, self.do_plots, self.infer_1D_sc, self.logger, custom_mask=None, use_interpolated_ps=self.use_interpolated_ps)
        return list(systems.values())

    def _generate_params_positions(self):
        '''Generate parameter positions mapping for each System.'''
        base_config = {
            'mu_r': 0,
            'p_b': 1,
            'p_bind': list(range(2, 10 if not self.DMS_mode else 6)),
            'm0': 10 if not self.DMS_mode else 6,
            'm1': 11 if not self.DMS_mode else 7,
        }
        for system in self.systems:
            params_pos = base_config.copy()
            if self.infer_1D_sc:
                params_pos['lambda_sc'] = self.lambdas_indices[system.sys_name]
            system.dict_with_params_pos = params_pos

    def write_log_file_header(self):
        with open(self.log_file_path,'w' if self.overwrite else 'a') as f:
            f.write(f'Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'ViennaRNA version: {RNA.__version__}\n')
            f.write(f'Description: {self.description}\n')
            f.write(f'Output suffix: {self.output_suffix}\n\n')
            for system in self.systems:
                f.write(f"System: {system.sys_name}\n")
                if system.exps_train:
                    train_desc = [exp.__dict__.get("short_description", "N/A") for exp in system.exps_train]
                    f.write(f"  Training Experiments: {train_desc}\n")
                if system.exps_val:
                    val_desc = [exp.__dict__.get("short_description", "N/A") for exp in system.exps_val]
                    f.write(f"  Validation Experiments: {val_desc}\n")
            f.write("\n")
            f.write(f"--- Fit Parameters ---\n")
            f.write(f"DMS mode: {self.DMS_mode}\n")
            f.write(f"Infer 1D soft constraints: {self.infer_1D_sc}\n")
            f.write(f"Fix physical params: {self.fix_physical_params}\n")
            f.write(f"Fix lambda_sc: {self.fix_lambda_sc}\n")
            f.write(f"Use interpolated pairing probabilities: {self.use_interpolated_ps}\n")
            f.write(f"Initial guess: {self.guess}\n")
            f.write(f"Max iterations: {self.max_iter}\n")
            f.write(f"Custom mask: {self.custom_mask}\n")
            f.write(f"\n--- Execution Settings ---\n")
            f.write(f"Root directory: {self.root_dir}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Log file path: {self.log_file_path}\n")
            f.write(f"Overwrite: {self.overwrite}\n")
            f.write(f"Debug mode: {self.debug}\n")
            f.write(f"Check gradient: {self.check_gradient}\n")
            f.write(f"Do plots: {self.do_plots}\n")
            f.write(f"Print to stdout: {self.print_to_std_out}\n\n")

    def initialize_guess_and_bounds(self, start_value=0.1, guess=None):
        """
        Initialize the parameter vector and corresponding bounds.
    
        The base parameters are:
            - mu_r (index 0): bound (-2, 5)
            - p_b  (index 1): bound (0, 10)
            - p_bind: 4 values if DMS_mode is True, 8 if False.
            - m0   (second last): bound (0.1, 10)
            - m1   (last): bound (0.00001, 1)
        If infer_1D_sc is True, then for each system (taken from the first training experiment)
        a block of lambda_sc parameters (length = N_seq) is appended with bounds (-1,1).
    
        The `guess` parameter controls the initial guess:
            - None: use a constant value (start_value) for base parameters, zeros for lambda_sc.
            - 'random': use random values in [0,1) (or [-0.5,0.5) for lambda_sc)
            - 'last': load initial guess from {output_dir}/params.txt.
            - Otherwise, if guess is a filepath ending in .txt, load that file.
        """
        assert guess in [None, 'random', 'last'] or guess.endswith('.txt'), "Invalid guess parameter"
        # Determine number of base parameters:
        # mu_r, p_b, (p_bind: 4 or 8), m0, m1
        base_param_count = 8 if self.DMS_mode else 12
        # Initialize bounds list for base parameters:
        bounds = []
        # mu_r and p_b
        bounds.extend([(-5, 5), (0, 10)])
        # p_bind bounds:
        n_p_bind = 4 if self.DMS_mode else 8
        for _ in range(n_p_bind):
            bounds.append((1e-6, 1))
        # m0 and m1 bounds:
        bounds.extend([(0, .1), (.1, 10)])
        
        # Initialize base guess vector
        if guess == 'random':
            # use a random guess for base parameters
            base_guess = np.random.rand(base_param_count)
            base_guess[0] = np.random.rand() - 1
        else:
            base_guess = np.ones(base_param_count) * start_value
        # Set m0 to 0
        base_guess[-2] = 0.0001
        
        # Start with the base guess
        initial_guess = base_guess.copy()
        
        # Append initial guesses and bounds for lambda_sc if needed
        if self.infer_1D_sc:
            for system in self.systems:
                # All experiments in the system share the same N_seq
                n_seq = system.N_seq
                # Choose random lambda_sc guess if requested, else zeros
                if guess == 'random':
                    #lambda_guess = np.random.rand(n_seq) - 0.5
                    lambda_guess = np.zeros(n_seq)
                else:
                    lambda_guess = np.zeros(n_seq)
                # Append the lambda_sc guesses for this system
                initial_guess = np.append(initial_guess, lambda_guess)
                # Extend bounds for each lambda_sc parameter
                bounds.extend([(-1, 1)] * n_seq)
        
        # Total number of parameters computed so far
        self.N_params_tot = len(initial_guess)
        
        # If a previous guess should be used, load it from file
        if guess == 'last':
            initial_guess = np.loadtxt(os.path.join(self.output_dir, 'params1D.txt'))
        elif isinstance(guess, str) and guess.endswith('.txt'):
            initial_guess = np.loadtxt(guess)
        
        # Ensure the loaded initial guess matches the expected parameter count
        try:
            assert len(initial_guess) == self.N_params_tot, f"Expected {self.N_params_tot} parameters, got {len(initial_guess)}"
        except AssertionError as e:
            self.logger.warning(e)
            self.logger.warning(f'Padding with zeros to match expected parameter count')
            # Pad with zeros if the loaded guess is shorter than expected
            if len(initial_guess) < self.N_params_tot:
                initial_guess = np.concatenate((initial_guess, np.zeros(self.N_params_tot - len(initial_guess))))
            elif len(initial_guess) > self.N_params_tot:
                raise ValueError(f"Loaded guess has more parameters than expected: {len(initial_guess)} vs {self.N_params_tot}")
        return initial_guess, bounds

    def _create_interim_result_string(self, final_params, reason_message):
        """
        Creates a string mimicking the Scipy OptimizeResult printout
        using the last known parameters after an interruption.
        """
        if final_params is None:
            return f"Optimization interrupted ({reason_message}), but no parameters were captured."
        try:
            current_fun, current_jac = self.multisys_loss_and_grad(final_params, compute_gradient=True)
        except Exception as e:
            self.logger.error(f"Could not calculate final fun/jac for summary: {e}")
            current_fun = 'N/A'
            current_jac = 'N/A'
        # Mimic the OptimizeResult attributes and formatting
        lines = []
        lines.append(f"     fun: {current_fun}" if isinstance(current_fun, (float, np.number)) else f"     fun: {current_fun}")
        lines.append(f" hess_inv: <8x8 LbfgsInvHessProduct with dtype=float64>") # added this line
        #lines.append(f"      jac: {np.array2string(current_jac, precision=8, suppress_small=True)}")
        lines.append(f"      jac: {current_jac}")
        lines.append(f" message: {reason_message}")
        lines.append(f"    nfev: 'N/A'")  # Not available from Scipy on interruption
        lines.append(f"     nit: {self.iteration_count}") # Iteration count from callback
        lines.append(f"    njev: 'N/A'")  # Not available from Scipy on interruption
        lines.append(f"  status: -1")     # Custom status for interruption
        lines.append(f" success: False")
        lines.append(f"        x: {final_params}")
        return "\n".join(lines)

    def fit(self):
        initial_guess, bounds = self.initialize_guess_and_bounds(guess=self.guess)
        self.logger.debug(f'Starting fitting process with {self.N_params_tot} parameters.')
        print(f'Starting fitting process of {self.systems[0].sys_name}, {datetime.datetime.now()}')
        self.logger.info(f'Initial guess:\n{initial_guess}')
        self.logger.info(f'Bounds:\n{bounds}')
        if len(initial_guess) != self.N_params_tot:
            raise ValueError(f'Expected {self.N_params_tot} parameters, got {len(initial_guess)}')
        if len(bounds) != self.N_params_tot:
            raise ValueError(f'Expected {self.N_params_tot} bounds, got {len(bounds)}')
        
        # Set stricter convergence criteria: reduce ftol to 1e-8 (default is 2.220446049250313e-09)
        #options = {'ftol': 0, 'gtol': 0}
        options=dict(ftol=np.nan, gtol=np.sqrt(np.finfo(float).tiny))
        if self.max_iter is not None:
            options['maxiter'] = self.max_iter

        # Record start time for 24h time limit
        self.fit_start_time = datetime.datetime.now()
        start_time = datetime.datetime.now()
        self.params_last_callback = np.copy(initial_guess)
        
        if self.debug:  # minimization outside of try block when debugging
            self.fit_result = minimize(self.multisys_loss_and_grad, initial_guess, 
                                       method='L-BFGS-B', bounds=bounds, jac=True, 
                                       options=options, callback=self.callback)
        else:
            try:
                self.fit_result = minimize(self.multisys_loss_and_grad, initial_guess, 
                                           method='L-BFGS-B', bounds=bounds, jac=True, 
                                           options=options, callback=self.callback)
                with open(self.log_file_path, 'a') as f:
                    self.logger.info(f'Fit result:\n {self.fit_result}')
            except KeyboardInterrupt as e:
                # Handle keyboard interrupt gracefully
                # compute loss and gradient for the last params
                self.logger.info('Fitting process interrupted by user with KeyboardInterrupt.\n')
                self.logger.info(self._create_interim_result_string(self.params_last_callback, "KeyboardInterrupt"))
            except Exception as e:
                err_msg = f'Error in fitting process: {e}'
                print(err_msg)
                self.logger.exception(err_msg)
                raise e
            finally:
                finished_msg = f'Finished fitting process of {self.systems[0].sys_name}, {datetime.datetime.now()}'
                print(finished_msg)
                self.logger.info(finished_msg)
                self.save_results()
                self.logger.info(f"Execution finished in {datetime.datetime.now() - start_time}")
        return self.fit_result if hasattr(self, 'fit_result') else None

    def pack_params(self, params_1d, system):
        '''Unpacks the 1D array of parameters into a dictionary.'''
        params = {}
        for param, pos in system.dict_with_params_pos.items():
            params[param] = params_1d[pos]
        # Convert the 1D array for p_bind into a dictionary.
        p_bind_dict = convert_p_bind_1d_to_dict(params['p_bind'], self.DMS_mode)
        if self.debug:
            # Check that conversion is reversible using the inverse function.
            p_bind_1d_recovered = convert_p_bind_dict_to_1d(p_bind_dict, self.DMS_mode)
            assert np.allclose(params['p_bind'], p_bind_1d_recovered), "p_bind conversion mismatch"
        params['p_bind'] = p_bind_dict
        if not self.infer_1D_sc:
            params['lambda_sc'] = None
        return params

    def map_system_grad_to_total(self, grad_system, system):
        '''Maps the gradient of a system to the total gradient.
        The first is a dict {sys_param_name: grad_wrt_param}, and the second is a 1d array of dim self.N_params_tot.
        We want to return a gradient with the same shape as the total one, mapped to the correct indices.
        Special handling is provided for the "p_bind" parameter, which is itself a dictionary.'''
        grad_newshape = np.zeros(self.N_params_tot)
        for param, pos in system.dict_with_params_pos.items():
            if param != 'p_bind':
                grad_val = grad_system[param]
                if isinstance(pos, int):
                    grad_array = np.asarray(grad_val)
                    if grad_array.size == 0:
                        grad_value = 0.0
                    elif grad_array.size == 1:
                        grad_value = float(grad_array.reshape(-1)[0])
                    else:
                        grad_value = float(grad_array.reshape(-1)[0])
                        self.logger.debug(
                            "Gradient for parameter '%s' had size %d; using first element for position %d.",
                            param, grad_array.size, pos
                        )
                    grad_newshape[pos] = grad_value
                else:
                    grad_array = np.asarray(grad_val).reshape(-1)
                    if len(pos) != grad_array.size:
                        raise ValueError(f"Gradient size {grad_array.size} for parameter '{param}' does not match mapped positions {pos}.")
                    for idx, value in zip(pos, grad_array):
                        grad_newshape[idx] = value
            else:
                # For p_bind, grad_system['p_bind'] is a dictionary.
                # Define the order for flattening according to DMS_mode.
                if self.DMS_mode:
                    order = [(0, 'A'), (0, 'C'), (0, 'G'), (0, 'U')]
                else:
                    order = [(0, 'A'), (0, 'C'), (0, 'G'), (0, 'U'),
                             (1, 'A'), (1, 'C'), (1, 'G'), (1, 'U')]
                grad_vals = np.array([grad_system['p_bind'][key] for key in order])
                # Ensure pos is a list so we can use indexing.
                if not isinstance(pos, list):
                    pos = [pos]
                # If pos indices are contiguous, we can use slicing.
                if np.all(np.diff(pos) == 1):
                    start = pos[0]
                    end = pos[-1] + 1
                    grad_newshape[start:end] = grad_vals
                else:
                    # Otherwise, assign element-by-element.
                    for idx, val in zip(pos, grad_vals):
                        grad_newshape[idx] = val
        if self.DMS_mode:
            # be sure to add (1,'G') to (0,'G') and (1,'U') to (0,'U') since now they are the same params
            grad_newshape[4] += grad_system['p_bind'][(1, 'G')]
            grad_newshape[5] += grad_system['p_bind'][(1, 'U')]
        return grad_newshape


    def debug_check_gradients(self, params_dict, exp_fit, grad_mut_profile, mut_profile_model):
        # Check gradients for mu_r, p_b, m0, m1 with central differences.
        dparam = 1e-2
        for param in ['mu_r', 'p_b', 'm0', 'm1']:
            params_dict_plus = copy.deepcopy(params_dict)
            params_dict_plus[param] += dparam
            mut_profile_plus, _ = exp_fit.mut_rate_and_its_grad(**params_dict_plus, compute_gradient=False)
            params_dict_minus = copy.deepcopy(params_dict)
            params_dict_minus[param] -= dparam
            mut_profile_minus, _ = exp_fit.mut_rate_and_its_grad(**params_dict_minus, compute_gradient=False)
            numerical_grad_wrt_param = (mut_profile_plus - mut_profile_minus) / (2 * dparam)
            plt.title(f'Gradient of mut profile wrt {param}')
            plt.plot(numerical_grad_wrt_param, label='numerical')
            plt.plot(grad_mut_profile[param], label='analytical', ls='--')
            plt.legend()
            plt.show()
        # Check gradients for p_bind parameters.
        for i, (key, _) in enumerate(params_dict['p_bind'].items()):
            dparam = 1e-4
            params_dict_plus = copy.deepcopy(params_dict)
            params_dict_plus['p_bind'][key] += dparam
            mut_profile_plus, _ = exp_fit.mut_rate_and_its_grad(**params_dict_plus, compute_gradient=False)
            numerical_grad_wrt_param = (mut_profile_plus - mut_profile_model) / dparam
            plt.title(f'Gradient of mut profile wrt p_bind[{key}]')
            plt.plot(numerical_grad_wrt_param, label='numerical')
            plt.plot(grad_mut_profile['p_bind'][:, i], label='analytical', ls='--')
            plt.legend()
            plt.show()
        # Check gradients for lambda_sc if needed.
        if self.infer_1D_sc:
            for j in np.random.randint(0, exp_fit.N_seq, 10):
                dparam = 0.02
                params_dict_plus = copy.deepcopy(params_dict)
                params_dict_plus['lambda_sc'][j] += dparam
                mut_profile_plus, _ = exp_fit.mut_rate_and_its_grad(**params_dict_plus, compute_gradient=False)
                params_dict_minus = copy.deepcopy(params_dict)
                params_dict_minus['lambda_sc'][j] -= dparam
                mut_profile_minus, _ = exp_fit.mut_rate_and_its_grad(**params_dict_minus, compute_gradient=False)
                numerical_grad_wrt_param = (mut_profile_plus - mut_profile_minus) / (2 * dparam)
                plt.title(f'Gradient of mut profile wrt lambda_sc[{j}]')
                plt.plot(numerical_grad_wrt_param, label='numerical')
                plt.plot(grad_mut_profile['lambda_sc'][j], label='analytical', ls='--')
                plt.legend()
                plt.show()

    @clocked
    def multisys_loss_and_grad(self, params_1D, compute_gradient=True):
        self.logger.debug(f'multysys_loss_and_grad called')
        losses_exp_fit = {}  # loss of each experiment fit
        loss_train = 0
        grad_tot = np.zeros_like(params_1D)  # total gradient (only training)
        for system in self.systems:
            # find params from each system
            params_dict = self.pack_params(params_1D, system)
            mut_profiles_dict = {}
            grad_mut_profiles_dict = {}
            for exp_fit in system.exp_fits_all:
                # if mut_profile not computed OR (in training, gradient needed but current gradient is not yet computed) then compute with gradient as required
                if (exp_fit.conc_mM not in mut_profiles_dict.keys() or 
                    (exp_fit.is_training and compute_gradient and grad_mut_profiles_dict.get(exp_fit.conc_mM) is None)):
                    if exp_fit.is_training and compute_gradient and grad_mut_profiles_dict.get(exp_fit.conc_mM) is None:
                        # need to compute gradient
                        mut_profile_model, grad_mut_profile = exp_fit.mut_rate_and_its_grad(
                            **params_dict, compute_gradient=True)
                        if self.debug and self.check_gradient:
                            self.debug_check_gradients(params_dict, exp_fit, grad_mut_profile, mut_profile_model)
                    else:   # do not need gradient (for now)
                        mut_profile_model, grad_mut_profile = exp_fit.mut_rate_and_its_grad(
                            **params_dict, compute_gradient=False)
                    mut_profiles_dict[exp_fit.conc_mM] = mut_profile_model
                    grad_mut_profiles_dict[exp_fit.conc_mM] = grad_mut_profile  # May be None if compute_gradient==False
                else:
                    mut_profile_model = mut_profiles_dict[exp_fit.conc_mM]
                    grad_mut_profile = grad_mut_profiles_dict[exp_fit.conc_mM]
                loss_exp, grad_exp = exp_fit.loss_and_grad(mut_profile_model, grad_mut_profile)
                # compare with finite differences
                if self.debug and self.check_gradient and grad_exp is not None:
                    grad_exp_ = self.map_system_grad_to_total(grad_exp, system)
                    grad_fd = np.zeros(len(grad_exp_))
                    for i, grad_anal in enumerate(grad_exp_):
                        if grad_anal is None:
                            continue
                        else:
                            params_plus = params_1D.copy()
                            # use 0.02 for lambda_sc
                            if i >= (8 if self.DMS_mode else 12):
                                increment = 0.02
                            else:
                                increment = 1e-4
                            params_plus[i] += increment
                            mut_rate_plus, _ = exp_fit.mut_rate_and_its_grad(**self.pack_params(params_plus, system), compute_gradient=False)
                            #params_minus = params_1D.copy()
                            #params_minus[i] -= 1e-4
                            #mut_rate_minus, _ = exp_fit.mut_rate_and_its_grad(**self.pack_params(params_minus, system), compute_gradient=False)
                            #loss_minus, _ = exp_fit.loss_and_grad(mut_rate_minus, None)
                            loss_plus, _ = exp_fit.loss_and_grad(mut_rate_plus, None)
                            grad_fd[i] = (loss_plus - loss_exp) / increment
                    # rel diff
                    # compare
                    plt.plot(grad_exp_, label='analytical')
                    plt.plot(grad_fd, label='finite differences', ls='--')
                    plt.legend()
                    plt.title(f'Gradient of {exp_fit.short_description}')
                    # plot relative difference in twin axis
                    plt.twinx()
                    plt.plot(np.abs(grad_exp_ - grad_fd) / np.abs(grad_exp_), color='r', label='rel diff')
                    plt.legend()
                    plt.show()
                losses_exp_fit[exp_fit.ID] = loss_exp
                if system.exp_fits_train:
                    if exp_fit in system.exp_fits_train:
                        loss_train += loss_exp
                        if compute_gradient:
                            grad_tot += self.map_system_grad_to_total(grad_exp, system)
        self.losses_exp_fit = losses_exp_fit
        assert grad_tot.shape == params_1D.shape, "Gradient shape mismatch!"
        # Put the gradient to zero for the physical parameters if fix_physical_params is True
        if self.fix_physical_params:
            grad_tot[:8 if self.DMS_mode else 12] = 0
        if self.fix_lambda_sc:
            # Set lambda_sc gradient to zero if not inferring 1D soft constraints
            grad_tot[8 if self.DMS_mode else 12:] = 0
        
        # create a subdirectory to store parameters at each call
        os.makedirs(os.path.join(self.output_dir, 'params_at_calls'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'loss_at_calls'), exist_ok=True)
        np.savetxt(os.path.join(self.output_dir, 'loss_at_calls', f'loss_call_{self.evaluation_count}.txt'), np.array([loss_train]))
        np.savetxt(os.path.join(self.output_dir, 'params_at_calls', f'params_call_{self.evaluation_count}.txt'), params_1D)
        if compute_gradient:
            os.makedirs(os.path.join(self.output_dir, 'grad_at_calls'), exist_ok=True)
            np.savetxt(os.path.join(self.output_dir, 'grad_at_calls', f'grad_call_{self.evaluation_count}.txt'), grad_tot)
        self.evaluation_count += 1
            
        return loss_train, grad_tot

    def plot_loss(self):
        plt.figure()
        # Get marker list from matplotlib rcParams or use a default list
        markers = plt.rcParams['axes.prop_cycle'].by_key().get('marker', 
              ['o', 'v', '^', '<', '>', 's', 'p', '*'])
        for i,system in enumerate(self.systems):
            marker = markers[i % len(markers)]
            if system.exp_fits_train is not None:
                total_loss_train_history = np.zeros(len(system.exp_fits_train[0].loss_history))
                for exp_fit in system.exp_fits_train:
                    total_loss_train_history += np.array(exp_fit.loss_history)
                normalised_loss = total_loss_train_history / (system.N_seq * len(system.exp_fits_train))
                ls = '-'
                color = 'b'
                training_or_val = 'train'
                label = f'{system.sys_name} ({training_or_val})'
                plt.plot(normalised_loss, label=label, linestyle=ls, color=color, marker=marker)
            if system.exp_fits_val is not None:
                total_loss_val_history = np.zeros(len(system.exp_fits_val[0].loss_history))
                for exp_fit in system.exp_fits_val:
                    total_loss_val_history += np.array(exp_fit.loss_history)
                normalised_loss = total_loss_val_history / (system.N_seq * len(system.exp_fits_val))
                ls = '--'
                color = 'r'
                training_or_val = 'val'
                label = f'{system.sys_name} ({training_or_val})'
                plt.plot(normalised_loss, label=label, linestyle=ls, color=color, marker=marker)
            # next marker from matplotlib
        plt.title('Loss history')
        plt.xlabel('Iteration')
        plt.ylabel('Loss/(N_seq*N_exps)')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'loss_history.png'), bbox_inches='tight')
        if plt.isinteractive():
            plt.show()
        else:
            plt.close()
    
    def plot_params(self):
        '''Plot the history of mu_r, p_b, p_bind, m0, m1'''
        _, axs = plt.subplots(3, 4, sharex=True, figsize=(16, 12))
        # pack params
        # plot mu_r, p_b, m0, m1, then p_bind
        for i, param in enumerate(['mu_r', 'p_b', 'm0', 'm1']):
            ax = axs[i//4, i%4]
            ax.plot(self.params_history[param])
            ax.set_title(param)
        # plot p_bind
        for i, (key, history) in enumerate(self.params_history['p_bind'].items()):
            ax = axs[(i//4)+1, i%4]
            ax.plot(history)
            ax.set_title(f'p_bind[{key}]')
        for ax in axs.flat:
            ax.set_xlabel('Iteration')
            ax.grid()
            ax.set_xlim(0, None)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'params_history.png'), bbox_inches='tight')
        if plt.isinteractive():
            plt.show()
        else:
            plt.close()

    
    def save_results(self):
        params_final_1D = self.fit_result.x if hasattr(self, 'fit_result') else self.params_last_callback
        self.callback(params_final_1D, last_callback=True)

    def callback(self, params, last_callback=False):
        if not last_callback: # Only count actual optimization steps
            self.iteration_count += 1
        if datetime.datetime.now() - self.fit_start_time > datetime.timedelta(hours=48) and not last_callback:
            # 24 hours elapsed; save results and stop optimization
            self.logger.info("24 hours elapsed; saving results before stopping optimization.")
            self.save_results()
            raise KeyboardInterrupt("Stopping optimization after 24 hours.")
        self.logger.info(f'Callback n.{self.iteration_count} at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} with losses: {self.losses_exp_fit}\n params: {list(params)}')
        np.savetxt(os.path.join(self.output_dir, 'params1D.txt'), params)
        if not last_callback:
            # update params
            self.params_last_callback = params
            # update loss history for each ExperimentFit
            for system in self.systems:
                for exp_fit in system.exp_fits_all:
                    exp_fit.loss_history.append(self.losses_exp_fit[exp_fit.ID])
            params_dict = self.pack_params(params, self.systems[0])
            # update params history dictionary
            for param, history in self.params_history.items():
                if param == 'p_bind':
                    for key, _ in params_dict[param].items():
                        self.params_history[param][key].append(params_dict[param][key])
                else:
                    history.append(params_dict[param])  
        # plots
        if self.do_plots:
            # do this only if: first callback or very last callback or last callback was more than 15s ago
            await_time = 30 if self.debug else 120
            if self.last_plot_callback_time is None or last_callback or (datetime.datetime.now() - self.last_plot_callback_time).seconds > await_time:
                self.plot_loss()
                self.plot_params()
                for system in self.systems:
                    system_param = self.pack_params(params, system)
                    # plot mut profile
                    system.plot_mut_profiles(**system_param, save_fig_path=os.path.join(self.output_dir, f'{system.sys_name}_mut_profile.png'))
                    if plt.isinteractive():
                        plt.show()
                    else:
                        plt.close()
                    # plot lambdas
                    # if there are training experiments in the system
                    if system.exp_fits_train:
                        system.plot_lambdas(system_param['lambda_sc'], save_fig_path=os.path.join(self.output_dir, f'{system.sys_name}_lambda_sc.png'))
                    # plot pairing probs
                    if last_callback or self.infer_1D_sc:
                        system.plot_pairing_probs(**system_param, save_fig_path=os.path.join(self.output_dir, f'{system.sys_name}_pairing_probs.png'))
                        if plt.isinteractive():
                            plt.show()
                        else:
                            plt.close()
                        system.plot_averages_vs_conc(system_param, save_fig_path=os.path.join(self.output_dir, f'{system.sys_name}_avg_vs_conc.png'))
                        if plt.isinteractive():
                            plt.show()
                        else:
                            plt.close()
                self.logger.info(f'Plots saved')
                plt.close('all')
                # update last callback time
                self.last_plot_callback_time = datetime.datetime.now()

    
#reload library
#importlib.reload(class_experiment)
#%%
# try to fit
if __name__ == '__main__':
    #cspA
    rna_struct_home = os.environ.get("RNA_STRUCT_HOME")
    experiments_cspA_10C = [Experiment(path) for path in Experiment.paths_to_cspA_10C_data_txt]
    experiments_cspA_10C_train = experiments_cspA_10C[:3]
    experiments_cspA_10C_val = experiments_cspA_10C[3:]
    experiment_cspA_10C_with_protein_path = f'{rna_struct_home}/data_validation_draco/SRR6507969/rf_map_draco_params_170_nt/rf_count/SRR6507969_sorted.txt'
    experiments_cspA_10C_with_protein = [Experiment(experiment_cspA_10C_with_protein_path),]
    experiment_cspA_37C_train = [Experiment(path) for path in Experiment.paths_to_cspa_37C_data_txt]
    #redmond
    experiments_Redmond = [Experiment(path) for path in Experiment.paths_to_redmond_ivt_data_txt]
    experiments_Redmond = [exp for exp in experiments_Redmond if exp.conc_mM != 85]
    experiments_Redmond_train = [exp for exp in experiments_Redmond if exp.system_name not in ['hc16', 'bact_RNaseP_typeA']]
    experiments_Redmond_val = [exp for exp in experiments_Redmond if exp.system_name in ['hc16', 'bact_RNaseP_typeA']]
    experiments_new_Redmond = [Experiment(path) for path in Experiment.paths_to_newseq_WT_data_txt]

    args_multi_sys ={
        #'experiments': experiment_cspA_37C_train,
        'experiments': experiments_new_Redmond,
        'validation_exps': None,
        #'output_suffix': 'nuovo',
        #'debug': True,
        'infer_1D_sc': True,
        'guess': 'fits/newseq_fix_1/newseqWT/params1D.txt',
        #'overwrite': True,
        #'check_gradient': True,
        'fix_physical_params': True,
        'use_interpolated_ps': True,
    }
    multi_sys = MultiSystemsFit(**args_multi_sys)
# %%
# try fit
if __name__ == '__main__':
    multi_sys.fit()
# %%
# screen -S name to create a screen
# Ctrl + a + d to detach

# screen -ls to see the screen
# screen -r name to reattach
# %%

# %%
