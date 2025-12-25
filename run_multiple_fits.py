#%%
import os
import importlib
import datetime
from multiprocessing import Pool
import class_experimentfit
from tqdm import tqdm
from class_experimentfit import MultiSystemsFit 

#importlib.reload(class_experimentfit)
def initialize_list_of_multisys_args(systems):
    # initialize experiments
    from class_experiment import Experiment
    rna_struct_home = os.environ.get("RNA_STRUCT_HOME", ".")
    #experiments_Redmond_train = [exp for exp in experiments_Redmond if exp.system not in ['hc16', 'bact_RNaseP_typeA']]
    #experiments_Redmond_val = [exp for exp in experiments_Redmond if exp.system in ['hc16', 'bact_RNaseP_typeA']]
    # initialize list of args
    list_of_args_multi_sys = []
    if systems == 'cspA':
        experiments_cspA_10C = [Experiment(path) for path in Experiment.paths_to_cspA_10C_data_txt]
        experiments_cspA_10C_train = experiments_cspA_10C
        experiments_cspA_10C_val = None
        #experiments_cspA_10C_train = experiments_cspA_10C[:3]
        #experiments_cspA_10C_val = experiments_cspA_10C[3:]
        experiment_cspA_10C_with_protein_path = f'{rna_struct_home}/data_validation_draco/SRR6507969/rf_map_draco_params_170_nt/rf_count/SRR6507969_sorted.txt'
        experiments_cspA_10C_with_protein = [Experiment(experiment_cspA_10C_with_protein_path),]
        experiment_cspA_37C_train = [Experiment(path) for path in Experiment.paths_to_cspa_37C_data_txt]
        list_of_args_multi_sys.append({
            'experiments': experiments_cspA_10C_train,
            'validation_exps': experiments_cspA_10C_val,
            'output_suffix': 'cspA_10C'})
        list_of_args_multi_sys.append({
            'experiments': experiments_cspA_10C_with_protein,
            'validation_exps': None,
            'output_suffix': 'cspA_with_protein'})
        list_of_args_multi_sys.append({
            'experiments': experiment_cspA_37C_train,
            'validation_exps': None,
            'output_suffix': 'cspA_37C'})
    elif systems == 'redmond': # fit each system separately
        experiments_Redmond = [Experiment(path) for path in Experiment.paths_to_redmond_ivt_data_txt]
        # remove exps at 85mM
        experiments_Redmond = [exp for exp in experiments_Redmond if exp.conc_mM != 85]
        # add redmond exps
        for system in ['hc16', 'bact_RNaseP_typeA', 'tetrahymena_ribozyme', 'HCV_IRES', 'V_chol_gly_riboswitch']:
            exps = [exp for exp in experiments_Redmond if exp.system == system]
            list_of_args_multi_sys.append({
                'experiments': exps,
                'validation_exps': None,
                'output_suffix': system})
    elif systems == 'redmond_together': # fit all systems together
        experiments_Redmond = [Experiment(path) for path in Experiment.paths_to_redmond_ivt_data_txt]
        experiments_Redmond = [exp for exp in experiments_Redmond if exp.conc_mM != 85]
        experiments_Redmond_train = [exp for exp in experiments_Redmond if exp.system not in ['hc16', 'bact_RNaseP_typeA']]
        experiments_Redmond_val = [exp for exp in experiments_Redmond if exp.system in ['hc16', 'bact_RNaseP_typeA']]
        list_of_args_multi_sys.append({
            'experiments': experiments_Redmond_train,
            'validation_exps': experiments_Redmond_val,
            'output_suffix': 'redmond_together'})
    elif systems == 'redmond_crossval': # fit 3 systems together and validate the other 2
        experiments_Redmond = [Experiment(path) for path in Experiment.paths_to_redmond_ivt_data_txt]
        experiments_Redmond = [exp for exp in experiments_Redmond if exp.conc_mM != 85]
        # take all combinations of 3 training systems
        from itertools import combinations
        systems = ['hc16', 'bact_RNaseP_typeA', 'tetrahymena_ribozyme', 'HCV_IRES', 'V_chol_gly_riboswitch']
        for train_systems in combinations(systems, 3):
            train_exps = [exp for exp in experiments_Redmond if exp.system in train_systems]
            val_exps = [exp for exp in experiments_Redmond if exp.system not in train_systems]
            list_of_args_multi_sys.append({
                'experiments': train_exps,
                'validation_exps': val_exps,
                'output_suffix': 'red_crossval_' + '_'.join(train_systems)
            })
    elif systems == 'cspAcomb':
        from class_experiment import initialise_combined_cspA_exp
        import numpy as np
        for pop_10 in np.linspace(0, 1, 6):
            exps = [initialise_combined_cspA_exp(pop_10=pop_10, normalise_mut_rate=True),]
            list_of_args_multi_sys.append({
                'experiments': exps,
                'validation_exps': None,
                'output_suffix': f'cspA_comb_{pop_10*100:.0f}'})
            assert np.isclose(exps[0].df['mut_rate'].mean(), 0.012180, atol=1e-3), f"Normalisation failed for pop_10={pop_10}"
        # fit also the 10C with protein
        experiments_cspA_10C_with_protein = [initialise_combined_cspA_exp(pop_10=None, is_exp_protein=True, normalise_mut_rate=True),]
        # assert temperature is close to the right one
        assert np.isclose(experiments_cspA_10C_with_protein[0].temp_C, (10 + 37) / 2, atol=0.1)
        list_of_args_multi_sys.append({
            'experiments': experiments_cspA_10C_with_protein,
            'validation_exps': None,
            'output_suffix': 'cspA_with_protein'})
    elif systems == 'cspAcomb_together':
        from class_experiment import initialise_combined_cspA_exp
        import numpy as np
        exps = [initialise_combined_cspA_exp(pop_10=pop_10, same_system=True, normalise_mut_rate=True) for pop_10 in np.linspace(0, 1, 6)]
        assert np.isclose(exps[0].df['mut_rate'].mean(), 0.012180, atol=1e-3), "Normalisation failed for pop_10=0"
        list_of_args_multi_sys.append({
            'experiments': exps,
            'validation_exps': None,
            'output_suffix': f'cspA_comb_together'})
    elif systems == 'cspAcomb_synthetic':
        from class_experiment import initialise_combined_cspA_exp
        import numpy as np
        for pop_10 in np.linspace(0, 1, 6):
            exps = [initialise_combined_cspA_exp(pop_10=pop_10, normalise_mut_rate=False, use_synthetic_data=True),]
            list_of_args_multi_sys.append({
                'experiments': exps,
                'validation_exps': None,
                'output_suffix': f'cspA_comb_{pop_10*100:.0f}'})
        # fit also the 10C with protein
        experiments_cspA_10C_with_protein = [initialise_combined_cspA_exp(pop_10=None, is_exp_protein=True, normalise_mut_rate=False, use_synthetic_data=True),]
        # assert temperature is close to the right one
        assert np.isclose(experiments_cspA_10C_with_protein[0].temp_C, (10 + 37) / 2, atol=0.1)
        list_of_args_multi_sys.append({
            'experiments': experiments_cspA_10C_with_protein,
            'validation_exps': None,
            'output_suffix': 'cspA_with_protein'})
    elif systems == 'cspAcomb_together_synthetic':
        from class_experiment import initialise_combined_cspA_exp
        import numpy as np
        exps = [initialise_combined_cspA_exp(pop_10=pop_10, same_system=True, normalise_mut_rate=False, use_synthetic_data=True) for pop_10 in np.linspace(0, 1, 6)]
        list_of_args_multi_sys.append({
            'experiments': exps,
            'validation_exps': None,
            'output_suffix': f'cspA_comb_together'})
    elif systems == 'synthetic_comb_together':
        from class_experiment import create_exp_synthetic_comb
        exps = [create_exp_synthetic_comb(pop1=pop1, same_system=True) for pop1 in [0, 0.2, 0.4, 0.6, 0.8, 1.0]]
        list_of_args_multi_sys.append({
            'experiments': exps,
            'validation_exps': None,
            'output_suffix': f'synthetic_comb'})
    elif systems == 'synthetic_comb':
        from class_experiment import create_exp_synthetic_comb
        for pop1 in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            exps = [create_exp_synthetic_comb(pop1=pop1, same_system=False),]
            list_of_args_multi_sys.append({
                'experiments': exps,
                'validation_exps': None,
                'output_suffix': f'synthetic_comb_{pop1*100:.0f}'})
    elif systems == 'newseqWT':
        exps = [Experiment(path) for path in Experiment.paths_to_newseq_WT_data_txt]
        list_of_args_multi_sys.append({
            'experiments': exps,
            'validation_exps': None,
            'use_interpolated_ps': True,
            'output_suffix': f'newseqWT'})
    else:
        raise ValueError(f"Unknown systems: {systems}")
    return list_of_args_multi_sys

def run_fit(multi_sys_instance):
    return multi_sys_instance.fit()

def safe_run_fit(multi_sys_instance):
    try:
        return multi_sys_instance.fit()
    except Exception as e:
        multi_sys_instance.logger.exception("Error during fit:", e)
        print(f"Error during fit: {e}")
        return e

def run_parallel_fits(list_of_args_multi_sys, infer_1D_sc, overwrite=False, root_dir=None, initial_guess=None, fit_mode='sequential', strict_convergence=False, fix_physical_params=False, fix_lambda_sc=False):
    '''Run multiple fits in parallel using multiprocessing and show progress with tqdm.'''
    # Disable printing to stdout
    list_of_args_multi_sys = [{**args, 'print_to_std_out': False} for args in list_of_args_multi_sys]
    if infer_1D_sc:
        list_of_args_multi_sys = [{**args, 'infer_1D_sc': True} for args in list_of_args_multi_sys]
        # Update output_suffix to include a tag
        list_of_args_multi_sys = [{**args, 'output_suffix': f"{args['output_suffix']}_with_lambdas"} 
                                    for args in list_of_args_multi_sys]
    # add overwrite
    if overwrite:
        list_of_args_multi_sys = [{**args, 'overwrite': True} for args in list_of_args_multi_sys]
    if root_dir:
        list_of_args_multi_sys = [{**args, 'root_dir': 'fits/' + root_dir} for args in list_of_args_multi_sys]
    # if root_dir exists raise error
    if root_dir and os.path.exists('fits/'+root_dir):
        raise ValueError(f"root_dir {'fits/'+root_dir} already exists. Please choose a different name.")
    # Create a list of MultiSystemsFit instances
    print("results will be saved in :")
    for args in list_of_args_multi_sys:
        print(args.get('root_dir', 'fits') +'/'+ args['output_suffix'])
    if initial_guess is not None:
        list_of_args_multi_sys = [{**args, 'guess': initial_guess} for args in list_of_args_multi_sys]
    
    # Handle fit_mode vs legacy flags (legacy flags take precedence for backwards compatibility)
    effective_fit_mode = fit_mode
    if fix_physical_params and fix_lambda_sc:
        raise ValueError("Cannot use both --fix-physical-params and --fix-lambda-sc together")
    elif fix_physical_params:
        effective_fit_mode = 'lambda_only'
        print("Note: --fix-physical-params is deprecated; use --fit-mode lambda_only instead")
    elif fix_lambda_sc:
        effective_fit_mode = 'physical_only'
        print("Note: --fix-lambda-sc is deprecated; use --fit-mode physical_only instead")
    
    # Add fit_mode to all args (only if not using legacy flags which set fix_* directly)
    if not (fix_physical_params or fix_lambda_sc):
        list_of_args_multi_sys = [{**args, 'fit_mode': effective_fit_mode} for args in list_of_args_multi_sys]
    else:
        # Legacy behavior: use fix_physical_params and fix_lambda_sc directly
        if fix_physical_params:
            list_of_args_multi_sys = [{**args, 'fix_physical_params': True} for args in list_of_args_multi_sys]
        if fix_lambda_sc:
            list_of_args_multi_sys = [{**args, 'fix_lambda_sc': True} for args in list_of_args_multi_sys]
    
    # Add strict_convergence to all args
    if strict_convergence:
        list_of_args_multi_sys = [{**args, 'strict_convergence': True} for args in list_of_args_multi_sys]

    multi_sys_list = [MultiSystemsFit(**args) for args in list_of_args_multi_sys]
    # Inside run_parallel_fits
    num_processes = min(len(multi_sys_list), os.cpu_count()) # Or make it an argument
    pool = Pool(processes=num_processes)
    try:
        results = list(tqdm(pool.imap(safe_run_fit, multi_sys_list), total=len(multi_sys_list)))
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Terminating pool...")
        pool.terminate()
        pool.join()
        results = None
    except Exception as e:
        print(f"An error occurred in the pool: {e}")
        pool.terminate()
        pool.join()
    else:
        pool.close()
        pool.join()
    return results
#%%
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run parallel fits with optional parameters."
    )
    parser.add_argument(
        "--infer-1D-sc",
        action="store_true",
        help="Set flag to use infer_1D_sc=True"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Enables overwriting of existing files"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        help="Root directory to save results"
    )
    parser.add_argument(
        "--initial-guess",
        type=str,
        default=None,
        help="Initial guess can be None, random, last, or a path to a file"
    )
    parser.add_argument(
        "--system",
        type=str,
        choices=['cspA', 'redmond', 'cspAcomb', 'cspAcomb_together', 'redmond_together', 'redmond_crossval', 'cspAcomb_synthetic', 'cspAcomb_together_synthetic', 'synthetic_comb', 'synthetic_comb_together', 'newseqWT'],
        required=True,
        help="Specify the system to analyze (cspA, redmond, cspAcomb, cspAcomb_together, redmond_together, redmond_crossval, cspAcomb_synthetic, cspAcomb_together_synthetic, synthetic_comb, synthetic_comb_together, newseqWT). Only one system can be selected."
    )
    parser.add_argument(
        "--fit-mode",
        type=str,
        choices=['simultaneous', 'physical_only', 'lambda_only', 'sequential'],
        default='sequential',
        help="Fitting strategy: sequential (default), simultaneous, physical_only, or lambda_only"
    )
    parser.add_argument(
        "--strict-convergence",
        action="store_true",
        default=False,
        help="Use very strict convergence (runs until timeout/numerical limit). Default uses scipy defaults."
    )
    # Legacy flags - kept for backwards compatibility
    parser.add_argument(
        "--fix-physical-params",
        action="store_true",
        default=False,
        help="(Legacy) Set flag to use fixed physical parameters. Prefer --fit-mode lambda_only"
    )
    parser.add_argument(
        "--fix-lambda-sc",
        action="store_true",
        default=False,
        help="(Legacy) Set flag to use fixed lambda_sc. Prefer --fit-mode physical_only"
    )
    args = parser.parse_args()

    print("Starting parallel fits...", datetime.datetime.now())
    list_of_args_multi_sys = initialize_list_of_multisys_args(systems=args.system)
    run_parallel_fits(
        list_of_args_multi_sys, 
        infer_1D_sc=args.infer_1D_sc, 
        root_dir=args.root_dir, 
        overwrite=args.overwrite,
        initial_guess=args.initial_guess,
        fit_mode=args.fit_mode,
        strict_convergence=args.strict_convergence,
        fix_physical_params=args.fix_physical_params,
        fix_lambda_sc=args.fix_lambda_sc
    )
    print("Parallel fits completed.", datetime.datetime.now())