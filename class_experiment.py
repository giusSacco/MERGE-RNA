#%%
import os
import time
import warnings
import logging

import uuid

import numpy as np
import pandas as pd

def dotbracket_to_matrix(dotbracket):
    '''Creates a pairing matrix from a dotbracket string.'''
    matrix = np.zeros((len(dotbracket), len(dotbracket)))
    # Stack for holding the indices of opening brackets
    stack = []
    for i, char in enumerate(dotbracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            j = stack.pop()
            matrix[i][j] = 1
            matrix[j][i] = 1
        elif char == '.':
            pass
        else:
            raise ValueError('Invalid character in dotbracket string')
    assert len(stack) == 0
    return matrix

# Redmond's code to convert binary to pandas, added only comments
def to_int(bytestring):
    ''' Convert a bytestring to an integer '''
    return int.from_bytes(bytestring, byteorder="little")

def bytes_to_seq(bytestring):
    ''' Convert a bytestring to a sequence of nucleotides '''
    sequence = []
    nts = ["A","C","G","T","N"]
    for byte in bytestring:
        high, low = byte>>4, byte&0x0F
        sequence.append(nts[high])
        sequence.append(nts[low])
    return sequence

def load_rc(filepath, sample_name, debug=False):
    """Load RNA structure data from .rc file and return it as a pandas DataFrame."""
    with open(filepath,"rb") as infile:
        # Same notation as RNA framework article here
        len_transcript_id = to_int(infile.read(4)) # read in the first 4 bytes to get the length of the transcript id, needed in the next line
        transcript_id = infile.read(len_transcript_id).decode("utf8") # read in the transcript id
        len_seq = to_int(infile.read(4)) # read in the length of the sequence
        seq = bytes_to_seq(infile.read(int((len_seq+1)/2)))

        counts = np.array([to_int(infile.read(4)) for i in range(len_seq)]) # read in the counts, which are 4 bytes each
        coverage = np.array([to_int(infile.read(4))for i in range(len_seq)]) # same for coverage
        num_reads_to_transcript = to_int(infile.read(4))
        num_reads_experiment = to_int(infile.read(4))
        rest_of_file = infile.read()
    
    # Redmond's comment: correct sequence if odd length due to way bytes are read in
    if len(seq)!=len(counts):
        len_seq=len_seq-1
        seq=seq[:-1]
    
    if debug:
        print("len_transcript_id", len_transcript_id,sep=":")
        print("transcript_id",transcript_id)
        print("len_seq",len_seq)
        print("counts",counts, len(counts))
        print("coverage",coverage, len(coverage))
        print("num_reads_to_transcript",num_reads_to_transcript)
        print("num_reads_experiment",num_reads_experiment)
        print("rest of file", rest_of_file)
        print(len(seq), seq)
        print(len(counts), counts)
        print(len(coverage), coverage)
        print(len(np.arange(1,len(seq)+1)))
    
    # Dictionary to store the data
    data = {'Sample' : sample_name,
            'mut_count' : counts,
            'wt_count' : coverage-counts,
            'total_count' : coverage,
            'mut_rate' : counts/coverage,
            'ref_nt' : list(seq),
            'pos' : range(1, len(seq)+1)}
    
    return pd.DataFrame(data)


def clocked(func):
    def wrapper(*args, **kwargs):
        # Check if the first argument (self) has a logger attribute.
        logger = getattr(args[0], 'logger', logging.getLogger(__name__))
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"Execution time of function {func.__name__}: {execution_time} seconds")
        return result
    return wrapper

#%%
class Experiment:
    newseqWT_fastq_names = [fastq_name.removesuffix('.fastq') for fastq_name in os.listdir(f"{os.environ.get('RNA_STRUCT_HOME')}/Newseq/fastq") if fastq_name.endswith('.fastq')]
    paths_to_newseq_WT_data_txt = [f"{os.environ.get('RNA_STRUCT_HOME')}/Newseq/RNAf_results/{fastq_name}/rf_map_draco_params/rf_count/{fastq_name}_sorted.txt" for fastq_name in newseqWT_fastq_names]
    paths_to_redmond_ivt_data_txt = [f'{os.environ.get("RNA_STRUCT_HOME")}/RedmondData/ivt_2/Rep{rep_num}_{conc_mM}mM/{system}/rfcount_ndni/LAST_MD_sorted.txt' for rep_num in range(1,3) for conc_mM in [0,8,17,34,57,85] for system in ['bact_RNaseP_typeA', 'hc16', 'HCV_IRES', 'tetrahymena_ribozyme', 'V_chol_gly_riboswitch']]
    SRRs_cspA_10C = ['SRR6123774','SRR6123775','SRR6507967', 'SRR6507968']
    paths_to_cspA_10C_data_txt = [f'{os.environ.get("RNA_STRUCT_HOME")}/data_validation_draco/{SRR}/rf_map_draco_params_170_nt/rf_count/{SRR}_sorted.txt' for SRR in SRRs_cspA_10C]
    SRRs_cspA_37C = ['SRR6123773','SRR6507966']
    paths_to_cspa_37C_data_txt = [f'{os.environ.get("RNA_STRUCT_HOME")}/data_validation_draco/{SRR}/rf_map_draco_params_170_nt/rf_count/{SRR}_sorted.txt' for SRR in SRRs_cspA_37C]
    SRRs_cspA_10C_with_protein = ['SRR6507969',]
    paths_to_cspA_10C_with_protein_data_txt = [f'{os.environ.get("RNA_STRUCT_HOME")}/data_validation_draco/{SRR}/rf_map_draco_params_170_nt/rf_count/{SRR}_sorted.txt' for SRR in SRRs_cspA_10C_with_protein]

    def __init__(self, path_to_file_with_info=None, seq=None, **kwargs):
        '''Initialize an Experiment object with data from a file or synthetic data.
        In the first case, only the path to a txt file correctly formatted (you can create it with Experiment.create_info_file)
          with information about the experiment is required.

        In the second case, a sequence must be provided.
        In order to generate sysnthetic data, the following parameters must be provided to the function generate_synthetic_data:
          - params_dict: dictionary with the parameters of the biophysical model
          - noise: whether to add noise to the mutation rates
          - coverage: read depth for all positions
          - eps_b: background mutation rates

        Additional optional keywords are:
          - reagent: reagent used in the experiment
          - conc_mM: concentration in mM
          - system_name: name of the system
          - rep_number: replicate number
          - short_description: short description of the experiment
          - long_description: long description of the experiment
          - temp_C: temperature in Celsius
        '''
        if path_to_file_with_info:
            self.path_to_info_txt = path_to_file_with_info
            if not os.path.exists(self.path_to_info_txt):
                raise FileNotFoundError(f"File {self.path_to_info_txt} not found")
            # load the data
            self.load_experiment_info() # info such as reagent, ... are saved as attributes
            self.reagent = self.__dict__['Reagent']
            conc_mM = self.__dict__['Concentration']
            self.conc_mM = None if conc_mM == 'None' else int(conc_mM)
            self.ID = self.__dict__['ID']   # unique identifier of the experiment
            self.path_to_rc = self.__dict__['Path to rc data']
            if '$RNA_STRUCT_HOME' in self.path_to_rc:   # ensures that the path is correct on different machines
                self.path_to_rc = os.path.expandvars(self.path_to_rc)
            self.df, self.raw_df = self.load_data()
            self.N_seq = len(self.df)
            self.seq = ''.join(self.df['ref_nt'])
            self.seq = self.seq.replace('T', 'U')
            self.temp_C = int(self.__dict__['Temperature'])
            self.temp_K = 273.15 + self.temp_C
            self.system_name = self.__dict__['system']
            self.rep_number = int(self.__dict__['Rep number'])
            self.short_description = self.__dict__['Short description']
            self.long_description = self.__dict__['Description']
            self.is_synthetic = False
        elif seq:
            print('No data file provided, using sequence and creating synthetic data')
            self.seq = seq
            self.N_seq = len(self.seq)
            self.temp_C = 37 if 'temp_C' not in kwargs else kwargs['temp_C']
            self.temp_K = 273.15 + self.temp_C
            self.reagent = 'Synthetic data' if 'reagent' not in kwargs else kwargs['reagent']
            self.conc_mM = None if 'conc_mM' not in kwargs else kwargs['conc_mM']
            self.ID = uuid.uuid4()
            self.system_name = 'Synthetic' if 'system_name' not in kwargs else kwargs['system_name']
            self.rep_number = None if 'rep_number' not in kwargs else kwargs['rep_number']
            self.short_description = None
            self.long_description = None
            self.is_synthetic = True
            #self.df = self.generate_synthetic_data(**kwargs)
            self.raw_df = None
        else:
            raise ValueError('Either path to file with information or sequence must be provided')
    '''
    def __str__(self):
        if self.short_description is None:
            return super().__str__()
        else:
            descr = self.short_description.replace(',', ' ')
            descr = self.__class__.__name__ + ': ' + descr
            return descr

    def __repr__(self):  
        if self.short_description is None:
            return super().__str__()
        else:
            descr = self.short_description.replace(',', ' ')
            descr = self.__class__.__name__ + ': ' + descr
            return self.__str__()
    '''

    def load_experiment_info(self):
        with open(self.path_to_info_txt, 'r') as file:
            for line in file:
                # Skip lines that are comments
                if line.startswith('#'):
                    continue
                key, value = line.strip().split(':', 1)
                setattr(self, key.strip(), value.strip())

    def add_pdb_ss_to_df(self, raw_df):
        """
        Add secondary structure information from PDB to the DataFrame.
        This method adds a column 'pdb_ss' to the DataFrame, which contains the secondary structure information
        from the PDB file corresponding to the sequence of the experiment.
        """
        from Bio.PDB import PDBParser
        
        # Import here to avoid circular imports
        pdbs_paths_dict = { #'bact_RNaseP_typeA': '2a2e.pdb',
                            'hc16': 'automated_models_final/hc16_model_1.pdb',
                            'tetrahymena_ribozyme': 'automated_models_final/Tetrahymena_ribozyme_model_1.pdb',
                            'V_chol_gly_riboswitch': 'automated_models_final/VC_gly_riboswitch_apo_half1_model_1.pdb'}
        
        # Get secondary structure for this system
        ss_pdb = None
        seq_pdb = None
        
        if self.system in ['hc16', 'tetrahymena_ribozyme', 'V_chol_gly_riboswitch']:
            pdb_path = pdbs_paths_dict[self.system]
            pdb_name = pdb_path.removeprefix('automated_models_final/')
            pdb_path = os.path.join(os.environ.get("RNA_STRUCT_HOME"), 'other', pdb_path)
            
            # Get sequence from PDB
            pdb_seq = ''
            pdb_parser = PDBParser().get_structure(pdb_name, pdb_path)
            for chain in pdb_parser.get_chains():
                for residue in chain:
                    pdb_seq += residue.get_resname()
            
            # Get secondary structure annotation
            ann_file_path = f'annotated_struct/{pdb_name.removesuffix(".pdb")}.ANNOTATE.dotbracket.out'
            ann_file_path = os.path.join(os.environ.get("RNA_STRUCT_HOME"),'other', ann_file_path)
            
            # Create annotated file if not present
            if not os.path.exists(ann_file_path):
                raise FileNotFoundError(f'Annotated file {ann_file_path} does not exist. Please run barnaba ANNOTATE on the PDB file first.')
            
            with open(ann_file_path) as f:
                lines = f.readlines()
                dotbracket = lines[3]
                # Remove up to .pdb
                dotbracket = dotbracket.split('.pdb')[1]
                # Remove spaces and newlines
                dotbracket = dotbracket.replace(' ', '').replace('\n', '')
                
                # Check presence of pseudoknots
                if any([c in dotbracket for c in ['{', '}', '[', ']']]):
                    print('Warning, pseudoknots present in the secondary structure')
            
            seq_pdb = pdb_seq.replace('T', 'U')
            ss_pdb = dotbracket
            
        elif self.system == 'HCV_IRES':
            with open(os.path.join(os.environ.get("RNA_STRUCT_HOME"), 'other', 'ires')) as f:
                seq_pdb = next(f).replace('\n', '')
                ss_pdb = next(f).replace('\n', '')
        
        # Add secondary structure column to DataFrame
        if ss_pdb is not None and seq_pdb is not None:
            # Verify sequence compatibility
            
            exp_fullseq = ''.join(list(raw_df['ref_nt']))
            # add also df['seq_pdb']
            if seq_pdb == exp_fullseq or seq_pdb == exp_fullseq[:len(seq_pdb)]:
                # Add pdb_ss column, truncating to match DataFrame length
                # add -- as padding if ss_pdb is shorter than the raw_df
                if len(ss_pdb) < len(raw_df):
                    seq_pdb = seq_pdb + '-' * (len(raw_df) - len(seq_pdb))
                    ss_pdb = ss_pdb + '-' * (len(raw_df) - len(ss_pdb))
                raw_df['pdb_ss'] = list(ss_pdb)
                raw_df['seq_pdb'] = list(seq_pdb)
            else:
                print(f'Sequence mismatch for {self.system}, cannot add PDB secondary structure')
                raw_df['pdb_ss'] = ['-'] * len(raw_df)
        else:
            print(f'No secondary structure available for {self.system}')
            raw_df['pdb_ss'] = ['-'] * len(raw_df)
        df = raw_df.dropna()
        # remove rows with coverage lower than the half-mean of the total coverage
        df = df[df['total_count'] >= np.mean(df['total_count'])/2]
        self.raw_df = raw_df
        self.df = df

    def load_data(self):
        # silence runtime warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw_df = load_rc(self.path_to_rc, self.__dict__['Short description'])
            # reactivate warnings
            warnings.simplefilter("default", RuntimeWarning)
        # substitute T with Us
        raw_df['ref_nt'] = raw_df['ref_nt'].replace('T', 'U')
        # remove rows with nans
        df = raw_df.dropna()
        # remove rows with coverage lower than the half-mean of the total coverage
        df = df[df['total_count'] >= np.mean(df['total_count'])/2]
        return df, raw_df


    def generate_synthetic_data(self, *, params_dict, noise=True,
                               coverage=10000, eps_b=None, **kwargs):
        """
        Generate synthetic probing data using the biophysical model.
        This method creates a synthetic dataset by:
        1. Using a biophysical model to predict mutation rates based on provided parameters
        2. Generating read counts with optional noise (binomial distribution)
        3. Creating a DataFrame similar to real experimental data
        
        Args:
            params_dict (dict): Dictionary containing model parameters. If None, 
                                       simple random data will be generated.
            noise (bool, optional): Whether to add noise to the mutation rates.
                                  True: Use binomial distribution for realistic noise.
                                  False: Use exact mutation rates (deterministic).
            coverage (int, optional): Read depth for all positions. Defaults to 10000.
            eps_b (np.ndarray, optional): Background mutation rates. If None, defaults to zeros.
            **kwargs: Additional parameters passed from __init__
        
        Returns:
            pd.DataFrame: DataFrame containing synthetic data
        """
        
        if eps_b is None:
            eps_b = np.zeros(len(self.seq))
        
        # Import here to avoid circular imports
        from class_experimentfit import ExperimentFit
        
        # Safely check for lambda_sc and determine infer_1D_sc
        if 'lambda_sc' in params_dict and params_dict['lambda_sc'] is not None:
            infer_1D_sc = True
        else:
            infer_1D_sc = False
        
        # Create temporary ExperimentFit to compute mutation rates
        exp_fit = ExperimentFit(self, infer_1D_sc=infer_1D_sc, eps_b=eps_b, is_training=True)
        mut_rate, _ = exp_fit.mut_rate_and_its_grad(**params_dict, compute_gradient=False)

        # Add noise to mutation rates
        if noise:
            mut_count = np.random.binomial(coverage, mut_rate)
        else:
            mut_count = np.round(mut_rate * coverage).astype(int)

        # Create DataFrame
        data = {'Sample' : self.short_description,
                'mut_count' : mut_count,
                'wt_count' : coverage - mut_count,
                'total_count' : coverage,
                'mut_rate' : mut_rate,
                'ref_nt' : list(self.seq),
                'pos' : range(1, len(self.seq)+1)}
        df = pd.DataFrame(data)
        self.df = df
        return df


    def create_info_file(output_txt_path, *,  path_to_rc, reagent, concentration,  temperature, system, rep_number,             
            description=None, short_description=None, SRR=None, path_to_full_reference_fasta=None):
        """
        Create an experiment info file with the required fields, asserting that each key is provided.
        The file format will be similar to:
        
        ### Chemical probing experiment
        ID:       <value>
        Description:  <value>
        Short description:  <value>
        Path to rc data:  <value>
        Reagent:  <value>
        Concentration:  <value>
        SRR:  <value>
        Path to full reference fasta:  <value>
        Temperature:  <value>
        system:  <value>
        Rep number:  <value>
        """
        # Build a dictionary of required fields using a combination of instance attributes and provided arguments.
        # Use self.ID for 'ID' assuming the instance was loaded from file or set previously.
        info = {
            "Description": description,
            "Short description": short_description,
            "Path to rc data": path_to_rc,
            "Reagent": reagent,
            "Concentration": concentration,
            "SRR": SRR,
            "Path to full reference fasta": path_to_full_reference_fasta,
            "Temperature": temperature,
            "system": system,
            "Rep number": rep_number
        }
        # create unique ID
        unique_id = uuid.uuid4()
        info["ID"] = unique_id
        
        # Write the info file
        with open(output_txt_path, "w") as f:
            f.write("### Chemical probing experiment\n")
            for key, value in info.items():
                f.write(f"{key}:\t{value}\n")
        print(f"Experiment info file created at {output_txt_path}")

# %%
def create_combined_cspA_df(pop_10, normalise_mut_rate, use_synthetic_data=False, noise=False):
    # SRR6123773 and SRR6123774: the ones they use in DRACO
    cspA_10C_path = [path for path in Experiment.paths_to_cspA_10C_data_txt if '6123774' in path]
    cspA_10C_path = cspA_10C_path[0]
    cspA_37C_path = [path for path in Experiment.paths_to_cspa_37C_data_txt if '6123773' in path]
    cspA_37C_path = cspA_37C_path[0]
    exp_10 = Experiment(cspA_10C_path)
    exp_37 = Experiment(cspA_37C_path)
    pop_37 = 1 - pop_10
    if use_synthetic_data:
        # create synthetic data for 10C and 37C experiments
        RNA_STRUCT_HOME = os.environ.get("RNA_STRUCT_HOME")
        path_to_params_10C = os.path.join(RNA_STRUCT_HOME,"fits/cspA_comb_norm/fix_1/cspA_comb_100_with_lambdas/params1D.txt")
        params_1D = np.loadtxt(path_to_params_10C)
        # import multysystems fit
        from class_experimentfit import MultiSystemsFit
        multisysfit = MultiSystemsFit([exp_10,], infer_1D_sc=True)
        # pack params
        params_dict = multisysfit.pack_params(params_1D, multisysfit.systems[0])
        exp_10.df = exp_10.generate_synthetic_data(params_dict=params_dict, noise=noise)

        # same for exp_37
        path_to_params_37C = os.path.join(RNA_STRUCT_HOME,"fits/cspA_comb_norm/fix_1/cspA_comb_0_with_lambdas/params1D.txt")
        params_1D = np.loadtxt(path_to_params_37C)
        multisysfit = MultiSystemsFit([exp_37,], infer_1D_sc=True)
        params_dict = multisysfit.pack_params(params_1D, multisysfit.systems[0])
        exp_37.df = exp_37.generate_synthetic_data(params_dict=params_dict, noise=noise)
    new_df = pd.DataFrame()
    # total coverage as average of both
    new_df['total_count'] = (exp_10.df['total_count'] + exp_37.df['total_count']) // 2
    # we combine as data_37 * (new_cov/old_cov_37) * pop_37 + data_10 * (new_cov/old_cov_10) * pop_10
    new_df['mut_count'] = exp_10.df['mut_count'] * (new_df['total_count'] / exp_10.df['total_count']) * pop_10
    new_df['mut_count'] += exp_37.df['mut_count'] * (new_df['total_count'] / exp_37.df['total_count']) * pop_37
    new_df['mut_count'] = np.round(new_df['mut_count']).astype(int)

    new_df['mut_rate'] = new_df['mut_count'] / new_df['total_count']
    new_df['ref_nt'] = exp_10.df['ref_nt']
    assert exp_10.df['ref_nt'].equals(exp_37.df['ref_nt'])
    new_df['Sample'] = f"{pop_10 * 100:.0f}% 10C cspA + {pop_37 * 100:.0f}% 37C"
    # following assertion will fail beacause of rounding, otherwise it is correct
    #assert np.allclose(new_df['mut_rate'], pop_10 * exp_10.df['mut_rate'] + pop_37 * exp_37.df['mut_rate'])

    if normalise_mut_rate:
        # force average to be equal to the 37C experiment, i.e. 0.012180
        corrective_factor = 0.012180 / new_df['mut_rate'].mean()
        # correct also mut_count
        new_df['mut_count'] = np.round(new_df['mut_count'] * corrective_factor)
        new_df['mut_rate'] = new_df['mut_count'] / new_df['total_count']

    return new_df
#%%
# %%
def initialise_combined_cspA_exp(pop_10, same_system=False, is_exp_protein=False, normalise_mut_rate=False, use_synthetic_data=False, noise=False):
    """
    Initialise the instance of Experiment with the combined cspA data
    """
    if not is_exp_protein:
        # create combined cspA df
        new_df = create_combined_cspA_df(pop_10, normalise_mut_rate=normalise_mut_rate, use_synthetic_data=use_synthetic_data, noise=noise)
        # create the experiment
        kwargs = {
            'seq':''.join(new_df['ref_nt'].values), 
            'temp_C': (10+37)/2,
            'reagent': 'DMS combined in vitro',
            'system_name': f'cspA_combined_{pop_10 * 100:.0f}%' if not same_system else 'cspA_combined',
        }
        exp = Experiment(**kwargs)
        exp.df = new_df
    else:
        # we still need to create a new experiment for consistency with temperature
        cspA_10C_with_protein_path = [path for path in Experiment.paths_to_cspA_10C_with_protein_data_txt if '6507969' in path]
        cspA_10C_with_protein_path = cspA_10C_with_protein_path[0]
        dummy_exp = Experiment(cspA_10C_with_protein_path)

        if not use_synthetic_data:
            new_df = dummy_exp.df.copy()
        else:
            # create synthetic data for experiment with protein
            RNA_STRUCT_HOME = os.environ.get("RNA_STRUCT_HOME")
            path_to_params_withprotein = os.path.join(RNA_STRUCT_HOME,"fits/cspA_comb_norm/fix_1/cspA_with_protein_with_lambdas/params1D.txt")
            params_1D = np.loadtxt(path_to_params_withprotein)
            # import multysystems fit
            from class_experimentfit import MultiSystemsFit
            multisysfit = MultiSystemsFit([dummy_exp,], infer_1D_sc=True)
            # pack params
            params_dict = multisysfit.pack_params(params_1D, multisysfit.systems[0])
            new_df = dummy_exp.generate_synthetic_data(params_dict=params_dict, noise=noise)
            if noise:
                # add noise to the mutation rates
                raise NotImplementedError('Noise is not implemented for the combined cspA with protein experiment')

        if normalise_mut_rate:
            # force average to be equal to the 37C experiment, i.e. 0.012180
            corrective_factor = 0.012180 / new_df['mut_rate'].mean()
            new_df['mut_count'] = np.round(new_df['mut_count'] * corrective_factor)
            new_df['mut_rate'] = new_df['mut_count'] / new_df['total_count']
        kwargs = {
            'seq':''.join(new_df['ref_nt'].values), 
            'temp_C': (10+37)/2,
            'reagent': 'DMS combined in vitro',
            'system_name': dummy_exp.system_name,
        }
        exp = Experiment(**kwargs)
        exp.df = new_df
    return exp

def create_exp_synthetic_comb(pop1, params_dict = None, same_system = False, noise=False, coverage=10000, eps_b=None):
    """
    Create a synthetic experiment for the bistable sequence that I designed for Redmond

    pop1: float, relative population for the first helix
    noise: bool, whether to add noise to the synthetic data
    """

    if params_dict is None:
        from class_experimentfit import MultiSystemsFit
        params_1D = np.loadtxt("fits/red_crossval_1/red_crossval_bact_RNaseP_typeA_tetrahymena_ribozyme_V_chol_gly_riboswitch/params1D.txt")
        exp = [Experiment(path_) for path_ in Experiment.paths_to_redmond_ivt_data_txt if Experiment(path_).conc_mM == 0]
        exp = exp[0]
        multi_exp = MultiSystemsFit([exp], validation_exps=None, infer_1D_sc=False)
        params_dict = multi_exp.pack_params(params_1D, multi_exp.systems[0])
        del params_1D
        del exp
        del multi_exp

    # seq from redmond's mail
    SEQ = "TAATACGACTCACTATAgggCATTATGCCACAGCCAATCCCCACTTCAACTCACAACTATTCCAAAAAATTGGAATAGTTGTGAGTTGAAGTGGGGATTAAAAAATCCCCACTTCAACTCACAACTATTCCAACCTCCAGCAGACCAT"
    SEQ = SEQ.upper().replace('T', 'U')

    # find the two occurrences of where we should put the constrains
    compl_seq = "CCCCACUUCAACUCACAACUAUUCC"
    start1 = SEQ.find(compl_seq)
    if start1 == -1:
        raise ValueError(f"Could not find helix 1 in sequence")
    end1 = start1 + len(compl_seq) -1
    indices1 = list(range(start1, end1+1))
    assert SEQ[indices1[0]:indices1[-1]+1] == compl_seq, f"Expected {compl_seq} but got {SEQ[indices1[0]:indices1[-1]+1]}"
    # find second occurrence of to_which_it_pairs
    start2 = SEQ.find(compl_seq, start1 + 1)
    if start2 == -1:
        raise ValueError(f"Could not find helix 2 in sequence")
    end2 = start2 + len(compl_seq) -1
    indices2 = list(range(start2, end2+1))
    assert SEQ[indices2[0]:indices2[-1]+1] == compl_seq, f"Expected {compl_seq} but got {SEQ[indices2[0]:indices2[-1]+1]}"
    
    # Create the synthetic experiment
    kwargs = {}
    kwargs['conc_mM'] = 50
    system_name = f'bistable_synthetic_{pop1*100:.0f}' if not same_system else 'bistable_synthetic'
    exp = Experiment(seq=SEQ, temp_C=37, reagent='DMS synthetic', system_name=system_name, **kwargs)
    cov1 = int(coverage * pop1)
    cov2 = int(coverage * (1 - pop1))
    for indices, cov in [(indices1, cov1), (indices2, cov2)]:
        params_dict_ = params_dict.copy()
        if params_dict_['lambda_sc'] is None:
            params_dict_['lambda_sc'] = np.zeros(len(SEQ))
        params_dict_['lambda_sc'][indices] += -2
        exp.df = exp.generate_synthetic_data(params_dict = params_dict_, coverage = cov, noise=noise, eps_b=eps_b)
        if indices == indices1:
            df = exp.df.copy()
        elif indices == indices2:
            df['mut_count'] += exp.df['mut_count']
            df['wt_count'] += exp.df['wt_count']
            df['total_count'] += exp.df['total_count']
            df['mut_rate'] = df['mut_count'] / df['total_count']
    exp.df = df
    exp.raw_df = df.copy()
    return exp