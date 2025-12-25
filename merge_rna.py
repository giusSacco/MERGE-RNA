#!/usr/bin/env python3
"""
MERGE-RNA: User-friendly command-line interface for fitting mutation profiles.

This script provides a simple way to run MERGE-RNA fits on custom data.
For pre-defined systems (cspA, redmond, etc.), use run_multiple_fits.py instead.

Usage examples:
    # Fit a single system from CSV files
    python merge_rna.py fit --csv data_0mM.csv data_20mM.csv data_40mM.csv \
        --system my_rna --temp 37 --output results/

    # Fit using a YAML configuration file
    python merge_rna.py fit --config experiments.yaml --output results/

    # Convert RNAframework .rc files to CSV for inspection
    python merge_rna.py convert --rc-files *.rc --output data/

    # Show info about an experiment file
    python merge_rna.py info --file experiment.csv
"""

import argparse
import os
import sys
import yaml
from pathlib import Path


def load_experiments_from_config(config_path):
    """Load experiments from a YAML configuration file.
    
    Expected YAML format:
    ```yaml
    systems:
      - name: my_rna
        temp_C: 37
        reagent: DMS
        experiments:
          - file: 0mM_rep1.csv
            conc_mM: 0
            rep: 1
          - file: 20mM_rep1.csv
            conc_mM: 20
            rep: 1
    
    # Optional settings
    mask_edges: [25, 25]  # or null for adaptive default
    infer_soft_constraints: true
    ```
    """
    from class_experiment import Experiment
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiments = []
    for system_config in config.get('systems', []):
        system_name = system_config['name']
        temp_C = system_config.get('temp_C', 37)
        # Note: Currently only DMS reagent is supported
        
        for exp_config in system_config['experiments']:
            file_path = exp_config['file']
            conc_mM = exp_config['conc_mM']
            rep = exp_config.get('rep', 1)
            
            # Determine file type and load accordingly
            if file_path.endswith('.csv'):
                exp = Experiment.from_csv(
                    file_path, 
                    system_name=system_name,
                    conc_mM=conc_mM,
                    temp_C=temp_C,
                    rep_number=rep
                )
            elif file_path.endswith('.txt'):
                # Assume it's an info file pointing to .rc data
                exp = Experiment(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}. Use .csv or .txt (info file)")
            
            experiments.append(exp)
    
    return experiments, config


def load_experiments_from_csvs(csv_files, system_name, temp_C):
    """Load experiments from a list of CSV files.
    
    CSV filenames should contain concentration info, e.g.:
    - 0mM_rep1.csv, 20mM_rep1.csv, etc.
    - data_0mM.csv, data_20mM.csv, etc.
    
    If concentration cannot be parsed from filename, user will be prompted.
    
    Note: Currently only DMS reagent is supported.
    """
    from class_experiment import Experiment
    import re
    
    experiments = []
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        
        # Try to extract concentration from filename
        conc_match = re.search(r'(\d+)\s*mM', filename, re.IGNORECASE)
        if conc_match:
            conc_mM = int(conc_match.group(1))
        else:
            # Try to extract any number as concentration
            nums = re.findall(r'\d+', filename)
            if nums:
                print(f"Could not parse concentration from '{filename}'.")
                print(f"Found numbers: {nums}")
                conc_mM = int(input(f"Enter concentration in mM for {filename}: "))
            else:
                conc_mM = int(input(f"Enter concentration in mM for {filename}: "))
        
        # Try to extract replicate number
        rep_match = re.search(r'rep\s*(\d+)|r(\d+)|_(\d+)\.csv$', filename, re.IGNORECASE)
        if rep_match:
            rep = int(next(g for g in rep_match.groups() if g is not None))
        else:
            rep = 1
        
        exp = Experiment.from_csv(
            csv_path,
            system_name=system_name,
            conc_mM=conc_mM,
            temp_C=temp_C,
            rep_number=rep
        )
        experiments.append(exp)
        print(f"  Loaded: {filename} -> {system_name}, {conc_mM} mM, rep {rep}")
    
    return experiments


def cmd_fit(args):
    """Run a fit on custom data."""
    from class_experimentfit import MultiSystemsFit
    
    print("=" * 60)
    print("MERGE-RNA Fitting")
    print("=" * 60)
    
    # Load experiments
    if args.config:
        print(f"\nLoading experiments from config: {args.config}")
        experiments, config = load_experiments_from_config(args.config)
        mask_edges = config.get('mask_edges')
        if mask_edges:
            mask_edges = tuple(mask_edges)
        infer_sc = config.get('infer_soft_constraints', args.infer_soft_constraints)
        # Override fit_mode from config if specified, otherwise use CLI arg
        fit_mode_from_config = config.get('fit_mode')
        if fit_mode_from_config:
            args.fit_mode = fit_mode_from_config
    elif args.csv:
        print(f"\nLoading {len(args.csv)} CSV files for system '{args.system}'")
        experiments = load_experiments_from_csvs(
            args.csv, 
            system_name=args.system,
            temp_C=args.temp
        )
        mask_edges = tuple(args.mask_edges) if args.mask_edges else None
        infer_sc = args.infer_soft_constraints
    else:
        print("Error: Either --config or --csv files must be provided")
        sys.exit(1)
    
    print(f"\nLoaded {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp.system_name}: {exp.conc_mM} mM, {exp.N_seq} nt, rep {exp.rep_number}")
    
    # Check for 0 mM control
    has_zero = any(exp.conc_mM == 0 for exp in experiments)
    if not has_zero:
        print("\n⚠️  WARNING: No 0 mM control experiment found!")
        print("   Background mutation rates (eps_b) will be set to zero.")
        print("   This may affect fit quality. Consider adding untreated control data.")
    
    # Display masking info
    if mask_edges:
        print(f"\nEdge masking: first {mask_edges[0]} and last {mask_edges[1]} nucleotides")
    else:
        print("\nEdge masking: adaptive default (will show warnings for short sequences)")
    
    # Determine fit mode
    fit_mode = args.fit_mode
    if fit_mode == 'sequential' and not infer_sc:
        print("\n⚠️  WARNING: --fit-mode sequential requires soft constraints.")
        print("   Automatically enabling --infer-soft-constraints")
        infer_sc = True
    
    print(f"\nFit mode: {fit_mode}")
    if fit_mode == 'sequential':
        print("   Phase 1: Fit physical parameters (mu_r, p_b, p_bind, m0, m1)")
        print("   Phase 2: Fix physical params, fit soft constraints (lambda_sc)")
    elif fit_mode == 'physical_only':
        print("   Fitting only physical parameters")
    elif fit_mode == 'lambda_only':
        print("   Fitting only soft constraints (lambda_sc)")
    else:
        print("   Fitting all parameters simultaneously")
    
    # Create output directory
    output_dir = args.output or 'fits/custom'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Confirm before starting
    if not args.yes:
        response = input("\nProceed with fit? [y/N]: ")
        if response.lower() not in ('y', 'yes'):
            print("Aborted.")
            sys.exit(0)
    
    # Create and run fit
    print("\n" + "=" * 60)
    print("Starting optimization...")
    print("=" * 60 + "\n")
    
    multi_sys = MultiSystemsFit(
        experiments=experiments,
        validation_exps=None,
        output_suffix=args.suffix or 'custom_fit',
        root_dir=output_dir,
        infer_1D_sc=infer_sc,
        fit_mode=fit_mode,
        mask_edges=mask_edges,
        overwrite=args.overwrite,
        guess=args.initial_guess,
        do_plots=not args.no_plots,
    )
    
    multi_sys.fit()
    
    print("\n" + "=" * 60)
    print("Fit completed!")
    print(f"Results saved to: {multi_sys.output_dir}")
    print("=" * 60)


def cmd_convert(args):
    """Convert RNAframework .rc files to CSV."""
    from class_experiment import rc_to_csv
    
    print("Converting .rc files to CSV...")
    
    output_dir = args.output or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    for rc_file in args.rc_files:
        output_path = os.path.join(output_dir, os.path.basename(rc_file).replace('.rc', '.csv'))
        rc_to_csv(rc_file, output_path)
    
    print(f"\nConverted {len(args.rc_files)} files to {output_dir}")


def cmd_info(args):
    """Show information about an experiment file."""
    from class_experiment import Experiment, load_csv, load_rc
    
    file_path = args.file
    
    print(f"\nFile: {file_path}")
    print("=" * 60)
    
    if file_path.endswith('.csv'):
        df = load_csv(file_path)
        print(f"Format: CSV")
    elif file_path.endswith('.rc'):
        df = load_rc(file_path, os.path.basename(file_path))
        print(f"Format: RNAframework binary (.rc)")
    elif file_path.endswith('.txt'):
        exp = Experiment(file_path)
        df = exp.df
        print(f"Format: Experiment info file (.txt)")
        print(f"System: {exp.system_name}")
        print(f"Concentration: {exp.conc_mM} mM")
        print(f"Temperature: {exp.temp_C}°C")
        print(f"Reagent: {exp.reagent}")
    else:
        print(f"Unknown file format: {file_path}")
        sys.exit(1)
    
    print(f"\nSequence length: {len(df)} nt")
    seq = ''.join(df['ref_nt'])
    print(f"Sequence (first 50): {seq[:50]}...")
    print(f"\nCoverage: min={df['total_count'].min()}, max={df['total_count'].max()}, mean={df['total_count'].mean():.0f}")
    print(f"Mutation rate: min={df['mut_rate'].min():.4f}, max={df['mut_rate'].max():.4f}, mean={df['mut_rate'].mean():.4f}")
    
    print("\nFirst 10 positions:")
    print(df.head(10).to_string())


def cmd_create_config(args):
    """Create a template YAML configuration file."""
    template = """# MERGE-RNA experiment configuration
# Edit this file to match your data
# Note: Currently only DMS reagent is supported

systems:
  - name: my_rna_system          # Name for this RNA (used for grouping)
    temp_C: 37                   # Temperature in Celsius
    experiments:
      # List all experiments for this system
      # Each should have: file, conc_mM, and optionally rep
      - file: data/0mM_rep1.csv
        conc_mM: 0               # Untreated control - IMPORTANT!
        rep: 1
      - file: data/0mM_rep2.csv
        conc_mM: 0
        rep: 2
      - file: data/20mM_rep1.csv
        conc_mM: 20
        rep: 1
      - file: data/20mM_rep2.csv
        conc_mM: 20
        rep: 2
      - file: data/40mM_rep1.csv
        conc_mM: 40
        rep: 1

# Optional: Add more systems for joint fitting
#  - name: another_rna
#    temp_C: 25
#    experiments:
#      - file: ...

# Masking settings (optional)
# mask_edges: [25, 25]          # Mask first/last N nucleotides
                                 # Set to null or omit for adaptive default

# Fitting settings
infer_soft_constraints: true     # Infer position-specific corrections
fit_mode: sequential             # Options: all, physical_only, lambda_only, sequential (recommended)
"""
    
    output_path = args.output or 'experiments.yaml'
    
    if os.path.exists(output_path) and not args.force:
        print(f"File {output_path} already exists. Use --force to overwrite.")
        sys.exit(1)
    
    with open(output_path, 'w') as f:
        f.write(template)
    
    print(f"Created template configuration: {output_path}")
    print("\nEdit this file to match your data, then run:")
    print(f"  python merge_rna.py fit --config {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MERGE-RNA: Fit chemical probing mutation profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # fit command
    fit_parser = subparsers.add_parser('fit', help='Run a fit on custom data')
    fit_input = fit_parser.add_mutually_exclusive_group()
    fit_input.add_argument('--config', type=str, help='Path to YAML configuration file')
    fit_input.add_argument('--csv', nargs='+', help='CSV files to fit (specify concentration in filename or be prompted)')
    fit_parser.add_argument('--system', type=str, default='custom', help='System name (default: custom)')
    fit_parser.add_argument('--temp', type=int, default=37, help='Temperature in Celsius (default: 37)')
    # Note: Currently only DMS reagent is supported
    fit_parser.add_argument('--output', '-o', type=str, help='Output directory (default: fits/custom)')
    fit_parser.add_argument('--suffix', type=str, help='Output subfolder name')
    fit_parser.add_argument('--infer-soft-constraints', '-sc', action='store_true', help='Infer position-specific soft constraints')
    fit_parser.add_argument('--fit-mode', type=str, default='sequential', 
                           choices=['simultaneous', 'physical_only', 'lambda_only', 'sequential'],
                           help='Fitting strategy: simultaneous, physical_only, lambda_only, or sequential (default, recommended)')
    fit_parser.add_argument('--mask-edges', nargs=2, type=int, metavar=('START', 'END'), help='Mask first START and last END nucleotides')
    fit_parser.add_argument('--initial-guess', type=str, help='Initial guess: None, random, last, or path to params file')
    fit_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output')
    fit_parser.add_argument('--no-plots', action='store_true', help='Disable diagnostic plots')
    fit_parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation prompt')
    
    # convert command
    convert_parser = subparsers.add_parser('convert', help='Convert .rc files to CSV')
    convert_parser.add_argument('--rc-files', nargs='+', required=True, help='RNAframework .rc files to convert')
    convert_parser.add_argument('--output', '-o', type=str, help='Output directory')
    
    # info command
    info_parser = subparsers.add_parser('info', help='Show information about an experiment file')
    info_parser.add_argument('--file', '-f', type=str, required=True, help='Path to experiment file (.csv, .rc, or .txt)')
    
    # create-config command
    config_parser = subparsers.add_parser('create-config', help='Create a template YAML configuration file')
    config_parser.add_argument('--output', '-o', type=str, help='Output path (default: experiments.yaml)')
    config_parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing file')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Dispatch to command handler
    commands = {
        'fit': cmd_fit,
        'convert': cmd_convert,
        'info': cmd_info,
        'create-config': cmd_create_config,
    }
    
    commands[args.command](args)


if __name__ == '__main__':
    main()
