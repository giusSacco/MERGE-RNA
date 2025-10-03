#%%
#interactive_arcplots.py
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
SAVE_TO_OVERLEAF = False
OTHERFIGS_FOLDER_PATH = "otherfigs"
OVERLEAF_FOLDER_PATH = None
SAVE_LOCAL_PLOTS = False  # Save static PNGs locally
# Save interactive arc-plot as standalone HTML
SAVE_INTERACTIVE_HTML = True
INTERACTIVE_HTML_OUTPUT = os.path.join(OTHERFIGS_FOLDER_PATH, "arcplot_explorer.html")
INTERACTIVE_HTML_DEFAULT_SYSTEM: Optional[str] = None  # e.g., "hc16" or None to auto-pick
INTERACTIVE_HTML_TOP_SOURCE_DEFAULT = "vienna"
INTERACTIVE_HTML_BOTTOM_SOURCE_DEFAULT = "model"
INTERACTIVE_HTML_TOP_OVERLAY_DEFAULT = "none"
INTERACTIVE_HTML_BOTTOM_OVERLAY_DEFAULT = "none"
# Import required libraries
from tqdm import tqdm
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import RNA

# Add RNA structure analysis path
from class_experiment import clocked, Experiment, initialise_combined_cspA_exp
from class_experimentfit import ExperimentFit, MultiSystemsFit
# Configuration
RNA_STRUCT_HOME = '.'
FOLDER = 'fits/red_crossval_1'  # folder where the fits are stored, change as needed
systems = ['hc16', 'bact_RNaseP_typeA', 'tetrahymena_ribozyme', 'HCV_IRES', 'V_chol_gly_riboswitch']
model_fit_params_folders = [f'fits/red_sep_fix_{i}' for i in range(1,4)]
optimal_lambda_magnitude_per_system = {}
physical_params_dict = {}
for (sys1, sys2, sys3) in tqdm(combinations(systems, 3), desc="Loading parameters"):
    subfolder = f'red_crossval_{sys1}_{sys2}_{sys3}'
    subfolder_fullpath = os.path.join(RNA_STRUCT_HOME, FOLDER, subfolder)
    params_file = os.path.join(subfolder_fullpath, 'params1D.txt')
    params_1D = np.loadtxt(params_file)
    physical_params_dict[(sys1, sys2, sys3)] = params_1D

#%%
# Define cache directory and control flag
CACHE_DIR = os.path.join(RNA_STRUCT_HOME, 'cache', 'interactive_arcplots')
force_recompute = False  # Set to True to force recomputation
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

for system in systems:
    if system == 'bact_RNaseP_typeA':
        print(f"Skipping {system} - no PDB data available")
        continue
    
    # Define cache file paths for this system
    cache_file = os.path.join(CACHE_DIR, f"{system}_results.npz")
    
    cached_data = None
    if os.path.exists(cache_file) and not force_recompute:
        tmp_cached_data = np.load(cache_file, allow_pickle=True)
        required_fields = {
            'model_fit_losses',
            'lambdas_magnitude',
            'all_losses',
            'optimal_lambda_magnitude',
            'model_bpps',
            'lambda_bpps',
        }
        if required_fields.issubset(set(tmp_cached_data.files)):
            print(f"Loading cached results for {system}...")
            cached_data = tmp_cached_data
        else:
            print(f"Cached results for {system} missing BPP data. Recomputing...")
            tmp_cached_data.close()

    if cached_data is not None:
        model_fit_losses = cached_data['model_fit_losses']
        lambdas_magnitude = cached_data['lambdas_magnitude']
        all_losses = cached_data['all_losses'].item()  # Convert back from numpy object array
        optimal_lambda_magnitude = cached_data['optimal_lambda_magnitude']
        model_bpps = [arr for arr in np.atleast_1d(cached_data['model_bpps'])]
        lambda_bpps = cached_data['lambda_bpps'].item()
        optimal_lambda_magnitude_per_system[system] = optimal_lambda_magnitude
        cached_data.close()
    else:
        print(f"Computing soft constraints analysis for {system}...")
        
        # Initialize experiments for this system
        exps = [Experiment(path) for path in Experiment.paths_to_redmond_ivt_data_txt 
                if Experiment(path).system == system]
        exps = [exp for exp in exps if exp.conc_mM != 85]  # remove 85mM data
        multi_exp = MultiSystemsFit(exps, infer_1D_sc=True)
        # loss from model fit
        exp = exps[0]
        exp_fit_zero = next(
            exp_fit_
            for exp_fit_ in multi_exp.systems[0].exp_fits_all
            if exp_fit_.conc_mM == 0 and exp_fit_.system == system
        )
        
        # Get model fit losses
        model_fit_losses = []
        model_bpps = []
        for folder in model_fit_params_folders:
            params_file = os.path.join(RNA_STRUCT_HOME, folder, f'{system}_with_lambdas','params1D.txt')
            params_1D = np.loadtxt(params_file)
            loss, _ = multi_exp.multisys_loss_and_grad(params_1D=params_1D, compute_gradient=False)
            # normalize
            loss = loss / (exp.N_seq - 50)
            loss /= 5*2
            # compute and store bpps
            lambda_sc = params_1D[multi_exp.lambdas_indices[system]]
            fc = RNA.fold_compound(exp.seq)
            RNA.cvar.temperature = exp_fit_zero.temp_C
            exp_fit_zero.apply_soft_constraints(lambda_sc, fc)
            fc.pf()
            bpp = np.array(fc.bpp())[1:,1:]   # base pair probability matrix. Ask Giovanni for the [1:,1:]
            model_bpps.append(bpp)
            model_fit_losses.append(loss)
        
        # Get sequence and PDB secondary structure
        exp.add_pdb_ss_to_df(exp.raw_df)
        ss_pdb = ''.join(exp.df['pdb_ss'])
        
        # Test different constraint magnitudes
        lambdas_magnitude = (np.linspace(0, np.sqrt(2), 15))**2  # quadratic spacing
        
        # Store all losses for plotting
        all_losses = {}
        lambda_bpps = {}
        
        for set_i_params, (systems_train, physical_params) in tqdm(enumerate(physical_params_dict.items()), 
                                                                  desc="Processing parameter sets", 
                                                                  total=len(physical_params_dict)):
            is_training = system in systems_train
            
            # Test both PDB and MFE structure constraints
            for structure in ['pdb1D', 'mfe1D']:
                # Define reference structure
                if structure == 'pdb1D':
                    ref_struct = ss_pdb
                elif structure == 'mfe1D':
                    fc = RNA.fold_compound(exp.seq)
                    RNA.cvar.temperature = exp_fit_zero.temp_C
                    ref_struct = fc.mfe()[0]
                
                # Convert structure to constraint vector
                is_paired = np.array([-1 if nb in '()' else 1 if nb == '.' else 0 for nb in ref_struct])
                
                losses = []
                bpps_for_lambdas = []
                for lambda_magnitude in lambdas_magnitude:
                    lambda_sc = is_paired * lambda_magnitude
                    params_1D = np.append(physical_params.copy(), lambda_sc)
                    loss, _ = multi_exp.multisys_loss_and_grad(params_1D=params_1D, compute_gradient=False)
                    # normalize
                    loss = loss / (exp.N_seq - 50)
                    loss /= 5*2
                    losses.append(loss)
                    
                    fc_sc = RNA.fold_compound(exp.seq)
                    RNA.cvar.temperature = exp_fit_zero.temp_C
                    exp_fit_zero.apply_soft_constraints(lambda_sc, fc_sc)
                    fc_sc.pf()
                    bpps_for_lambdas.append(np.array(fc_sc.bpp())[1:, 1:])
                
                # Store results for plotting
                key = f"{structure}_{set_i_params}_{is_training}"
                all_losses[key] = losses
                lambda_bpps[key] = bpps_for_lambdas
                
                # Find optimal lambda for PDB structure in first parameter set
                if structure == 'pdb1D' and set_i_params == 0:
                    optimal_index = np.argmin(losses)
                    optimal_lambda_magnitude = lambdas_magnitude[optimal_index]
                    optimal_lambda_magnitude_per_system[system] = optimal_lambda_magnitude
            
            #if set_i_params > 0:  # Process only the first parameter set to save time
                #break
        
        # Save results to cache
        np.savez(cache_file, 
                 model_fit_losses=model_fit_losses,
                 lambdas_magnitude=lambdas_magnitude,
                 all_losses=all_losses,
                 optimal_lambda_magnitude=optimal_lambda_magnitude,
                 model_bpps=np.array(model_bpps, dtype=object),
                 lambda_bpps=np.array([lambda_bpps], dtype=object))
    
    # Generate the plot
    print(f"Generating plot for {system}...")
    plt.figure(figsize=(10, 8))
    
    # Plot model fit losses
    for i, loss in enumerate(model_fit_losses):
        plt.axhline(y=loss, color='green', linestyle='-.', lw=2, 
                    label=f'Model fit {model_fit_params_folders[i].split("_")[-1]} set')
    
    # Plot losses for different constraint magnitudes
    if 'all_losses' in locals():
        for key, losses in all_losses.items():
            structure, set_i, is_training = key.split('_')
            set_i_params = int(set_i)
            marker = markers[set_i_params % len(markers)]
            is_training = is_training.lower() == 'true'
            color = 'red' if structure == 'pdb1D' else 'blue'
            ls = '--' if is_training else '-'
            
            plt.plot(lambdas_magnitude, losses, color=color, linestyle=ls, 
                    marker=marker, markersize=6, alpha=0.7, 
                    label=f'{structure}_set{set_i_params}' if set_i_params < 3 else "")
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='PDB structure'),
        Line2D([0], [0], color='blue', lw=2, label='MFE structure'),
        Line2D([0], [0], color='black', linestyle='-', lw=2, label='Validation set'),
        Line2D([0], [0], color='black', linestyle='--', lw=2, label='Training set'),
        Line2D([0], [0], color='black', marker='o', linestyle='None', 
               markersize=5, label='Different parameter sets'),
        Line2D([0], [0], color='green', linestyle='-.', lw=2, label='model fit')
    ]
    
    plt.xlabel('Soft Constraint Magnitude')
    plt.ylabel('Loss')
    plt.title(f'Loss for {system.replace("_", " ")} with 1D soft constraints')
    plt.legend(handles=legend_elements, loc='best')
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Save to Overleaf if enabled
    if SAVE_TO_OVERLEAF and OVERLEAF_FOLDER_PATH:
        os.makedirs(OVERLEAF_FOLDER_PATH, exist_ok=True)
        plt.savefig(f"{OVERLEAF_FOLDER_PATH}/fig2b_{system}.pdf", dpi=300, bbox_inches='tight')
    # Save locally to OTHERFIGS_FOLDER_PATH if enabled
    if SAVE_LOCAL_PLOTS:
        os.makedirs(OTHERFIGS_FOLDER_PATH, exist_ok=True)
        plt.savefig(f"{OTHERFIGS_FOLDER_PATH}/fig2b_{system}.png", dpi=300, bbox_inches='tight')
    plt.show()
# %%


#%%
# Interactive arc plot explorer powered by precomputed BPPs
try:
    import ipywidgets as widgets
    from IPython.display import display
    from ipywidgets.embed import embed_minimal_html
    from matplotlib.patches import Arc
    from matplotlib.lines import Line2D
except ImportError:  # pragma: no cover - widgets not installed
    widgets = None
    Arc = None
    Line2D = None
    embed_minimal_html = None

# Optional Plotly backend for interactive, zoomable arcs
try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None

from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_SOURCE_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("Model", "model"),
    ("Vienna", "vienna"),
    ("Reference-informed", "reference_informed"),
    ("Reference", "reference"),
)

_LABEL_BY_KEY: Dict[str, str] = {value: label for (label, value) in _SOURCE_OPTIONS}

_SOURCE_COLORS: Dict[str, str] = {
    "model": "#1f77b4",  # blue
    "vienna": "#2ca02c",  # green
    "reference_informed": "#d62728",  # red
    "reference": "#ff7f0e",  # orange
}

_LAMBDA_SOURCES = {"vienna", "reference_informed"}

_ARC_THRESHOLD = 0.05
_ARC_HEIGHT_RATIO = 1.0
_ARC_MAX_LINEWIDTH = 3.0
_OVERLAY_ALPHA = 0.4
_OVERLAY_LINESTYLE = "--"
_ARC_POINTS = 80  # polyline resolution for Plotly arcs (higher = smoother)

_EMPTY_SEGMENTS = {
    "x": [None],
    "y": [None],
    "text": [None],
    "line": {"color": "rgba(0,0,0,0)", "width": 1.0, "dash": "solid"},
}


def _discover_cached_systems() -> List[str]:
    """Return systems with available precomputed BPP caches."""
    if not os.path.isdir(CACHE_DIR):
        return []
    systems_with_cache = []
    for filename in os.listdir(CACHE_DIR):
        if filename.endswith("_results.npz"):
            systems_with_cache.append(filename.replace("_results.npz", ""))
    return sorted(systems_with_cache)


def _pick_lambda_series(lambda_bpps: Dict[str, Iterable[np.ndarray]], prefix: str) -> Tuple[Optional[str], List[np.ndarray]]:
    """Select a representative lambda->BPP series for a prefix (pdb1D/mfe1D)."""
    candidates = [key for key in lambda_bpps.keys() if key.startswith(prefix)]
    if not candidates:
        return None, []

    def _sort_key(entry: str) -> Tuple[int, int]:
        parts = entry.split("_")
        set_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 999
        is_training = 0 if len(parts) > 2 and parts[2].lower() == "true" else 1
        return (is_training, set_idx)

    selected_key = sorted(candidates, key=_sort_key)[0]
    series = lambda_bpps[selected_key]
    # Ensure we work with a plain Python list of ndarrays
    return selected_key, [np.array(mat) for mat in np.atleast_1d(series)]


@lru_cache(maxsize=None)
def _load_sequence_for_system(system: str) -> str:
    """Grab the first available sequence for a system (excluding 85 mM)."""
    for path in Experiment.paths_to_redmond_ivt_data_txt:
        exp = Experiment(path)
        if exp.system == system and exp.conc_mM != 85:
            return exp.seq
    raise ValueError(f"No experiment sequence found for system '{system}'.")


@lru_cache(maxsize=None)
def _load_reference_structure(system: str) -> str:
    """Load the reference dot-bracket structure used for soft constraints."""
    for path in Experiment.paths_to_redmond_ivt_data_txt:
        exp = Experiment(path)
        if exp.system == system and exp.conc_mM != 85:
            exp.add_pdb_ss_to_df(exp.raw_df)
            if hasattr(exp, "df") and "pdb_ss" in exp.df:
                return "".join(exp.df["pdb_ss"])
    raise ValueError(f"Reference structure unavailable for system '{system}'.")


def _dotbracket_to_pairs(structure: str) -> List[Tuple[int, int]]:
    """Convert a dot-bracket structure into paired index tuples."""
    openers = {
        "(": ")",
        "[": "]",
        "{": "}",
        "<": ">",
    }
    closers = {v: k for k, v in openers.items()}
    stack: Dict[str, List[int]] = {ch: [] for ch in openers.keys()}
    pairs: List[Tuple[int, int]] = []
    for idx, ch in enumerate(structure):
        if ch in openers:
            stack[ch].append(idx)
        elif ch in closers:
            opener = closers[ch]
            if not stack[opener]:
                continue
            start = stack[opener].pop()
            pairs.append((start, idx))
    return pairs


def _structure_to_matrix(length: int, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    matrix = np.zeros((length, length))
    for i, j in pairs:
        if 0 <= i < length and 0 <= j < length:
            matrix[i, j] = 1.0
            matrix[j, i] = 1.0
    return matrix


@lru_cache(maxsize=None)
def _load_precomputed_bundle(system: str) -> Dict[str, np.ndarray]:
    """Load precomputed BPP arrays and metadata for a system."""
    cache_file = os.path.join(CACHE_DIR, f"{system}_results.npz")
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"Expected cache file '{cache_file}' not found. "
            "Run the preprocessing step to generate BPP caches."
        )

    with np.load(cache_file, allow_pickle=True) as cached_data:
        lambdas = np.array(cached_data["lambdas_magnitude"], dtype=float)
        lambda_bpps_raw = cached_data["lambda_bpps"].item()
        model_bpps_raw = [np.array(arr) for arr in np.atleast_1d(cached_data["model_bpps"])]
        optimal_lambda = float(cached_data["optimal_lambda_magnitude"]) if "optimal_lambda_magnitude" in cached_data else None

    vienna_key, vienna_series = _pick_lambda_series(lambda_bpps_raw, "mfe1D")
    reference_informed_key, reference_informed_series = _pick_lambda_series(lambda_bpps_raw, "pdb1D")

    sequence = _load_sequence_for_system(system)
    reference_structure = _load_reference_structure(system)
    reference_pairs = _dotbracket_to_pairs(reference_structure)
    reference_matrix = _structure_to_matrix(len(sequence), reference_pairs)

    bundle: Dict[str, object] = {
        "sequence": sequence,
        "lambdas": lambdas,
        "optimal_lambda": optimal_lambda,
        "model_bpp": model_bpps_raw[0] if model_bpps_raw else None,
        "vienna_bpps": vienna_series,
        "reference_informed_bpps": reference_informed_series,
        "reference_key": reference_informed_key,
        "reference_structure": reference_structure,
        "reference_matrix": reference_matrix,
        "reference_pairs": reference_pairs,
        "vienna_key": vienna_key,
        "reference_informed_key": reference_informed_key,
    }

    if bundle["model_bpp"] is None:
        raise ValueError(f"Model BPP data missing for system '{system}'.")
    if not vienna_series:
        raise ValueError(f"Vienna lambda-series missing in cache for system '{system}'.")
    if not reference_informed_series:
        raise ValueError(f"Reference-informed lambda-series missing in cache for system '{system}'.")

    if len(reference_structure) != len(sequence):
        raise ValueError(
            f"Reference structure length ({len(reference_structure)}) does not match sequence length ({len(sequence)})."
        )

    return bundle


def _closest_lambda_value(lambdas: Sequence[float], target: Optional[float]) -> float:
    values = np.asarray(lambdas, dtype=float)
    if values.size == 0:
        raise ValueError("No lambda magnitudes available.")
    if target is None:
        return float(values[0])
    idx = int(np.argmin(np.abs(values - target)))
    return float(values[idx])


def _lambda_index(lambdas: Sequence[float], value: float) -> int:
    lambdas = np.asarray(lambdas, dtype=float)
    return int(np.argmin(np.abs(lambdas - value)))


def _lambda_step(lambdas: Sequence[float]) -> float:
    lambdas_arr = np.asarray(sorted(set(np.asarray(lambdas, dtype=float))))
    if lambdas_arr.size <= 1:
        return 0.01
    diffs = np.diff(lambdas_arr)
    positive_diffs = diffs[diffs > 0]
    return float(np.min(positive_diffs)) if positive_diffs.size else 0.01


def _plot_half_arcs(
    ax: plt.Axes,
    matrix: np.ndarray,
    n: int,
    *,
    top: bool,
    color: str,
    threshold: float = _ARC_THRESHOLD,
    linestyle: str = "-",
    alpha: float = 0.85,
) -> None:
    if Arc is None:
        raise RuntimeError("matplotlib Arc patch unavailable; ensure matplotlib is installed.")

    matrix = np.asarray(matrix)
    if matrix.shape[0] != n:
        raise ValueError("BPP matrix size does not match sequence length.")

    theta1, theta2 = (0, 180) if top else (180, 360)
    for i in range(n - 1):
        for j in range(i + 1, n):
            prob = float(matrix[i, j])
            if np.isnan(prob) or prob <= threshold:
                continue
            width = j - i
            if width <= 0:
                continue
            height = width * _ARC_HEIGHT_RATIO
            linewidth = min(prob, 1.0) * _ARC_MAX_LINEWIDTH
            arc = Arc(
                xy=((i + j) / 2.0, 0.0),
                width=width,
                height=height,
                angle=0.0,
                theta1=theta1,
                theta2=theta2,
                edgecolor=color,
                linewidth=linewidth,
                fill=False,
                alpha=alpha,
                linestyle=linestyle,
            )
            ax.add_patch(arc)


def _render_arcplot(
    sequence: str,
    layers: Sequence[Dict[str, Any]],
    top_text: str,
    bottom_text: str,
    top_color: str,
    bottom_color: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    n = len(sequence)
    ax.plot([0, n - 1], [0, 0], color="#333333", linewidth=1.2)

    legend_handles = []
    seen_legends = set()

    for layer in layers:
        _plot_half_arcs(
            ax,
            layer["matrix"],
            n,
            top=layer["top"],
            color=layer["color"],
            threshold=layer.get("threshold", _ARC_THRESHOLD),
            linestyle=layer.get("linestyle", "-"),
            alpha=layer.get("alpha", 0.85),
        )
        legend_key = layer.get("legend_key")
        if Line2D is not None and legend_key and legend_key not in seen_legends:
            seen_legends.add(legend_key)
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=_SOURCE_COLORS[legend_key],
                    lw=2,
                    linestyle=layer.get("legend_linestyle", "-"),
                    label=_LABEL_BY_KEY.get(legend_key, legend_key.replace("_", " ").title()),
                )
            )

    ax.set_xlim(-2, n + 1)
    radius = n / 2 + 5
    ax.set_ylim(-radius, radius)
    ax.set_yticks([])
    ax.set_xlabel("Position (nt)")
    ax.set_title("Interactive arc plot (precomputed BPPs)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    ax.text(0.01, 0.95, top_text, transform=ax.transAxes, fontsize=11, color=top_color, va="top")
    ax.text(0.01, 0.05, bottom_text, transform=ax.transAxes, fontsize=11, color=bottom_color, va="bottom")

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    return fig


def _make_arc_polyline(i: int, j: int, top: bool, n: int) -> Tuple[np.ndarray, np.ndarray]:
    center_x = (i + j) / 2.0
    width = j - i
    height = width * _ARC_HEIGHT_RATIO
    # parameterize a half-ellipse
    t = np.linspace(0, np.pi, _ARC_POINTS)
    if not top:
        t = t + np.pi
    x = center_x + (width / 2.0) * np.cos(t)
    y = (height / 2.0) * np.sin(t)
    return x, y


def _render_arcplot_plotly(
    sequence: str,
    layers: Sequence[Dict[str, Any]],
    top_text: str,
    bottom_text: str,
    top_color: str,
    bottom_color: str,
) -> Any:
    if go is None:
        raise ImportError("Plotly is required for zoomable HTML. Install plotly.")

    n = len(sequence)
    fig = go.Figure()

    # baseline
    fig.add_trace(
        go.Scatter(
            x=[0, n - 1],
            y=[0, 0],
            mode="lines",
            line=dict(color="#333", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # draw layers
    for layer in layers:
        matrix = np.asarray(layer["matrix"])
        threshold = layer.get("threshold", _ARC_THRESHOLD)
        top = bool(layer["top"])
        color = layer["color"]
        alpha = layer.get("alpha", 0.85)
        # encode alpha in rgba
        if color.startswith("#") and len(color) == 7:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            color_rgba = f"rgba({r},{g},{b},{alpha})"
        else:
            color_rgba = color

        for i in range(n - 1):
            for j in range(i + 1, n):
                prob = float(matrix[i, j])
                if np.isnan(prob) or prob <= threshold:
                    continue
                x, y = _make_arc_polyline(i, j, top=top, n=n)
                width = max(1.0, prob * _ARC_MAX_LINEWIDTH * 1.2)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(color=color_rgba, width=width, dash="dash" if layer.get("linestyle") == _OVERLAY_LINESTYLE else "solid"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    radius = n / 2 + 5
    fig.update_layout(
        width=1400,
        height=800,
        margin=dict(l=40, r=20, t=60, b=40),
        title="Interactive arc plot (precomputed BPPs)",
        xaxis=dict(range=[-2, n + 1], showgrid=True, zeroline=False),
        yaxis=dict(range=[-radius, radius], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        dragmode="pan",
    )
    # captions
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.97, showarrow=False, text=top_text, font=dict(color=top_color))
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.03, showarrow=False, text=bottom_text, font=dict(color=bottom_color))
    return fig


def _color_with_alpha(hex_color: str, alpha: float) -> str:
    if hex_color and hex_color.startswith("#") and len(hex_color) == 7:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color


def _matrix_to_plotly_segments(
    matrix: np.ndarray,
    *,
    top: bool,
    color: str,
    threshold: float,
    linestyle: str = "solid",
    alpha: float = 0.9,
) -> Dict[str, Any]:
    n = matrix.shape[0]
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    text: List[Optional[str]] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            prob = float(matrix[i, j])
            if np.isnan(prob) or prob <= threshold:
                continue
            arc_x, arc_y = _make_arc_polyline(i, j, top=top, n=n)
            hover = f"{i + 1}-{j + 1} (p={prob:.3f})"
            xs.extend(arc_x.tolist())
            xs.append(None)
            ys.extend(arc_y.tolist())
            ys.append(None)
            text.extend([hover] * len(arc_x))
            text.append(None)

    if not xs:
        xs = [None]
        ys = [None]
        text = [None]

    rgba = _color_with_alpha(color, alpha)
    return {
        "x": xs,
        "y": ys,
        "text": text,
        "line": {
            "color": rgba,
            "width": 2.2,
            "dash": "dash" if linestyle == _OVERLAY_LINESTYLE else "solid",
        },
    }


def _export_segments_for_source(
        bundle: Dict[str, Any],
        source: str,
        lambda_values: Sequence[float],
) -> Dict[str, Any]:
        segments_top: List[Dict[str, Any]] = []
        segments_bottom: List[Dict[str, Any]] = []
        overlay_top: List[Dict[str, Any]] = []
        overlay_bottom: List[Dict[str, Any]] = []
        lambda_display: List[Optional[float]] = []

        for lambda_value in lambda_values:
                matrix, lam_val = _matrix_from_source(bundle, source, float(lambda_value))
                segments_top.append(
                        _matrix_to_plotly_segments(
                                matrix,
                                top=True,
                                color=_SOURCE_COLORS[source],
                                threshold=_ARC_THRESHOLD,
                                alpha=0.95,
                        )
                )
                segments_bottom.append(
                        _matrix_to_plotly_segments(
                                matrix,
                                top=False,
                                color=_SOURCE_COLORS[source],
                                threshold=_ARC_THRESHOLD,
                                alpha=0.95,
                        )
                )

                overlay_threshold = 0.0 if source == "reference" else _ARC_THRESHOLD
                overlay_top.append(
                        _matrix_to_plotly_segments(
                                matrix,
                                top=True,
                                color=_SOURCE_COLORS[source],
                                threshold=overlay_threshold,
                                linestyle=_OVERLAY_LINESTYLE,
                                alpha=_OVERLAY_ALPHA,
                        )
                )
                overlay_bottom.append(
                        _matrix_to_plotly_segments(
                                matrix,
                                top=False,
                                color=_SOURCE_COLORS[source],
                                threshold=overlay_threshold,
                                linestyle=_OVERLAY_LINESTYLE,
                                alpha=_OVERLAY_ALPHA,
                        )
                )

                lambda_display.append(float(lam_val) if lam_val is not None else None)

        return {
                "label": _LABEL_BY_KEY[source],
                "color": _SOURCE_COLORS[source],
                "requiresLambda": source in _LAMBDA_SOURCES and len(lambda_values) > 1,
                "lambdaDisplay": lambda_display,
                "primary": {"top": segments_top, "bottom": segments_bottom},
                "overlay": {"top": overlay_top, "bottom": overlay_bottom},
        }


def _prepare_system_payload(system: str) -> Dict[str, Any]:
        bundle = _load_precomputed_bundle(system)
        lambda_array = np.asarray(bundle["lambdas"], dtype=float)
        if lambda_array.size:
                lambda_values = [float(value) for value in lambda_array.tolist()]
        else:
                lambda_values = [0.0]

        try:
                default_lambda_value = _closest_lambda_value(lambda_values, bundle.get("optimal_lambda"))
        except (ValueError, TypeError):
                default_lambda_value = lambda_values[0]

        default_lambda_index = _lambda_index(lambda_values, default_lambda_value) if lambda_values else 0

        sources_payload = {
                source: _export_segments_for_source(bundle, source, lambda_values)
                for (_, source) in _SOURCE_OPTIONS
        }

        sequence = bundle.get("sequence", "")

        return {
                "sequence": sequence,
                "sequenceLength": len(sequence),
                "lambdaValues": lambda_values,
                "lambdaLabels": [f"{value:.4f}" for value in lambda_values],
                "defaultLambdaIndex": default_lambda_index,
                "hasLambdaSlider": len(lambda_values) > 1,
                "sources": sources_payload,
        }


def _render_arcplot_html(payload: Dict[str, Any]) -> str:
        """Render the standalone HTML file for the arc plot explorer."""
        json_payload = json.dumps(payload, separators=(",", ":"))
        # Basic XSS mitigation
        json_payload = json_payload.replace("<", "\\u003c").replace(">", "\\u003e")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{payload.get("title", "RNA Arc Plot Explorer")}</title>
    <style>
        :root {{
            color-scheme: light;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{ margin: 0; padding: 0; background: #f8f9fa; color: #212529; }}
        #app {{ max-width: 1400px; margin: 0 auto; padding: 1.5rem; }}
        h1 {{ font-size: 2rem; margin: 0 0 0.5rem; font-weight: 600; }}
        .subtitle {{ margin: 0 0 1.5rem; color: #6c757d; }}
        .controls {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.2rem;
            background: #fff;
            padding: 1.2rem;
            border-radius: 0.75rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid #dee2e6;
        }}
        .controls label {{
            display: flex;
            flex-direction: column;
            font-size: 0.9rem;
            font-weight: 500;
        }}
        .controls select, .controls input[type=range] {{
            margin-top: 0.5rem;
            padding: 0.5rem 0.75rem;
            border-radius: 0.375rem;
            border: 1px solid #ced4da;
            background: #fff;
            font-size: 1rem;
        }}
        .lambda-readout {{ font-size: 0.85rem; margin-top: 0.4rem; color: #495057; }}
        #arcplot {{ margin-top: 2rem; background: #fff; border-radius: 0.75rem; box-shadow: 0 8px 15px rgba(0, 0, 0, 0.07); }}
        .tips {{ margin-top: 1.5rem; font-size: 0.9rem; color: #495057; }}
    </style>
</head>
<body>
    <div id="app">
        <h1>{payload.get("title", "RNA Arc Plot Explorer")}</h1>
        <p class="subtitle">Explore precomputed RNA base-pair probabilities. Pan or zoom, and hover over arcs for details.</p>
        <div class="controls">
            <label>System <select id="system-select"></select></label>
            <label>Top Source <select id="top-source"></select></label>
            <label>Bottom Source <select id="bottom-source"></select></label>
            <label>Top Overlay <select id="top-overlay"></select></label>
            <label>Bottom Overlay <select id="bottom-overlay"></select></label>
            <label id="lambda-control" style="display: none;">
                λ Magnitude
                <input type="range" id="lambda-slider" min="0" max="0" step="1" value="0" />
                <div class="lambda-readout" id="lambda-readout">λ ≈ 0.0000</div>
            </label>
        </div>
        <div id="arcplot"></div>
        <div class="tips">Tip: Drag to pan, scroll to zoom, and double-click to reset.</div>
    </div>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script>
        (function() {{
            const payload = {json_payload};
            const dom = {{
                system: document.getElementById('system-select'),
                topSource: document.getElementById('top-source'),
                bottomSource: document.getElementById('bottom-source'),
                topOverlay: document.getElementById('top-overlay'),
                bottomOverlay: document.getElementById('bottom-overlay'),
                lambdaControl: document.getElementById('lambda-control'),
                lambdaSlider: document.getElementById('lambda-slider'),
                lambdaReadout: document.getElementById('lambda-readout'),
                plot: document.getElementById('arcplot'),
            }};

            const state = {{
                system: payload.initialSystem,
                lambdaIndex: 0,
                topSource: payload.defaults.topSource,
                bottomSource: payload.defaults.bottomSource,
                topOverlay: payload.defaults.topOverlay,
                bottomOverlay: payload.defaults.bottomOverlay,
            }};

            function populateSelect(el, options, selectedValue) {{
                el.innerHTML = '';
                options.forEach(opt => {{
                    const option = document.createElement('option');
                    option.value = opt.value;
                    option.textContent = opt.label;
                    el.appendChild(option);
                }});
                el.value = selectedValue;
            }}

            function setupControls() {{
                const systemOptions = Object.keys(payload.systems).map(key => ({{ label: key, value: key }}));
                populateSelect(dom.system, systemOptions, payload.initialSystem);
                populateSelect(dom.topSource, payload.sourceOptions, payload.defaults.topSource);
                populateSelect(dom.bottomSource, payload.sourceOptions, payload.defaults.bottomSource);
                const overlayOptions = [{{ label: 'None', value: 'none' }}, ...payload.sourceOptions];
                populateSelect(dom.topOverlay, overlayOptions, payload.defaults.topOverlay);
                populateSelect(dom.bottomOverlay, overlayOptions, payload.defaults.bottomOverlay);

                dom.system.addEventListener('change', e => {{ state.system = e.target.value; update(); }});
                dom.topSource.addEventListener('change', e => {{ state.topSource = e.target.value; update(); }});
                dom.bottomSource.addEventListener('change', e => {{ state.bottomSource = e.target.value; update(); }});
                dom.topOverlay.addEventListener('change', e => {{ state.topOverlay = e.target.value; update(); }});
                dom.bottomOverlay.addEventListener('change', e => {{ state.bottomOverlay = e.target.value; update(); }});
                dom.lambdaSlider.addEventListener('input', e => {{
                    state.lambdaIndex = parseInt(e.target.value, 10);
                    update();
                }});
            }}

            function getSystemData() {{
                return payload.systems[state.system];
            }}

            function update() {{
                const systemData = getSystemData();
                if (!systemData) return;

                const needsSlider = systemData.hasLambdaSlider && [
                    state.topSource, state.bottomSource, state.topOverlay, state.bottomOverlay
                ].some(src => src !== 'none' && systemData.sources[src]?.requiresLambda);

                dom.lambdaControl.style.display = needsSlider ? '' : 'none';

                const maxIndex = Math.max(0, systemData.lambdaValues.length - 1);
                if (state.lambdaIndex > maxIndex) state.lambdaIndex = maxIndex;

                dom.lambdaSlider.max = maxIndex;
                dom.lambdaSlider.value = state.lambdaIndex;
                dom.lambdaReadout.textContent = `λ ≈ ${{systemData.lambdaLabels[state.lambdaIndex] || '0.0000'}}`;

                const traces = [
                    getSegments(state.topSource, false, true),
                    getSegments(state.bottomSource, false, false),
                    getSegments(state.topOverlay, true, true),
                    getSegments(state.bottomOverlay, true, false),
                ];
                Plotly.restyle(dom.plot, {{
                    'x': traces.map(t => t.x),
                    'y': traces.map(t => t.y),
                    'text': traces.map(t => t.text),
                    'line': traces.map(t => t.line),
                    'visible': [true, true, state.topOverlay !== 'none', state.bottomOverlay !== 'none'],
                }}, [1, 2, 3, 4]);

                const n = systemData.sequenceLength;
                const radius = n / 2 + 5;
                Plotly.relayout(dom.plot, {{
                    'xaxis.range': [-2, n + 1],
                    'yaxis.range': [-radius, radius],
                    'annotations[0].text': getCaption('Top', state.topSource, state.topOverlay),
                    'annotations[1].text': getCaption('Bottom', state.bottomSource, state.bottomOverlay),
                }});
                Plotly.restyle(dom.plot, {{ x: [[0, n - 1]], y: [[0, 0]] }}, [0]);
            }}

            function getSegments(sourceKey, isOverlay, isTop) {{
                const systemData = getSystemData();
                if (!sourceKey || sourceKey === 'none' || !systemData) return payload.emptySegments;
                const source = systemData.sources[sourceKey];
                if (!source) return payload.emptySegments;
                const segments = isOverlay ? source.overlay : source.primary;
                const orientation = isTop ? 'top' : 'bottom';
                return segments[orientation][state.lambdaIndex] || payload.emptySegments;
            }}

            function getCaption(pos, srcKey, ovKey) {{
                const systemData = getSystemData();
                if (!systemData) return '';
                const source = systemData.sources[srcKey];
                let caption = `${{pos}}: ${{source ? source.label : '—'}}`;
                if (source && source.requiresLambda) {{
                    const lam = source.lambdaDisplay[state.lambdaIndex];
                    if (lam !== null) caption += ` (λ≈${{lam.toFixed(3)}})`;
                }}
                if (ovKey && ovKey !== 'none') {{
                    const overlay = systemData.sources[ovKey];
                    if (overlay) {{
                        caption += ` + ${{overlay.label}} [dashed]`;
                        if (overlay.requiresLambda) {{
                            const lam = overlay.lambdaDisplay[state.lambdaIndex];
                            if (lam !== null) caption += ` (λ≈${{lam.toFixed(3)}})`;
                        }}
                    }}
                }}
                return caption;
            }}

            function initPlot() {{
                const systemData = getSystemData();
                const n = systemData.sequenceLength;
                const radius = n / 2 + 5;
                const traces = [
                    {{ type: 'scattergl', mode: 'lines', x: [0, n - 1], y: [0, 0], line: {{ color: '#333', width: 1.2 }}, hoverinfo: 'skip' }},
                    ...Array(4).fill(payload.emptySegments).map(s => ({{ ...s, type: 'scattergl', mode: 'lines', hoverinfo: 'text' }})),
                ];
                const layout = {{
                    margin: {{ l: 40, r: 20, t: 80, b: 40 }},
                    paper_bgcolor: '#fff',
                    plot_bgcolor: '#fff',
                    dragmode: 'pan',
                    hovermode: 'closest',
                    showlegend: false,
                    xaxis: {{ range: [-2, n + 1], showgrid: true, zeroline: false }},
                    yaxis: {{ range: [-radius, radius], showgrid: false, zeroline: false, scaleanchor: 'x', scaleratio: 1, visible: false }},
                    annotations: [
                        {{ xref: 'paper', yref: 'paper', x: 0.01, y: 0.98, showarrow: false, text: '', font: {{ size: 14 }}, xanchor: 'left' }},
                        {{ xref: 'paper', yref: 'paper', x: 0.01, y: 0.02, showarrow: false, text: '', font: {{ size: 14 }}, xanchor: 'left' }},
                    ],
                }};
                Plotly.newPlot(dom.plot, traces, layout, {{ responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }});
                state.lambdaIndex = systemData.defaultLambdaIndex;
                update();
            }}

            setupControls();
            initPlot();
        }})();
    </script>
</body>
</html>"""
def _build_plotly_arc_widget(
    bundle: Dict[str, Any],
    lambdas: Sequence[float],
    *,
    top_source_default: str,
    bottom_source_default: str,
    top_overlay_default: str,
    bottom_overlay_default: str,
    lambda_default_value: float,
) -> Any:
    if widgets is None:
        raise ImportError("ipywidgets not available for HTML export.")
    if go is None:
        raise ImportError("plotly is required for HTML export.")

    sequence = bundle["sequence"]

    lambda_values = np.array(lambdas, dtype=float)
    lambda_options = (
        [(f"{lam:.4f}", float(lam)) for lam in lambda_values]
        if lambda_values.size
        else [("0.0000", 0.0)]
    )
    lambda_default_value = float(lambda_default_value) if lambda_values.size else 0.0

    top_source_dd = widgets.Dropdown(options=_SOURCE_OPTIONS, value=top_source_default, description="Top")
    bottom_source_dd = widgets.Dropdown(options=_SOURCE_OPTIONS, value=bottom_source_default, description="Bottom")

    overlay_choices = [("None", "none")] + list(_SOURCE_OPTIONS)
    top_overlay_dd = widgets.Dropdown(options=overlay_choices, value=top_overlay_default, description="Top overlay")
    bottom_overlay_dd = widgets.Dropdown(options=overlay_choices, value=bottom_overlay_default, description="Bottom overlay")

    lambda_slider = widgets.SelectionSlider(
        options=lambda_options,
        value=lambda_default_value,
        description="λ",
        continuous_update=False,
        layout=widgets.Layout(width="360px"),
    )

    fig = go.FigureWidget()
    n = len(sequence)
    with fig.batch_update():
        fig.add_scattergl(
            x=[0, n - 1],
            y=[0, 0],
            mode="lines",
            line=dict(color="#333", width=1.2),
            hoverinfo="skip",
            showlegend=False,
        )
        for _ in range(4):  # Four layers: top, bottom, top_overlay, bottom_overlay
            fig.add_scattergl(
                x=[None], y=[None], mode="lines", hoverinfo="text", showlegend=False
            )

        radius = n / 2 + 5
        fig.update_layout(
            width=1400,
            height=800,
            margin=dict(l=40, r=20, t=60, b=40),
            title="Interactive arc plot (precomputed BPPs)",
            xaxis=dict(range=[-2, n + 1], showgrid=True, zeroline=False),
            yaxis=dict(range=[-radius, radius], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
            dragmode="pan",
            template="plotly_white",
            annotations=[
                dict(xref="paper", yref="paper", x=0.01, y=0.97, text="", showarrow=False, font_size=14),
                dict(xref="paper", yref="paper", x=0.01, y=0.03, text="", showarrow=False, font_size=14),
            ],
        )

    def _matrix_for(source: str, lambda_value: float) -> Tuple[np.ndarray, Optional[float]]:
        # reuse existing helper
        return _matrix_from_source(bundle, source, lambda_value)

    def _update_slider_visibility() -> None:
        requires_slider = False
        for selection in (
            top_source_dd.value,
            bottom_source_dd.value,
            top_overlay_dd.value,
            bottom_overlay_dd.value,
        ):
            if selection in _LAMBDA_SOURCES:
                requires_slider = True
                break
        lambda_slider.layout.display = "" if requires_slider and lambda_values.size > 1 else "none"

    def _apply_trace(trace_index: int, segments: Dict[str, Any], visible: bool) -> None:
        trace = fig.data[trace_index]
        trace.update(
            x=segments["x"],
            y=segments["y"],
            text=segments["text"],
            line=segments["line"],
            visible=visible,
        )

    def _caption_for(pos_label: str, source: str, overlay: str, lam_val: Optional[float], ov_lam_val: Optional[float]) -> str:
        caption = f"{pos_label}: {_LABEL_BY_KEY.get(source, '—')}"
        if source in _LAMBDA_SOURCES and lam_val is not None:
            caption += f" (λ≈{lam_val:.3f})"
        if overlay != "none":
            caption += f" + {_LABEL_BY_KEY.get(overlay, '—')} [dashed]"
            if overlay in _LAMBDA_SOURCES and ov_lam_val is not None:
                caption += f" (λ≈{ov_lam_val:.3f})"
        return caption

    def _update(_=None) -> None:
        lambda_value = lambda_slider.value if lambda_values.size > 1 else 0.0
        top_source = top_source_dd.value
        bottom_source = bottom_source_dd.value
        top_overlay = top_overlay_dd.value
        bottom_overlay = bottom_overlay_dd.value

        _update_slider_visibility()

        top_matrix, top_lambda_val = _matrix_for(top_source, lambda_value)
        bottom_matrix, bottom_lambda_val = _matrix_for(bottom_source, lambda_value)
        top_ov_matrix, top_ov_lambda_val = _matrix_for(top_overlay, lambda_value)
        bottom_ov_matrix, bottom_ov_lambda_val = _matrix_for(bottom_overlay, lambda_value)

        top_segments = _matrix_to_plotly_segments(top_matrix, top=True, color=_SOURCE_COLORS[top_source], threshold=_ARC_THRESHOLD)
        bottom_segments = _matrix_to_plotly_segments(bottom_matrix, top=False, color=_SOURCE_COLORS[bottom_source], threshold=_ARC_THRESHOLD)
        top_ov_segments = _matrix_to_plotly_segments(top_ov_matrix, top=True, color=_SOURCE_COLORS.get(top_overlay, "#000"), threshold=_ARC_THRESHOLD, linestyle=_OVERLAY_LINESTYLE, alpha=_OVERLAY_ALPHA)
        bottom_ov_segments = _matrix_to_plotly_segments(bottom_ov_matrix, top=False, color=_SOURCE_COLORS.get(bottom_overlay, "#000"), threshold=_ARC_THRESHOLD, linestyle=_OVERLAY_LINESTYLE, alpha=_OVERLAY_ALPHA)

        with fig.batch_update():
            _apply_trace(1, top_segments, True)
            _apply_trace(2, bottom_segments, True)
            _apply_trace(3, top_ov_segments, top_overlay != "none")
            _apply_trace(4, bottom_ov_segments, bottom_overlay != "none")
            fig.layout.annotations[0].text = _caption_for("Top", top_source, top_overlay, top_lambda_val, top_ov_lambda_val)
            fig.layout.annotations[1].text = _caption_for("Bottom", bottom_source, bottom_overlay, bottom_lambda_val, bottom_ov_lambda_val)

    # initial update
    _update_slider_visibility()
    _update()

    # connect callbacks
    lambda_slider.observe(_update, names="value")
    top_source_dd.observe(_update, names="value")
    bottom_source_dd.observe(_update, names="value")
    top_overlay_dd.observe(_update, names="value");
    bottom_overlay_dd.observe(_update, names="value");

    controls = widgets.VBox([
        widgets.HBox([top_source_dd, bottom_source_dd]),
        widgets.HBox([top_overlay_dd, bottom_overlay_dd]),
        lambda_slider,
    ]);

    return widgets.VBox([controls, fig]);


def _matrix_from_source(bundle: Dict[str, object], source: str, lambda_value: float) -> Tuple[np.ndarray, Optional[float]]:
    lambdas = bundle["lambdas"]
    if source == "model":
        return bundle["model_bpp"], None
    if source == "reference":
        return bundle["reference_matrix"], None
    series_key = {
        "vienna": "vienna_bpps",
        "reference_informed": "reference_informed_bpps",
    }.get(source)
    if series_key is None:
        # Handle 'none' or other invalid sources gracefully
        return np.zeros((len(bundle["sequence"]), len(bundle["sequence"]))), None
    series = bundle[series_key]
    if not series:
        return np.zeros((len(bundle["sequence"]), len(bundle["sequence"]))), None
    idx = _lambda_index(lambdas, lambda_value)
    return series[idx], float(lambdas[idx])


def create_precomputed_arcplot_widget(
    default_system: Optional[str] = None,
    *,
    display_widget: bool = True,
) -> Any:
    """Build (and optionally display) the interactive arc plot widget."""
    if widgets is None:
        raise ImportError(
            "ipywidgets is required for the interactive arc plot. Install it via 'pip install ipywidgets'."
        )

    available_systems = _discover_cached_systems()
    if not available_systems:
        raise RuntimeError(
            "No precomputed caches found in '{CACHE_DIR}'. "
            "Run the preprocessing section of interactive_arcplots.py first."
        )

    if default_system is None:
        default_system = available_systems[0]
    elif default_system not in available_systems:
        raise ValueError(
            f"System '{default_system}' has no precomputed cache. Available: {available_systems}."
        )

    bundle = _load_precomputed_bundle(default_system)
    lambdas = bundle["lambdas"]
    preferred_lambda = _closest_lambda_value(lambdas, bundle.get("optimal_lambda"))
    lambda_step = max(_lambda_step(lambdas), 1e-6)

    system_dd = widgets.Dropdown(options=available_systems, value=default_system, description="System")
    top_source = widgets.ToggleButtons(options=_SOURCE_OPTIONS, value="vienna", description="Top")
    bottom_source = widgets.ToggleButtons(options=_SOURCE_OPTIONS, value="model", description="Bottom")

    def _build_slider(description: str) -> Any:
        slider = widgets.FloatSlider(
            value=preferred_lambda,
            min=float(np.min(lambdas)),
            max=float(np.max(lambdas)),
            step=lambda_step,
            description=description,
            continuous_update=False,
            readout_format=".4f",
            layout=widgets.Layout(width="360px"),
        )
        return slider

    top_lambda = _build_slider("Top λ")
    bottom_lambda = _build_slider("Bottom λ")

    overlay_choices = [("None", "none")] + list(_SOURCE_OPTIONS)
    top_overlay = widgets.Dropdown(options=overlay_choices, value="none", description="Top overlay")
    bottom_overlay = widgets.Dropdown(options=overlay_choices, value="none", description="Bottom overlay")

    output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", padding="0.5rem"))

    def _toggle_slider_visibility(selection_widget: Any, slider: Any) -> None:
        slider.layout.display = "" if selection_widget.value in _LAMBDA_SOURCES else "none"

    def _snap_slider(slider: Any, lambdas_arr: Sequence[float]) -> float:
        snapped = _closest_lambda_value(lambdas_arr, slider.value)
        if abs(snapped - slider.value) > 1e-8:
            slider.value = snapped
        return snapped

    def _render(_: Optional[dict] = None) -> None:
        with output:
            output.clear_output(wait=True)
            try:
                current_bundle = _load_precomputed_bundle(system_dd.value)
                current_lambdas = current_bundle["lambdas"]
                snapped_top = _snap_slider(top_lambda, current_lambdas)
                snapped_bottom = _snap_slider(bottom_lambda, current_lambdas)

                top_matrix, top_lambda_val = _matrix_from_source(current_bundle, top_source.value, snapped_top)
                bottom_matrix, bottom_lambda_val = _matrix_from_source(current_bundle, bottom_source.value, snapped_bottom)

                layers = [
                    {
                        "matrix": top_matrix,
                        "top": True,
                        "color": _SOURCE_COLORS[top_source.value],
                        "legend_key": top_source.value,
                    },
                    {
                        "matrix": bottom_matrix,
                        "top": False,
                        "color": _SOURCE_COLORS[bottom_source.value],
                        "legend_key": bottom_source.value,
                    },
                ]

                top_overlay_matrix = None
                top_overlay_lambda = None
                if top_overlay.value != "none":
                    top_overlay_matrix, top_overlay_lambda = _matrix_from_source(
                        current_bundle, top_overlay.value, snapped_top
                    )
                    layers.push(
                        {
                            "matrix": top_overlay_matrix,
                            "top": True,
                            "color": _SOURCE_COLORS[top_overlay.value],
                            "threshold": 0.0 if top_overlay.value == "reference" else _ARC_THRESHOLD,
                            "linestyle": _OVERLAY_LINESTYLE,
                            "alpha": _OVERLAY_ALPHA,
                            "legend_key": top_overlay.value,
                            "legend_linestyle": _OVERLAY_LINESTYLE,
                        }
                    )

                bottom_overlay_matrix = None
                bottom_overlay_lambda = None
                if bottom_overlay.value != "none":
                    bottom_overlay_matrix, bottom_overlay_lambda = _matrix_from_source(
                        current_bundle, bottom_overlay.value, snapped_bottom
                    )
                    layers.push(
                        {
                            "matrix": bottom_overlay_matrix,
                            "top": False,
                            "color": _SOURCE_COLORS[bottom_overlay.value],
                            "threshold": 0.0 if bottom_overlay.value == "reference" else _ARC_THRESHOLD,
                            "linestyle": _OVERLAY_LINESTYLE,
                            "alpha": _OVERLAY_ALPHA,
                            "legend_key": bottom_overlay.value,
                            "legend_linestyle": _OVERLAY_LINESTYLE,
                        }
                    )

                top_label = _LABEL_BY_KEY[top_source.value]
                bottom_label = _LABEL_BY_KEY[bottom_source.value]
                top_text = f"Top: {top_label}"
                bottom_text = f"Bottom: {bottom_label}"
                if top_lambda_val is not None:
                    top_text += f" (λ≈{top_lambda_val:.3f})"
                if bottom_lambda_val is not None:
                    bottom_text += f" (λ≈{bottom_lambda_val:.3f})"
                if top_overlay.value != "none":
                    overlay_label = _LABEL_BY_KEY[top_overlay.value]
                    top_text += f" + {overlay_label}"
                    if top_overlay_lambda is not None:
                        top_text += f" (λ≈{top_overlay_lambda:.3f})"
                    top_text += " [dashed]"
                if bottom_overlay.value != "none":
                    overlay_label = _LABEL_BY_KEY[bottom_overlay.value]
                    bottom_text += f" + {overlay_label}"
                    if bottom_overlay_lambda is not None:
                        bottom_text += f" (λ≈{bottom_overlay_lambda:.3f})"
                    bottom_text += " [dashed]"

                # Prefer Matplotlib within widget to keep filters visible; Plotly reserved for export
                fig = _render_arcplot(
                    current_bundle["sequence"],
                    layers,
                    top_text,
                    bottom_text,
                    _SOURCE_COLORS[top_source.value],
                    _SOURCE_COLORS[bottom_source.value],
                )
                plt.show()
                plt.close(fig)
            except Exception as exc:  # pragma: no cover - interactive feedback path
                print(f"Unable to update arc plot: {exc}")

    def _configure_sliders(lambdas_arr: Sequence[float], default_value: float) -> None:
        min_val = float(np.min(lambdas_arr))
        max_val = float(np.max(lambdas_arr))
        step_val = max(_lambda_step(lambdas_arr), 1e-6)
        for slider in (top_lambda, bottom_lambda):
            slider.min = min_val
            slider.max = max_val
            slider.step = step_val
            slider.value = default_value

    def _on_system_change(change: dict) -> None:
        if change.get("name") != "value":
            return
        new_bundle = _load_precomputed_bundle(change["new"])
        new_default = _closest_lambda_value(new_bundle["lambdas"], new_bundle.get("optimal_lambda"))
        _configure_sliders(new_bundle["lambdas"], new_default)
        _render()

    def _on_source_change(change: dict, slider: Any) -> None:
        if change.get("name") != "value":
            return
        _toggle_slider_visibility(change["owner"], slider)
        _render()

    system_dd.observe(_on_system_change, names="value")
    top_source.observe(lambda change: _on_source_change(change, top_lambda), names="value")
    bottom_source.observe(lambda change: _on_source_change(change, bottom_lambda), names="value")
    top_overlay.observe(_render, names="value")
    bottom_overlay.observe(_render, names="value")
    top_lambda.observe(_render, names="value")
    bottom_lambda.observe(_render, names="value")

    _toggle_slider_visibility(top_source, top_lambda)
    _toggle_slider_visibility(bottom_source, bottom_lambda)
    _render()

    controls = widgets.VBox([
        system_dd,
        widgets.HBox([top_source, top_lambda]),
        widgets.HBox([top_overlay]),
        widgets.HBox([bottom_source, bottom_lambda]),
        widgets.HBox([bottom_overlay]),
    ])
    ui = widgets.VBox([controls, output])

    if display_widget:
        display(ui)

    return ui;


def launch_precomputed_arcplot(default_system: Optional[str] = None) -> None:
    """Convenience wrapper to display the widget in IPython sessions."""
    create_precomputed_arcplot_widget(default_system=default_system, display_widget=True)


def export_precomputed_arcplot_html(
    output_path: str,
    default_system: Optional[str] = None,
    *,
    title: str = "RNA Arc Plot Explorer",
) -> None:
    """Persist the interactive explorer to a standalone HTML file."""
    available_systems = _discover_cached_systems()
    if not available_systems:
        raise RuntimeError("No cached systems available for export.")

    initial_system = default_system or available_systems[0]
    if initial_system not in available_systems:
        raise ValueError(
            f"System '{initial_system}' has no precomputed cache. Available: {available_systems}."
        )

    systems_payload: Dict[str, Any] = {}
    print(f"Exporting HTML for systems: {available_systems}")
    for system in available_systems:
        try:
            systems_payload[system] = _prepare_system_payload(system)
            print(f"Successfully prepared payload for '{system}'")
        except Exception as e:
            print(f"Could not process system '{system}': {e}")

    # Add this for debugging
    import json
    print("--- Payload for HTML ---")
    print(json.dumps({k: v for k, v in systems_payload.items() if k == initial_system}, indent=2, default=lambda o: f"<non-serializable: {type(o).__name__}>"))
    print("------------------------")

    payload = {
        "title": title,
        "initialSystem": initial_system,
        "systems": systems_payload,
        "defaults": {
            "topSource": INTERACTIVE_HTML_TOP_SOURCE_DEFAULT,
            "bottomSource": INTERACTIVE_HTML_BOTTOM_SOURCE_DEFAULT,
            "topOverlay": INTERACTIVE_HTML_TOP_OVERLAY_DEFAULT,
            "bottomOverlay": INTERACTIVE_HTML_BOTTOM_OVERLAY_DEFAULT,
        },
        "sourceOptions": [{"label": label, "value": value} for (label, value) in _SOURCE_OPTIONS],
        "emptySegments": _EMPTY_SEGMENTS,
    }

    html = _render_arcplot_html(payload)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        stream.write(html)


if widgets is not None:
    try:  # pragma: no cover - notebook detection
        __IPYTHON__  # type: ignore  # noqa: F821
        # Automatically expose the widget when run in an interactive notebook context.
        launch_precomputed_arcplot()
    except NameError:
        # Not running inside IPython; skip auto-launch.
        pass

# Optional: auto-export interactive widget to HTML when running as a script
try:
    if SAVE_INTERACTIVE_HTML:
        os.makedirs(OTHERFIGS_FOLDER_PATH, exist_ok=True)
        export_precomputed_arcplot_html(
            INTERACTIVE_HTML_OUTPUT,
            default_system=INTERACTIVE_HTML_DEFAULT_SYSTEM,
            title="RNA Arc Plot Explorer",
        )
except Exception as _exc:
    # Keep script robust even if export fails (e.g., ipywidgets not installed)
    print(f"Interactive HTML export skipped: {_exc}")



# %%
