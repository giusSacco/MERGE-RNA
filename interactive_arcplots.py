#%%
#interactive_arcplots.py
import json
import os
import tempfile
import webbrowser
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
SAVE_TO_OVERLEAF = False
OTHERFIGS_FOLDER_PATH = "otherfigs"
OVERLEAF_FOLDER_PATH = None
SAVE_LOCAL_PLOTS = False  # Save static PNGs locally
# Import required libraries
from tqdm import tqdm
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import RNA
try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional dependency
    go = None

# Use a non-interactive backend for notebooks and TkAgg for command-line
def is_notebook() -> bool:
    """
    Returns True if the code is running in a Jupyter notebook, False otherwise.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type
    except (ImportError, NameError, AttributeError):
        return False      # Not in an IPython shell

if not is_notebook():
    import matplotlib
    matplotlib.use('TkAgg')

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

#%%
# Interactive arc plot explorer powered by precomputed BPPs
try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError:  # pragma: no cover - widgets not installed
    widgets = None
    display = None

from functools import lru_cache


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

_DEFAULT_FIGURE_SIZE: Tuple[float, float] = (6.3, 4.0)
_DEFAULT_FIGURE_DPI: float = 105.0

_LAMBDA_SOURCES = {"vienna", "reference_informed"}

_ARC_THRESHOLD = 0.05
_ARC_HEIGHT_RATIO = 1.0
_ARC_MAX_LINEWIDTH = 3.0
_OVERLAY_ALPHA = 0.4
_OVERLAY_LINESTYLE = "--"

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


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(ch * 2 for ch in hex_color)
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color '{hex_color}'")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _render_arcplot(
    sequence: str,
    layers: Sequence[Dict[str, Any]],
    top_text: str,
    bottom_text: str,
    top_color: str,
    bottom_color: str,
    *,
    figure_size: Tuple[float, float] = _DEFAULT_FIGURE_SIZE,
    figure_dpi: float = _DEFAULT_FIGURE_DPI,
) -> Any:
    if go is None:
        raise RuntimeError(
            "Plotly is required for interactive arc plot export. Install it via 'pip install plotly'."
        )

    n = len(sequence)
    fig = go.Figure()

    baseline_color = "rgba(51, 51, 51, 0.8)"
    fig.add_trace(
        go.Scatter(
            x=[0, n - 1],
            y=[0, 0],
            mode="lines",
            line=dict(color=baseline_color, width=1.5),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    seen_legends: set[str] = set()

    for layer in layers:
        matrix = np.asarray(layer["matrix"])
        if matrix.shape[0] != n:
            raise ValueError("BPP matrix size does not match sequence length.")

        threshold = layer.get("threshold", _ARC_THRESHOLD)
        linestyle = layer.get("linestyle", "-")
        alpha = layer.get("alpha", 0.85)
        rgba_color = _hex_to_rgba(layer["color"], alpha)
        top_half = bool(layer["top"])
        legend_key = layer.get("legend_key")
        legend_name = _LABEL_BY_KEY.get(legend_key, legend_key.replace("_", " ").title()) if legend_key else None

        for i in range(n - 1):
            for j in range(i + 1, n):
                prob = float(matrix[i, j])
                if np.isnan(prob) or prob <= threshold:
                    continue

                width = j - i
                if width <= 0:
                    continue

                height = width * _ARC_HEIGHT_RATIO
                center_x = (i + j) / 2.0
                radius_x = width / 2.0
                radius_y = height / 2.0

                samples = max(16, min(80, int(width * 6)))
                theta = (
                    np.linspace(0, np.pi, samples)
                    if top_half
                    else np.linspace(np.pi, 2 * np.pi, samples)
                )
                xs = center_x + radius_x * np.cos(theta)
                ys = radius_y * np.sin(theta)

                line_width = max(0.75, min(_ARC_MAX_LINEWIDTH, prob * _ARC_MAX_LINEWIDTH))
                dash_style = "dash" if linestyle == _OVERLAY_LINESTYLE else "solid"
                hover_label = f"{legend_name or 'Arc'}: {i + 1}–{j + 1}, p={prob:.3f}"

                showlegend = legend_key is not None and legend_key not in seen_legends
                if showlegend:
                    seen_legends.add(legend_key)

                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color=rgba_color, width=line_width, dash=dash_style),
                        hoverinfo="text",
                        hovertext=[hover_label] * len(xs),
                        showlegend=showlegend,
                        name=legend_name if showlegend else None,
                    )
                )

    radius = n / 2 + 5
    fig.update_xaxes(range=[-2, n + 1], showgrid=True, zeroline=False, title="Position (nt)")
    fig.update_yaxes(range=[-radius, radius], visible=False, zeroline=False)

    fig.update_layout(
        title="Interactive arc plot (precomputed BPPs)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        margin=dict(l=40, r=40, t=80, b=40),
        width=int(figure_size[0] * figure_dpi),
        height=int(figure_size[1] * figure_dpi),
        legend=dict(title="Source", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.97,
        text=top_text,
        showarrow=False,
        font=dict(color=top_color, size=12),
        align="left",
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.03,
        text=bottom_text,
        showarrow=False,
        font=dict(color=bottom_color, size=12),
        align="left",
    )

    return fig

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
    figure_size: Tuple[float, float] = _DEFAULT_FIGURE_SIZE,
    figure_dpi: float = _DEFAULT_FIGURE_DPI,
) -> Any:
    """Build (and optionally display) the interactive arc plot widget."""
    if widgets is None:
        raise ImportError(
            "ipywidgets is required for the interactive arc plot. Install it via 'pip install ipywidgets'."
        )
    if go is None:
        raise ImportError(
            "plotly is required for the interactive arc plot. Install it via 'pip install plotly'."
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

    system_dd = widgets.Dropdown(
        options=available_systems,
        value=default_system,
        description="System",
        layout=widgets.Layout(width="100%"),
    )
    top_source = widgets.ToggleButtons(
        options=_SOURCE_OPTIONS,
        value="vienna",
        description="Top",
        layout=widgets.Layout(width="100%"),
        style=dict(button_width="6.2rem"),
    )
    bottom_source = widgets.ToggleButtons(
        options=_SOURCE_OPTIONS,
        value="model",
        description="Bottom",
        layout=widgets.Layout(width="100%"),
        style=dict(button_width="6.2rem"),
    )

    def _build_slider(description: str) -> Any:
        slider = widgets.FloatSlider(
            value=preferred_lambda,
            min=float(np.min(lambdas)),
            max=float(np.max(lambdas)),
            step=lambda_step,
            description=description,
            continuous_update=False,
            readout_format=".4f",
            layout=widgets.Layout(width="100%"),
        )
        return slider

    top_lambda = _build_slider("Top λ")
    bottom_lambda = _build_slider("Bottom λ")

    overlay_choices = [("None", "none")] + list(_SOURCE_OPTIONS)
    top_overlay = widgets.Dropdown(
        options=overlay_choices,
        value="none",
        description="Top overlay",
        layout=widgets.Layout(width="100%"),
    )
    bottom_overlay = widgets.Dropdown(
        options=overlay_choices,
        value="none",
        description="Bottom overlay",
        layout=widgets.Layout(width="100%"),
    )

    message_box = widgets.Output(
        layout=widgets.Layout(border="0px", padding="0.2rem 0", max_height="56px", overflow="hidden")
    )
    plot_container = widgets.Output(
        layout=widgets.Layout(flex="1 1 auto", min_width="460px", height="100%")
    )
    latest_render_info: Optional[Dict[str, Any]] = None
    open_browser_button = widgets.Button(
        description="Open in browser",
        icon="external-link",
        tooltip="Render the current figure in a standalone browser window",
        layout=widgets.Layout(width="100%"),
    )

    def _toggle_slider_visibility(selection_widget: Any, slider: Any) -> None:
        slider.layout.display = "" if selection_widget.value in _LAMBDA_SOURCES else "none"

    def _snap_slider(slider: Any, lambdas_arr: Sequence[float]) -> float:
        snapped = _closest_lambda_value(lambdas_arr, slider.value)
        if abs(snapped - slider.value) > 1e-8:
            slider.value = snapped
        return snapped

    def _render(_: Optional[dict] = None) -> None:
        nonlocal latest_render_info
        message_box.clear_output(wait=True)
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
                layers.append(
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
                layers.append(
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

            fig = _render_arcplot(
                current_bundle["sequence"],
                layers,
                top_text,
                bottom_text,
                _SOURCE_COLORS[top_source.value],
                _SOURCE_COLORS[bottom_source.value],
                figure_size=figure_size,
                figure_dpi=figure_dpi,
            )
            latest_render_info = {
                "system": system_dd.value,
                "top_source": top_source.value,
                "bottom_source": bottom_source.value,
                "top_lambda": snapped_top,
                "bottom_lambda": snapped_bottom,
                "top_overlay": top_overlay.value,
                "bottom_overlay": bottom_overlay.value,
            }
            plot_widget = go.FigureWidget(fig)
            with plot_container:
                plot_container.clear_output(wait=True)
                display(plot_widget)
        except Exception as exc:  # pragma: no cover - interactive feedback path
            latest_render_info = None
            with message_box:
                print(f"Unable to update arc plot: {exc}")
            with plot_container:
                plot_container.clear_output(wait=True)

    def _open_in_browser(_: Any) -> None:
        nonlocal latest_render_info
        try:
            current_bundle = _load_precomputed_bundle(system_dd.value)
            current_lambdas = current_bundle["lambdas"]
            snapped_top = _snap_slider(top_lambda, current_lambdas)
            snapped_bottom = _snap_slider(bottom_lambda, current_lambdas)

            top_lambda_index = _lambda_index(current_lambdas, snapped_top)
            bottom_lambda_index = _lambda_index(current_lambdas, snapped_bottom)

            vienna_series = [np.asarray(mat).tolist() for mat in current_bundle["vienna_bpps"]]
            reference_informed_series = [np.asarray(mat).tolist() for mat in current_bundle["reference_informed_bpps"]]

            export_payload: Dict[str, Any] = {
                "sequence": current_bundle["sequence"],
                "lambdas": np.asarray(current_lambdas, dtype=float).tolist(),
                "model_matrix": np.asarray(current_bundle["model_bpp"], dtype=float).tolist(),
                "reference_matrix": np.asarray(current_bundle["reference_matrix"], dtype=float).tolist(),
                "vienna_bpps": vienna_series,
                "reference_informed_bpps": reference_informed_series,
                "source_options": list(_SOURCE_OPTIONS),
                "overlay_options": [
                    ("None", "none"),
                    *_SOURCE_OPTIONS,
                ],
                "colors": _SOURCE_COLORS,
                "labels": _LABEL_BY_KEY,
                "initial": {
                    "system": system_dd.value,
                    "top_source": top_source.value,
                    "bottom_source": bottom_source.value,
                    "top_overlay": top_overlay.value,
                    "bottom_overlay": bottom_overlay.value,
                    "top_lambda_index": top_lambda_index,
                    "bottom_lambda_index": bottom_lambda_index,
                },
                "settings": {
                    "arc_threshold": _ARC_THRESHOLD,
                    "overlay_threshold": 0.0,
                    "arc_height_ratio": _ARC_HEIGHT_RATIO,
                    "arc_max_linewidth": _ARC_MAX_LINEWIDTH,
                    "overlay_alpha": _OVERLAY_ALPHA,
                },
                "figure": {
                    "width": max(int(figure_size[0] * figure_dpi * 1.45), 880),
                    "height": max(int(figure_size[1] * figure_dpi * 1.55), 620),
                },
            }

            config_json = json.dumps(export_payload)
            html_content = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>Arc plot: {system_dd.value}</title>
    <script src=\"https://cdn.plot.ly/plotly-2.32.0.min.js\"></script>
    <style>
        body {{
            font-family: 'Inter', 'Segoe UI', sans-serif;
            margin: 0;
            padding: 1.5rem;
            background: #f6f7fb;
            color: #1d2733;
        }}
        .container {{
            max-width: 1280px;
            margin: 0 auto;
        }}
        h2 {{
            margin-top: 0;
            font-weight: 600;
        }}
        .browser-layout {{
            display: flex;
            align-items: stretch;
            gap: 1.4rem;
        }}
        .controls-pane {{
            flex: 0 0 320px;
            max-width: 320px;
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
        }}
        .control-block {{
            background: #ffffff;
            border: 1px solid #dfe3eb;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
        }}
        .control-block label {{
            font-size: 0.9rem;
            font-weight: 600;
            display: block;
            margin-bottom: 0.35rem;
        }}
        .control-block select,
        .control-block input[type=\"range\"] {{
            width: 100%;
            margin-bottom: 0.55rem;
        }}
        .lambda-readout {{
            font-size: 0.82rem;
            text-align: right;
            margin-top: -0.35rem;
            margin-bottom: 0.2rem;
            color: #536175;
        }}
        .plot-pane {{
            flex: 1 1 auto;
            min-width: 520px;
        }}
        #plot {{
            width: 100%;
            min-height: 580px;
        }}
        .note {{
            font-size: 0.85rem;
            color: #6c7a92;
            margin-bottom: 0.8rem;
        }}
        @media (max-width: 1024px) {{
            .browser-layout {{
                flex-direction: column;
            }}
            .controls-pane {{
                flex: 1 1 auto;
                max-width: none;
                flex-direction: row;
                flex-wrap: wrap;
                gap: 0.75rem;
            }}
            .control-block {{
                flex: 1 1 280px;
            }}
            .plot-pane {{
                min-width: 0;
            }}
        }}
    </style>
</head>
<body>
    <div class=\"container\">
        <h2>Arc plot for {system_dd.value}</h2>
        <p class="note">Use the dropdowns and λ sliders to explore alternative models. Re-run the notebook export to refresh this view.</p>
        <div class="browser-layout">
            <div class="controls-pane">
                <div class="control-block">
                    <label for="top-source">Top source</label>
                    <select id="top-source"></select>
                    <div id="top-lambda-wrapper">
                        <label for="top-lambda">Top λ index</label>
                        <input type="range" id="top-lambda" min="0" max="{max(len(current_lambdas) - 1, 0)}" step="1" />
                        <div id="top-lambda-readout" class="lambda-readout"></div>
                    </div>
                    <label for="top-overlay">Top overlay</label>
                    <select id="top-overlay"></select>
                </div>
                <div class="control-block">
                    <label for="bottom-source">Bottom source</label>
                    <select id="bottom-source"></select>
                    <div id="bottom-lambda-wrapper">
                        <label for="bottom-lambda">Bottom λ index</label>
                        <input type="range" id="bottom-lambda" min="0" max="{max(len(current_lambdas) - 1, 0)}" step="1" />
                        <div id="bottom-lambda-readout" class="lambda-readout"></div>
                    </div>
                    <label for="bottom-overlay">Bottom overlay</label>
                    <select id="bottom-overlay"></select>
                </div>
            </div>
            <div class="plot-pane">
                <div id="plot"></div>
            </div>
        </div>
    </div>
    <script>
        const CONFIG = {config_json};
    </script>
    <script>
        (function() {{
            const cfg = CONFIG;
            const lambdaValues = cfg.lambdas;
            const n = cfg.sequence.length;
            const sourceOptions = cfg.source_options;
            const overlayOptions = cfg.overlay_options;
            const colors = cfg.colors;
            const labels = cfg.labels;
            const settings = cfg.settings;
            const figure = cfg.figure;

            const topSourceSelect = document.getElementById('top-source');
            const topLambdaSlider = document.getElementById('top-lambda');
            const topLambdaReadout = document.getElementById('top-lambda-readout');
            const topLambdaWrapper = document.getElementById('top-lambda-wrapper');
            const topOverlaySelect = document.getElementById('top-overlay');

            const bottomSourceSelect = document.getElementById('bottom-source');
            const bottomLambdaSlider = document.getElementById('bottom-lambda');
            const bottomLambdaReadout = document.getElementById('bottom-lambda-readout');
            const bottomLambdaWrapper = document.getElementById('bottom-lambda-wrapper');
            const bottomOverlaySelect = document.getElementById('bottom-overlay');

            function populateSelect(select, options) {{
                select.innerHTML = '';
                options.forEach(function(option) {{
                    const opt = document.createElement('option');
                    opt.value = option[1];
                    opt.textContent = option[0];
                    select.appendChild(opt);
                }});
            }}

            populateSelect(topSourceSelect, sourceOptions);
            populateSelect(bottomSourceSelect, sourceOptions);
            populateSelect(topOverlaySelect, overlayOptions);
            populateSelect(bottomOverlaySelect, overlayOptions);

            topSourceSelect.value = cfg.initial.top_source;
            bottomSourceSelect.value = cfg.initial.bottom_source;
            topOverlaySelect.value = cfg.initial.top_overlay;
            bottomOverlaySelect.value = cfg.initial.bottom_overlay;

            topLambdaSlider.value = cfg.initial.top_lambda_index || 0;
            bottomLambdaSlider.value = cfg.initial.bottom_lambda_index || 0;

            const lambdaCount = lambdaValues.length;
            if (lambdaCount <= 1) {{
                topLambdaWrapper.style.display = 'none';
                bottomLambdaWrapper.style.display = 'none';
            }} else {{
                topLambdaWrapper.style.display = '';
                bottomLambdaWrapper.style.display = '';
            }}

            function lambdaLabel(idx) {{
                idx = Math.max(0, Math.min(lambdaCount - 1, Number(idx) || 0));
                return `λ ≈ ${{lambdaValues[idx].toFixed(4)}}`;
            }}

            function updateLambdaReadouts() {{
                if (topLambdaReadout) {{
                    topLambdaReadout.textContent = lambdaLabel(topLambdaSlider.value);
                }}
                if (bottomLambdaReadout) {{
                    bottomLambdaReadout.textContent = lambdaLabel(bottomLambdaSlider.value);
                }}
            }}

            updateLambdaReadouts();

            function getMatrix(source, lambdaIndex) {{
                if (source === 'model') {{
                    return cfg.model_matrix;
                }}
                if (source === 'reference') {{
                    return cfg.reference_matrix;
                }}
                const idx = Math.max(0, Math.min(lambdaCount - 1, lambdaIndex));
                if (source === 'vienna') {{
                    return cfg.vienna_bpps[idx];
                }}
                if (source === 'reference_informed') {{
                    return cfg.reference_informed_bpps[idx];
                }}
                return null;
            }}

            function buildArcs(matrix, opts) {{
                if (!matrix) {{
                    return [];
                }}
                const threshold = opts.threshold;
                const topHalf = !!opts.top;
                const color = opts.color;
                const legendKey = opts.legendKey || null;
                const linestyle = opts.linestyle || 'solid';
                const alpha = opts.alpha || 0.85;
                const showLegend = opts.showLegend;
                const strokeColor = rgba(color, alpha);

                const traces = [];
                const seen = new Set();

                for (let i = 0; i < n - 1; i++) {{
                    for (let j = i + 1; j < n; j++) {{
                        const prob = matrix[i][j];
                        if (!prob || !isFinite(prob) || prob <= threshold) {{
                            continue;
                        }}
                        const width = j - i;
                        if (width <= 0) {{
                            continue;
                        }}
                        const height = width * settings.arc_height_ratio;
                        const centerX = (i + j) / 2.0;
                        const radiusX = width / 2.0;
                        const radiusY = height / 2.0;
                        const samples = Math.max(16, Math.min(120, Math.round(width * 6)));
                        const xs = new Array(samples);
                        const ys = new Array(samples);
                        for (let s = 0; s < samples; s++) {{
                            const theta = topHalf
                                ? (Math.PI * s) / (samples - 1)
                                : Math.PI + (Math.PI * s) / (samples - 1);
                            xs[s] = centerX + radiusX * Math.cos(theta);
                            ys[s] = radiusY * Math.sin(theta);
                        }}
                        const lineWidth = Math.max(0.75, Math.min(settings.arc_max_linewidth, prob * settings.arc_max_linewidth));
                        const hoverLabel = `${{legendKey ? labels[legendKey] : 'Arc'}}: ${{i + 1}}–${{j + 1}}, p=${{prob.toFixed(3)}}`;
                        const dashStyle = linestyle === '--' ? 'dash' : 'solid';
                        const legendName = legendKey ? labels[legendKey] : undefined;
                        const shouldShowLegend = legendKey && !seen.has(legendKey) && showLegend;
                        if (legendKey) {{
                            seen.add(legendKey);
                        }}
                        traces.push({{
                            x: xs,
                            y: ys,
                            mode: 'lines',
                            line: {{ color: strokeColor, width: lineWidth, dash: dashStyle }},
                            hoverinfo: 'text',
                            hovertext: Array(samples).fill(hoverLabel),
                            showlegend: shouldShowLegend,
                            name: shouldShowLegend ? legendName : undefined,
                        }});
                    }}
                }}
                return traces;
            }}

            function rgba(hex, alpha) {{
                if (hex.startsWith('rgba')) {{
                    return hex;
                }}
                let h = hex.replace('#', '');
                if (h.length === 3) {{
                    h = h.split('').map(ch => ch + ch).join('');
                }}
                const r = parseInt(h.substring(0, 2), 16);
                const g = parseInt(h.substring(2, 4), 16);
                const b = parseInt(h.substring(4, 6), 16);
                return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
            }}

            function renderPlot() {{
                const topSource = topSourceSelect.value;
                const bottomSource = bottomSourceSelect.value;
                const topOverlay = topOverlaySelect.value;
                const bottomOverlay = bottomOverlaySelect.value;
                const topLambdaIdx = Math.max(0, Math.min(lambdaCount - 1, Number(topLambdaSlider.value) || 0));
                const bottomLambdaIdx = Math.max(0, Math.min(lambdaCount - 1, Number(bottomLambdaSlider.value) || 0));

                const layers = [];

                const topMatrix = getMatrix(topSource, topLambdaIdx);
                const bottomMatrix = getMatrix(bottomSource, bottomLambdaIdx);

                if (!topMatrix || !bottomMatrix) {{
                    return;
                }}

                layers.push(...buildArcs(topMatrix, {{
                    top: true,
                    color: colors[topSource],
                    legendKey: topSource,
                    threshold: settings.arc_threshold,
                    showLegend: true,
                }}));

                layers.push(...buildArcs(bottomMatrix, {{
                    top: false,
                    color: colors[bottomSource],
                    legendKey: bottomSource,
                    threshold: settings.arc_threshold,
                    showLegend: true,
                }}));

                if (topOverlay !== 'none') {{
                    const overlayMatrix = getMatrix(topOverlay, topLambdaIdx);
                    if (overlayMatrix) {{
                        layers.push(...buildArcs(overlayMatrix, {{
                            top: true,
                            color: rgba(colors[topOverlay], settings.overlay_alpha),
                            legendKey: topOverlay,
                            threshold: topOverlay === 'reference' ? settings.overlay_threshold : settings.arc_threshold,
                            linestyle: '--',
                            alpha: settings.overlay_alpha,
                            showLegend: true,
                        }}));
                    }}
                }}

                if (bottomOverlay !== 'none') {{
                    const overlayMatrix = getMatrix(bottomOverlay, bottomLambdaIdx);
                    if (overlayMatrix) {{
                        layers.push(...buildArcs(overlayMatrix, {{
                            top: false,
                            color: rgba(colors[bottomOverlay], settings.overlay_alpha),
                            legendKey: bottomOverlay,
                            threshold: bottomOverlay === 'reference' ? settings.overlay_threshold : settings.arc_threshold,
                            linestyle: '--',
                            alpha: settings.overlay_alpha,
                            showLegend: true,
                        }}));
                    }}
                }}

                const traces = [];
                const baselineColor = 'rgba(51, 51, 51, 0.85)';
                traces.push({{
                    x: [0, n - 1],
                    y: [0, 0],
                    mode: 'lines',
                    line: {{ color: baselineColor, width: 1.6 }},
                    hoverinfo: 'skip',
                    showlegend: false,
                }});

                traces.push(...layers);

                const radius = n / 2 + 5;
                const topLabel = labels[topSource] + (['vienna', 'reference_informed'].includes(topSource) ? ` (λ≈${{lambdaValues[topLambdaIdx].toFixed(3)}})` : '');
                const bottomLabel = labels[bottomSource] + (['vienna', 'reference_informed'].includes(bottomSource) ? ` (λ≈${{lambdaValues[bottomLambdaIdx].toFixed(3)}})` : '');

                let topAnnotation = `Top: ${{topLabel}}`;
                if (topOverlay !== 'none') {{
                    const overlayName = labels[topOverlay];
                    topAnnotation += ` + ${{overlayName}}`;
                    if (['vienna', 'reference_informed'].includes(topOverlay)) {{
                        topAnnotation += ` (λ≈${{lambdaValues[topLambdaIdx].toFixed(3)}})`;
                    }}
                    topAnnotation += ' [dashed]';
                }}

                let bottomAnnotation = `Bottom: ${{bottomLabel}}`;
                if (bottomOverlay !== 'none') {{
                    const overlayName = labels[bottomOverlay];
                    bottomAnnotation += ` + ${{overlayName}}`;
                    if (['vienna', 'reference_informed'].includes(bottomOverlay)) {{
                        bottomAnnotation += ` (λ≈${{lambdaValues[bottomLambdaIdx].toFixed(3)}})`;
                    }}
                    bottomAnnotation += ' [dashed]';
                }}

                const layout = {{
                    title: {{ text: `Interactive arc plot (browser view)` }},
                    width: figure.width,
                    height: figure.height,
                    plot_bgcolor: '#ffffff',
                    paper_bgcolor: '#ffffff',
                    hovermode: 'closest',
                    margin: {{ l: 60, r: 40, t: 80, b: 40 }},
                    xaxis: {{ range: [-2, n + 1], title: 'Position (nt)', showgrid: true, zeroline: false }},
                    yaxis: {{ range: [-radius, radius], visible: false, zeroline: false }},
                    legend: {{ orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1.0 }},
                    annotations: [
                        {{
                            xref: 'paper', yref: 'paper', x: 0.01, y: 0.98,
                            text: topAnnotation, showarrow: false,
                            font: {{ color: colors[topSource], size: 13 }},
                            align: 'left'
                        }},
                        {{
                            xref: 'paper', yref: 'paper', x: 0.01, y: 0.04,
                            text: bottomAnnotation, showarrow: false,
                            font: {{ color: colors[bottomSource], size: 13 }},
                            align: 'left'
                        }}
                    ],
                }};

                const config = {{ responsive: true, displaylogo: false, scrollZoom: true }};
                Plotly.react('plot', traces, layout, config);
            }}

            function toggleLambdaVisibility(select, wrapper) {{
                if (!wrapper) return;
                const value = select.value;
                if (value === 'vienna' || value === 'reference_informed') {{
                    wrapper.style.display = '';
                }} else {{
                    wrapper.style.display = 'none';
                }}
            }}

            topSourceSelect.addEventListener('change', function() {{
                toggleLambdaVisibility(topSourceSelect, topLambdaWrapper);
                renderPlot();
            }});
            bottomSourceSelect.addEventListener('change', function() {{
                toggleLambdaVisibility(bottomSourceSelect, bottomLambdaWrapper);
                renderPlot();
            }});
            topOverlaySelect.addEventListener('change', renderPlot);
            bottomOverlaySelect.addEventListener('change', renderPlot);
            topLambdaSlider.addEventListener('input', function() {{
                updateLambdaReadouts();
                renderPlot();
            }});
            bottomLambdaSlider.addEventListener('input', function() {{
                updateLambdaReadouts();
                renderPlot();
            }});

            toggleLambdaVisibility(topSourceSelect, topLambdaWrapper);
            toggleLambdaVisibility(bottomSourceSelect, bottomLambdaWrapper);
            renderPlot();
    }})();
    </script>
</body>
</html>"""

            with tempfile.NamedTemporaryFile("w", suffix="_arcplot.html", delete=False, encoding="utf-8") as tmp:
                tmp.write(html_content)
                temp_path = tmp.name

            webbrowser.open(f"file://{temp_path}")
            with message_box:
                message_box.clear_output(wait=True)
                print("Opened interactive browser view in a new tab/window.")
        except Exception as exc:  # pragma: no cover - renderer feedback path
            with message_box:
                message_box.clear_output(wait=True)
                print(f"Unable to launch browser renderer: {exc}")

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
    open_browser_button.on_click(_open_in_browser)

    _toggle_slider_visibility(top_source, top_lambda)
    _toggle_slider_visibility(bottom_source, bottom_lambda)
    _render()

    controls_column = widgets.VBox(
        [
            system_dd,
            widgets.VBox([top_source, top_lambda], layout=widgets.Layout(gap="0.25rem")),
            top_overlay,
            widgets.VBox([bottom_source, bottom_lambda], layout=widgets.Layout(gap="0.25rem")),
            bottom_overlay,
            open_browser_button,
            message_box,
        ],
        layout=widgets.Layout(
            width="240px",
            min_width="220px",
            padding="0.35rem",
            gap="0.3rem",
            border="1px solid #ddd",
            border_radius="6px",
        ),
    )

    ui = widgets.HBox(
        [controls_column, plot_container],
        layout=widgets.Layout(width="100%", gap="0.6rem", align_items="stretch"),
    )

    if display_widget:
        display(ui)

    return ui;


def launch_precomputed_arcplot(
    default_system: Optional[str] = None,
    *,
    figure_size: Tuple[float, float] = _DEFAULT_FIGURE_SIZE,
    figure_dpi: float = _DEFAULT_FIGURE_DPI,
) -> None:
    """Convenience wrapper to display the widget in IPython sessions."""
    create_precomputed_arcplot_widget(
        default_system=default_system,
        display_widget=True,
        figure_size=figure_size,
        figure_dpi=figure_dpi,
    )
if widgets is not None:
    try:  # pragma: no cover - notebook detection
        __IPYTHON__  # type: ignore  # noqa: F821
        # Automatically expose the widget when run in an interactive notebook context.
        launch_precomputed_arcplot()
    except NameError:
        # Not running inside IPython; skip auto-launch.
        pass

# %%
