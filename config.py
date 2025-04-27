# --- config.py ---
"""
config.py

Defines simulation parameters for the Gillespie-based affinity maturation model,
including physical, demographic, mutation, and fitness-landscape settings.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Type alias for functions that generate landscape parameters over time
# Arguments: time array/float, center value, width value, amplitude value
LandscapeFunc = Callable[[Union[np.ndarray, float], float, float, float], Union[np.ndarray, float]]

# -----------------------------------------------------------------------------
# Default Time-Dependent Functions (Constant Strategies)
# -----------------------------------------------------------------------------

def _const_center_func(
    t: Union[np.ndarray, float],
    center_value: float = 0.0,
) -> Union[np.ndarray, float]:
    """
    Returns a constant center value, ignoring extra keyword args.
    """
    if isinstance(t, np.ndarray):
        return np.full_like(t, center_value, dtype=float)
    return float(center_value)


def _const_width_func(
    t: Union[np.ndarray, float],
    width_value: float = 2.0,  # Updated default to match SimulationParams
    min_width: float = 1.0,
) -> Union[np.ndarray, float]:
    """
    Returns a constant width value, enforcing a minimum.

    Args:
        t: Time point(s).
        width_value: The constant base value for the width.
        min_width: The minimum allowed width.

    Returns:
        Constant width value (scalar or array matching t), >= min_width.
    """
    value = max(width_value, min_width)
    if isinstance(t, np.ndarray):
        return np.full_like(t, value, dtype=float)
    return float(value)

# -----------------------------------------------------------------------------
# Simulation Parameters Dataclass
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class SimulationParams:
    """
    Container for all input and derived parameters for the affinity maturation simulation.

    Uses standardized parameter names consistent across the pipeline.
    """

    # -- Physical & Temporal Resolution --
    L: int = 21  #: Half-range of the affinity grid (total grid size = 2*L + 1).
    max_time: float = 400.0  #: Total physical time to simulate (tf).
    n_iterations: int = 200  #: Number of iterations (each with birth -> death+mutation regime).

    # -- Demographic Rates --
    lambda_death: float = 1e-3  #: Per-cell baseline death rate (apoptosis, λ).
    # Note: Effective birth rate is driven by fitness V(x,t) via F_total_t.
    # A baseline birth rate (rho) seems unused in the primary fitness mechanism.

    # -- Mutation Dynamics --
    mu_total: float = 0.1  #: Total mutation rate (mu), driving diffusion D = mu_total / 2.
    q_mutation_bias: float = 0.5  #: Bias in mutation direction (0.5 = symmetric, no drift).

    # -- Initial Population --
    pop0_center_coord: float = 8.0  #: Mean coordinate of the starting B cell population (mu_0).
    pop0_sigma: float = 1.0  #: Standard deviation of the starting germline distribution (sigma_0).
    pop0_total_N: int = 100  #: Total number of cells initially (N0).

    # -- Fitness Landscape (Vaccine Protocol) --
    F_total_t: float = 0.08  #: Total fitness strength (integrated birth rate) per iteration.
    min_fitness_width: float = 1.0  #: Minimum allowed fitness profile width (sigma_min).
    # Default constant values if time-dependent functions are not specified:
    center_value_default: float = 0.0  #: Default constant center if center_func is _const_center_func.
    width_value_default: float = 2.0  #: Default constant width if width_func is _const_width_func.
    # Functions defining the time-dependent landscape shape:
    center_func: LandscapeFunc = _const_center_func  #: Function V(t) -> center coordinate index.
    width_func: LandscapeFunc = _const_width_func  #: Function V(t) -> width (sigma_v).

    # -- Simulation Control --
    num_simulations: int = 10  #: Number of stochastic replicates per parameter set.
    max_population_cap: int = 5000  #: Cap on total population size across all bins.

    # -- Compute & Reproducibility --
    device: str = "cpu"  #: Computational device ("cpu" or "cuda").
    seed: int = 6317  #: Random seed for reproducibility.

    # -- I/O & Plotting --
    output_base_dir: Path = Path("simulation_results")  #: Base directory for saving outputs.
    param_file: Path = Path("parameters.pkl")  #: Standard filename for parameter serialization.
    reaction_matrices_file: Path = Path("reaction_matrices.npy")  #: Standard filename for reaction matrices.
    results_prefix: str = "ssa_results"  #: Prefix for simulation result files (e.g., ssa_results_strategy.pkl).
    analysis_tag: str = "analysis_results"  #: Tag for naming analysis outputs (e.g., analysis_results_dataframe.csv).
    plot_font_size_title: int = 18  #: Font size for plot titles.
    plot_font_size_label: int = 16  #: Font size for axis labels.
    plot_font_size_tick: int = 14  #: Font size for tick labels.
    plot_font_size_legend: int = 12  #: Font size for legends.
    plot_dpi: int = 150  #: Resolution for saved figures.

    # -- Derived Fields (computed post-init) --
    n_bins: int = field(init=False)  #: Number of affinity bins (2L + 1).
    pop0_center_idx: int = field(init=False)  #: Index corresponding to pop0_center_coord.
    diffusion_coefficient: float = field(init=False)  #: Effective diffusion D = mu_total / 2.
    n_regimes: int = field(init=False)  #: Total number of regimes (2 per iteration).
    regime_duration: float = field(init=False) #: Duration of a single regime (birth OR death/mut).
    dt_physical: float = field(init=False) #: Physical time duration of one full iteration (birth + death/mut). Used in analytics.
    regime_end_times: NDArray[np.float_] = field(init=False)  #: Cumulative end times of all regimes.
    simulation_duration: float = field(init=False)  #: Total simulation time across all regimes (should match max_time).

    def __post_init__(self) -> None:
        """Computes derived attributes after initialization."""
        # Grid setup
        self.n_bins = 2 * self.L + 1
        # Map coordinate mu_0 to grid index. Index L corresponds to coordinate 0.
        self.pop0_center_idx = self.L + int(round(self.pop0_center_coord))

        if not (0 <= self.pop0_center_idx < self.n_bins):
            logger.error(
                "Initial center index %d (from coord %.2f) is out of grid bounds [0, %d]. "
                "Check L and pop0_center_coord.",
                self.pop0_center_idx, self.pop0_center_coord, self.n_bins - 1
            )
            raise ValueError("Initial population center out of bounds.")

        # Mutation-driven diffusion
        self.diffusion_coefficient = self.mu_total / 2.0

        # Time regimes
        self.n_regimes = 2 * self.n_iterations
        if self.n_regimes <= 0:
             raise ValueError("n_iterations must be positive.")
        self.regime_duration = self.max_time / self.n_regimes # Duration of one regime
        self.dt_physical = self.max_time / self.n_iterations # Duration of one full iteration
        # Calculate cumulative end times for each regime
        self.regime_end_times = np.cumsum(np.full(self.n_regimes, self.regime_duration))
        self.simulation_duration = float(self.regime_end_times[-1])
        # Verify consistency
        if not np.isclose(self.simulation_duration, self.max_time):
             logger.warning(
                 "Calculated simulation duration (%.4f) differs slightly from max_time (%.4f)",
                 self.simulation_duration, self.max_time
             )

        # Validate mutation bias
        if not (0.0 <= self.q_mutation_bias <= 1.0):
            raise ValueError("q_mutation_bias must be between 0 and 1.")

        logger.info("Derived parameters calculated:")
        logger.info(f"  n_bins = {self.n_bins}")
        logger.info(f"  pop0_center_idx = {self.pop0_center_idx}")
        logger.info(f"  diffusion_coefficient D = {self.diffusion_coefficient:.4f}")
        logger.info(f"  n_regimes = {self.n_regimes}")
        logger.info(f"  regime_duration = {self.regime_duration:.4f}")
        logger.info(f"  dt_physical (iteration duration) = {self.dt_physical:.4f}")
        logger.info(f"  simulation_duration = {self.simulation_duration:.4f}")


    @classmethod
    def default(cls) -> SimulationParams:
        """Creates a new instance with default settings."""
        logger.info("Creating SimulationParams instance with default values.")
        return cls()

    def to_pickle(self, path: Union[str, Path]) -> None:
        """
        Serializes all *initializable* fields (non-callable attributes used
        to construct the object) to a pickle file. Excludes derived fields.

        Args:
            path: Destination filepath for the pickle file.
        """
        data_to_save: dict[str, any] = {}
        for f in fields(self):
            # Only save fields that are part of __init__
            if f.init:
                attr_value = getattr(self, f.name)
                # Exclude functions, save paths as strings for portability
                if callable(attr_value):
                     # Store function name for reference, won't be reloaded directly
                     data_to_save[f.name] = f"{attr_value.__module__}.{attr_value.__name__}"
                elif isinstance(attr_value, Path):
                    data_to_save[f.name] = str(attr_value)
                else:
                    data_to_save[f.name] = attr_value

        # Ensure output directory exists
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "wb") as fh:
                pickle.dump(data_to_save, fh)
            logger.info("Initializable parameters saved to %s", path)
        except Exception as e:
            logger.error("Failed to save parameters to %s: %s", path, e, exc_info=True)
            raise

    @classmethod
    def from_pickle(cls, path: Union[str, Path]) -> SimulationParams:
        """
        Loads simulation parameters from a pickle file created by `to_pickle`.

        Note: Time-dependent functions (`center_func`, `width_func`) are not
        restored from the pickle file (only their names are saved). The loaded
        instance will retain the default functions defined in this class.
        If custom functions were used, they need to be reassigned manually
        after loading, or the `SimulationParams` instance needs modification
        to handle function deserialization/reassignment based on saved names.

        Args:
            path: Path to the pickle file containing saved parameters.

        Returns:
            A SimulationParams instance initialized with loaded values.
        """
        path = Path(path)
        if not path.exists():
             logger.error("Parameter file not found: %s", path)
             raise FileNotFoundError(f"Parameter file not found: {path}")

        try:
            with open(path, "rb") as fh:
                saved_data = pickle.load(fh)
        except Exception as e:
            logger.error("Failed to load parameters from %s: %s", path, e, exc_info=True)
            raise

        # Filter saved data to only include fields defined in the class __init__
        valid_init_fields = {f.name for f in fields(cls) if f.init}
        init_kwargs = {}
        for key, value in saved_data.items():
            if key in valid_init_fields:
                 # Handle Path conversion back from string
                 field_type = cls.__annotations__.get(key, None)
                 if isinstance(field_type, type) and issubclass(field_type, Path):
                      init_kwargs[key] = Path(value)
                 # Skip loading functions - use defaults
                 elif key in ["center_func", "width_func"]:
                      logger.warning("Skipping loading function '%s' from pickle. Using class default.", key)
                      # Let dataclass handle default assignment
                      pass
                 else:
                      init_kwargs[key] = value
            else:
                 logger.warning("Ignoring unknown parameter '%s' found in %s.", key, path)

        try:
            # Create instance using filtered kwargs, allowing defaults for missing/skipped fields
            instance = cls(**init_kwargs)
            logger.info("Parameters loaded from %s", path)
            # __post_init__ is automatically called to compute derived fields.
            return instance
        except TypeError as e:
             logger.error("Error creating SimulationParams instance from loaded data: %s", e)
             logger.error("Loaded kwargs: %s", init_kwargs)
             raise


