# Affinity Maturation Simulation Code for `Minimal framework for optimizing vaccination protocols targeting highly mutable pathogens` (Phys Rev E 110, 064137 (2024))

This repository contains the Python code used to perform the simulations presented in the publication:

**Mahdisoltani, Saeed, et al. "Minimal framework for optimizing vaccination protocols targeting highly mutable pathogens." Physical Review E 110.6 (2024): 064137.
DOI: https://doi.org/10.1103/PhysRevE.110.064137

## Overview

The code simulates a minimal birth-mutation-death model for the affinity maturation of B cells in germinal centers. It focuses on protocols aimed at generating broadly neutralizing antibodies (bnAbs) against mutable pathogens. The model employs a shape-space representation for B cell receptors (BCRs) and antigens, tracking the population dynamics through stochastic simulations.

Key processes modeled include:
*   **B Cell Replication:** Driven by a fitness landscape determined by the vaccine protocol (antigen presentation), which varies over time.
*   **Somatic Hypermutation:** Modeled as diffusion in the shape-space.
*   **Apoptosis:** Baseline B cell death rate.
*   **Selection:** Implicitly occurs through affinity-dependent replication rates.

The simulations are implemented using the Gillespie Stochastic Simulation Algorithm (SSA), accelerated with PyTorch, allowing for efficient parallel execution of multiple replicates.

## Code Structure

This repository contains the following main components:

*   **`simulation.ipynb`:** A Jupyter Notebook serving as the main interface. It allows users to:
    *   Configure simulation parameters (loading defaults from `config.py` and setting specific ranges for sweeps).
    *   Define and execute simulation loops corresponding to different vaccination strategies (e.g., varying initial conditions, fitness landscape center/width strategies, as analyzed in the paper's Figures 4 and 5).
    *   Call the core simulation functions from `funcs.py`.
    *   Save detailed simulation results (final population states, full trajectories, run parameters) using Python's pickle format.
    *   Load saved results into a Pandas DataFrame for analysis.
    *   Generate key plots (e.g., equivalents of Figs 4 and 5 from the paper) directly within the notebook.
*   **`funcs.py`:** A Python module containing the core simulation functions:
    *   `initial_pop_gaussian`: Generates the initial B cell population distribution (discrete Gaussian).
    *   `gauss_reaction_matrix`: Constructs the time-dependent reaction rate tensor based on specified fitness landscape parameters (center, width, strength) and demographic rates (death, mutation), implementing the alternating birth vs. death/mutation regimes.
    *   `ssa`: Implements the Gillespie SSA using PyTorch, handling stochastic events, time evolution across regimes, population caps, and extinction. This version uses full tensors and masking for operations.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Jupyter Notebook or Jupyter Lab

### Dependencies

The following Python libraries are required:
*   NumPy
*   PyTorch (CPU version is sufficient, GPU optional)
*   SciPy (for `scipy.stats.norm`)
*   Matplotlib (for plotting)
*   Pandas (for data analysis)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/saeedmsol/Gillespie-stochastic-simulation-algorithm.git
    cd [repository-directory]
    ```
2.  **Create Environment (Recommended):** Create a virtual environment (e.g., using `conda` or `venv`) to manage dependencies.
    ```bash
    # Example using conda
    conda create -n affinity_sim python=3.9
    conda activate affinity_sim
    ```
3.  **Install Dependencies:**
    ```bash
    pip install numpy torch scipy matplotlib pandas jupyterlab # Or jupyter
    ```
    *(Note: Ensure PyTorch installation command matches your system/CUDA requirements if using GPU. See [pytorch.org](https://pytorch.org/))*
4.  **Ensure `config.py` is present:** Place the `config.py` file defining `SimulationParams` in the same directory as `simulation.ipynb` and `funcs.py`.

### How to Run

1.  **Launch Jupyter:**
    ```bash
    jupyter lab simulation.ipynb
    # or
    # jupyter notebook simulation.ipynb
    ```
2.  **Run Notebook Cells:** Execute the cells in the notebook sequentially from top to bottom.
    *   The initial cells set up imports and parameters.
    *   The simulation loop cells (marked "Running Simulation Loops") will execute the simulations. **This may take significant time** depending on the number of parameter combinations, `num_simulations` set in `config.py`, and `n_iterations`. Progress logs will be printed.
    *   Subsequent cells load the generated data and produce plots.

## Output

The notebook will create an output directory named `simulation_results_original_YYYYMMDD_HHMMSS` (timestamped) in the same location as the notebook. This directory contains:

*   **`/pickles/`:** Contains `.pkl` files, one for each simulation set executed. Each file includes:
    *   `'final_population'`: PyTorch tensor (CPU) of the population state at the end of each replicate.
    *   `'trajectory'`: PyTorch tensor (CPU) of the population state recorded at t=0 and after each simulation regime boundary.
    *   `'run_parameters'`: A dictionary containing all parameters specific to that simulation run (initial conditions, strategy names, rates, etc.).
*   **`/plots/`:** Contains `.png` files for the generated plots (e.g., Fig 4 panels, Fig 5).
*   **`analysis_summary_original.csv`:** A CSV file summarizing the key parameters and the calculated mean/std deviation of the final bnAb count (population in the target bin x=0) for each simulation set.

## Key Parameters

Most simulation parameters are defined in the `SimulationParams` dataclass within `config.py`. Key parameters influencing the simulation include:

*   `L`: Half-range of the shape-space grid.
*   `max_time`: Physical duration (`tf`) of the affinity maturation process being simulated.
*   `n_iterations`: Number of birth/death+mutation cycles within `max_time`.
*   `lambda_death`: Baseline B cell death rate.
*   `mu_total`: Total mutation rate (determining diffusion).
*   `q_mutation_bias`: Bias in mutation direction (0.5 for symmetric).
*   `pop0_center_coord`, `pop0_sigma`, `pop0_total_N`: Initial germline population properties.
*   `F_total_t`: Strength of the fitness landscape (total birth rate potential).
*   `min_fitness_width`: Minimum allowed width (`sigma_min`) for fitness strategies.
*   `num_simulations`: Number of stochastic replicates per parameter set.
*   `max_population_cap`: Simulation cap for total B cell population.
*   `device`: `'cpu'` or `'cuda'` for PyTorch execution.

The ranges for parameters swept in `simulation.ipynb` (like `F_tot_list`, `pop0_center_coord_list`, etc.) can be modified directly in the notebook cells.

## Implementation Details

*   **Gillespie Algorithm:** The core stochastic simulation uses the Gillespie SSA, implemented to handle time-dependent reaction rates by switching between pre-calculated rate matrices for different regimes.
*   **PyTorch:** Used for efficient tensor operations, enabling parallel simulation of many replicates. The SSA implementation provided (`funcs.py`) uses full-tensor operations with boolean masking.
*   **Fitness Strategies:** The notebook calculates time-dependent fitness center (`centers_arr`) and width (`radii_arr`) trajectories based on formulas derived from mean-field analysis in the associated paper (e.g., `x*`, BCH width). These arrays are passed to `gauss_reaction_matrix`.

## Limitations

*   The model is a simplification of the complex germinal center reaction (e.g., no explicit T cell help, antigen decay/presentation dynamics, continuous B cell entry).
*   Mutations are modeled as unbiased diffusion (unless `q_mutation_bias` != 0.5), potentially omitting sequence-specific biases.
*   The fitness landscape is assumed to be representable by a time-varying Gaussian profile.

## License

This project is licensed under the MIT License 

## Citation

If you use this code in your research or software, please cite the following publication:

@article{PhysRevE.110.064137,
  title = {Minimal framework for optimizing vaccination protocols targeting highly mutable pathogens},
  author = {Mahdisoltani, Saeed and Murugan, Pranav and Chakraborty, Arup K. and Kardar, Mehran},
  journal = {Phys. Rev. E},
  volume = {110},
  issue = {6},
  pages = {064137},
  numpages = {18},
  year = {2024},
  month = {Dec},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.110.064137},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.110.064137}
}