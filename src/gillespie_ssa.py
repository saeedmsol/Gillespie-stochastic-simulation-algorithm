import torch
import numpy as np
from typing import Union

class GillespieSSA:
    def __init__(self, 
                 initial_population: np.ndarray, 
                 rxn_matrices: np.ndarray, 
                 reaction_durations: np.ndarray, 
                 n_experiments: int, 
                 max_time: float, 
                 max_population: int, 
                 device: str = 'cpu'):
        """
        Initializes the GillespieSSA simulation.

        Args:
            initial_population (np.ndarray): Initial population distribution across bins.
            rxn_matrices (np.ndarray): Reaction rate matrices for all regimes (shape: [n_bins, n_rxn_channels, n_regimes]).
            reaction_durations (np.ndarray): End times for each reaction regime.
            n_experiments (int): Number of experiments to simulate in parallel.
            max_time (float): Maximum simulation time.
            max_population (int): Maximum allowed population before stopping the simulation.
            device (str): The computation device ('cpu' or 'cuda').
        """
        self.population = torch.tensor(initial_population[:, None], dtype=torch.int32, device=device)
        self.population = self.population.repeat(1, n_experiments).to(device=device)
        self.total_population = torch.sum(self.population, dim=0, dtype=torch.int64)
        self.n_experiments = n_experiments
        self.max_time = max_time
        self.max_population = max_population
        self.device = device

        self.current_time = torch.zeros(n_experiments, dtype=torch.float32, device=device)
        self.running_experiment = torch.ones(n_experiments, dtype=torch.bool, device=device)
        self.regime_endtimes = torch.tensor(reaction_durations, dtype=torch.float32, device=device)

        self.n_bins, self.n_rxn_channels, self.n_regimes = rxn_matrices.shape
        self.rxn_mat_full = torch.tensor(rxn_matrices, device=device, dtype=torch.float32)
        self.n_timepoints = 1 + self.n_regimes

        self.trajectory = torch.zeros(self.n_bins, n_experiments, self.n_timepoints, dtype=torch.int32, device=device)
        self.trajectory[:, :, 0] = self.population

        # Initialize other needed tensors for Gillespie
        self.rxn_mat_current = torch.zeros(self.n_bins, self.n_rxn_channels, self.n_experiments, device=device)
        self.regime_index_mat = torch.zeros(n_experiments, dtype=torch.int64, device=device)

        self.propensity_mat = torch.zeros(self.n_bins, self.n_rxn_channels, self.n_experiments, device=device)
        self.tot_propensity = torch.zeros(n_experiments, dtype=torch.double, device=device)

        self.tau = torch.zeros(n_experiments, dtype=torch.float32, device=device)  # Time for each experiment
        self.rng = torch.zeros(1000, n_experiments, device=device, dtype=torch.double)  # Random numbers
        self.rng_counter = 999  # Start at 999 to refill rng in the first step

    def perform_gillespie_step(self) -> None:
        """
        Performs one Gillespie step by calculating the reaction time, selecting reactions, and updating populations.

        Args:
            None

        Returns:
            None
        """
        # Step 1: Check if random numbers need to be replenished
        if self.rng_counter >= 999:
            self.rng = torch.rand(1000, self.n_experiments, device=self.device)
            self.rng_counter = 0

        # Step 2: Update the reaction regime based on current time
        self.update_reaction_regime()

        # Step 3: Compute propensities based on current populations and reaction rates
        self.compute_propensities()

        # Step 4: Calculate time to next reaction for each experiment
        self.calculate_reaction_times()

        # Step 5: Select reactions and update populations
        self.select_and_apply_reactions()

        # Step 6: Update the running status of experiments
        self.update_experiment_status()

        # Increment random number generator counter
        self.rng_counter += 1

    def update_reaction_regime(self) -> None:
        """
        Update the current reaction regime based on current time for each experiment.

        Args:
            None

        Returns:
            None
        """
        rxn_regime_mat = torch.lt(self.current_time[:, None], self.regime_endtimes[None, :])
        torch.argmax(rxn_regime_mat, dim=1, out=self.regime_index_mat)
        torch.index_select(self.rxn_mat_full, dim=2, index=self.regime_index_mat, out=self.rxn_mat_current)

    def compute_propensities(self) -> None:
        """
        Calculate the propensities for each reaction channel in each bin for all experiments.
        """
        torch.mul(self.rxn_mat_current, self.population[:, None, :], out=self.propensity_mat)
        torch.sum(self.propensity_mat, dim=(0, 1), out=self.tot_propensity)

    def calculate_reaction_times(self) -> None:
        """
        Calculate the time to the next reaction for each experiment using random numbers.
        """
        log_rng = torch.log(1 / self.rng[self.rng_counter, :])
        self.tau = log_rng / self.tot_propensity

    def select_and_apply_reactions(self) -> None:
        """
        Select which reaction occurs and update the population accordingly.
        """
        # Cumulative sum of propensities for each reaction in each experiment
        cumsum_propensity = torch.cumsum(self.propensity_mat.view(-1, self.n_experiments), dim=0)

        # Random selection of reaction based on weighted propensities
        rank_propensity = self.rng[self.rng_counter, :] * self.tot_propensity
        gt_cumsum_propensity = torch.lt(cumsum_propensity, rank_propensity)

        # Get the index of the selected reaction
        rxn_combined_index = torch.sum(gt_cumsum_propensity, dim=0)

        # Convert combined index to bin and reaction type
        rxn_bin_index = torch.div(rxn_combined_index, self.n_rxn_channels, rounding_mode='floor')
        operation_index = rxn_combined_index % self.n_rxn_channels

        # Determine the action (birth, death, left-mutation, right-mutation)
        self.update_population(rxn_bin_index, operation_index)

    def update_population(self, rxn_bin_index: torch.Tensor, operation_index: torch.Tensor) -> None:
        """
        Update population based on selected reaction type (birth, death, mutation).

        Args:
            rxn_bin_index (torch.Tensor): The index of the bin where the reaction occurs.
            operation_index (torch.Tensor): The type of reaction (0 for birth, 1 for death, 2 for left mutation, 3 for right mutation).

        Returns:
            None
        """
        birth_experiment_bool = operation_index == 0
        death_experiment_bool = operation_index == 1
        left_experiment_bool = operation_index == 2
        right_experiment_bool = operation_index == 3

        # Adjust population based on reactions
        increase_index = rxn_bin_index + right_experiment_bool.to(self.device).long()
        decrease_index = rxn_bin_index - left_experiment_bool.to(self.device).long()

        # Update population for birth, death, and mutation events
        self.population.scatter_add_(0, rxn_bin_index.view(1, -1),
                                     (-1 * death_experiment_bool + birth_experiment_bool).long())
        self.population.scatter_add_(0, increase_index.view(1, -1), right_experiment_bool.long())
        self.population.scatter_add_(0, decrease_index.view(1, -1), left_experiment_bool.long())

    def update_experiment_status(self) -> None:
        """
        Check for completion of experiments (extinction or maximum population) and update the running status.
        """
        self.total_population = torch.sum(self.population, dim=0)
        
        # Handle extinction (population goes to 0)
        extinction_condition = self.total_population == 0
        self.running_experiment[extinction_condition] = False

        # Handle reaching max population
        max_population_condition = self.total_population >= self.max_population
        self.running_experiment[max_population_condition] = False

        # Update time and set time limit for running experiments
        self.current_time += self.tau * self.running_experiment.float()

        # Check if max_time is reached
        time_limit_condition = self.current_time >= self.max_time
        self.running_experiment[time_limit_condition] = False
