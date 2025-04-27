# --- funcs.py ---
"""
Contains the implementations of simulation functions:
- initial_pop_gaussian: Generates initial population.
- gauss_reaction_matrix: Generates reaction rate tensor.
- ssa: Performs the Gillespie SSA using full tensors and masking.
"""
import logging
import time
from typing import Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import norm
from torch import Tensor

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Initial Condition Generation
# --------------------------------------------------------------------------

def gaussian_discrete (x: NDArray[np.int_], area: int, mean: int, sigma: float) -> NDArray[np.int_]:
    """
    Creates a discrete approximation of a Gaussian distribution over integer bins.
    Uses rounding and a single central correction. 

    Args:
        x: 1D array of bin indices.
        area: Desired total population (sum of returned counts).
        mean: Bin index corresponding to the mean of the Gaussian.
        sigma: Standard deviation of the Gaussian in index units.

    Returns:
        1D integer array of counts per bin summing to area.
    """
    if area <= 0: return np.zeros_like(x, dtype=int)
    if sigma <= 1e-9: # Handle near-zero sigma
        result = np.zeros_like(x, dtype=int)
        if 0 <= mean < len(x):
            result[mean] = area
        return result

    prob_mass = norm.pdf(x, mean, sigma)
    pdf_sum = np.sum(prob_mass)
    if pdf_sum < 1e-12: # Prevent division by zero
        result = np.zeros_like(x, dtype=int)
        if 0 <= mean < len(x):
            result[mean] = area
        return result

    normalization_factor = area / pdf_sum
    prob_mass_normalized = prob_mass * normalization_factor
    result = np.round(prob_mass_normalized).astype(int) # Round the probabilities
    current_sum = np.sum(result)
    diff = area - current_sum
    # Add any difference to the center bin (ensure index is valid)
    if 0 <= mean < len(result):
        result[mean] += diff
        # Ensure counts don't go negative after correction (unlikely but possible)
        result[mean] = max(0, result[mean])
    elif diff != 0:
        logger.warning("Mean index %d out of bounds, cannot apply rounding correction of %d", mean, diff)

    # Final check/correction if previous step failed or made center negative
    final_sum = np.sum(result)
    if final_sum != area:
         logger.warning("Final population sum %d differs from target %d after central correction. Attempting fallback.", final_sum, area)
         # Fallback: assign difference to first available non-zero bin or center if possible
         correction = area - final_sum
         if 0 <= mean < len(result):
             result[mean] += correction
             result[mean] = max(0, result[mean])
         else: # If mean invalid, try adding to first bin
              result[0] += correction
              result[0] = max(0, result[0])
         # Check one last time
         if np.sum(result) != area:
              logger.error("FATAL: Could not match target population sum %d. Final sum: %d", area, np.sum(result))


    return result

def initial_pop_gaussian (n_bins: int, tot_pop_t0: int, mean_t0_idx: int, sigma_t0: float) -> NDArray[np.int_]:
    """Generates initial population using gaussian_discrete."""
    logger.debug("Generating initial population: N=%d, mean_idx=%d, sigma=%.2f",
                 tot_pop_t0, mean_t0_idx, sigma_t0)
    return gaussian_discrete(np.arange(n_bins), tot_pop_t0, mean_t0_idx, sigma_t0)

# --------------------------------------------------------------------------
# Reaction Matrix Generation 
# --------------------------------------------------------------------------

def gauss_reaction_matrix (n_bins: int, n_iterations: int,
                           centers_xarr: NDArray[np.float_], # Note: formula uses L+coord. Okay.
                           radii_arr: NDArray[np.float_],
                           F_tot_arr: NDArray[np.float_], # Fitness strength per iteration
                           rho: float, # Death rate
                           mu_tot: float, # Total mutation rate
                           q: float # Mutation bias
                           ) -> NDArray[np.float_]:
    """
    Generates the reaction rate tensor 
    (internal functions, column_stack, repeat).

    Args:
        n_bins: Number of bins.
        n_iterations: Number of birth/death cycles.
        centers_xarr: Array (size n_iterations) of center locations (as *indices*).
        radii_arr: Array (size n_iterations) of fitness widths (sigma_v).
        F_tot_arr: Array (size n_iterations) of total fitness strength per iteration.
        rho: Baseline death rate (lambda).
        mu_tot: Total mutation rate (mu).
        q: Mutation bias towards center (0.5 = symmetric).

    Returns:
        Reaction tensor shape (n_bins, 4, n_regimes), dtype=float.
    """
    logger.debug("Generating reaction matrix: %d iterations", n_iterations)
    D = mu_tot/2.0 # Not explicitly used inside
    x_arr = np.arange(n_bins)

    # Input validation assertions 
    assert centers_xarr.size == n_iterations, f"Center array size {centers_xarr.size} != n_iterations {n_iterations}"
    assert radii_arr.size == n_iterations, f"Radii array size {radii_arr.size} != n_iterations {n_iterations}"
    assert F_tot_arr.size == n_iterations, f"Fitness array size {F_tot_arr.size} != n_iterations {n_iterations}"

    # --- Internal Helper Functions  ---
    def birth_gaussian (F_tot: float, center_idx: float, sigma: float) -> NDArray[np.float_]:
        if sigma <= 1e-9: # Handle zero width
             rates = np.zeros_like(x_arr, dtype=float)
             center_int = int(round(center_idx))
             if 0 <= center_int < n_bins: rates[center_int] = F_tot
             return rates
        g = norm.pdf(x_arr, center_idx, sigma)
        g_sum = np.sum(g)
        if g_sum < 1e-12: return np.zeros_like(x_arr, dtype=float)
        return F_tot * g / g_sum

    def death(n_bins_local: int, rho_local: float) -> NDArray[np.float_]:
        return np.full(n_bins_local, rho_local)

    def left(n_bins_local: int, q_local: float, mu_local: float) -> NDArray[np.float_]:
        mid = n_bins_local // 2
        left_vec = np.zeros(n_bins_local, dtype=float)
        if mid > 0: # Ensure slices are valid
             left_vec[1:mid] = q_local * mu_local
        if 0 <= mid < n_bins_local:
             left_vec[mid] = mu_local / 2.0
        if mid + 1 < n_bins_local:
             left_vec[mid+1:] = (1.0 - q_local) * mu_local
        # Boundary condition: left[0] is implicitly 0
        return left_vec

    def right(n_bins_local: int, q_local: float, mu_local: float) -> NDArray[np.float_]:
        mid = n_bins_local // 2
        right_vec = np.zeros(n_bins_local, dtype=float)
        if mid > 0:
             right_vec[:mid] = (1.0 - q_local) * mu_local
        if 0 <= mid < n_bins_local:
             right_vec[mid] = mu_local / 2.0
        if mid + 1 < n_bins_local -1: # Exclude last bin for right mutation source
             right_vec[mid + 1:-1] = q_local * mu_local
        # Boundary condition: right[-1] is implicitly 0
        return right_vec

    # --- Assemble Arrays  ---
    birth_rates_per_iter = [birth_gaussian(F_tot_arr[i], centers_xarr[i], radii_arr[i])
                            for i in range(n_iterations)]
    birth_array = np.repeat(np.column_stack(birth_rates_per_iter), 2, axis=1)
    birth_array[:, 1::2] = 0 # Zero out odd columns (death/mutation regime)

    death_rates_per_iter = [death(n_bins, rho) for _ in range(n_iterations)]
    death_array = np.repeat(np.column_stack(death_rates_per_iter), 2, axis=1)
    death_array[:, 0::2] = 0 # Zero out even columns (birth regime)

    left_rates_per_iter = [left(n_bins, q, mu_tot) for _ in range(n_iterations)]
    left_array = np.repeat(np.column_stack(left_rates_per_iter), 2, axis=1)
    left_array[:, 0::2] = 0 # Zero out even columns (birth regime)

    right_rates_per_iter = [right(n_bins, q, mu_tot) for _ in range(n_iterations)]
    right_array = np.repeat(np.column_stack(right_rates_per_iter), 2, axis=1)
    right_array[:, 0::2] = 0 # Zero out even columns (birth regime)

    # --- Concatenate into Final Tensor ---
    # Stack along a new axis (axis=1) to create the channels dimension
    reaction_tensor = np.stack(
        (birth_array, death_array, left_array, right_array),
        axis=1
    )
    # could np.concatenate with [:, None], which achieves the same shape.
    # np.stack is slightly more direct here. Shape: (n_bins, 4, n_regimes)

    logger.debug("Generated reaction matrix with shape %s", reaction_tensor.shape)
    return reaction_tensor

# --------------------------------------------------------------------------
# Gillespie SSA 
# --------------------------------------------------------------------------

def ssa (pop_t0: NDArray[np.int_], # Expects 1D initial pop
         rxn_matrices: NDArray[np.float_], # Shape (n_bins, 4, n_regimes)
         reaction_durations: NDArray[np.float_], # Cumulative end times of regimes
         n_experiments: int,
         max_sim_time: float, # Absolute max time cutoff
         max_population: int,
         device: str
         ) -> Tuple[Tensor, Tensor]:
    """
    Performs the Gillespie SSA:
    - Full tensor operations with masking.
    - int32 population type.
    - Specific edge case handling for reaction selection.
    - No explicit population clamping (relies on scatter logic).
    - Returns PyTorch Tensors.

    Args:
        pop_t0: 1D NumPy array for initial population state.
        rxn_matrices: 3D NumPy array of reaction rates.
        reaction_durations: 1D NumPy array of cumulative regime end times.
        n_experiments: Number of simulation replicates.
        max_sim_time: Absolute simulation time limit.
        max_population: Population cap.
        device: PyTorch device ('cpu' or 'cuda').

    Returns:
        Tuple (final_population_tensor, trajectory_tensor).
    """
    ssa_start_time = time.time()
    logger.info("Starting SSA...")
    dev = torch.device(device) # Use shorter name

    # --- Initialize population tensors ---
    if pop_t0.ndim != 1: raise ValueError("pop_t0 must be a 1D array")
    initial_population = torch.tensor(pop_t0[:, None], dtype=torch.int32, device=dev) 
    population = initial_population.repeat(1, n_experiments)
    tot_population = torch.sum(population, dim=0, dtype=torch.int64,) # Use int64 for sum

    # --- Time and control tensors ---
    current_time = torch.zeros(n_experiments, dtype=torch.float32, device=dev)
    running_experiment_bool = torch.ones(n_experiments, dtype=torch.bool, device=dev)
    regime_endtimes = torch.tensor(reaction_durations, dtype=torch.float32, device=dev)
    current_regime_exp_bool = torch.ones(n_experiments, dtype=torch.bool, device=dev) # Tracks if reaction happens *before* boundary
    # experiment_index is used as a temporary boolean mask, renamed tmp_mask for clarity
    tmp_mask = torch.zeros(n_experiments, dtype=torch.bool, device=dev)

    # --- Reaction matrix tensor ---
    rxn_mat_full = torch.tensor(rxn_matrices, dtype=torch.float32, device=dev)
    n_bins = rxn_mat_full.size(0)
    n_rxn_channels = rxn_mat_full.size(1) # Should be 4
    n_regimes = rxn_mat_full.size(2)
    n_timepoints = 1 + n_regimes
    assert n_rxn_channels == 4, "Reaction matrix must have 4 channels"

    # --- Workspace tensors (full size) ---
    rxn_mat_current = torch.zeros(n_bins, n_rxn_channels, n_experiments, dtype=torch.float32, device=dev)
    rxn_regime_mat = torch.zeros(n_experiments, n_regimes, dtype=torch.bool, device=dev) # Use bool for clarity
    regime_index_mat = torch.zeros(n_experiments, dtype=torch.long, device=dev) # Use long for indexing

    # --- Trajectory recorder ---
    trajectory = torch.zeros((n_bins, n_experiments, n_timepoints), dtype=torch.int32, device=dev)
    trajectory[:, :, 0] = population.clone() # Record initial state

    # --- Propensity tensors ---
    propensity_mat = torch.zeros_like(rxn_mat_current, dtype=torch.float64, device=dev) # Use float64
    cumsum_propensity = torch.zeros(n_bins*n_rxn_channels, n_experiments, dtype=torch.float64, device=dev)
    tot_propensity = torch.zeros(n_experiments, dtype=torch.float64, device=dev)

    # --- Reaction selection tensors ---
    rank_propensity = torch.zeros(n_experiments, dtype=torch.double, device=dev)
    # gt_cumsum_propensity = torch.zeros(n_bins * n_rxn_channels, n_experiments, dtype=torch.int16, device=dev) # int16 seems too small
    gt_cumsum_propensity = torch.zeros(n_bins * n_rxn_channels, n_experiments, dtype=torch.bool, device=dev) # Use bool
    # rxn_combined_index = torch.zeros(n_experiments, dtype=torch.int16, device=dev) # Use long for index
    rxn_combined_index = torch.zeros(n_experiments, dtype=torch.long, device=dev)
    rxn_bin_index   = torch.zeros((1, n_experiments), dtype=torch.long, device=dev) # Use long
    increase_index  = torch.zeros((1, n_experiments), dtype=torch.long, device=dev)
    operation_index = torch.zeros((1, n_experiments), dtype=torch.long, device=dev)

    # Reaction type bools (match population dtype for scatter)
    birth_experiment_bool = torch.zeros_like(rxn_bin_index, dtype=population.dtype, device=dev)
    death_experiment_bool = torch.zeros_like(rxn_bin_index, dtype=population.dtype, device=dev)
    left_experiment_bool  = torch.zeros_like(rxn_bin_index, dtype=population.dtype, device=dev)
    right_experiment_bool = torch.zeros_like(rxn_bin_index, dtype=population.dtype, device=dev)

    # Edge case handling tensors
    edgeCase_index = torch.zeros(n_experiments, dtype=torch.bool, device=dev)
    exp_range = torch.arange(n_experiments, dtype=torch.long, device=dev)

    # --- RNG tensors ---
    n_rng = 1000
    log_rng = torch.zeros(n_experiments, dtype=torch.float64, device=dev) # Use float64
    tau = torch.zeros(n_experiments, dtype=torch.float64, device=dev) # Use float64
    rng = torch.zeros(n_rng, n_experiments, dtype=torch.double, device=dev)

    # --- Counters ---
    rng_counter = n_rng - 1
    gillespie_step = 0

    # --- Main Loop ---
    logger.info("Starting SSA loop...")
    while True:
        # --- Refresh RNG Buffer ---
        if rng_counter > n_rng - 2: # Need two numbers per step (time, reaction)
            torch.rand(n_rng, n_experiments, device=dev, dtype=torch.double, out=rng)
            rng_counter = 0

        gillespie_step += 1

        # --- Determine current regime and rates ---
        torch.lt(current_time[:, None], regime_endtimes[None, :], out=rxn_regime_mat)
        torch.argmax(rxn_regime_mat.int(), dim=1, out=regime_index_mat) # Convert bool->int for argmax
        torch.index_select(rxn_mat_full, dim=2, index=regime_index_mat, out=rxn_mat_current)

        # --- Calculate propensities ---
        torch.mul(rxn_mat_current, population[:, None, :].to(rxn_mat_current.dtype), out=propensity_mat) # Match dtypes
        propensity_mat = propensity_mat.to(torch.float64) # Convert back to float64
        torch.sum(propensity_mat, dim=(0, 1), out=tot_propensity)

        # Add 1 to propensity of inactive simulations
        tot_propensity_safe = tot_propensity + (~running_experiment_bool).to(tot_propensity.dtype)
        # Ensure non-zero for division
        tot_propensity_safe = torch.clamp(tot_propensity_safe, min=1e-30)


        # --- Sample tau ---
        # Avoid log(0) or log(negative)
        current_rng_time = torch.clamp(rng[rng_counter, :], min=1e-30)
        torch.log(1.0 / current_rng_time, out=log_rng)
        torch.div(log_rng, tot_propensity_safe, out=tau)
        rng_counter += 1

        # --- Check for boundary crossing ---
        potential_next_time = current_time + tau.to(current_time.dtype) * running_experiment_bool # Apply mask
        torch.ge(potential_next_time, regime_endtimes[regime_index_mat], out=tmp_mask)
        # tmp_mask is True if next reaction >= boundary time (or if inactive)

        # Record trajectory if boundary hit by an *active* simulation
        boundary_hit_active_mask = tmp_mask & running_experiment_bool
        if torch.any(boundary_hit_active_mask):
            boundary_indices = torch.where(boundary_hit_active_mask)[0]
            regimes_ended = regime_index_mat[boundary_indices]
            timepoint_to_record = regimes_ended + 1 # 0 is initial state
            # Ensure indices are valid before assignment
            valid_tp_mask = timepoint_to_record < trajectory.shape[2]
            if torch.any(valid_tp_mask):
                 valid_boundary_indices = boundary_indices[valid_tp_mask]
                 valid_timepoint_idx = timepoint_to_record[valid_tp_mask]
                 trajectory[:, valid_boundary_indices, valid_timepoint_idx] = population[:, valid_boundary_indices].clone()

        # Update current_regime_exp_bool: True if reaction happens before boundary
        current_regime_exp_bool = ~tmp_mask

        # --- Update time and check for max_time ---
        current_time = torch.min(potential_next_time, regime_endtimes[regime_index_mat])
        # Check if max_sim_time reached for active sims
        max_time_reached_mask = (current_time >= max_sim_time - 1e-6) & running_experiment_bool # Use tolerance
        if torch.any(max_time_reached_mask):
            running_experiment_bool[max_time_reached_mask] = False
            current_time[max_time_reached_mask] = max_sim_time # Set exactly to max_time

        # --- Check if all finished ---
        if (~running_experiment_bool).all():
            logger.info("All experiments finished at step %d.", gillespie_step)
            ssa_end_time = time.time()
            logger.info("SSA finished in %.2f seconds.", ssa_end_time - ssa_start_time)
            return (population, trajectory) # Return final population and trajectory

        # --- Skip reaction update if no active sim had a reaction before boundary ---
        if (~(running_experiment_bool & current_regime_exp_bool)).all():
            continue

        # --- Select Reaction ---
        prop_flat = propensity_mat.view(-1, n_experiments)
        torch.cumsum(prop_flat, dim=0, out=cumsum_propensity)
        current_rng_rxn = rng[rng_counter, :]
        torch.mul(current_rng_rxn, tot_propensity, out=rank_propensity) # Use tot_propensity here

        # using torch.lt + torch.sum
        torch.lt(cumsum_propensity, rank_propensity[None, :], out=gt_cumsum_propensity)
        torch.sum(gt_cumsum_propensity, dim=0, dtype=torch.long, out=rxn_combined_index)
        rng_counter += 1

        # --- Apply Edge Case Correction ---
        torch.ge(rxn_combined_index, n_bins * n_rxn_channels, out=edgeCase_index)
        if torch.any(edgeCase_index):
            logger.debug("Applying edge case fix at step %d", gillespie_step)
            # This finds the index of the last non-zero element when flipped
            delta = torch.argmax(
                (torch.flip(prop_flat[:, edgeCase_index], [0]) > 1e-12).int(), dim=0 # Use tolerance > 0
            )
            # Calculate the correct index: (n_bins*n_channels - 1) - delta
            corrected_index = (n_bins * n_rxn_channels - 1) - delta
            rxn_combined_index[edgeCase_index] = corrected_index.to(rxn_combined_index.dtype)


        # --- Convert reaction index to updates ---
        torch.floor_divide(rxn_combined_index[None, :], n_rxn_channels, out=rxn_bin_index)
        torch.remainder(rxn_combined_index[None, :], n_rxn_channels, out=operation_index)

        # Determine reaction types
        torch.eq(operation_index, 0, out=birth_experiment_bool)
        torch.eq(operation_index, 1, out=death_experiment_bool)
        torch.eq(operation_index, 2, out=left_experiment_bool)
        torch.eq(operation_index, 3, out=right_experiment_bool)

        # Determine target bin index for increase (clamp to handle boundaries)
        increase_index = torch.clamp(
            rxn_bin_index + right_experiment_bool.long() - left_experiment_bool.long(),
            0, n_bins - 1
        )

        # --- Apply scatter updates (masked) ---
        # Mask ensures updates only apply if sim is running AND reaction happened before boundary
        update_mask = (running_experiment_bool & current_regime_exp_bool).to(population.dtype)[None, :]

        # Decrement source bin
        population.scatter_add_(0, rxn_bin_index, -1 * update_mask)

        # Increment target bin (amount depends on reaction type)
        increment_amount = (1 + birth_experiment_bool - death_experiment_bool) * update_mask
        population.scatter_add_(0, increase_index, increment_amount)

        # --- Check extinction ---
        torch.sum(population, dim=0, out=tot_population)
        extinct_mask = (tot_population <= 0) & running_experiment_bool # Use <= 0 for safety
        if torch.any(extinct_mask):
            logger.debug("Extinction detected at step %d", gillespie_step)
            running_experiment_bool[extinct_mask] = False
            # Fill rest of trajectory with zeros for extinct simulations
            extinct_indices = torch.where(extinct_mask)[0]
            for exp_idx in extinct_indices:
                start_fill_tp = regime_index_mat[exp_idx] + 1
                if start_fill_tp < trajectory.shape[2]:
                     trajectory[:, exp_idx, start_fill_tp:] = 0

        # --- Check population cap ---
        escape_mask = (tot_population >= max_population) & running_experiment_bool
        if torch.any(escape_mask):
            logger.debug("Population cap hit at step %d", gillespie_step)
            running_experiment_bool[escape_mask] = False
            # Fill rest of trajectory with final state for capped simulations
            escape_indices = torch.where(escape_mask)[0]
            for exp_idx in escape_indices:
                start_fill_tp = regime_index_mat[exp_idx] + 1
                final_state_escape = population[:, exp_idx].clone() # Get state when cap was hit
                if start_fill_tp < trajectory.shape[2]:
                     trajectory[:, exp_idx, start_fill_tp:] = final_state_escape[:, None]


        # --- Final check if all finished ---
        if (~running_experiment_bool).all():
            logger.info("All experiments finished at step %d.", gillespie_step)
            ssa_end_time = time.time()
            logger.info("SSA finished in %.2f seconds.", ssa_end_time - ssa_start_time)
            return (population, trajectory) # Return final population and trajectory

        # --- Periodic Logging ---
        if gillespie_step % 10000 == 0: # Log less frequently maybe
            n_active = torch.sum(running_experiment_bool).item()
            mean_t_active = torch.mean(current_time[running_experiment_bool]).item() if n_active > 0 else max_sim_time
            logger.debug("Step: %d, Active: %d, Mean Time Active: %.2f", gillespie_step, n_active, mean_t_active)