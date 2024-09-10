import numpy as np
import torch
from scipy.stats import norm


############# initial condition and rxn arrays ######################

@np.vectorize
def gaussian_fn(i, mu, sigma):
    return np.exp(-1 * (i - mu) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.math.pi * sigma ** 2)

def gaussian_discrete (x, area, mean, sigma):
    prob_mass = norm.pdf(x, mean, sigma)
    normalization_factor = area / np.sum(prob_mass)
    prob_mass_normalized = prob_mass * normalization_factor
    result = np.round(prob_mass_normalized).astype(int)           # Round the probabilities to integers to mimic a discrete distribution
    result[mean] += area - np.sum(result)                         # add any differences between the desired area and the resulting one to the center
    return result  

def initial_pop_gaussian (n_bins, tot_pop_t0, mean_t0, sigma_t0):
    return gaussian_discrete(np.arange(n_bins), tot_pop_t0, mean_t0, sigma_t0)


def gauss_reaction_matrix (n_bins, n_iterations, centers_xarr, radii_arr, F_tot_arr, rho, mu_tot, q):

    D = mu_tot/2     # Diffusion cnst = (1/2) \int y^2 \mu_y
    
    assert centers_xarr.size == n_iterations            # centers are given as indexes
    assert radii_arr.size == n_iterations 
    
    def birth_gaussian (n_bins, F_tot, center_idx, sigma):  
        return F_tot * gaussian_fn(np.arange(n_bins), center_idx, sigma) / np.sum(gaussian_fn(np.arange(n_bins), center_idx, sigma))

    def death(n_bins, rho):
        return np.array([rho] * n_bins)  # rho: death rate

    def left(n_bins, q, mu):  # q and (1-q) divide the total mutation rate, mu, for inward/outward mutations
        return np.array([0] + [q * mu] * (int(n_bins / 2) - 1) + [mu / 2] + [(1 - q) * mu] * int(n_bins / 2))

    def right(n_bins, q, mu):
        return np.array([(1 - q) * mu] * (int(n_bins / 2) - 1) + [mu / 2] + [q * mu] * int(n_bins / 2) + [0])
    
    
    birth_array = np.repeat(np.column_stack([birth_gaussian(n_bins, F_tot_arr[i], centers_xarr[i], radii_arr[i]) for i in np.arange(n_iterations)]), 
                            2, axis=1)
    birth_array[:,1::2] = 0   
    # birth_array shape: (n_bins, n_regimes)
    # puts the birth rate arrays in columns, each column repeated next to itself, then the odd columns are set to zero
    # fitness and death - mutation - fitness and death - ... ; the matrix has elements zero for every other column starting from column 1

    death_array = np.repeat(np.column_stack([death(n_bins, rho) for i in np.arange(n_iterations)]), 2, axis=1)
    death_array[:,1::2] = 0

    left_array = np.repeat(np.column_stack([left(n_bins, q, mu_tot) for i in np.arange(n_iterations)]), 2, axis=1)
    left_array[:,0::2] = 0

    right_array = np.repeat(np.column_stack([right(n_bins, q, mu_tot) for i in np.arange(n_iterations)]), 2, axis=1)
    right_array[:,0::2] = 0
    
    return np.concatenate((birth_array[:, None], death_array[:, None], left_array[:, None], right_array[:, None]), axis=1)
    # shape: (n_bins, n_rxn_channels, n_regimes); 
    
    
##########################  Gillespie stochastic algorithm  ##########################

def ssa (pop_t0, rxn_matrices, reaction_durations, n_experiments, max_time, max_population, device):
    
    ##########  initialize population tensors #########
    
    initial_population = torch.tensor(pop_t0[:, None], dtype=torch.int32, device=device)        # shape: (n_bins,)
    population = initial_population.repeat(1, n_experiments).to(device=device)                  # shape: (n_bins, n_experiments)
    tot_population = torch.sum(population, dim=0, dtype=torch.int64).to(device=device)          # shape: (n_experiments,)

    current_time = torch.zeros(n_experiments, dtype=torch.float32, device=device)               # shape: (n_experiments,) # tracks the Gillespie time for individual experiments
    running_experiment_bool = torch.ones(n_experiments, dtype=torch.bool, device=device)        # if False, experiment has ended
    regime_endtimes = torch.tensor(reaction_durations, dtype=torch.float32, device=device)      # shape: (n_regimes,) as [end_time1, end_time2, ..., t_max] # excludes 0
                                                                   # recording the population at the end of each regime (=2*iterations) and at the beginning

    current_regime_exp_bool = torch.ones(n_experiments, dtype=torch.bool, device=device)        # if False, the next rxn time is beyond the current rxn regime's end time
    experiment_index = torch.ones(n_experiments, dtype=torch.bool, device=device)               # True: experiments that receive update in a Gillespie step

    rxn_mat_full = torch.tensor(rxn_matrices, device=device, dtype=torch.float32)               # Shape: (n_bins, n_rxn_channels, n_regimes); 
    # rxn_mat_full has the rates for all reactions in all regimes (alternating between birth only and death+mutation regimes)
    n_bins = rxn_mat_full.size(0)           # same as population.size(0)
    n_rxn_channels = rxn_mat_full.size(1)   # n_rxn_channels = 4 (birth, death, left-mutation, right-mutation)
    n_regimes = rxn_mat_full.size(2)        # n_regimes = 2*n_iteration; each iteration is one round of birth followed by one round of mutation+death
    n_timepoints = 1 + n_regimes 
    
    rxn_mat_current = torch.zeros(n_bins, n_rxn_channels, n_experiments, device=device, dtype=torch.float32)
    # sub-array of the rxn_mat_full that corresponds to the current rxn regime
    rxn_regime_mat = torch.zeros(n_experiments, n_regimes, dtype=torch.int32, device=device)
    # tracks what rxn regime each experiment is in; filled with torch.lt (current_time, regime_end_times)
    regime_index_mat = torch.zeros(n_experiments, dtype=torch.int64, device=device)
    # index for next Gillespie rxn in each running experiment; filled with torch.argmax(rxn_regime_mat, dim=1)
    

    trajectory = torch.zeros(*population.size(), n_timepoints, dtype=torch.int32, device=device)    # shape: (n_bins, n_experiments, n_timepoints)
    trajectory[:, :, 0] = population    # first element: initial population 

    
    
                                ############## initialize propensity tensors ##############

    propensity_mat = torch.zeros(*rxn_mat_current.size(), device=device)           # shape: (n_bins, n_rxn_channels, n_experiments)
    cumsum_propensity = torch.zeros(n_bins*n_rxn_channels, n_experiments, dtype=torch.double, device=device)
    tot_propensity = torch.zeros(n_experiments, dtype=torch.double, device=device)

    
    ############    initialize reaction selection tensors
    
    rank_propensity = torch.zeros(n_experiments, device=device, dtype=torch.double)      # for choosing rxns in Gillespie updates
    gt_cumsum_propensity = torch.zeros(n_bins * n_rxn_channels, n_experiments, dtype=torch.int16, device=device)
    
    rxn_combined_index = torch.zeros(n_experiments, dtype=torch.int16, device=device)     # for each experiment: a combined index that determines the bin and rxn type
    rxn_bin_index   = torch.zeros(1, n_experiments, dtype=torch.int64, device=device)
    increase_index  = torch.zeros(1, n_experiments, dtype=torch.int64, device=device)    # used in updating the population
    operation_index = torch.zeros(1, n_experiments, dtype=torch.int64, device=device)    # which rxn channel

    birth_experiment_bool = torch.zeros_like(rxn_bin_index, dtype=population.dtype, device=device)     #True: experiment has a birth event
    death_experiment_bool = torch.zeros_like(rxn_bin_index, dtype=population.dtype, device=device)     #True: experiment has a death event
    left_experiment_bool  = torch.zeros_like(rxn_bin_index, dtype=population.dtype, device=device)      #True: experiment has a l_mutation event
    right_experiment_bool = torch.zeros_like(rxn_bin_index, dtype=population.dtype, device=device)     #True: experiment has a r_mutation event
    
    edgeCase_index = torch.zeros(n_experiments, dtype=torch.bool, device=device)                # to deal with edge cases (in choosing rxns) with random number generator
    exp_range = torch.arange(n_experiments, dtype=torch.int64, device=device)


    #########  initialize random number generator tensors ##########
    n_rng = 1000
    log_rng = torch.zeros(n_experiments, device=device)
    tau = torch.zeros(n_experiments, device=device)       # for Gillespie times
    rng = torch.zeros(n_rng, n_experiments, device=device, dtype=torch.double)
    # generating n_rng (=1000) random numbers for each experiment  # generating in bulk amortizes rng cost

    ###### initialize rng_counter counters
    
    rng_counter = n_rng-1           # for re-generating random numbers
    gillespie_step = 0  
    
    while True: 
        
        if rng_counter > n_rng-2:
            torch.rand(n_rng, n_experiments, device=device, dtype=torch.double, out=rng)       # shape: (n_random_updates, n_experiments); each Gillespie update uses two random numbers (time, rxn)
            rng_counter = 0

        gillespie_step += 1

        ############  determine the current reaction regime and reaction matrix  ##########

        torch.lt(current_time[:, None], regime_endtimes[None, :], out=rxn_regime_mat)          # rxn_regime_mat shape: (n_experiments, n_regimes)
        torch.argmax(rxn_regime_mat, dim=1, out=regime_index_mat)                              # regime_index_mat shape: (n_experiments,); argmax picks the smallest index from multiple maxima
        torch.index_select(rxn_mat_full, dim=2, index=regime_index_mat, out=rxn_mat_current)   # Shape: rxn_mat_full : (n_bins, n_rxn_channels, n_regimes); index 2 in the rxn matrix: rxn regime
        

        ############ determine propensities ###################

        torch.mul(rxn_mat_current, population[:, None, :], out=propensity_mat)                 # propensity_mat shape: (n_bins, n_rxn_channels, n_experiments)
        torch.sum(propensity_mat, dim=(0, 1), out=tot_propensity)                              # tot_propensity shape: (n_experiments); sums over bins (0) and rxn channels (1)

        tot_propensity = tot_propensity + ~running_experiment_bool
        # remove zero elements before computing the inverse
        # Finished simulations are updated as before but the updates will be masked (through running_experiment_bool)

        ###########  sample tau ###############

        # takes the log of the step'th generated ranodm number for each experiment
        torch.log(1 / rng[rng_counter, :], out=log_rng)
        torch.div(log_rng, tot_propensity, out=tau)
        rng_counter += 1

        ###### track if rxn time was set to the beginning of the next regime 

        torch.ge((current_time + tau)*running_experiment_bool, regime_endtimes[regime_index_mat], out=experiment_index)
        # if True: next rxn time > current regime's end time. Move to the next rxn regime
        # Note: equality only happens for the last regime; in that case, we will have already set current_time to t_End - 1e-6 (see below) which will cause expr_idx to become False

        if experiment_index.any():
            trajectory[:, experiment_index, regime_index_mat[experiment_index]+1] = population [:,experiment_index]
            # +1 as intial population is the first element
        
        current_regime_exp_bool[experiment_index] = False
        current_regime_exp_bool[~experiment_index] = True

        ##################  updating the time and running_experiment_bool 
        
        current_time = torch.min(current_time + tau*running_experiment_bool, regime_endtimes[regime_index_mat])
        torch.logical_and((current_time >= max_time), running_experiment_bool, out=experiment_index)
        running_experiment_bool[experiment_index] = False
        
        current_time[experiment_index] = max_time - 1e-6    
        #these are still going through the update process but their updates are masked; 
        #with -1e-6, the torch.lt(current_time, regime_end_times) doesn't give the wrong index (0)
        
        
        ##########  if all experiments are completed, end the simulation
        
        if (~running_experiment_bool).all():
            return (population, trajectory)


        ################    if all experiments have gone to the next regime, skip the rest and go to the next rxn step
        
        if (~current_regime_exp_bool).all():
            continue

        
        ########################################################
        ########## compute reaction indices for all experiments
        
        
        torch.cumsum(propensity_mat.view(-1, n_experiments), dim=0, out=cumsum_propensity)    # cumulatively sums all rxn rates in all bins for every experiment
        torch.mul(rng[rng_counter, :], tot_propensity, out=rank_propensity)                   # multiply the generated random number with the cumulative sum
        torch.lt(cumsum_propensity, rank_propensity, out=gt_cumsum_propensity)                # find where the generated random results lie in the axis of reaction rates
        torch.sum(gt_cumsum_propensity, dim=0, out=rxn_combined_index)                        # for each experiment, a combined index that determines the (bin and rxn type) together

        rng_counter += 1

        
        # adjust reaction indexing for rng ~ 1.000 due to precision errors in cumsum
        # i.e. if rng-weighted propensity > max propensity, select last nonzero propensity rxn
        # these errors appear to be uniform in sign so shouldn't affect ensemble trajectories
        torch.ge(rxn_combined_index, n_bins * n_rxn_channels, out=edgeCase_index)
        if edgeCase_index.any():
            delta = torch.argmax(
                (torch.flip(propensity_mat.view(-1, n_experiments)[:, edgeCase_index], [0]) > 0).to(torch.int16), dim=0)
            rxn_combined_index.scatter_(0, exp_range[edgeCase_index],
                                        (delta + (rxn_combined_index - n_bins * n_rxn_channels)[edgeCase_index] + 1).to(
                                            torch.int16) * (-1), reduce='add')

        

        ############### convert reaction index to population updates
        
        torch.floor_divide(rxn_combined_index[None, :], n_rxn_channels, out=rxn_bin_index)           # bin for rxn update 
        torch.remainder(rxn_combined_index[None, :], n_rxn_channels, out=operation_index)            # rxn channel for rxn update

        torch.eq(operation_index, 0, out=birth_experiment_bool)
        torch.eq(operation_index, 1, out=death_experiment_bool)
        torch.eq(operation_index, 2, out=left_experiment_bool)
        torch.eq(operation_index, 3, out=right_experiment_bool)

        torch.add(rxn_bin_index, right_experiment_bool, out=increase_index)
        torch.sub(increase_index, left_experiment_bool, out=increase_index)
        # increase_index records birth bins, death bins, and the target sites of a mutation bin

        
        ##### scatter updates
        
        population.scatter_(0, rxn_bin_index * running_experiment_bool[None, :],
                            -1 * (running_experiment_bool * current_regime_exp_bool).to(population.dtype)[None, :], 
                            reduce='add')
        
        population.scatter_(0, increase_index * running_experiment_bool[None, :],
                            (1 + birth_experiment_bool - death_experiment_bool) * running_experiment_bool * current_regime_exp_bool, 
                            reduce='add')

        # for birth and death only, increase_index is the same as rxn_bin_index; so the above would subtract one, and then add 1 back, plus the birth-death term
        # for left and right mutations, the rxn_bin will decrease by one, while the adjacent bins increase by one

        
        ####### extinction 

        torch.sum(population, dim=0, out=tot_population)
        torch.logical_and((tot_population == 0), running_experiment_bool, out=experiment_index)
        
        if experiment_index.any():
            running_experiment_bool[experiment_index] = False 
            
            for exp_idx in torch.where(experiment_index)[0]:          # iterates over n_experiment and fills the rest of trajectory with the final population
                for rgm_idx in np.arange(1+regime_index_mat[exp_idx], trajectory.size(2)):
                   trajectory[:, exp_idx, rgm_idx] = 0 


        
        ####### escape (reach max_population)

        torch.logical_and((torch.ge(tot_population, max_population)), running_experiment_bool, out=experiment_index)
        
        if experiment_index.any():
            running_experiment_bool[experiment_index] = False
            
            for exp_idx in torch.where(experiment_index)[0]:
               for rgm_idx in np.arange(regime_index_mat[exp_idx]+1, trajectory.size(2)):
                   trajectory[:, exp_idx, rgm_idx] = population[:,exp_idx]

        ##########  if all experiments are completed (with extinction and escape)
        if (~running_experiment_bool).all():
            return (population, trajectory)