import numpy as np
from scipy.stats import norm

class PopulationInitializer:
    def __init__(self, n_bins: int, total_population: int, mean: float, sigma: float):
        """
        Initializes the PopulationInitializer with parameters for a Gaussian distribution.

        Args:
            n_bins (int): The number of bins (discrete states or spatial positions).
            total_population (int): Total population to distribute across the bins.
            mean (float): The mean (center) of the Gaussian distribution.
            sigma (float): The standard deviation (spread) of the Gaussian distribution.
        """
        self.n_bins = n_bins
        self.total_population = total_population
        self.mean = mean
        self.sigma = sigma

    @staticmethod
    def gaussian_discrete(x: np.ndarray, area: int, mean: float, sigma: float) -> np.ndarray:
        """
        Generates a discrete Gaussian distribution that sums up to the specified area (total population).

        Args:
            x (np.ndarray): Array of bin indices (discrete states or positions).
            area (int): Total area (population) to distribute across the bins.
            mean (float): The mean (center) of the Gaussian distribution.
            sigma (float): The standard deviation (spread) of the Gaussian distribution.
        
        Returns:
            np.ndarray: An array of integers representing the discrete population in each bin.
        """
        prob_mass = norm.pdf(x, mean, sigma)  # Gaussian PDF values
        normalization_factor = area / np.sum(prob_mass)  # Normalization factor to match total population
        prob_mass_normalized = prob_mass * normalization_factor
        result = np.round(prob_mass_normalized).astype(int)  # Round to integers for discrete population
        result[mean] += area - np.sum(result)  # Adjust to ensure total sum equals the desired population
        return result

    def initialize_population(self) -> np.ndarray:
        """
        Initializes a population distributed according to a Gaussian profile.

        Args: None
        
        Returns:
            np.ndarray: An array of integers representing the initial population distribution across the bins.
        """
        x = np.arange(self.n_bins)  # Array of bin indices
        return self.gaussian_discrete(x, self.total_population, self.mean, self.sigma)
