import numpy as np
from scipy.stats import norm

class ReactionMatrix:
    def __init__(self, n_bins: int, n_iterations: int, centers: np.ndarray, radii: np.ndarray, 
                 F_tot: np.ndarray, rho: float, mu_tot: float, q: float):
        """
        Initializes the ReactionMatrix with parameters for reaction rates and Gaussian distribution parameters.

        Args:
            n_bins (int): Number of bins (discrete spatial points or states).
            n_iterations (int): Number of iterations or regimes in the simulation.
            centers (np.ndarray): Array of center indices for the Gaussian distribution in each iteration.
            radii (np.ndarray): Array of standard deviations (spread) of the Gaussian distribution in each iteration.
            F_tot (np.ndarray): Array of total birth rates for each iteration.
            rho (float): Constant death rate for all bins.
            mu_tot (float): Total mutation rate (for left and right mutations).
            q (float): Fraction of the mutation rate allocated to left mutations.
        """
        self.n_bins = n_bins
        self.n_iterations = n_iterations
        self.centers = centers
        self.radii = radii
        self.F_tot = F_tot
        self.rho = rho
        self.mu_tot = mu_tot
        self.q = q

        # Precompute death rates since they do not change over iterations
        self.death_array = self.death()

        # Precompute mutation rates since they do not change over iterations
        self.left_array = self.mutation_rates('left')
        self.right_array = self.mutation_rates('right')

    def birth_gaussian(self, F_tot: float, center_idx: int, sigma: float) -> np.ndarray:
        """
        Computes the Gaussian-distributed birth rates across the bins.

        Args:
            F_tot (float): Total birth rate for this iteration.
            center_idx (int): Center bin index for the Gaussian distribution.
            sigma (float): Standard deviation (spread) of the Gaussian distribution.

        Returns:
            np.ndarray: An array of birth rates, distributed according to a Gaussian profile.
        """
        x_arr = np.arange(self.n_bins)
        g = norm.pdf(x_arr, loc=center_idx, scale=sigma)
        return F_tot * g / np.sum(g)

    def death(self) -> np.ndarray:
        """
        Generates a constant death rate for all bins.

        Returns:
            np.ndarray: An array of constant death rates for each bin.
        """
        return np.full(self.n_bins, self.rho)

    def mutation_rates(self, direction: str) -> np.ndarray:
        """
        Computes the mutation rates for each bin based on the mutation rate and the `q` fraction.

        Args:
            direction (str): Either 'left' or 'right', specifying the direction of the mutation.

        Returns:
            np.ndarray: An array of mutation rates for each bin based on the specified direction.
        """
        mid = int(self.n_bins / 2)
        mutation_vec = np.zeros(self.n_bins)

        if direction == 'left':
            mutation_vec[1:mid] = self.q * self.mu_tot
            mutation_vec[mid] = self.mu_tot / 2
            mutation_vec[mid + 1:] = (1 - self.q) * self.mu_tot
        elif direction == 'right':
            mutation_vec[:mid] = (1 - self.q) * self.mu_tot
            mutation_vec[mid] = self.mu_tot / 2
            mutation_vec[mid + 1:] = self.q * self.mu_tot
        else:
            raise ValueError("Direction must be either 'left' or 'right'.")

        return mutation_vec

    def generate_reaction_matrix(self) -> np.ndarray:
        """
        Generates the complete reaction matrix with birth, death, left, and right mutation rates.

        Returns:
            np.ndarray: A 3D array where the first dimension represents bins, the second represents
                        reaction channels (birth, death, left-mutation, right-mutation), and the third 
                        dimension represents the iterations or regimes.
        """
        # Preallocate arrays for birth rates and reaction matrices
        birth_array = np.zeros((self.n_bins, self.n_iterations * 2))
        death_array = np.repeat(self.death_array[:, None], self.n_iterations * 2, axis=1)
        left_array = np.repeat(self.left_array[:, None], self.n_iterations * 2, axis=1)
        right_array = np.repeat(self.right_array[:, None], self.n_iterations * 2, axis=1)

        # Fill birth array for each iteration
        for i in range(self.n_iterations):
            birth = self.birth_gaussian(self.F_tot[i], self.centers[i], self.radii[i])
            birth_array[:, 2 * i] = birth  # Only fill every second column (birth channels)

        # Combine all reaction matrices into one 3D array
        return np.stack([birth_array, death_array, left_array, right_array], axis=1)
