import unittest
import torch
import numpy as np
from gillespie.gillespie_ssa import GillespieSSA  # Import your GillespieSSA class from its module

class TestGillespieSSA(unittest.TestCase):

    def setUp(self):
        """
        Set up test variables and initialize a GillespieSSA instance.
        """
        # Parameters for the simulation
        self.n_bins = 10
        self.n_rxn_channels = 4
        self.n_regimes = 5
        self.n_experiments = 3
        self.max_time = 10.0
        self.max_population = 1000
        self.device = 'cpu'

        # Randomly generated initial population, reaction matrices, and reaction durations
        self.initial_population = np.random.randint(0, 100, size=self.n_bins)
        self.rxn_matrices = np.random.rand(self.n_bins, self.n_rxn_channels, self.n_regimes)
        self.reaction_durations = np.linspace(0, self.max_time, self.n_regimes)

        # Initialize the GillespieSSA instance
        self.gillespie = GillespieSSA(
            initial_population=self.initial_population,
            rxn_matrices=self.rxn_matrices,
            reaction_durations=self.reaction_durations,
            n_experiments=self.n_experiments,
            max_time=self.max_time,
            max_population=self.max_population,
            device=self.device
        )

    def test_initialization(self):
        """
        Test whether the GillespieSSA class initializes correctly.
        """
        self.assertEqual(self.gillespie.population.shape, (self.n_bins, self.n_experiments))
        self.assertEqual(self.gillespie.total_population.shape, (self.n_experiments,))
        self.assertEqual(self.gillespie.current_time.shape, (self.n_experiments,))
        self.assertEqual(self.gillespie.trajectory.shape, (self.n_bins, self.n_experiments, 1 + self.n_regimes))

    def test_update_reaction_regime(self):
        """
        Test the update of the reaction regime based on the current time.
        """
        # Simulate a scenario where current time is halfway through the reaction duration
        self.gillespie.current_time = torch.tensor([2.5, 5.0, 7.5], dtype=torch.float32, device=self.device)
        self.gillespie.update_reaction_regime()

        # Check if regime index mat has been updated correctly
        self.assertTrue(torch.all(self.gillespie.regime_index_mat < self.n_regimes))

    def test_compute_propensities(self):
        """
        Test the computation of propensities for each reaction channel.
        """
        self.gillespie.update_reaction_regime()  # Ensure current reaction regime is set
        self.gillespie.compute_propensities()

        # Assert that the total propensity has the correct shape
        self.assertEqual(self.gillespie.tot_propensity.shape, (self.n_experiments,))
        # Propensities should be non-negative
        self.assertTrue(torch.all(self.gillespie.tot_propensity >= 0))

    def test_calculate_reaction_times(self):
        """
        Test the calculation of reaction times for each experiment.
        """
        self.gillespie.update_reaction_regime()
        self.gillespie.compute_propensities()
        self.gillespie.calculate_reaction_times()

        # Reaction times should be non-negative
        self.assertTrue(torch.all(self.gillespie.tau >= 0))

    def test_select_and_apply_reactions(self):
        """
        Test the selection and application of reactions on the population.
        """
        initial_population = self.gillespie.population.clone()  # Save initial population
        self.gillespie.update_reaction_regime()
        self.gillespie.compute_propensities()
        self.gillespie.calculate_reaction_times()
        self.gillespie.select_and_apply_reactions()

        # Ensure population changes after applying reactions
        self.assertFalse(torch.equal(self.gillespie.population, initial_population))

    def test_update_population(self):
        """
        Test population updates for a single reaction.
        """
        # Simulate a simple birth reaction at bin 0
        rxn_bin_index = torch.tensor([0], dtype=torch.long)
        operation_index = torch.tensor([0])  # Birth reaction

        initial_population = self.gillespie.population.clone()
        self.gillespie.update_population(rxn_bin_index, operation_index)

        # Ensure population in bin 0 increased
        self.assertTrue(torch.all(self.gillespie.population[0] > initial_population[0]))

    def test_update_experiment_status(self):
        """
        Test the updating of experiment statuses based on extinction or max population.
        """
        # Simulate extinction (all populations set to 0)
        self.gillespie.population[:] = 0
        self.gillespie.update_experiment_status()

        # Ensure all experiments are no longer running
        self.assertTrue(torch.all(self.gillespie.running_experiment == False))

        # Simulate max population
        self.gillespie.population[:] = self.max_population + 1
        self.gillespie.update_experiment_status()

        # Ensure all experiments are no longer running
        self.assertTrue(torch.all(self.gillespie.running_experiment == False))


if __name__ == '__main__':
    unittest.main()
