"""
Solution object.

:author: Max Milazzo
"""


import math
import random
import numpy as np
from scipy.stats import maxwell


MOVE_ATTEMPTS = 5
# number of "attempts" molecule has to move to valid new location before being
# skipped and not moving locations


PERCENT_STDEV_EST = 0.15
# the standard deviation for molecule displacement is used to generate the
# maxwell distribution that simulates molecular movement -- this standard
# deviation is assumed to be about 15% of the average value


MAXWELL_TEST_RVS = 1000
# it is extremely slow and inefficient to calculate random variables directly
# from the maxwell distribution, so instead, 1000 random variables are
# generated at the start of the simulation and randomly selected from a list;
# this allows the program to simulate a maxwell distribution but achieve much
# higher performance


M_TO_PM = 1e+12
# meters to picometers conversion factor


class Solution:
    """
    Solution model class definition.
    
    Solution.solution and Solution.step_time are intended to be publicly
    accessible.
    """
    
    def __init__(self, mol_solute, mol_solvent, sim_size, solute_diameter=460,
            solute_diffusion=1.33e-9, step_time=1e-9):
        """
        :param mol_solute: moles of solute in solution
        :param mol_solvent: moles of solvent in solution
        :param sim_size: the simulation will take place in a roughly NxNxN
            environment, where "1" = 1 pm
        :param solute_diameter: diameter of solute molecules (pm); Na+ ion
            dissolved in room temperature H2O is set as default
        :param solute_diffusion: diffusion constant of solute (m^2/s); Na+ ion
            dissolved in room temperature H2O is set as default
        :param step_time: the time (s) that passes between each rendered step
            of the simulation
        """
        
        self.step_time = step_time
        # create step_time publicly accessible variable
 
        self.solute_percent = mol_solute / (mol_solute + mol_solvent)
        # calculate percent of solute in solution
        
        self.scaled_sim_size = int(sim_size / solute_diameter)
        # scale the simulation size so that the smallest unit is the solute
        # molecular diameter
        
        self.avg_step_move = M_TO_PM * math.sqrt(
            2 * step_time * solute_diffusion
        ) / solute_diameter
        # calculate the average distance a solute molecule will move in any
        # direction using passed data and diffusion equation, then scale
        
        maxwell_scale = math.sqrt(math.pi / (3 * math.pi - 8))
        m = 2 * maxwell_scale * math.sqrt(2 / math.pi)
        maxwell_loc = self.avg_step_move - m
        self.maxwell = maxwell(loc=maxwell_loc, scale=maxwell_scale)
        # generate maxwell distribution
        
        self.maxwell_rvs = [
            self.maxwell.rvs() for _ in range(MAXWELL_TEST_RVS)
        ]
        # generate list of maxwell random variables
        
        self.solution = np.zeros(
            (
                self.scaled_sim_size, self.scaled_sim_size,
                self.scaled_sim_size
            )
        )
        # create a new "empty" solution containing only solvent

        self._mix_solution()
        # generate initial "mixed" solution
    
        
    def _mix_solution(self):
        """
        Generates the initial solution.
        """
        
        self.solution_indices = []
        # holds solute molecules location and direction
        
        for x in range(self.scaled_sim_size):
            for y in range(self.scaled_sim_size):
                for z in range(self.scaled_sim_size):
                    if random.random() < self.solute_percent:
                        self.solution_indices.append((x, y, z))
                        self.solution[x][y][z] = 1
                        # generate solute molecules based on percent occurance
                        # in solution; coordinate data for solute molecules is
                        # stored as tuples, and all other coordinates are
                        # assumed to be occupied by solvent
    
    
    def _gen_location(self, cur_pos):
        """
        Generates a new possible solute molecule location update using a random
        direction and the average solute molecule movement per time step.
        
        :param cur_pos: the (x, y, z) tuple coordinates of the molecule being
            updated to a new location
            
        :return: the new generated possible updated location
        """
        
        direction = (
            random.uniform(-1, 1), random.uniform(-1, 1),
            random.uniform(-1, 1)
        )
        # generate a vector in R3 to represent a random direction in the
        # solution space
        
        distance = random.choice(self.maxwell_rvs)
        
        if distance < 0:
            distance *= -1
        
        scalar = math.sqrt(
            distance ** 2 / (
                direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2
            )
        )
        # calculate the scalar value that multiplies with the direction vector
        # such that the resulting vector has the correct distance
        
        #print("sc", scalar)
        
        return (
            int(cur_pos[0] + scalar * direction[0]),
            int(cur_pos[1] + scalar * direction[1]),
            int(cur_pos[2] + scalar * direction[2])
        )
        # return final possible update coordinates
        

    def _update_molecule(self, cur_pos):
        """
        Updates the location for a single molecule for one time step.
        
        :param cur_pos: the (x, y, z) tuple coordinates of the molecule being
            updated to a new location
            
        :return: the new updated molecule location
        """
        
        new_pos = (cur_pos[0], cur_pos[1], cur_pos[2])
        cur_attempts = 0
        
        while self.solution[new_pos[0]][new_pos[1]][new_pos[2]] == 1:
            new_pos = self._gen_location(cur_pos)
            # generate new position until valid coordinates (no solvent already
            # present at location) is found
            
            if new_pos[0] >= len(self.solution):
                new_pos = (len(self.solution) - 1, new_pos[1], new_pos[2])
                
            if new_pos[1] >= len(self.solution[0]):
                new_pos = (new_pos[0], len(self.solution[0]) - 1, new_pos[1])
                
            if new_pos[2] >= len(self.solution[0][0]):
                new_pos = (new_pos[0], new_pos[1], len(self.solution[0][0]) - 1)
            
            cur_attempts += 1
            
            if cur_attempts == MOVE_ATTEMPTS:
                return cur_pos
                # move failed
        
        self.solution[cur_pos[0]][cur_pos[1]][cur_pos[2]] = 0
        self.solution[new_pos[0]][new_pos[1]][new_pos[2]] = 1
        # update molecule position in solution
        
        return new_pos


    def step(self):
        """
        Advances one time step.
        """
        
        new_solution_indices = []
        
        for solvent_molecule in self.solution_indices:
            new_solution_indices.append(
                self._update_molecule(solvent_molecule)
            )
            # update each solvent molecule and add to new index list
            
        self.solution_indices = new_solution_indices
        # update index list


    def sample(self, dim1_frac=0.1):
        """
        Returns a small sample of the solution space.
        
        :param 1D_dim_frac: fraction of 1-dimensional solution data fetched.
        """
        
        sample_size = int(dim1_frac * len(self.solution[0][0]))
        # calculate amount of solution space to return based on size
        
        return self.solution[0][0][0:sample_size]