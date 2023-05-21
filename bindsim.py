"""
Molecular binding simulator.

:author: Max Milazzo
"""


import os
from Solution import Solution
from multiprocessing import Process


NUM_SIMS = 10
# number of simultaneous simulations running


SIM_DATA_DIR = "simdata"
# simulation data file save directory


def run_sim(sim_num):
    """
    Runs a binding simulation.
    """
    
    print(f"Simulation #{sim_num} starting.")
    
    mixture = Solution(mol_solute=1, mol_solvent=10, sim_size=1e+5)
    # create a 100x100x100 nm environment
    
    step_count = 0
    
    while mixture.solution[0][0][0] != 1:
    # assuming binding molecule is at position (0, 0, 0) and is the same size
    # as a solute molecule (for calculation simplicity)
    
        mixture.step()
        step_count += 1
    
    data_file = os.path.join(SIM_DATA_DIR, "SIMDATA" + str(sim_num) + ".txt")
    # construct unique data file name
    
    with open(data_file, "w") as f:
        f.write(
            "STEPS: " + str(step_count) + "\n" +
            "SECONDS: " + str(step_count * mixture.step_time)
        )
        # write simulation results to file
        
    print(f"Simulation #{sim_num} complete.")


def main():
    """
    Program entry point.
    """
    
    for i in range(NUM_SIMS):
        Process(
            target=run_sim, args=(i,)
        ).start()
        # start a new process for each simulation


if __name__ == "__main__":
    main()