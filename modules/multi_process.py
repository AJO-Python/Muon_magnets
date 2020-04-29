import numpy as np
import time
from multiprocessing import Pool

import modules.functions as func
from modules.ActiveEnsemble import ActiveEnsemble

"""
TODO:
Perceived problem: Fields returned from multiprocessing are not in order.
        therefore they do not get assigned back to muons correctly
    Attempted solution: Have the particles feel the dipoles and return
        the particles instead of just the fields
        The ensemble can be reconstructed using the particles that have
        "felt" the dipoles.
        Reload the ensemble in __main__ from the temporary ActiveEnsemble
    Additionally: Utilising feel_dipole_grid to see if list comprehension
        and np.sum() results in quicker calculation. (May be quicker to just
        accept the slow calculation. Can always run on my PC overnight)

Current issues: Circular imports from
    __main__ -> Ensemble -> multiprocess -> ActiveEnsemble -> Ensemble
    
REMEMBER:
    I was a smarticle particle and used git branches how they were meant to
    be used. If it comes to it just revert back to the working version
    
    Also, double check whether the fields are returned in order or not
    MORE IMPORTANTLY: Does it actually make a difference? The muon locations
    are not used after I get the fields. Each muon just relaxes in a defined field,
    independent of location? Check with Sean if I can figure it out tomorrow
"""


def calc_fields_faster(*args):
    particle_chunk, dipoles = args
    for p in particle_chunk:
        p.feel_dipole_grid(dipoles)
    return particle_chunk


def MP_fields(particles, dipoles,
              silent=False, result_list=[]):
    """
    :param str folder_name: Folder to save to
    :param object particles: Ensemble
    :param object dipoles: Dipole_grid
    :param bool silent: print % completion if False
    """

    def start():
        particles.chunk_for_processing()

        print("Starting pool...")
        p = Pool(processes=8)
        print("Dispatching tasks...")
        workers = []
        for i, chunk in enumerate(particles.chunks):
            input_tuple = (chunk, dipoles)
            workers.append(p.apply_async(calc_fields_faster,
                                         args=input_tuple))
            print(f"Started task {i + 1}/{len(particles.chunks)}")
        print("-------------------")
        if not silent:
            while True:
                incomplete_count = sum(1 for x in workers if not x.ready())
                if incomplete_count == 0:
                    print("Finished calculations")
                    break
                print(f"{(len(particles.chunks) - incomplete_count) / len(particles.chunks) * 100} % Complete")
                time.sleep(1)
        for worker in workers:
            res = worker.get()
            result_list.append(res)
        p.close()
        p.join()
        return result_list

    # Main
    print("Booting up...")
    result_list = start()
    print("Creating new ensemble")
    results = np.concatenate(result_list, axis=0)
    active_particles = ActiveEnsemble(particles.run_name, results)
    print("Saving active ensemble")
    active_particles.save_ensemble()
    print("Finished multiprocessing")
    print("========")
