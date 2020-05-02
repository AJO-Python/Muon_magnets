import numpy as np
from multiprocessing import Pool
import modules.functions as func
import time


def calc_fields(*args):
    """

    :param tuple args: array of particles and array of dipoles
    :return:
    """
    particle_chunk, dipoles = args
    chunk_fields = []
    for i, p in enumerate(particle_chunk):
        for d in dipoles:
            p.feel_dipole(d)
        chunk_fields.append(p.field)
    return chunk_fields


def MP_fields(folder_name, particles, dipoles,
              silent=False):
    """
    DOES NOT PRESERVE ORDER OF INPUT MUONS TO OUTPUT FIELDS
    :param str folder_name: Folder to save to
    :param array particles: array of muons
    :param array dipoles: array of dipoles
    :param int chunk_size: Size of chunk to break particles into
    :param bool silent: print % completion if False
    """

    def start():
        chunks = np.array_split(particles, 16)
        print("Starting pool...")
        p = Pool(processes=8)
        print("Dispatching tasks...")
        workers = []
        for i, chunk in enumerate(chunks):
            input_tuple = (chunk, dipoles)
            workers.append(p.apply_async(calc_fields,
                                         args=input_tuple))
            print(f"Started task {i + 1}/{len(chunks)}")
        print("-------------------")
        if not silent:
            while True:
                incomplete_count = sum(1 for x in workers if not x.ready())
                if incomplete_count == 0:
                    print("Finished calculations")
                    break
                print(f"{(len(chunks) - incomplete_count) / len(chunks) * 100} % Complete")
                time.sleep(1)
        for worker in workers:
            res = worker.get()
            result_list.append(res)
        p.close()
        p.join()
        return result_list

    # Main
    result_list = []
    print("Booting up...")
    result_list = start()
    print("Saving data to file...")
    results = np.concatenate(result_list, axis=0)
    func.save_array(folder_name, "muon_fields", muon_fields=results)
    print("Finished multiprocessing")
    print("========")
