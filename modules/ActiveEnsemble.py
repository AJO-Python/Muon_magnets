# -*-coding: UTF-8-*-
import numpy as np
import modules.functions as func
from modules.ensemble import Ensemble


class ActiveEnsemble(Ensemble):

    def __init__(self, run_name, particles):
        self.muons = particles
        self.N = len(particles)
        self.run_name = run_name
        self.fields = np.array([p.field for p in self.muons])
        self.magnitudes = np.array([func.get_mag(f) for f in self.fields])
        self.create_field_dict()
        self.loc = np.array([p.loc for p in self.muons])

    def save_ensemble(self):
        func.save_object(self.run_name, "active_ensemble_obj", self.__dict__)
