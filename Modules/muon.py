import numpy as np
import Modules.functions as func

class Muon:
    mass = 1.883531627e-28
    charge = 1
    mass_energy = 105.6583745e6
    halflife = 2.2969811e-6
    gamma_u = 2*np.pi*135.5e6
    decay_const = np.log(2)/halflife
    
    def __init__(self):
        """
        All values for muons in SI units
        """
        self.lifetime = float(self.inv_decay(np.random.rand(1)))
        
    def inv_decay(self, U):
        """
        Inverse of the decay equation
        Takes a number U={0, 1} and returns decay time
        """
        return -(np.log(U)) / Muon.decay_const
    
    def get_spin_polarisation(self, field, theta):
        w = func.larmor_freq(field, Muon.gamma_u)
        t = self.lifetime
        return np.cos(theta)**2 + (np.sin(theta)**2)*np.cos(w*t)
    
    def get_decay_orientation(self, field):
        """Return orientation as total revolutions"""
        return func.larmor_freq(field)*self.lifetime