import numpy as np
import functions as func
class Muon:
    def __init__(self):
        """
        All values for muons in SI units
        """
        self.mass = 1.883531627e-28
        self.charge = 1
        self.mass_energy = 105.6583745e6
        self.halflife = 2.2969811e-6
        self.gamma_u = 2*np.pi*135.5e6
        self.decay_const = np.log(2)/self.halflife
        self.lifetime = float(self.inv_decay(np.random.rand(1)))
        
    def inv_decay(self, U):
        """
        Inverse of the decay equation
        Takes a number U={0, 1} and returns decay time
        """
        return -(np.log(U)) / self.decay_const
    
    def get_spin_polarisation(self, field, theta):
        w = func.larmor_freq(field, self.gamma_u)
        t = self.lifetime
        return np.cos(theta)**2 + (np.sin(theta)**2)*np.cos(w*t)