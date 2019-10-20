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
        self.inv_decay(np.random.rand())
    
    def inv_decay(self):
        """
        Inverse of the decay equation
        Takes a number U={0, 1} and returns decay time
        """
        U = np.random.rand(0, 1)
        self.lifetime = -(np.log
                          (U)) / U