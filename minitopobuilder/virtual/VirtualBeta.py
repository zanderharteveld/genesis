"""
.. codeauthor:: Zander Harteveld <zandermilanh@gmail.com>
.. codeauthor:: Jaume Bonet      <jaume.bonet@gmail.com>

.. affiliation::
    Laboratory of Protein Design and Immunoengineering <lpdi.epfl.ch>
    Bruno Correia <bruno.correia@epfl.ch>

.. class:: VirtualStructure
"""

# Standard Libraries

# External Libraries
import numpy as np
import pandas as pd

# This Library
from .VirtualStructure import VirtualStructure

class VirtualBeta( VirtualStructure ):
    """
    """
    # The pleating causes the distance between alpha[i] and alpha[i+2] to be
    # approximately 6, rather than the 7.6 (2 × 3.8) expected from
    # two fully extended trans peptides.
    _ATOMTYPES    = ("N", "CA", "C", "O")#, "H")

    _ATOM_CA_DIST = {"N": 5.000, "CA": 3.800, "C": 2.600, "O": 2.500} #"H": 2.5}
    _RADIUS       = {"N": 0.300, "CA": 1.100, "C": 0.300, "O": 0.200} #"H": 0.068}
    _SHIFT        = {"N": 0.400, "CA": 0.000, "C": 0.500, "O": 1.900} #"H": 1.35}

    # CHOP780202 beta-sheet propensity AAindex (Chou-Fasman, 1978b)
    # TO 0: G -> 0.75; P -> 0.55; C -> 1.19
    _AA_STAT = [("A", 0.83), ("L", 1.30), ("R", 0.93), ("K", 0.74), ("N", 0.89),
                ("M", 1.05), ("D", 0.54), ("F", 1.38), ("C", 0.00), ("P", 0.00),
                ("Q", 1.10), ("S", 0.75), ("E", 0.37), ("T", 1.19), ("G", 0.00),
                ("W", 1.37), ("H", 0.87), ("Y", 1.47), ("I", 1.60), ("V", 1.70)]

    _TYPE = "E"

    def __init__(self, residues, centre = [0., 0., 0.], start_atomcount = 0, start_residuecount = 0, chain = "A"):
        #super(VirtualBeta, self).__init__(residues, centre, start_atomcount, start_residuecount, chain)
        super().__init__(residues, centre, start_atomcount, start_residuecount, chain)
        self.last_orientation = self._RADIUS["CA"]

        self.atoms     = []
        self.atomtypes = []
        count = 0
        for x in range( len(self.points) ):
            count += 1
            self.last_orientation *= -1
            for atomtype in self._ATOMTYPES:
                points = np.copy(self.points[x]) \
                + np.array([self._SHIFT[atomtype] * self.last_orientation, \
                            self._ATOM_CA_DIST[atomtype] - self._ATOM_CA_DIST["CA"], \
                            self._RADIUS[atomtype] * self.last_orientation])
                self.atoms.append(points)
                self.atomtypes.append(atomtype)


    def get_frame(self):
        """
        """
        self.d = {
            'x': [],
            'y': [],
            'z': [],
            'id': [],
            'chain': [],
            'atomtype': [],
            'atomnum': [],
            'res1aa': [],
            'res3aa': []
        }
        atom_count = self.start_atomcount
        residue_count = self.start_residuecount
        for i in range(len(self.atoms)):
            if i%4==0:
                residue_count += 1
            atom_count += 1

            self.d['x'].append(self.atoms[i][0])
            self.d['y'].append(self.atoms[i][1])
            self.d['z'].append(self.atoms[i][2])
            self.d['res1aa'].append('V')
            self.d['res3aa'].append('VAL')
            self.d['chain'].append('A')
            self.d['atomtype'].append(self.atomtypes[i])
            self.d['id'].append(residue_count)
            self.d['atomnum'].append(atom_count)

        return pd.DataFrame(self.d)
