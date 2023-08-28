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

class VirtualHelix( VirtualStructure ):
    _MAX_AA_DIST  = 1.5
    _ATOMTYPES    = ("N", "CA", "C", "O")#, "H")

    _ATOM_CA_DIST = {"N": 0.841, "CA": 0, "C": -1.029, "O": -2.248, "H": 1.839}
    _RADIUS       = {"N": 1.5, "CA": 2.3, "C": 1.8, "O": 2.1, "H": 1.5}
    _ANGLES       = {"N": -28.3, "CA": 100, "C": 28.9, "O": 24.5, "H": -22.5}

    # CHOP780201 alpha-helix propensity AAindex (Chou-Fasman, 1978b)
    # TO 0: G -> 0.57; P -> 0.57; C -> 0.70
    _AA_STAT = [("A", 1.42), ("L", 1.21), ("R", 0.98), ("K", 1.16), ("N", 0.67),
                ("M", 1.45), ("D", 1.01), ("F", 1.13), ("C", 0.00), ("P", 0.00),
                ("Q", 1.11), ("S", 0.77), ("E", 1.51), ("T", 0.83), ("G", 0.00),
                ("W", 1.08), ("H", 1.00), ("Y", 0.69), ("I", 1.08), ("V", 1.06)]

    _TYPE = "H"

    def __init__(self, residues, centre = [0., 0., 0.], start_atomcount = 0, start_residuecount = 0, chain = "A"):
        #super(VirtualHelixAlpha, self).__init__(residues, centre, start_atomcount, start_residuecount, chain)
        super().__init__(residues, centre, start_atomcount, start_residuecount, chain)
        self.edge_angles = [0., 0.]

        self.d = {
            'x': [],
            'y': [],
            'z': [],
            'chain': [],
            'atomtype': [],
            'residuenum': [],
            'res1aa': [],
            'res3aa': []
        }

        self.atoms = []
        self.atomtypes = []
        count = 0
        for x in range(len(self.points)):
            count += 1
            for atomtype in self._ATOMTYPES:
                if atomtype == "CA":
                    angle = self._ANGLES["CA"] * x
                else:
                    angle = self._ANGLES["CA"] * x + self._ANGLES[atomtype]
                point = np.copy(self.points[x]) + np.array([self._RADIUS[atomtype], self._ATOM_CA_DIST[atomtype], 0.])
                self._tilt_y_point_from_centre(self.points[x], point, np.radians(angle))

                self.atoms.append(point)
                self.atomtypes.append(atomtype)

    def _tilt_y_point_from_centre(self, centre, point, angle):
        tmp_point = point[0] - centre[0] , point[2] - centre[2]
        tmp_point = ( tmp_point[0] * np.cos(angle) - tmp_point[1] * np.sin(angle),
                      tmp_point[1] * np.cos(angle) + tmp_point[0] * np.sin(angle))
        tmp_point = tmp_point[0] + centre[0] , tmp_point[1] + centre[2]
        point[0]  = tmp_point[0]
        point[2]  = tmp_point[1]

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
            self.d['atomnum'].append(atom_count)
            self.d['id'].append(residue_count)

        return pd.DataFrame(self.d)
