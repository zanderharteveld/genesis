"""
.. codeauthor:: Zander Harteveld <zandermilanh@gmail.com>
.. codeauthor:: Jaume Bonet      <jaume.bonet@gmail.com>

.. affiliation::
    Laboratory of Protein Design and Immunoengineering <lpdi.epfl.ch>
    Bruno Correia <bruno.correia@epfl.ch>

.. class:: VirtualStructure
"""

# Standard Libraries
import copy

# External Libraries
from collections import Iterable
from random import random
from bisect import bisect
import numpy as np
import pandas as pd
import scipy.spatial
from transforms3d.euler import euler2mat, mat2euler

# This Library


class VirtualStructure( object ):
    """
    """
    _MAX_AA_DIST = 3.2
    _ATOMTYPE     = ("N", "CA", "C", "O")
    _STRING_X    = "HETATM{0:>5d}  X     X {2}{0:>4d} {1[0]:>11.3f}{1[1]:>8.3f}{1[2]:>8.3f}  1.00"

    _STRING_ATOMS   = {
                        "N": "ATOM  {4:>5d}  N   {3:>3} {2}{0:>4d} {1[0]:>11.3f}{1[1]:>8.3f}{1[2]:>8.3f}  1.00",
                        "CA": "ATOM  {4:>5d}  CA  {3:>3} {2}{0:>4d} {1[0]:>11.3f}{1[1]:>8.3f}{1[2]:>8.3f}  1.00",
                        "C": "ATOM  {4:>5d}  C   {3:>3} {2}{0:>4d} {1[0]:>11.3f}{1[1]:>8.3f}{1[2]:>8.3f}  1.00",
                        "O": "ATOM  {4:>5d}  O   {3:>3} {2}{0:>4d} {1[0]:>11.3f}{1[1]:>8.3f}{1[2]:>8.3f}  1.00",
                        "H": "ATOM  {4:>5d}  H   {3:>3} {2}{0:>4d} {1[0]:>11.3f}{1[1]:>8.3f}{1[2]:>8.3f}  1.00",
                        }

    _A123 = {"C": "CYS", "D": "ASP", "S": "SER", "Q": "GLN", "K": "LYS",
             "I": "ILE", "P": "PRO", "T": "THR", "F": "PHE", "N": "ASN",
             "G": "GLY", "H": "HIS", "L": "LEU", "R": "ARG", "W": "TRP",
             "A": "ALA", "V": "VAL", "E": "GLU", "Y": "TYR", "M": "MET"}

    def __init__(self, residues, centre = [0., 0., 0.], start_atomcount = 0, start_residuecount = 0, chain = "A"):
        self.residues = int(residues)
        self.chain    = chain
        self.centre   = np.array(centre, dtype="float64")
        self.start_atomcount = start_atomcount
        self.start_residuecount = start_residuecount

        self.max_dist = float(self._MAX_AA_DIST * self.residues)
        self.edges    = [np.copy(self.centre) + np.array([0, self.max_dist / 2, 0]),
                         np.copy(self.centre) - np.array([0, self.max_dist / 2, 0])]

        self.points   = []
        for x in range(self.residues):
            self.points.append(np.copy(self.edges[0]) - np.array([0, self._MAX_AA_DIST * x, 0]) )
        self.atoms    = []
        self.atomtypes = []
        self.ca_atoms = []
        self.atom = None
        self.Rapplied = np.eye(3)
        self.is_inverted = False
        self.spinned     = 0
        self.sequence    = None
        self.ref         = None
        self.name        = None

    # BOOLEANS
    def in_origin(self):
        return np.allclose(self.centre, [0., 0., 0.])

    def goes_up(self):
        if len(self.atoms) > 0:
            return self.atoms[0][1] < self.atoms[-1][1]
        elif len(self.points) > 0:
            return self.points[0][1] < self.points[-1][1]
    
    def display_atoms(self):
        if len(self.atoms) > 0:
            print('atoms', self.atoms)
        elif len(self.points) > 0:
            print('points', self.points)

    #def get_direction(self):
    #    self.atoms[-1] - self.atoms[-1][1]

    def goes_down(self):
        return not self.goes_up()

    # GETTERS
    def get_type(self):
        return self._TYPE

    def up_is_1(self):
        return 1 if self.goes_up() else 0

    # TILT
    def tilt_x_degrees(self, angle): self.tilt_degrees(x_angle = angle)
    def tilt_y_degrees(self, angle): self.tilt_degrees(y_angle = angle)
    def tilt_z_degrees(self, angle): self.tilt_degrees(z_angle = angle)

    def tilt_degrees(self, x_angle = 0, y_angle = 0, z_angle = 0, store = True):
        if x_angle == 0 and y_angle == 0 and z_angle == 0: return
        self.tilt_radiants(x_angle = np.radians(x_angle),
                           y_angle = np.radians(y_angle),
                           z_angle = np.radians(z_angle), store = store)

    def tilt_x_radiants(self, angle): self.tilt_radiants(x_angle = angle)
    def tilt_y_radiants(self, angle): self.tilt_radiants(y_angle = angle)
    def tilt_z_radiants(self, angle): self.tilt_radiants(z_angle = angle)

    def tilt_radiants(self, x_angle = 0, y_angle = 0, z_angle = 0, store = True):
        Rx = euler2mat(x_angle, 0, 0, "sxyz")
        Ry = euler2mat(0, y_angle, 0, "sxyz")
        Rz = euler2mat(0, 0, z_angle, "sxyz")
        R  = np.dot(Rz, np.dot(Rx, Ry))

        tmpctr = np.array([0., 0., 0.])
        fixpos = not np.allclose(self.centre, tmpctr)
        tmpctr = np.copy(self.centre)
        if fixpos: self.shift(x = -tmpctr[0], y = -tmpctr[1], z = -tmpctr[2])
        self.apply_matrix(R)
        if fixpos: self.shift(x = tmpctr[0], y = tmpctr[1], z = tmpctr[2])

        if store: self.Rapplied = np.dot(self.Rapplied, R)

    def apply_matrix(self, R):
        if len(self.edges):  self.edges  = np.dot(self.edges,  R)
        if len(self.points): self.points = np.dot(self.points, R)
        if len(self.atoms):  self.atoms  = np.dot(self.atoms,  R)

    # ROTATE ON AXIS
    def spin_radians(self, angle): self.spin_degrees(np.degrees(angle))

    def spin_degrees(self, angle):
        if np.allclose(self.Rapplied, np.eye(3)):
            self.tilt_degrees(y_angle = angle, store = False)
        else:
            euler1 = mat2euler(self.Rapplied)
            euler2 = mat2euler(self.Rapplied.transpose())
            self.tilt_radiants(euler2[0], euler2[1], euler2[2])
            self.tilt_degrees(y_angle = angle, store = False)
            self.tilt_radiants(euler1[0], euler1[1], euler1[2])
        self.spinned += angle

    def remove_movement_memory(self):
        self.Rapplied = np.eye(3)

    # SHIFT
    def shift_x(self, x): self.shift(x, 0., 0.)
    def shift_y(self, y): self.shift(0., y, 0.)
    def shift_z(self, z): self.shift(0., 0., z)

    def shift(self, x = 0., y = 0., z = 0.):
        t = np.array(x) if isinstance(x, Iterable) else np.array([x, y, z])
        self.centre += t
        self.edges  += t
        self.points += t
        if len(self.atoms): self.atoms += t

    def shift_to_origin(self):
        anti = np.copy(self.centre) if not self.in_origin() else np.array([0., 0., 0.])
        self.shift(-anti)
        return anti

    def get_center(self):
        return np.array(self.points).mean(axis=0)

    # FUNC
    def invert_direction(self):
        if np.allclose(self.Rapplied, np.eye(3)):
            self.tilt_degrees(x_angle = 180, y_angle = 180, store = False)
        else:
            euler1 = mat2euler(self.Rapplied)
            euler2 = mat2euler(self.Rapplied.transpose())
            self.tilt_radiants(euler2[0], euler2[1], euler2[2])
            self.tilt_degrees(x_angle = 180, y_angle = 180, store = False)
            self.tilt_radiants(euler1[0], euler1[1], euler1[2])
        self.is_inverted = not self.is_inverted

    def atom_points(self, atom = 1, seq = None):
        count = 0
        data  = []
        d = {"C": "CYS", "D": "ASP", "S": "SER", "Q": "GLN", "K": "LYS",
             "I": "ILE", "P": "PRO", "T": "THR", "F": "PHE", "N": "ASN",
             "G": "GLY", "H": "HIS", "L": "LEU", "R": "ARG", "W": "TRP",
             "A": "ALA", "V": "VAL", "E": "GLU", "Y": "TYR", "M": "MET"}
        if seq is None and self.sequence is not None: seq = self.sequence
        if seq is None: seq = "G"
        else:           seq = seq.upper()

        for x, (points, atomtype) in enumerate(zip(self.atoms, self.atomtypes)):
            data.append(self._STRING_ATOMS[atomtype].format(atom + count, points, self.chain, d[seq[count]], atom + x))
            if (1 + x)%len(self._ATOMTYPE)==0:
                count += 1
        return "\n".join(data)

    def atom_data(self, atom = 1, seq = None):
        count = 0
        data = {
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

        d = {"C": "CYS", "D": "ASP", "S": "SER", "Q": "GLN", "K": "LYS",
             "I": "ILE", "P": "PRO", "T": "THR", "F": "PHE", "N": "ASN",
             "G": "GLY", "H": "HIS", "L": "LEU", "R": "ARG", "W": "TRP",
             "A": "ALA", "V": "VAL", "E": "GLU", "Y": "TYR", "M": "MET"}
        if seq is None and self.sequence is not None: seq = self.sequence
        if seq is None: seq = "G"
        else:           seq = seq.upper()

        for x, (points, atomtype) in enumerate(zip(self.atoms, self.atomtypes)):
            #data.append(self._STRING_ATOMS[atomtype].format(atom + count, points, self.chain, d[seq[count]], atom + x))
            data['x'].append( float(points[0]) )
            data['y'].append( float(points[1]) )
            data['z'].append( float(points[2]) )
            data['id'].append( atom + count )
            data['atomnum'].append( atom + x )
            data['atomtype'].append( atomtype )
            data['res1aa'].append( seq[count] )
            data['res3aa'].append( d[seq[count]] )
            data['chain'].append( self.chain )

            if (1 + x)%len(self._ATOMTYPE)==0:
                count += 1
        return pd.DataFrame(data)

    def eigenvectors(self, modules=[2.0, 2.0, 2.0]):
        def point_distance( point1, point2 ):
            return np.linalg.norm(point1 - point2)

        coordinates = np.array(self.points)
        center = coordinates.mean(axis=0)
        A  = np.asmatrix(np.zeros((3, 3)))
        P  = coordinates - center
        for p in P:
            r = np.asmatrix(p)
            A += r.transpose() * r
        val, EigVc = np.linalg.eigh(A)
        vectors = []
        for axis, module in enumerate(modules):
            t = np.asarray(EigVc[:, axis]).reshape(3)
            vectors.append([np.around(np.asarray(center + (module / 2.0) * t, dtype=np.float32), decimals=3),
                            np.around(center, decimals=3),
                            np.around(np.asarray(center - (module / 2.0) * t, dtype=np.float32), decimals=3)])

        ## Correct direction Major Axis
        if point_distance(vectors[2][0], coordinates[-1]) < point_distance(vectors[2][0], coordinates[0]):
            vectors[2] = np.flip(vectors[2], axis=0)

        return np.asarray(vectors)

    def change_basis(self, eigenvectors):
        # centering points
        evecs_cen = [eigenvectors - eigenvectors[1] for eigenvectors in eigenvectors]
        evecs_sys = np.array([eigenvectors[2] - eigenvectors[0] for eigenvectors in evecs_cen])

        points_new = []
        for i in range(len(self.points)):
            points_new.append(evecs_sys.dot(self.points[i]))
        self.points = points_new

    def create_val_sequence(self):
        if self.sequence is None:
            self.sequence = ""
            for x in range(self.residues):
                self.sequence += "V"
