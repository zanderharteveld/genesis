"""
.. codeauthor:: Zander Harteveld <zandermilanh@gmail.com>
.. codeauthor:: Jaume Bonet      <jaume.bonet@gmail.com>

.. affiliation::
    Laboratory of Protein Design and Immunoengineering <lpdi.epfl.ch>
    Bruno Correia <bruno.correia@epfl.ch>

.. class:: FakeForm
"""

# Standard Libraries
import math
import os

# External Libraries
import numpy as np
import pandas as pd
import scipy

# This Library


class Constraint( object ):
    """single Constraint"""
    def __init__(self, num1, num2, value, ctype="AtomPair", atm1="CA",
                 atm2="CA", func="HARMONIC", dev=3.0, tag="TAG"):
        self.ctype = ctype
        self.atm1  = atm1
        self.num1  = num1
        self.atm2  = atm2
        self.num2  = num2
        self.func  = func
        self.value = value
        self.dev   = dev
        self.tag   = tag

    def __str__(self):
        atm1 = "{0.atm1} {0.num1}".format(self)
        atm2 = "{0.atm2} {0.num2}".format(self)
        func = "{0.func} {0.value:.2f} {0.dev:.1f} {0.tag}".format(self)
        return "{0.ctype} {1} {2} {3}".format(self, atm1, atm2, func)


class ConstraintSet( object ):
    """ConstraintSet"""
    def __init__(self):
        self.constraints = []
        self.constrnsidx = {}

    @staticmethod
    def parse(filename):
        c = ConstraintSet()
        with open(filename) as fd:
            for line in fd:
                l = line.strip().split()
                c.add_constraint(l[2], l[4], l[6], l[0], l[1], l[3], l[5], l[7], l[8])
        return c

    def add_constraint(self, num1, num2, value, ctype="AtomPair", atm1="CA",
                       atm2="CA", func="HARMONIC", dev=3.0, tag="TAG"):
        c = Constraint(int(num1), int(num2), value, ctype, atm1, atm2, func, dev, tag)
        self.constraints.append(c)
        self.constrnsidx.setdefault(int(num1), {})[int(num2)] = c
        self.constrnsidx.setdefault(int(num2), {})[int(num1)] = c

    def has_contact(self, r1, r2):
        return int(r1) in self.constrnsidx and int(r2) in self.constrnsidx[int(r1)]

    def get_contact(self, r1, r2):
        return self.constrnsidx[int(r1)][int(r2)]

    def __getitem__(self, key):
        return self.constraints[key]

    def __len__(self):
        return len(self.constraints)

    def __str__(self):
        text = []
        for x in range(len(self.constraints)):
            text.append(str(self.constraints[x]))
        return "\n".join(text)


class Loops( object ):
    def __init__(self):
        self.loops = []

    def add_loop(self, ini, end):
        self.loops.append((ini, end))

    def __str__(self):
        text = []
        text.append("#LOOP start end cutpoint skip-rate extend")
        for l in self.loops:
            text.append("LOOP {0[0]} {0[1]} 0 0.0 1".format(l))
        return "\n".join(text) + "\n"



class Form( object ):
    """docstring for Form"""
    def __init__(self, identifier, sslist, l_linkers):
        self.l_linkers = l_linkers
        self.sslist  = sslist
        self.id      = identifier
        self.seq_str = []
        self.inits   = []
        self.const   = ConstraintSet()
        self.loops   = Loops()
        self.order   = []

    def set_order(self, data):
        order = {}
        for x in range(len(data)):
            order[data[x]] = x + 1
        refs  = filter(None, [x.ref for x in self.sslist])
        self.order = [order[x] for x in refs]

    def make_loops(self):
        for x in range(len(self.sslist)):
            #for i,aa_type in enumerate(self.sslist[x].atomtypes):
                #if aa_type ==
            if self.sslist[x].ref is not None:
                self.loops.add_loop(self.inits[x], self.inits[x] + (len(self.sslist[x].atoms)/4) - 1) # Make cleaner with residue object !
            #if self.sslist[x].ref is not None:
                #self.loops.add_loop(self.inits[x], self.inits[x] + len(self.sslist[x].atoms) - 1)

    def make_constraints(self):
        for x in range(len(self.sslist)):
            y = self.sslist[x]
            p = self.inits[x]
            inner_range = 1 if y.get_type() == 'C' else (2 if y.get_type() == 'E' else 5)
            for r1 in range(1,len(y.atoms),4): # only take Calphas
                for r2 in range(r1 + inner_range*4, len(y.atoms),20): # 1 go to Calphas
                    d = scipy.spatial.distance.euclidean(y.atoms[r1], y.atoms[r2])
                    self.const.add_constraint(num1 = p + r1/4, num2 = p + r2/4, value = d, dev=1.5, tag="INNER")

        for x in range(len(self.sslist)):
            px = self.inits[x]
            sx = self.sslist[x]
            for y in range(x + 1, len(self.sslist)):
                py = self.inits[y]
                sy = self.sslist[y]
                for r1 in range(1,len(sx.atoms),4): # only take Calphas
                    for r2 in range(1,len(sy.atoms),4):
                        d = scipy.spatial.distance.euclidean(sx.atoms[r1], sy.atoms[r2])
                        self.const.add_constraint(num1 = r1/4 + px, num2 = r2/4 + py, value = d, dev=3.0, tag="OUTER")

    #def _check_invert(self):  # TODO: wrong
    #    count1 = 0
    #    for x in range(len(self.sslist)):
    #        if self.sslist[x].ref is not None:
    #            count1 = x
    #    return 0 if count1 % 2 == 1 else 1

    def prepare_coords(self):
        #inv = self._check_invert()
        #for x in range(len(self.sslist)):
        #    if x % 2 == inv:
        #        self.sslist[x].struc.invert_direction()

        if self.l_linkers:
            assert len(self.l_linkers) == len(self.sslist) + 1, \
            "Uppps, did you forget to add the length of the termini (otherwise specify by 'x')."

        if self.l_linkers != None and self.l_linkers[0] != "x":
            if self.l_linkers[0] > 0:
                i = self.l_linkers[0]
                self.inits.append(i)
                for x in range(self.l_linkers[0]):
                    self.seq_str.append(("G", "C", "X"))
            else:# self.l_linkers[0] == 0:
                i = 1
                self.inits.append(i)
        else:
            i = 2
            self.seq_str.append(("G", "C", "X"))
            self.inits.append(i)

        for x in range(len(self.sslist) - 1):
            if self.sslist[x].sequence is None:
                self.sslist[x].create_val_sequence()
            for xx in self.sslist[x].sequence:
                self.seq_str.append((xx, self.sslist[x].get_type(), "S"))
            i += len(self.sslist[x].sequence)
            #d = scipy.spatial.distance.euclidean(self.sslist[x].atoms[-1], self.sslist[x + 1].atoms[0])
            if self.l_linkers != None and self.l_linkers[x+1] != "x":
                #if (len(self.sslist)-1) < len(self.l_linkers) or self.l_linkers[0] > 0:
                d = self.l_linkers[x+1]
                #else:
                    #d = self.l_linkers[x]
            else:
                d = scipy.spatial.distance.euclidean(self.sslist[x].atoms[-3], self.sslist[x + 1].atoms[1])
                d = int(math.ceil(d / 3.))
            i += d
            for yy in range(d):
                self.seq_str.append(("G", "C", "X"))
            self.inits.append(i)
        if self.sslist[-1].sequence is None:
                self.sslist[-1].create_val_sequence()
        for xx in self.sslist[-1].sequence:
            self.seq_str.append((xx, self.sslist[-1].get_type(), "S"))
        if self.l_linkers != None and (self.l_linkers[-1] != "x" and self.l_linkers[-1] != 0):
            #if (len(self.sslist)-1) < len(self.l_linkers):
            for x in range(self.l_linkers[-1]):
                self.seq_str.append(("G", "C", "X"))
        else:
            self.seq_str.append(("G", "C", "X"))

    def to_sequence(self):
        return ">" + self.id + "\n" + "".join([x[0] for x in self.seq_str])

    def to_psipred_ss(self):
        text = []
        text.append("# PSIPRED VFORMAT (PSIPRED V2.6 by David Jones)\n")
        sse = [x[1] for x in self.seq_str]
        seq = [x[0] for x in self.seq_str]
        for i in range(len(sse)):
            pH, pE, pC = 0, 0, 0
            if sse[i] == 'C': pC = 1
            else:
                edge = False
                if self.l_linkers != None and self.l_linkers[-1] == 0:
                    if sse[i] != sse[i - 1] or (sse[i] != sse[i - 2] and sse[i] == 'E'): edge = True
                else:
                    if sse[i] != sse[i - 1] or (sse[i] != sse[i - 2] and sse[i] == 'E'): edge = True
                    if sse[i] != sse[i + 1] or (sse[i] != sse[i + 2] and sse[i] == 'E'): edge = True
                if edge:
                    pC = 0.3
                    if sse[i] == 'E': pE = 0.7
                    else:             pH = 0.7
                else:
                    if sse[i] == 'E': pE = 1
                    else:             pH = 1

            line = "{0:>4} {1} {2}   {3:0.3f}  {4:0.3f}  {5:0.3f}".format(i + 1, seq[i], sse[i], pC, pH, pE)
            text.append(line)
        return '\n'.join(text)

    def to_pdb(self):
        data = []
        # ssdef = []
        for x in range(len(self.sslist)):
            data.append(self.sslist[x].atom_points(atom = self.inits[x]))
        return "\n".join(data)

    def to_frame(self):
        data = []
        for x in range(len(self.sslist)):
            data.append(self.sslist[x].atom_data(atom = self.inits[x]))
        return pd.concat(data).reset_index(drop=True)

    def __contains__(self, query):
        for x in self.sslist:
            if x.get_type().upper() == query.upper():
                return True
        return False
