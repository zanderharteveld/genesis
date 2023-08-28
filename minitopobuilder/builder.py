"""
.. codeauthor:: Zander Harteveld <zandermilanh@gmail.com>

.. affiliation::
    Laboratory of Protein Design and Immunoengineering <lpdi.epfl.ch>
    Bruno Correia <bruno.correia@epfl.ch>

.. function:: builders
"""

# Standard Libraries
import re
import copy
import itertools

# External Libraries
import numpy as np
import pandas as pd
import networkx as nx
from transforms3d.euler import euler2mat, mat2euler

# This Library
from .virtual.VirtualStructure import VirtualStructure
from .virtual.VirtualBeta import VirtualBeta
from .virtual.VirtualHelix import VirtualHelix
from .form.FakeForm import FakeForm
from .form.Form import Form, Constraint, ConstraintSet, Loops
from .form.SecondaryStructure import SecondaryStructure
#from topogocurvy.utils import rotation_matrix_from_vectors, superimpose


__all__ = [
    'build_forms', 'prepare_forms',
    'build_generic',
    'build_similar', 'build_flat'
    ]



def build_forms(form, check_forms=False, connectivity=None, links=None, link_distance_add=2., verbose=0):
    """
    Build sample and build from forms.
    """
    _DEF_Z_DISTANCE = 11.
    _DEF_X_DISTANCE = {"H": 11.,
                       "E": 5.}

    _LINK_DISTANCE  = (np.sqrt(2 * (_DEF_Z_DISTANCE * _DEF_Z_DISTANCE))) + link_distance_add

    _CONNECTIVITY   = [c[:3] for c in connectivity.split('.')] if connectivity else None # topology string
    _LINKER_LENGTH  = [int(l) if l != 'x' else l for l in links.split('.')] if links else None # ['x', 8, 8, 4, 'x', 2] -- x for unknown length

    data = _prepare_form_string(form, connectivity=_CONNECTIVITY, links=_LINKER_LENGTH)
    layer_sets = []
    for lyr in sorted(data["layers"]):
        layer_sets.append( data["layers"][lyr] )

    layers = []
    shapes = []
    for x in range(len(layer_sets)):
        layers.append([])
        width = 0
        for y in range(len(layer_sets[x])):
            ss   = layer_sets[x][y]
            if "shift_y" not in ss: ss["shift_y"] = 0.0
            if "shift_z" not in ss: ss["shift_z"] = _DEF_Z_DISTANCE * x
            else: ss["shift_z"] += (_DEF_Z_DISTANCE * x)
            if "shift_x" not in ss:
                xdist = _DEF_X_DISTANCE["E"] if ss["type"] == "E" else _DEF_X_DISTANCE["H"]
                xdist = 0.0 if y == 0 else xdist
                ss["shift_x"] = width + xdist
            else:
                ss["shift_x"] += width
            if "tilt_x" not in ss: ss["tilt_x"] = 0.0
            if "tilt_y" not in ss: ss["tilt_y"] = 0.0
            if "tilt_z" not in ss: ss["tilt_z"] = 0.0
            if "length" not in ss:
                if ss["type"] == "H": ss["length"] = 10
                else: ss["length"] = 5
            secstr = SecondaryStructure(ss)
            vs = _prepare(ss)
            #if secstr.ref is not None:
            #    ref = None
            #    for mtf in data["motifs"]:
            #        if mtf["id"] == secstr.get_ref_motif():
            #            for z in mtf["segments"]:
            #                if z["id"] == secstr.get_ref_segment():
            #                    ref = z
            #                    break
            #    vs.atoms = ref["coordinates"]
            #else:
            #    vs.tilt_y_degrees(ss["tilt_y"])
            #    vs.tilt_degrees(ss["tilt_x"], 0, ss["tilt_z"])
            vs.tilt_y_degrees(ss["tilt_y"])
            vs.tilt_degrees(ss["tilt_x"], 0, ss["tilt_z"])
            #xdist = _DEF_X_DISTANCE["E"] if vs.get_type() == "E" else _DEF_X_DISTANCE["H"]
            vs.shift(ss["shift_x"], ss["shift_y"], ss["shift_z"])
            width = vs.centre[0]
            secstr.add_structure(vs)
            layers[-1].append(secstr)
            shapes.append(vs)

    if _CONNECTIVITY:
        if verbose == 0: print("Building connectivity {}".format(_CONNECTIVITY))
        data.setdefault("forms", [])
        okforms = []
        forms = _create_forms_by_specification(layers, _LINK_DISTANCE, _CONNECTIVITY, verbose)
        if verbose == 0: print("forms created:", str(len(forms)))
    else:
        forms = _create_forms(layers, _LINK_DISTANCE, _CONNECTIVITY, verbose)
        if verbose == 0: print("forms created:", str(len(forms)))

    # GRAPHIC REPRESENTATIONS
    #vs = VisualForms(okforms if options.hurry else forms)
    #vs.make_svg(data)

    okforms = []
    for _, f in enumerate(forms):
        f.evaluate()
        if check_forms is True:
            if f.do: okforms.append(f)
        else: okforms.append(f)
        if _ > 0 and _ % 100 == 0 and verbose == 0:
            print("{0} out of {1} evaluated ({2} ok)".format(_, len(forms), len(okforms)))
    if verbose == 0: print("{0} evaluated ({1} ok)".format(len(forms), len(okforms)))
    data['forms']  = okforms
    return data


def prepare_forms(data, directions=None, two_way=True, action='memory'):
    """
    Creates the Forms from the container.
    Only use the two_way=False if your topology is symmetric, i.e. if the lengths of the
    secondary structure elements are the same.
    """
    sizes = {}
    for lyr in sorted(data["layers"].keys()):
        for i, sse in enumerate(data['layers'][lyr]):
            if i == 0:
                size = 0
                continue
            if sse['type'] == 'E':
                size += 5.
            else:
                size += 11.
        sizes[lyr] = size

    shifts = {}
    for i in range(len(sizes.keys()) - 1):
        a = sizes[sorted(sizes.keys())[i]]
        b = sizes[sorted(sizes.keys())[i + 1]]
        if a > b:
            shifts[sorted(sizes.keys())[i]] = 0.
            shifts[sorted(sizes.keys())[i + 1]] = (a - b) / 2
        else:
            shifts[sorted(sizes.keys())[i]] = (b - a) / 2
            shifts[sorted(sizes.keys())[i + 1]] = 0.
    
    structures = {}
    for l in sorted(data["layers"]):
        for xi, x in enumerate(data["layers"][l]):
            structures[x["id"]] = _prepare(x)
            structures[x["id"]].name = x["id"]
            structures[x["id"]].remove_movement_memory()
            structures[x["id"]].create_val_sequence()
            structures[x["id"]].tilt_y_degrees(x["tilt_y"])
            structures[x["id"]].tilt_degrees(x["tilt_x"], 0, x["tilt_z"])
            
            #if xi == 0.:
            xshift = shifts[l]
            structures[x["id"]].shift(x["shift_x"] + xshift, x["shift_y"], x["shift_z"])
            
            #else:
            #    structures[x["id"]].shift(x["shift_x"], x["shift_y"], x["shift_z"])

    sketches = {
        'description': [],
        'direction': [],
        'naive': [],
    }
    custom_directions = False
    # forward + direction
    for x in data['forms']:
        #if x.do:
        if not directions:
            custom_directions = True
            directions = [0 if i%2==0 else 1 for i in range(len(x.id.split("_")))]
            print('+ directions are {}'.format(directions))
        directions = [1 if dr == 0 else 0 for dr in directions] # flip for good convention
        sslist  = []
        refsegs = {}
        for cy, y in enumerate(x.id.split("_")):
            sslist.append(copy.deepcopy(structures[y]))
            #print('self up', sslist[-1].up_is_1())
            #if sslist[-1].up_is_1() != x.turn[cy]:
            if directions[cy] == 1:
                sslist[-1].invert_direction()
            #if sslist[-1].up_is_1() != x.turn[cy]:
            #if sslist[-1].ref is not None:
            #    refsegs[sslist[-1].ref] = sslist[-1].atoms
        if "links" not in data.keys(): #data["config"]["l_linkers"] = None
            f = Form(x.id, sslist, None)
        else:
            f = Form(x.id, sslist, data["links"])
        f.prepare_coords()
        #order = []
        #for mtf in data["motifs"]:
        #    for sgm in mtf["segments"]:
        #        order.append(mtf["id"] + "." + sgm["id"])
        #f.set_order(order)
        #f.make_loops()
        #f.make_constraints()

        if action == 'write':
            wdir = os.path.join('form_sketch', x.id)
            os.makedirs(wdir, exist_ok=True)
            seqF, ssF, pdbF, loopF, cnstF = _name_files(wdir)
            #with open(seqF, "w") as fd: fd.write(f.to_sequence())
            #with open(ssF, "w") as fd: fd.write(f.to_psipred_ss())
            with open(pdbF, "w") as fd: fd.write(f.to_pdb())
            #with open(loopF, "w") as fd: fd.write(str(f.loops))
            #with open(cnstF, "w") as fd: fd.write(str(f.const))
     
        frame = f.to_frame()
        sketches['description'].append( '.'.join([str(s) for s in x.sslist]) )
        sketches['direction'].append( '+' )
        sketches['naive'].append(  f.to_frame() )
        forward_directions = directions
    
    if two_way == True:
        # forward - direction
        for x in data['forms']:
            sslist = []
            refsegs = {}
            for cy, y in enumerate(x.id.split("_")):
                if custom_directions == True:
                    directions = [1 if i%2==0 else 0 for i in range(len(x.id.split("_")))]
                    print('- directions are {}'.format(directions))
                    custom_directions = False
                directions = [1 if dr == 0 else 0 for dr in directions]
                sslist.append(copy.deepcopy(structures[y]))
                if directions[cy] == 1:
                    sslist[-1].invert_direction()

            if "links" not in data.keys():
                f = Form(x.id, sslist, None)
            else:
                f = Form(x.id, sslist, data["links"])
            f.prepare_coords()
            #order = []
            #f.set_order(order)

            if action == 'write':
                wdir = os.path.join('form_sketch', x.id)
                os.makedirs(wdir, exist_ok=True)
                seqF, ssF, pdbF, loopF, cnstF = _name_files(wdir)
                with open(pdbF, "w") as fd: fd.write(f.to_pdb())

            sketches['description'].append( '.'.join([str(s) for s in x.sslist]) )
            sketches['direction'].append( '-' )
            sketches['naive'].append(  f.to_frame() )

            # reset directions
            #if forward_directions[-1] == 1: # we have to turn once
            #    for x in data['forms']:
            #        sslist = []
            #        refsegs = {}
            #        for cy, y in enumerate(x.id.split("_")):
            #            if custom_directions == True:
            #                directions = [1 if i%2==0 else 0 for i in range(len(x.id.split("_")))]
            #            sslist.append(copy.deepcopy(structures[y]))
            #            if directions[cy] == 1:
            #                sslist[-1].invert_direction()
            #        break
    return pd.DataFrame( sketches )


def _name_files(wdir):
    return (os.path.join(wdir, "sequence.fa"),
            os.path.join(wdir, "structure.ss2"),
            os.path.join(wdir, "sketch.pdb"),
            os.path.join(wdir, "design.loops"),
            os.path.join(wdir, "constraints.cst"))


def build_generic(topology, shift_noise = None, tilt_noise = None):
    """
    This is a mini-topobuilder - solely essentials to build sketches. Max 4 layers.

    :param string topology:   Topology string in sequence defining layer, element, ss type, direction, length.
    :param float shift_noise: List of dicts with noise values per ss element to translate.
    :param float tilt_noise:  List of dicts with noise values per ss element to tilt

    :return: :class:`~pandas.DataFrame` - a sketch.
    """
     # Global defaults
    _DEF_Z_DISTANCE = {"A": 0., "B": 11., "C": 22., "D": 33.}
    _DEF_X_DISTANCE = {"H": 11., "E": 5.}
    _CONNECTIVITY   = None
    _LINKER_LENGTH  = None

    # Local defaults
    start_atomcount = 0
    start_residuecount = 0

    # Collections
    collection = []

    # Precalculations
    tpg = topology.split('.')

    info = {}
    for lyr in list("ABCD"):
        size = [_DEF_X_DISTANCE[n[2]] * (int(n[1]) - 1) for n in tpg if n[0] == lyr]
        if size != []:
            info[lyr] = max( size )

    for tpgi in tpg:
        tpg_layer  = tpgi[0]
        tpg_number = int( tpgi[1] ) - 1
        tpg_type   = tpgi[2]
        tpg_direc  = tpgi[3]
        tpg_length = int( tpgi[4:] )

        # build
        if tpg_type == "E":
            vs = VirtualBeta(tpg_length, [0., 0., 0.],
                             start_atomcount=start_atomcount,
                             start_residuecount=start_residuecount,
                             chain = 'A')
        if tpg_type == "H":
            vs = VirtualHelix(tpg_length, [0., 0., 0.],
                              start_atomcount=start_atomcount,
                              start_residuecount=start_residuecount,
                              chain = 'A')

        start_atomcount    += (tpg_length * 4)
        start_residuecount += (tpg_length * 1)

        try:
            if info["A"] != info[tpg_layer] and tpg_layer != "A":
                recentering = (info["A"] / 2.) - (info[tpg_layer] / 2.)
            else:
                recentering = 0.
        except:
            recentering = 0.

        try:
            if tpg_layer == 'A':
                centering = 0.
            else:
                center = np.mean(info['A'])
                centering = (center + recentering) - np.mean(info[tpg_layer])
        except:
            centering = 0.

        # Shift
        vs.shift(x=_DEF_X_DISTANCE[tpg_type]*tpg_number + recentering,
                 y=0.,
                 z=_DEF_Z_DISTANCE[tpg_layer])

        # Invert
        if tpg_direc == '-':
            vs.invert_direction()

        # Add noise
        if shift_noise:
            vs.shift(x=float(np.random.randint(-1,1)), y=float(np.random.randint(-7,7)), z=0.)

        if tilt_noise:
            vs.tilt_degrees(x_angle=float(np.random.randint(-20,20)), y_angle=0., z_angle=0.)

        vs = vs.get_frame()
        vs = vs.assign(layer=[tpg_layer]*len(vs), element=[tpg_number]*len(vs))
        collection.append(vs)

    collection = pd.concat( collection )

    # centering by layers
    #if len(collection[collection.layer == 'A'] > len(collection[collection.layer == 'B']:
    #if not collection[collection.layer == 'B'].empty:
        #center1 = collection[collection.layer == 'A'][['x']].mean().values
        #center2 = collection[collection.layer == 'B'][['x']].mean().values
        #diff = center1 - center2
        #collection[collection.layer == 'B'].loc[:, 'x'] -= diff[0]
        #collection[collection.layer == 'B'].loc[:, 'y'] += diff[1]
        #collection[collection.layer == 'B'].loc[:, 'z'] += diff[2]

    #return pd.concat( collection )
    return collection


def build_similar(pdbframe, shift_noise = None, tilt_noise = None):
    """
    This is a mini-topobuilder - solely essentials to build sketches. Max 4 layers.

    :param string topology:   Topology string in sequence defining layer, element, ss type, direction, length.
    :param float shift_noise: List of dicts with noise values per ss element to translate.
    :param float tilt_noise:  List of dicts with noise values per ss element to tilt

    :return: :class:`~pandas.DataFrame` - a sketch.
    """
    # Local defaults
    start_atomcount = 0
    start_residuecount = 0

    # Collections
    collection = []

    # Precalculations
    pdb_ca = pdbframe[pdbframe.atomtype == 'CA']
    dssp_str = ''.join(pdb_ca.dssp3.values)
    combs = ["".join(g) for k, g in itertools.groupby(dssp_str)]
    combs = [[c[0], len(c)] for c in combs]

    prev, post = 0, 0
    for i, comb in enumerate(combs):
        post += comb[1]
        piece = pdb_ca.iloc[prev:post]
        prev += comb[1]

        tpg_type   = comb[0]
        tpg_length = int( comb[1] )

        # build
        if tpg_type == "E":
            vs = VirtualBeta(tpg_length, [0., 0., 0.],
                             start_atomcount=start_atomcount,
                             start_residuecount=start_residuecount,
                             chain = 'A')
        elif tpg_type == "H":
            vs = VirtualHelix(tpg_length, [0., 0., 0.],
                              start_atomcount=start_atomcount,
                              start_residuecount=start_residuecount,
                              chain = 'A')
        else:
            continue

        start_atomcount    += (tpg_length * 4)
        start_residuecount += (tpg_length * 1)


        vs = vs.get_frame()
        vs, rms = superimpose(piece, vs, ca_only=True)

        # Add noise
        if tilt_noise:
            xyz = vs[['x', 'y', 'z']].values
            vs = vs.drop(columns=['x', 'y', 'z'])

            Rx = euler2mat(float(np.random.randint(-5,5)), 0, 0, "sxyz")
            Ry = euler2mat(0, float(np.random.randint(-5,5)), 0, "sxyz")
            Rz = euler2mat(0, 0, float(np.random.randint(-5,5)), "sxyz")
            R  = np.dot(Rz, np.dot(Rx, Ry))

            xyz_cen = xyz - np.mean(xyz, axis=0)
            xyz_rot = xyz_cen.dot(R)
            xyz_new = xyz_rot + np.mean(xyz, axis=0)

            vs = pd.concat([vs, pd.DataFrame(xyz_new, columns=['x', 'y', 'z'])], axis=1)

        # Add noise
        #if shift_noise:
        #    vs['x'] += float(np.random.randint(-2,2))
        #    vs['y'] += float(np.random.randint(-2,2))
        #    vs['z'] += float(np.random.randint(-2,2))

        #if tilt_noise:
        #    vs.tilt_degrees(x_angle=float(np.random.randint(-20,20)), y_angle=0., z_angle=0.)

        collection.append(vs)

    collection = pd.concat( collection )
    return collection


def build_flat(pdbframe, shift_noise = None, tilt_noise = None):
    """
    This is a mini-topobuilder - solely essentials to build sketches. Max 4 layers.

    :param string topology:   Topology string in sequence defining layer, element, ss type, direction, length.
    :param float shift_noise: List of dicts with noise values per ss element to translate.
    :param float tilt_noise:  List of dicts with noise values per ss element to tilt

    :return: :class:`~pandas.DataFrame` - a sketch.
    """
    # Local defaults
    start_atomcount = 0
    start_residuecount = 0

    # Collections
    collection = []

    # Precalculations
    pdb_ca = pdbframe[pdbframe.atomtype == 'CA']
    dssp_str = ''.join(pdb_ca.dssp3.values)
    combs = [''.join(g) for k, g in itertools.groupby(dssp_str)]
    combs = [[c[0], len(c)] for c in combs]

    prev, post = 0, 0
    for i, comb in enumerate(combs):
        post += comb[1]
        if i == 0:
            piece_first = pdb_ca.iloc[prev:post]
        piece = pdb_ca.iloc[prev:post]
        com_piece = piece[['x', 'y', 'z']].mean().values
        prev += comb[1]

        tpg_type   = comb[0]
        tpg_length = int( comb[1] )

        # build
        if tpg_type == "E":
            vs = VirtualBeta(tpg_length, [0., 0., 0.],
                             start_atomcount=start_atomcount,
                             start_residuecount=start_residuecount,
                             chain = 'A')
        elif tpg_type == "H":
            vs = VirtualHelix(tpg_length, [0., 0., 0.],
                              start_atomcount=start_atomcount,
                              start_residuecount=start_residuecount,
                              chain = 'A')
        else:
            continue

        start_atomcount    += (tpg_length * 4)
        start_residuecount += (tpg_length * 1)

        vs = vs.get_frame()
        vs, rms = superimpose(piece_first, vs, ca_only=True)

        # Add noise
        #if shift_noise:
        #    vs.shift(x=float(np.random.randint(-1,1)), y=float(np.random.randint(-7,7)), z=0.)

        #if tilt_noise:
        #    vs.tilt_degrees(x_angle=float(np.random.randint(-20,20)), y_angle=0., z_angle=0.)

        collection.append(vs)

    collection = pd.concat( collection )
    return collection


def _prepare(sse):
    """
    """
    if sse["type"] == "E":
        vs = VirtualBeta(sse["length"], [0., 0., 0.])
    if sse["type"] == "H":
        vs = VirtualHelix(sse["length"], [0., 0., 0.])
    return vs


def _prepare_form_string(form, connectivity=None, links=None):
    formstr = form.split('.')

    data = {'layers': {}}
    layers = set([l[0] for l in formstr])

    for lr in layers:
        data['layers'][lr] = []
        for sse in formstr:
            if sse.startswith(lr):
                dl = {}
                dl['id']   = sse
                dl['type'] = sse[2]

                if len(sse) > 3:
                    if '+' in sse or '-' in sse:
                         dl['length'] = int(sse.replace('-', '+').split('+')[-1])
                    else:
                         dl['length'] = int(sse[3:])
                else:
                    if sse[2] == 'H': dl['length'] = 10
                    else: dl['length'] = 5

                data['layers'][lr].append(dl)
    if connectivity:
        data['connectivity'] = connectivity
    if links:
        data['links'] = links
    return data


def _create_graph( layers, distance, connectivity ):
    G = nx.Graph()
    for lyr1 in range(len(layers)):
        for lyr2 in range(len(layers)):
            if abs(lyr1 - lyr2) <= 1:  # Only consecutive layers
                for col1 in range(len(layers[lyr1])):
                    for col2 in range(len(layers[lyr2])):
                        if abs(col1 - col2) <= 1:  # Only consecutive columns
                            G.add_edge(layers[lyr1][col1],
                                       layers[lyr2][col2], object=SecondaryStructure)
    for lyr1 in layers:
        for lyr2 in layers:
            for sse1 in lyr1:
                for sse2 in lyr2:
                    if sse1 != sse2 and sse1.twoD_distance(sse2) <= distance:
                        G.add_edge(sse1, sse2)

    return G


def _search_paths(G, n1, n2, verbose):
    forms = []
    for path in nx.all_simple_paths(G, n1, n2):
        if len(path) == nx.number_of_nodes(G):
            f = FakeForm(copy.deepcopy(path))
            forms.append(f)
            path.reverse()
            f = FakeForm(copy.deepcopy(path))
            forms.append(f)
    if verbose == 0: print(n1.desc, "->", n2.desc, "  ", len(forms), "folds obtained")
    return forms


def _create_forms( layers, distance, connectivity, verbose ):
    G = _create_graph(layers, distance, connectivity)
    forms = []

    for _1, n1 in enumerate(G.nodes()):
        for _2, n2 in enumerate(G.nodes()):
            if _1 < _2:
                forms.extend(_search_paths(G, n1, n2, verbose))
    return forms


def _create_forms_by_specification( layers, distance, connectivity, verbose ):
    connect = {}
    for i,lyr in enumerate( layers ):
        l = map(str, lyr)
        for j,sse in enumerate(l):
            insertion_ind = connectivity.index(sse[:3])
            connect[insertion_ind] = [i, j]
    #connect = list(connect.values())
    connect = [connect[k] for k in sorted(connect)]

    path = []
    for i in range(len(connect)):
        lyr1, col1 = connect[i][0], connect[i][1]
        path.append(layers[lyr1][col1])
    forms = []
    if len(connectivity)==2:
        n1 = path[0]
        n2 = path[-1]
        forms.extend(_search_paths(G, n1, n2, verbose))
    else:
        f = FakeForm(copy.deepcopy(path))
        forms.append(f)
        path.reverse()
        f = FakeForm(copy.deepcopy(path))
        forms.append(f)
    return forms



