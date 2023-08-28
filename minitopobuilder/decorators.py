"""
.. codeauthor:: Zander Harteveld <zandermilanh@gmail.com>

.. affiliation::
    Laboratory of Protein Design and Immunoengineering <lpdi.epfl.ch>
    Bruno Correia <bruno.correia@epfl.ch>

.. function:: pdb_to_info
.. function:: add_cbeta
.. function:: find_objects
"""

# Standard Libraries
import random

# External Libraries
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# This Library
#from topogocurvy.utils import change_basis

__all__ = ['pdb_to_info', 'find_objects', 'find_sse',
           'add_cbeta', 'add_valine', 'make_residue', 'add_loops',
           'remove_overlaps']


def find_objects(pdbca, obj_sse):
    """
    """
    count, first = 0, 0
    boundaries = []

    _prev = obj_sse.astype(int)
    for i in range(len(pdbca) - 1):
        if i > 0:
            _prev = obj_sse[i - 1]

        _val = obj_sse[i]
        _nxt = obj_sse[i + 1]

        if first == 0:
            #objects.append(count)
            boundaries.append(-1)
            first += 1
        else:
            if _val == _prev and _val == _nxt:
                #objects.append(count)
                boundaries.append(1 * count)
            elif _val != _prev and _val == _nxt:
                count += 1
                #objects.append(count)
                boundaries.append(-1 * count)
            else:
                #objects.append(count)
                boundaries.append(-1 * count)
    boundaries.append(-1)
    return np.array( boundaries )


def find_sse( pdb, sse=['E', 'H'] ):
    sse_sets, sse_cnts, sse_objs = {}, [], []
    count = 0
    pdbca = pdb[pdb.atomtype.isin(['CA',])]
    for i in range(len(pdbca)):
        if i == 0:
            dssp_prev = pdbca.iloc[i].dssp3
            sse_set, sse_cnt = [], []
            sse_set.append(int(pdbca.iloc[i].id))
            sse_cnt.append(count)
        else:
            if pdbca.iloc[i].dssp3 == dssp_prev:
                sse_set.append(int(pdbca.iloc[i].id))
                sse_cnt.append(count)
                if pdbca.iloc[i].dssp3 in sse:
                    sse_objs.append(1)
                else:
                    sse_objs.append(-1)
                dssp_prev = pdbca.iloc[i].dssp3
            else:
                sse_sets[count] = sse_set
                sse_cnts.extend(sse_cnt)

                count += 1
                sse_set, sse_cnt = [], []

                sse_set.append(int(pdbca.iloc[i].id))
                sse_cnt.append(count)
                if pdbca.iloc[i].dssp3 in sse:
                    sse_objs.append(1)
                else:
                    sse_objs.append(-1)
                dssp_prev = pdbca.iloc[i].dssp3
    # last one
    sse_sets[count] = sse_set
    sse_cnts.extend(sse_cnt)

    count += 1
    sse_set, sse_cnt = [], []

    sse_set.append(int(pdbca.iloc[i].id))
    sse_cnt.append(count)
    if pdbca.iloc[i].dssp3 in sse:
        sse_objs.append(1)
    else:
        sse_objs.append(-1)
    dssp_prev = pdbca.iloc[i].dssp3
    return sse_sets, np.array(sse_cnts), sse_objs


def pdb_to_info(pdb):
    """
    """
    d_info = {
        'obj': [],
        'length': [],
        'type': [],
        'directionality': [],
        'means': [],
        'local_evecs': []
    }

    pdb   = pdb[~pdb.obj.isna()].drop_duplicates(['atomtype', 'id'])
    pdbbb = pdb[pdb.atomtype.isin(['CA', 'CB', 'O', 'N', 'C'])]

    pdbcaH = pdb[(pdb.atomtype.isin(['CA',])) & (pdb.dssp.isin(['H',])) & (pdb.bound == 1.)]
    #sizes = pdbcaH.groupby('obj').size()
    #sizes_selected = [ind for (i, ind) in zip(sizes, sizes.index) if i > 2]
    #pdbcaH = pdbcaH[(pdbcaH.bound == 1.) & (pdbcaH.obj.isin(sizes_selected))]

    pdbcaE = pdb[(pdb.atomtype.isin(['CA',])) & (pdb.dssp.isin(['E',])) & (pdb.bound == 1.)]
    #sizes = pdbcaE.groupby('obj').size()
    #sizes_selected = [ind for (i, ind) in zip(sizes, sizes.index) if i > 2]
    #pdbcaE = pdbcaH[(pdbcaE.bound == 1.) & (pdbcaE.obj.isin(sizes_selected))]

    pdbcaEH = pdb[(pdb.atomtype.isin(['CA',])) & (pdb.dssp.isin(['H', 'E']))]

    if len(pdbcaE) > ( len(pdbcaH) / 5 ):
        _, evecs, evecs_cen = change_basis(pdbcaE,)
    else:
        _, evecs, evecs_cen = change_basis(pdbcaH,)

    pdb_b, _, _ = change_basis(pdbcaEH,)

    for i in range( int( max(pdb_b.obj) + 1 ) ):
        local = pdb_b[pdb_b.obj == i]

        if len(local) == 0.:
            continue
        # local eigenvectors
        _, local_evecs, local_evecs_cen = change_basis(local,)

        # directionality
        if len(local) > 3:
            ri = local.iloc[1][['x', 'y', 'z']].values
            rj = local.iloc[-2][['x', 'y', 'z']].values
        if len(local) > 1:
            ri = local.iloc[0][['x', 'y', 'z']].values
            rj = local.iloc[-1][['x', 'y', 'z']].values
        else:
            ri = local.iloc[0][['x', 'y', 'z']].values
            rj = local.iloc[0][['x', 'y', 'z']].values
        rv = rj - ri
        mf = evecs_cen[-1][-1] - evecs_cen[-1][0]

        if rv.dot(mf) >= 0.:
            d_info['directionality'].append(1)
        else:
            d_info['directionality'].append(-1)

        d_info['obj'].append(i)
        d_info['length'].append(len(local))
        d_info['type'].append(local.iloc[0].dssp)
        d_info['means'].append(local[['x', 'y', 'z']].mean().values)
        d_info['local_evecs'].append([local_evecs])
        #d_info['means'].append(local[['x', 'y', 'z']].median().values)

    return pd.DataFrame(d_info), pdbbb, evecs


def rotation_matrix(axis, angle):
    """
    Euler-Rodrigues formula for rotation matrix
    """
    # Normalize the axis
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])


def add_cbeta( pdb ):
    """
    """
    protein = []
    distance = 1.52
    angle = np.deg2rad(122.6)
    dihedral = np.deg2rad(140.)

    dtypes = {
        'id': int, 'atomnum': int,
        'res3aa': str, 'res1aa': str, 'atomtype': str, 'chain': str,
        'x': float, 'y': float, 'z': float }

    atomnum = 1
    for i in pdb.id.drop_duplicates().values:
        atomnums = np.array(range(atomnum, atomnum + 4))
        atomnum += 4

        residue       = pdb[pdb.id == i]
        residue       = residue.assign(atomnum = atomnums)
        res3aa        = residue.res3aa.values[0]
        res1aa        = residue.res1aa.values[0]
        atomtypes     = residue.atomtype.values
        chain         = residue.chain.values[0]
        secstruct     = residue.secstruct.values[0]
        layer         = residue.layer.values[0]
        layer_element = residue.layer_element.values[0]
        #id_         = residue.id.values[0]

        N  = residue[residue.atomtype == 'N'][['x', 'y', 'z']].values[0]
        C  = residue[residue.atomtype == 'C'][['x', 'y', 'z']].values[0]
        CA = residue[residue.atomtype == 'CA'][['x', 'y', 'z']].values[0]

        q = np.array(CA, dtype='f8') # atom 1, 0
        r = np.array(N, dtype='f8') # atom 2, 1
        s = np.array(C, dtype='f8') # atom 3, 2

        # Vector pointing from q to r
        a = r - q
        # Vector pointing from s to r
        b = r - s

        # Vector of length distance pointing from q to r
        d = distance * a / np.sqrt(np.dot(a, a))
        # Vector normal to plane defined by q, r, s
        normal = np.cross(a, b)

        # Rotate d by the angle around the normal to the plane defined by q, r, s
        d = np.dot(rotation_matrix(normal, angle), d)
        # Rotate d around a by the dihedral
        d = np.dot(rotation_matrix(a, dihedral), d)

        # Add d to the position of q to get the new coordinates of the atom
        p = q + d

        p = np.array([i, atomnum, res3aa, res1aa, 'CB', chain, secstruct,
                      layer, layer_element, p[0], p[1], p[2]])
        p = pd.DataFrame(p, index=['id', 'atomnum', 'res3aa', 'res1aa',
                                   'atomtype', 'chain', 'secstruct', 'layer', 'layer_element',
                                   'x', 'y', 'z']).T

        protein.append(pd.concat([residue, p], sort=False))

    return pd.concat(protein, sort=False).astype(dtypes)


def add_valine( pdb ):
    """
    """
    protein = []

    # Cbeta
    distance = 1.52
    angle = np.deg2rad(110.3)
    dihedral = np.deg2rad(-122.6)

    # Cgamma 1
    distance1 = 2.7
    angle1 = np.deg2rad(90.2)
    dihedral1 = np.deg2rad(-120.6) # 123.3

    # Cgamma 2
    distance2 = 2.5
    angle2 = np.deg2rad(144.1)
    dihedral2 = np.deg2rad(-145.5) # 149.5

    for i in pdb.id.drop_duplicates().values:
        residue = pdb[pdb.id == i]

        res3aa      = residue.res3aa.values[0]
        res1aa      = residue.res1aa.values[0]
        atomtypes   = residue.atomtype.values
        chain       = residue.chain.values[0]
        #id_         = residue.id.values[0]

        N  = residue[residue.atomtype == 'N'][['x', 'y', 'z']].values[0]
        C  = residue[residue.atomtype == 'C'][['x', 'y', 'z']].values[0]
        CA = residue[residue.atomtype == 'CA'][['x', 'y', 'z']].values[0]

        q = np.array(CA, dtype='f8') # atom 1, 0
        r = np.array(N, dtype='f8') # atom 2, 1
        s = np.array(C, dtype='f8') # atom 3, 2

        # Vector pointing from q to r
        a = r - q
        # Vector pointing from s to r
        b = r - s

        # Vector of length distance pointing from q to r
        d  = distance * a / np.sqrt(np.dot(a, a))
        d1 = distance1 * a / np.sqrt(np.dot(a, a))
        d2 = distance2 * a / np.sqrt(np.dot(a, a))

        # Vector normal to plane defined by q, r, s
        normal = np.cross(a, b)

        # Rotate d by the angle around the normal to the plane defined by q, r, s
        d  = np.dot(rotation_matrix(normal, angle), d)
        d1 = np.dot(rotation_matrix(normal, angle1), d1)
        d2 = np.dot(rotation_matrix(normal, angle2), d2)

        # Rotate d around a by the dihedral
        d  = np.dot(rotation_matrix(a, dihedral), d)
        d1 = np.dot(rotation_matrix(a, dihedral1), d1)
        d2 = np.dot(rotation_matrix(a, dihedral2), d2)

        # Add d to the position of q to get the new coordinates of the atom
        p  = q + d
        p1 = q + d1
        p2 = q + d2

        p  = np.array([i, res3aa, res1aa, 'CB',  chain, p[0],  p[1],  p[2]])
        p1 = np.array([i, res3aa, res1aa, 'CG1', chain, p1[0], p1[1], p1[2]])
        p2 = np.array([i, res3aa, res1aa, 'CG2', chain, p2[0], p2[1], p2[2]])

        p  = pd.DataFrame(p,  index=['id', 'res3aa', 'res1aa', 'atomtype', 'chain', 'x', 'y', 'z']).T
        p1 = pd.DataFrame(p1, index=['id', 'res3aa', 'res1aa', 'atomtype', 'chain', 'x', 'y', 'z']).T
        p2 = pd.DataFrame(p2, index=['id', 'res3aa', 'res1aa', 'atomtype', 'chain', 'x', 'y', 'z']).T

        protein.append(pd.concat([residue, p],  sort=False))
        protein.append(pd.concat([residue, p1], sort=False))
        protein.append(pd.concat([residue, p2], sort=False))

    return pd.concat(protein, sort=False).drop_duplicates()


def make_residue():
    """
    """
    # generate residue object
    d_residue = {
        'xs': [-8.5540e-01, 6.1600e-02,  2.9600e-02, 1.0696e+00 ], #, -3.0540e-01],
        'ys': [1.5222e+00,  4.3720e-01, -6.2880e-01, -1.1498e+00], #, -1.8080e-01],
        'zs': [1.0000e-03,  3.0400e-01, -7.6600e-01, -1.1940e+00], #, 1.6550e+00],
        'atomtype': ['N', 'CA', 'C', 'O'], #, 'CB'],
        'chain': ['A', 'A', 'A', 'A'],
        'id': [1, 1, 1, 1],
        'res1aa': ['V', 'V', 'V', 'V'],
        'res3aa': ['VAL', 'VAL', 'VAL', 'VAL']
    }
    residue = pd.DataFrame( d_residue )

    # randomly rotate residue
    # Random rotations
    x_rots = random.randint(-180, 180)
    y_rots = random.randint(-180, 180)
    z_rots = random.randint(-180, 180)

    r = R.from_euler( 'xyz', [x_rots, y_rots, z_rots], degrees=True)
    c = r.apply( residue[['xs', 'ys', 'zs']].astype(float).values )

    residue['x'] = c[:, 0]
    residue['y'] = c[:, 1]
    residue['z'] = c[:, 2]

    return residue.drop(labels=['xs', 'ys', 'zs'], axis=1)


def add_loops( naive, verbose=False ):
    """
    """
    chunks = []
    d = {
        'atomtype': [],
        'chain': [],
        'res1aa': [],
        'res3aa': [],
        'id': [],
        'secstruct': [],
        'layer': [],
        'layer_element': [],
        'x': [],
        'y': [],
        'z': []}

    loops = []
    naiveca = naive[naive.atomtype == 'CA']
    for i in range(1, len(naiveca)):
        layer_prev, layer                 = naiveca.iloc[i - 1]['layer'],         naiveca.iloc[i]['layer']
        layer_element_prev, layer_element = naiveca.iloc[i - 1]['layer_element'], naiveca.iloc[i]['layer_element']
        if layer_prev != layer or layer_element_prev != layer_element:
            lent =  naiveca.iloc[i]['id'] - naiveca.iloc[i - 1]['id']
            loops.append( [lent, naiveca.iloc[i - 1]['id'], naiveca.iloc[i]['id']] )

    stop_prev = 0
    bag = []
    for i in range(len(loops)):
        # get loop informations
        loop  = loops[i]
        lent, start, stop = loop[0], loop[1], loop[2]
        if verbose == True:
            print('loop {}: length {}, from {} - {}'.format(i + 1, lent, start, stop))

        # get sketch chunks
        naive_part = naive[naive['id'].between(stop_prev, start, inclusive='both')]
        d['x'].extend(naive_part['x'].values)
        d['y'].extend(naive_part['y'].values)
        d['z'].extend(naive_part['z'].values)
        d['id'].extend(naive_part['id'].values)
        d['chain'].extend(naive_part['chain'].values)
        d['atomtype'].extend(naive_part['atomtype'].values)
        d['res1aa'].extend(naive_part['res1aa'].values)
        d['res3aa'].extend(naive_part['res3aa'].values)
        d['secstruct'].extend(naive_part['secstruct'].tolist())
        d['layer'].extend(naive_part['layer'].tolist())
        d['layer_element'].extend(naive_part['layer_element'].tolist())

        p1 = naive[(naive.atomtype == 'CA') &
                   (naive.id == start)][['x', 'y', 'z']].astype(float).values
        p2 = naive[(naive.atomtype == 'CA') &
                   (naive.id == stop)][['x', 'y', 'z']].astype(float).values

        vec  = p2 - p1
        vals = np.linspace(p1, p2, lent)
        if verbose == True:
            print('inserting residues...')

        chain = naive[naive.id == start].chain.drop_duplicates().values[0]
        for j in range(lent - 1):
            # generate residue
            residue = make_residue()

            if verbose == True:
                print('{} residue'.format(j+1))
            rs = residue[['x', 'y', 'z']].astype(float).values + p1 + ( (j + .5) / lent ) * vec
            #rs = residue[['x', 'y', 'z']].astype(float).values + p1 + ((j/lent)) * v12

            d['x'].extend(rs[:, 0])
            d['y'].extend(rs[:, 1])
            d['z'].extend(rs[:, 2])
            d['id'].extend([start + j + 1] * 4)
            d['chain'].extend([chain] * 4)
            d['atomtype'].extend(['N', 'CA', 'C', 'O'])
            d['res1aa'].extend(['G'] * 4)
            d['res3aa'].extend(['GLY'] * 4)
            d['secstruct'].extend(['L'] * 4)
            d['layer'].extend(['X'] * 4)
            d['layer_element'].extend([i] * 4)

        stop_prev = stop

    if verbose == True:
        print('appending last chunk...')
    naive_part = naive[naive['id'] >= stop]
    d['x'].extend(naive_part['x'].values)
    d['y'].extend(naive_part['y'].values)
    d['z'].extend(naive_part['z'].values)
    d['id'].extend(naive_part['id'].values)
    d['chain'].extend(naive_part['chain'].values)
    d['atomtype'].extend(naive_part['atomtype'].values)
    d['res1aa'].extend(naive_part['res1aa'].values)
    d['res3aa'].extend(naive_part['res3aa'].values)
    d['secstruct'].extend(naive_part['secstruct'].values)
    d['layer'].extend(naive_part['layer'].values)
    d['layer_element'].extend(naive_part['layer_element'].values)

    return pd.DataFrame(d)


def remove_overlaps( df ):
    """
    """
    container = []
    starter = True
    for i in range(len(df)):
        row = df.iloc[i]
        if starter == False or (row.dssp3 == 'H' or row.dssp3 == 'E'):
            container.append(pd.DataFrame(row).T)
            starter = False

    t = pd.concat(container).iloc[::-1]

    container = []
    starter = True
    for i in range(len(t)):
        row = t.iloc[i]
        if starter == False or (row.dssp3 == 'H' or row.dssp3 == 'E'):
            container.append(pd.DataFrame(row).T)
            starter = False

    return pd.concat(container).iloc[::-1]
