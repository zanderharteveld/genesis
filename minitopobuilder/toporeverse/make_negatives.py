"""
.. codeauthor:: Zander Harteveld <zandermilanh@gmail.com>

.. affiliation::
    Laboratory of Protein Design and Immunoengineering <lpdi.epfl.ch>
    Bruno Correia <bruno.correia@epfl.ch>
"""

# Standard Libraries
import os
import sys
import math
import argparse
import glob
import textwrap
from pathlib import Path
import subprocess 
# External Libraries
import numpy as np
import pandas as pd
import torch

# This library
#sys.path.append("/work/upcorreia/users/hartevel/bin/topoGoCurvy/")
#import topogocurvy as tgc


def create_parser():
    """
    Create a CLI parser.
    :return: the parser object.
    """
    parse = argparse.ArgumentParser()
    parse.add_argument("--topology",     "-t", type=str, nargs=1, help="Topology of interested.")
    parse.add_argument("--architecture", "-a", type=str, nargs=1, help="Architecture of interested.")
    #parse.add_argument("--slurm", action="store_true", help="Use slurm accelerated parallel system.")
    #parse.add_argument("--partition", type=str, nargs=1, default=['serial'], help="Partition for slurm cluster system (default: serial).")
    return parse


def parse_args(parser):
	"""
	Parse the arguments of a parser object.
	:param parser: the parser object.
	:return: the specified arguments
	"""
	args = parser.parse_args()
	return args


def make_script(top_folder, chunks):
    #fcsv   = './{}/processed01_csvs'.format(top_folder)
    fpdb    = './{}/processed01_pdbs'.format(top_folder)
    ftorch  = './{}/processed01_torch'.format(top_folder)
    outnegs = './{}/processed01_negatives'.format(top_folder)
    os.makedirs(outnegs, exist_ok=True)
    
    # create python script
    script = textwrap.dedent("""\
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
# pyrosetta related
from pyrosetta import *
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.protocols.relax import FastRelax
#init('-hb_cen_soft -relax:default_repeats 5 -default_max_cycles 200 -out:level 100 -ignore_unrecognized_res true')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
splitfile = str(sys.argv[1])
number = str(sys.argv[2])
weight = random.uniform(0.65, 0.95)
print(splitfile,number)
with open(splitfile, 'r') as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]
for line in lines:
    try:
        info = torch.load('../processed01_torch/{1}.pt'.format(line))
    except: continue
    info['negative'] = {1}
    info['negative']['minSketch'] = {1}
    info['negative']['minNative'] = {1}
    info['negative']['relax1Native'] = {1}
    info['negative']['relax2Native'] = {1}
    info['negative']['relax3Native'] = {1}
    info['negative']['relax4Native'] = {1}
    info['negative']['relax5Native'] = {1}
    info['negative']['relax6Native'] = {1}
    info['negative']['relax7Native'] = {1}
    info['negative']['relax8Native'] = {1}
    init('-hb_cen_soft -relax:default_repeats 5 -default_max_cycles 200 -ignore_unrecognized_res true -use_time_as_seed true -seed_offset {1}'.format(number))
    # Set up energy function with affinity for atom-pair-constraints, dihedrals and angles
    sfxn = get_fa_scorefxn()
    stm = pyrosetta.rosetta.core.scoring.ScoreTypeManager()
    atom_pair_constraint = stm.score_type_from_name("atom_pair_constraint")
    sfxn.set_weight(atom_pair_constraint, 1.0)
    angle_constraint = stm.score_type_from_name("angle_constraint")
    sfxn.set_weight(angle_constraint, 1.0)
    dihedral_constraint = stm.score_type_from_name("dihedral_constraint")
    sfxn.set_weight(dihedral_constraint, 1.0)
    hbond_lr_bb = stm.score_type_from_name("hbond_lr_bb")
    sfxn.set_weight(hbond_lr_bb, 3.0)
    hbond_sr_bb = stm.score_type_from_name("hbond_sr_bb")
    sfxn.set_weight(hbond_lr_bb, 1.5)
    try:
        ssStr = ''.join(info['topology_ss'])
        sequence = ssStr.replace('L', 'A').replace('H', 'L').replace('E', 'L')
        pose_sketch = pose_from_pdb('../processed01_pdbs/{1}-naive.pdb'.format(line))
        pose_native = pose_from_pdb('../processed01_pdbs/{1}-native.pdb'.format(line))
        pose_fresh1 = pose_from_sequence(''.join(info['res_type']))
        pose_fresh2 = pose_from_sequence(''.join(info['res_type'])) 
    except: continue
    # Movemap
    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)
    minMover = MinMover(mmap, sfxn, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    minMover.max_iter(1000)
    relaxMover = FastRelax(5)
    relaxMover.set_movemap(mmap)
    relaxMover.set_scorefxn(sfxn)
    ## SKETCH CONSTRAINTS
    # Secondary structure selector
    # to not put constraints onto the loops
    secStrucSelector1 = pyrosetta.rosetta.core.select.residue_selector.SecondaryStructureSelector()
    secStrucSelector1.set_use_dssp(False)
    secStrucSelector1.set_pose_secstruct('{1}'.format(ssStr))
    secStrucSelector1.set_minE(1)
    secStrucSelector1.set_minH(1)
    secStrucSelector1.set_include_terminal_loops(True)
    secStrucSelector1.set_selected_ss('HE')
    # Create constraints
    # distances
    constGen1 = pyrosetta.rosetta.protocols.fold_from_loops.constraint_generator.SegmentedAtomPairConstraintGenerator()
    constGen1.set_residue_selector(secStrucSelector1)
    constGen1.set_do_inner(True)
    constGen1.set_inner_ca_only(True)
    constGen1.set_inner_min_seq_sep(1)
    constGen1.set_inner_sd(.5)
    constGen1.set_inner_use_harmonic_function(True)
    constGen1.set_do_outer(True)
    constGen1.set_outer_ca_only(True)
    constGen1.set_outer_max_distance(40)
    constGen1.set_outer_sd(2.0)
    constGen1.set_outer_use_harmonic_function(True)
    constGen1.set_outer_weight(1.0)
    # torsions
    dihGen1 = pyrosetta.rosetta.protocols.constraint_generator.DihedralConstraintGenerator()
    tphi = pyrosetta.rosetta.core.id.MainchainTorsionType.phi_dihedral
    tpsi = pyrosetta.rosetta.core.id.MainchainTorsionType.phi_dihedral
    tomega = pyrosetta.rosetta.core.id.MainchainTorsionType.omega_dihedral
    dihGen1.set_residue_selector(secStrucSelector1)
    dihGen1.set_torsion_type(tphi)
    dihGen1.set_torsion_type(tpsi)
    dihGen1.set_torsion_type(tomega)
    distance_constraints1 = constGen1.apply(pose_native)
    dihedral_constraints1 = dihGen1.apply(pose_native)
    runpose11 = pose_fresh1.clone()
    _ = runpose11.add_constraints(dihedral_constraints1)
    _ = runpose11.add_constraints(distance_constraints1)
    minMover.apply(runpose11)
    runpose11.dump_pdb('./{1}-negativeMinSketch-{1}.pdb'.format(line, number))
    xca = torch.stack([
        torch.tensor(runpose11.residue(ik+1).xyz("CA")) for ik in range(len(runpose11.sequence()))])
    xc = torch.stack([
        torch.tensor(runpose11.residue(ik+1).xyz("C")) for ik in range(len(runpose11.sequence()))])
    xn = torch.stack([
        torch.tensor(runpose11.residue(ik+1).xyz("N")) for ik in range(len(runpose11.sequence()))])
    xo = torch.stack([
        torch.tensor(runpose11.residue(ik+1).xyz("O")) for ik in range(len(runpose11.sequence()))])    
    info['negative']['minSketch'][str(number)] = {1}
    info['negative']['minSketch'][str(number)]['CA'] = xca
    info['negative']['minSketch'][str(number)]['C' ] = xc
    info['negative']['minSketch'][str(number)]['N' ] = xn
    info['negative']['minSketch'][str(number)]['O' ] = xo
    ## NATIVE CONSTRAINTS
    # Secondary structure selector
    # to not put constraints onto the loops
    secStrucSelector2 = pyrosetta.rosetta.core.select.residue_selector.SecondaryStructureSelector()
    secStrucSelector2.set_use_dssp(True)
    secStrucSelector2.set_minE(1)
    secStrucSelector2.set_minH(1)
    secStrucSelector2.set_include_terminal_loops(True)
    secStrucSelector2.set_selected_ss('L')
    secStrucSelector2.set_selected_ss('HE')
    # Create constraints
    # distances
    constGen2 = pyrosetta.rosetta.protocols.fold_from_loops.constraint_generator.SegmentedAtomPairConstraintGenerator()
    constGen2.set_residue_selector(secStrucSelector2)
    constGen2.set_do_inner(True)
    constGen2.set_inner_ca_only(True)
    constGen2.set_inner_min_seq_sep(1)
    constGen2.set_inner_sd(1.)
    constGen2.set_inner_use_harmonic_function(True)
    constGen2.set_do_outer(True)
    constGen2.set_outer_ca_only(True)
    constGen2.set_outer_max_distance(40)
    constGen2.set_outer_sd(2.0)
    constGen2.set_outer_use_harmonic_function(True)
    constGen2.set_outer_weight(.9)
    # torsions
    dihGen2 = pyrosetta.rosetta.protocols.constraint_generator.DihedralConstraintGenerator()
    tphi = pyrosetta.rosetta.core.id.MainchainTorsionType.phi_dihedral
    tpsi = pyrosetta.rosetta.core.id.MainchainTorsionType.phi_dihedral
    tomega = pyrosetta.rosetta.core.id.MainchainTorsionType.omega_dihedral
    dihGen2.set_residue_selector(secStrucSelector2)
    dihGen2.set_torsion_type(tphi)
    dihGen2.set_torsion_type(tpsi)
    dihGen2.set_torsion_type(tomega) 
    distance_constraints2 = constGen2.apply(pose_native)
    dihedral_constraints2 = dihGen2.apply(pose_native)
    runpose21 = pose_fresh2.clone()
    _ = runpose21.add_constraints(dihedral_constraints2)
    _ = runpose21.add_constraints(distance_constraints2)
    minMover.apply(runpose21)
    runpose21.dump_pdb('./{1}-negativeMinNative-{1}.pdb'.format(line,number))
    xca = torch.stack([
        torch.tensor(runpose21.residue(ik+1).xyz("CA")) for ik in range(len(runpose21.sequence()))])
    xc = torch.stack([
        torch.tensor(runpose21.residue(ik+1).xyz("C")) for ik in range(len(runpose21.sequence()))])
    xn = torch.stack([
        torch.tensor(runpose21.residue(ik+1).xyz("N")) for ik in range(len(runpose21.sequence()))])
    xo = torch.stack([
        torch.tensor(runpose21.residue(ik+1).xyz("O")) for ik in range(len(runpose21.sequence()))])
    info['negative']['minNative'][str(number)] = {1}
    info['negative']['minNative'][str(number)]['CA'] = xca
    info['negative']['minNative'][str(number)]['C' ] = xc
    info['negative']['minNative'][str(number)]['N' ] = xn
    info['negative']['minNative'][str(number)]['O' ] = xo
    ## RELAX WITH REPULSIONS 1
    custom_relax1 = pyrosetta.rosetta.std.vector_std_string(
        ["repeat 1", 
         "ramp_repack_min 2.5  2.0     0.1",
         "accept_to_best", 
         "endrepeat"])
    relaxMover2 = FastRelax(1)
    relaxMover2.set_movemap(mmap)
    relaxMover2.set_scorefxn(sfxn)
    relaxMover2.set_script_from_lines(custom_relax1)
    runpose31 = pose_native.clone()
    for i in range(len(runpose31.sequence())):
        mutate = rosetta.protocols.simple_moves.MutateResidue(i+1,'VAL')
        mutate.apply(runpose31)
    relaxMover2.apply(runpose31)
    runpose31.dump_pdb('./{1}-negativeRelax1Native-{1}.pdb'.format(line,number))
    xca = torch.stack([
        torch.tensor(runpose31.residue(ik+1).xyz("CA")) for ik in range(len(runpose31.sequence()))])
    xc = torch.stack([
        torch.tensor(runpose31.residue(ik+1).xyz("C")) for ik in range(len(runpose31.sequence()))])
    xn = torch.stack([
        torch.tensor(runpose31.residue(ik+1).xyz("N")) for ik in range(len(runpose31.sequence()))])
    xo = torch.stack([
        torch.tensor(runpose31.residue(ik+1).xyz("O")) for ik in range(len(runpose31.sequence()))])
    info['negative']['relax1Native'][str(number)] = {1}
    info['negative']['relax1Native'][str(number)]['CA'] = xca
    info['negative']['relax1Native'][str(number)]['C' ] = xc
    info['negative']['relax1Native'][str(number)]['N' ] = xn
    info['negative']['relax1Native'][str(number)]['O' ] = xo
    ## RELAX WITH REPULSIONS 2
    custom_relax2 = pyrosetta.rosetta.std.vector_std_string(
            ["repeat 1", 
             "ramp_repack_min 3.5  3.0     0.1", 
             "accept_to_best", 
             "endrepeat"])
    relaxMover3 = FastRelax(1)
    relaxMover3.set_movemap(mmap)
    relaxMover3.set_scorefxn(sfxn)
    relaxMover3.set_script_from_lines(custom_relax2)
    runpose32 = pose_native.clone()
    for i in range(len(runpose32.sequence())):
        mutate = rosetta.protocols.simple_moves.MutateResidue(i+1,'VAL')
        mutate.apply(runpose32)
    relaxMover3.apply(runpose32)
    runpose32.dump_pdb('./{1}-negativeRelax2Native-{1}.pdb'.format(line,number))
    xca = torch.stack([
        torch.tensor(runpose32.residue(ik+1).xyz("CA")) for ik in range(len(runpose32.sequence()))])
    xc = torch.stack([
        torch.tensor(runpose32.residue(ik+1).xyz("C")) for ik in range(len(runpose32.sequence()))])
    xn = torch.stack([
        torch.tensor(runpose32.residue(ik+1).xyz("N")) for ik in range(len(runpose32.sequence()))])
    xo = torch.stack([
        torch.tensor(runpose32.residue(ik+1).xyz("O")) for ik in range(len(runpose32.sequence()))])
    info['negative']['relax2Native'][str(number)] = {1}
    info['negative']['relax2Native'][str(number)]['CA'] = xca
    info['negative']['relax2Native'][str(number)]['C' ] = xc
    info['negative']['relax2Native'][str(number)]['N' ] = xn
    info['negative']['relax2Native'][str(number)]['O' ] = xo
    ## RELAX WITH REPULSIONS 3
    custom_relax3 = pyrosetta.rosetta.std.vector_std_string(
            ["repeat 1",
             "ramp_repack_min 2.5  2.0     0.1",
             "accept_to_best",
             "endrepeat"])
    relaxMover4 = FastRelax(1)
    relaxMover4.set_movemap(mmap)
    relaxMover4.set_scorefxn(sfxn)
    relaxMover4.set_script_from_lines(custom_relax3)
    runpose33 = pose_native.clone()
    for i in range(len(runpose33.sequence())):
        mutate = rosetta.protocols.simple_moves.MutateResidue(i+1,'VAL')
        mutate.apply(runpose33)
    relaxMover4.apply(runpose33)
    runpose33.dump_pdb('./{1}-negativeRelax3Native-{1}.pdb'.format(line,number))
    xca = torch.stack([
        torch.tensor(runpose33.residue(ik+1).xyz("CA")) for ik in range(len(runpose33.sequence()))])
    xc = torch.stack([
        torch.tensor(runpose33.residue(ik+1).xyz("C")) for ik in range(len(runpose33.sequence()))])
    xn = torch.stack([
        torch.tensor(runpose33.residue(ik+1).xyz("N")) for ik in range(len(runpose33.sequence()))])
    xo = torch.stack([
        torch.tensor(runpose33.residue(ik+1).xyz("O")) for ik in range(len(runpose33.sequence()))])
    info['negative']['relax3Native'][str(number)] = {1}
    info['negative']['relax3Native'][str(number)]['CA'] = xca
    info['negative']['relax3Native'][str(number)]['C' ] = xc
    info['negative']['relax3Native'][str(number)]['N' ] = xn
    info['negative']['relax3Native'][str(number)]['O' ] = xo
    ## RELAX WITH REPULSIONS 4
    custom_relax4 = pyrosetta.rosetta.std.vector_std_string(
            ["repeat 1",
             "ramp_repack_min 2.0  1.0     0.1",
             "accept_to_best",
             "endrepeat"])
    relaxMover5 = FastRelax(1)
    relaxMover5.set_movemap(mmap)
    relaxMover5.set_scorefxn(sfxn)
    relaxMover5.set_script_from_lines(custom_relax4)
    runpose34 = pose_native.clone()
    for i in range(len(runpose34.sequence())):
        mutate = rosetta.protocols.simple_moves.MutateResidue(i+1,'VAL')
        mutate.apply(runpose34)
    relaxMover5.apply(runpose34)
    runpose34.dump_pdb('./{1}-negativeRelax4Native-{1}.pdb'.format(line,number))
    xca = torch.stack([
        torch.tensor(runpose34.residue(ik+1).xyz("CA")) for ik in range(len(runpose34.sequence()))])
    xc = torch.stack([
        torch.tensor(runpose34.residue(ik+1).xyz("C")) for ik in range(len(runpose34.sequence()))])
    xn = torch.stack([
        torch.tensor(runpose34.residue(ik+1).xyz("N")) for ik in range(len(runpose34.sequence()))])
    xo = torch.stack([
        torch.tensor(runpose34.residue(ik+1).xyz("O")) for ik in range(len(runpose34.sequence()))])
    info['negative']['relax4Native'][str(number)] = {1}
    info['negative']['relax4Native'][str(number)]['CA'] = xca
    info['negative']['relax4Native'][str(number)]['C' ] = xc
    info['negative']['relax4Native'][str(number)]['N' ] = xn
    info['negative']['relax4Native'][str(number)]['O' ] = xo
    ## RELAX WITH REPULSIONS 5
    custom_relax5 = pyrosetta.rosetta.std.vector_std_string(
            ["repeat 1",
             "ramp_repack_min 1.5  1.5     0.1",
             "accept_to_best",
             "endrepeat"])
    relaxMover6 = FastRelax(1)
    relaxMover6.set_movemap(mmap)
    relaxMover6.set_scorefxn(sfxn)
    relaxMover6.set_script_from_lines(custom_relax5)
    runpose35 = pose_native.clone()
    for i in range(len(runpose35.sequence())):
        mutate = rosetta.protocols.simple_moves.MutateResidue(i+1,'VAL')
        mutate.apply(runpose35)
    relaxMover6.apply(runpose35)
    runpose35.dump_pdb('./{1}-negativeRelax5Native-{1}.pdb'.format(line,number))
    xca = torch.stack([
        torch.tensor(runpose35.residue(ik+1).xyz("CA")) for ik in range(len(runpose35.sequence()))])
    xc = torch.stack([
        torch.tensor(runpose35.residue(ik+1).xyz("C")) for ik in range(len(runpose35.sequence()))])
    xn = torch.stack([
        torch.tensor(runpose35.residue(ik+1).xyz("N")) for ik in range(len(runpose35.sequence()))])
    xo = torch.stack([
        torch.tensor(runpose35.residue(ik+1).xyz("O")) for ik in range(len(runpose35.sequence()))])
    info['negative']['relax5Native'][str(number)] = {1}
    info['negative']['relax5Native'][str(number)]['CA'] = xca
    info['negative']['relax5Native'][str(number)]['C' ] = xc
    info['negative']['relax5Native'][str(number)]['N' ] = xn
    info['negative']['relax5Native'][str(number)]['O' ] = xo
    ## RELAX WITH REPULSIONS 6
    custom_relax6 = pyrosetta.rosetta.std.vector_std_string(
            ["repeat 1",
             "ramp_repack_min 1.5  1.0     0.1",
             "accept_to_best",
             "endrepeat"])
    relaxMover7 = FastRelax(1)
    relaxMover7.set_movemap(mmap)
    relaxMover7.set_scorefxn(sfxn)
    relaxMover7.set_script_from_lines(custom_relax6)
    runpose36 = pose_native.clone()
    for i in range(len(runpose36.sequence())):
        mutate = rosetta.protocols.simple_moves.MutateResidue(i+1,'VAL')
        mutate.apply(runpose36)
    relaxMover7.apply(runpose36)
    runpose36.dump_pdb('./{1}-negativeRelax6Native-{1}.pdb'.format(line,number))
    xca = torch.stack([
        torch.tensor(runpose36.residue(ik+1).xyz("CA")) for ik in range(len(runpose36.sequence()))])
    xc = torch.stack([
        torch.tensor(runpose36.residue(ik+1).xyz("C")) for ik in range(len(runpose36.sequence()))])
    xn = torch.stack([
        torch.tensor(runpose36.residue(ik+1).xyz("N")) for ik in range(len(runpose36.sequence()))])
    xo = torch.stack([
        torch.tensor(runpose36.residue(ik+1).xyz("O")) for ik in range(len(runpose36.sequence()))])
    info['negative']['relax6Native'][str(number)] = {1}
    info['negative']['relax6Native'][str(number)]['CA'] = xca
    info['negative']['relax6Native'][str(number)]['C' ] = xc
    info['negative']['relax6Native'][str(number)]['N' ] = xn
    info['negative']['relax6Native'][str(number)]['O' ] = xo
    ## RELAX WITH REPULSIONS 7
    custom_relax7 = pyrosetta.rosetta.std.vector_std_string(
            ["repeat 1",
             "ramp_repack_min 1.2  0.9     0.1",
             "accept_to_best",
             "endrepeat"])
    relaxMover8 = FastRelax(1)
    relaxMover8.set_movemap(mmap)
    relaxMover8.set_scorefxn(sfxn)
    relaxMover8.set_script_from_lines(custom_relax7)
    runpose37 = pose_native.clone()
    for i in range(len(runpose37.sequence())):
        mutate = rosetta.protocols.simple_moves.MutateResidue(i+1,'VAL')
        mutate.apply(runpose37)
    relaxMover8.apply(runpose37)
    runpose37.dump_pdb('./{1}-negativeRelax7Native-{1}.pdb'.format(line,number))
    xca = torch.stack([
        torch.tensor(runpose37.residue(ik+1).xyz("CA")) for ik in range(len(runpose37.sequence()))])
    xc = torch.stack([
        torch.tensor(runpose37.residue(ik+1).xyz("C")) for ik in range(len(runpose37.sequence()))])
    xn = torch.stack([
        torch.tensor(runpose37.residue(ik+1).xyz("N")) for ik in range(len(runpose37.sequence()))])
    xo = torch.stack([
        torch.tensor(runpose37.residue(ik+1).xyz("O")) for ik in range(len(runpose37.sequence()))])
    info['negative']['relax7Native'][str(number)] = {1}
    info['negative']['relax7Native'][str(number)]['CA'] = xca
    info['negative']['relax7Native'][str(number)]['C' ] = xc
    info['negative']['relax7Native'][str(number)]['N' ] = xn
    info['negative']['relax7Native'][str(number)]['O' ] = xo
    ## RELAX WITH REPULSIONS 8
    custom_relax8 = pyrosetta.rosetta.std.vector_std_string(
            ["repeat 1",
             "ramp_repack_min 0.9  0.9     0.1",
             "accept_to_best",
             "endrepeat"])
    relaxMover9 = FastRelax(1)
    relaxMover9.set_movemap(mmap)
    relaxMover9.set_scorefxn(sfxn)
    relaxMover9.set_script_from_lines(custom_relax8)
    runpose38 = pose_native.clone()
    for i in range(len(runpose38.sequence())):
        mutate = rosetta.protocols.simple_moves.MutateResidue(i+1,'VAL')
        mutate.apply(runpose38)
    relaxMover9.apply(runpose38)
    runpose38.dump_pdb('./{1}-negativeRelax8Native-{1}.pdb'.format(line,number))
    xca = torch.stack([
        torch.tensor(runpose38.residue(ik+1).xyz("CA")) for ik in range(len(runpose38.sequence()))])
    xc = torch.stack([
        torch.tensor(runpose38.residue(ik+1).xyz("C")) for ik in range(len(runpose38.sequence()))])
    xn = torch.stack([
        torch.tensor(runpose38.residue(ik+1).xyz("N")) for ik in range(len(runpose38.sequence()))])
    xo = torch.stack([
        torch.tensor(runpose38.residue(ik+1).xyz("O")) for ik in range(len(runpose38.sequence()))])
    info['negative']['relax8Native'][str(number)] = {1}
    info['negative']['relax8Native'][str(number)]['CA'] = xca
    info['negative']['relax8Native'][str(number)]['C' ] = xc
    info['negative']['relax8Native'][str(number)]['N' ] = xn
    info['negative']['relax8Native'][str(number)]['O' ] = xo
    torch.save(info, './{1}-{1}.pt'.format(line,number))""".format(top_folder, "{}")) 

    with open("./{}/{}/_processing_negatives.py".format(top_folder, 'processed01_negatives'), "w") as f:
        f.write(script)
    return "./{}/{}/_processing_negatives.py".format(top_folder, 'processed01_negatives')


def create_slurm_file(top_folder, chunks, n_arrays):
    """
    """
    # Create header
    _header_ = """#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition=serial
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --time 24:00:00
#SBATCH --array=1-{}

# activat env
source /work/lpdi/users/hartevel/venv/pytorch_cpu_gcc54/bin/activate && export PYTHONPATH=''

bash execNEGS.sh ${{SLURM_ARRAY_TASK_ID}}

""".format(n_arrays)
    # header for execution script
    _exec_header_ = """#!/bin/bash

SLURM_ARRAY_TASK_ID=$1

"""

    # Create core
    _core_, count = [], 0
    for ii,chk in enumerate(chunks):
        _core_.append('if (( ${{SLURM_ARRAY_TASK_ID}} == {} )); then\n'.format(ii))
        for ik in range(2):
            _core_.append('python -u ./_processing_negatives.py _modeling_negatives_split{}.list {}\n'.format(ii,ik))
        _core_.append('fi\n')
         
    # Assemble everything
    with open('./{}/{}/submitNEGS.slurm'.format(top_folder, 'processed01_negatives'), 'w') as f:
        f.write(_header_)
    with open('./{}/{}/execNEGS.sh'.format(top_folder, 'processed01_negatives'), 'w') as f:
        f.write(_exec_header_)
        f.writelines(_core_)
    return 'submitNEGS.slurm'


def submit_nowait_slurm(top_folder, slurm_file):
    """
    """
    command = ['sbatch']
    command.append(str(slurm_file))
    os.chdir('{}/processed01_negatives/'.format(top_folder))
    p = subprocess.run(command, stdout=subprocess.PIPE)
    os.chdir('../../')

def main():
    """
    Main execution point.
    """
    # Parse arguments
    args = parse_args(create_parser())
    topology = args.topology[0]
    architecture = args.architecture[0]
    #partition    = args.partition[0]

    #wdir = os.getcwd()

    # Create top folder level
    top_folder = "./{}".format(architecture)
    os.makedirs(top_folder, exist_ok=True)
   
    outnegs = './{}/processed01_negatives'.format(top_folder)
    os.makedirs(outnegs, exist_ok=True)
    # Check if checkpoint file exists
    chkpt = Path("./{}/make_negatives.chkpt".format(top_folder))
    if chkpt.is_file():
        print("Found checkpoint {}".format(chkpt))
        sys.exit()

    n_arrays = 500
    filenames = [g.strip().replace('.pt', '') for g in glob.iglob(os.path.join(top_folder, 'processed01_torch/*.pt'))]
    n_per_array = math.ceil( len(filenames) / n_arrays ) 
    chunks    = [filenames[i:i + n_per_array] for i in range(0, len(filenames), n_per_array)] 

    # write splitfiles down
    for i,chunk in enumerate(chunks):
        with open('{}/_modeling_negatives_split{}.list'.format(outnegs, i), 'w') as f:
            for item in chunk:
                f.write('{}\n'.format(os.path.basename(item)))

    # make main python script that will take in the splitfiles
    scriptname = make_script(top_folder, chunks)
    print(scriptname)

    # make submission slurm file
    slurm_file = create_slurm_file(top_folder, chunks, n_arrays)
    print(slurm_file)
    submit_nowait_slurm(top_folder, slurm_file)

    # Create checkpoint file
    with open("./{}/make_negatives.chkpt".format(top_folder), 'w') as f:
        f.write("make_negatives DONE")


if __name__ == "__main__":
	main()



