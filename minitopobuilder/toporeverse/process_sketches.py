"""
.. codeauthor:: Zander Harteveld <zandermilanh@gmail.com>

.. affiliation::
    Laboratory of Protein Design and Immunoengineering <lpdi.epfl.ch>
    Bruno Correia <bruno.correia@epfl.ch>
"""

# Standard Libraries
import os
import sys
import argparse
import time
import math
import subprocess
import glob
import textwrap
from pathlib import Path
from itertools import groupby

# External Libraries
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import torch

# This library
sys.path.append("/work/upcorreia/users/hartevel/bin/topoGoCurvy/")
import topogocurvy as tgc


def create_parser():
    """
    Create a CLI parser.
    :return: the parser object.
    """
    parse = argparse.ArgumentParser()
    parse.add_argument("--topology",     "-t", type=str, nargs=1, help="Topology of interested.")
    parse.add_argument("--architecture", "-a", type=str, nargs=1, help="Architecture of interested.")
    parse.add_argument("--slurm", action="store_true", help="Use slurm accelerated parallel system.")
    parse.add_argument("--partition", type=str, nargs=1, default=['serial'], help="Partition for slurm cluster system (default: serial).")
    return parse

def parse_args(parser):
	"""
	Parse the arguments of a parser object.
	:param parser: the parser object.
	:return: the specified arguments
	"""
	args = parser.parse_args()
	return args

def load_naive_csv(csvfile):
    # load input
    frame = pd.read_csv(csvfile, index_col=0, converters={'match': eval}).reset_index(drop=True)

    # Process input
    topology = os.path.basename(csvfile).split('-')[0].split('.')
    topology_match = [[t[:3]] * 4 * int(t[3:]) for t in topology]
    topology_match = [item for sublist in topology_match for item in sublist]
    secstruct      = [t[-1] for t in topology_match]
    layer          = [t[0] for t in topology_match]
    layer_element  = [int(t[1]) for t in topology_match]
    frame = frame.assign(secstruct=secstruct, layer=layer, layer_element=layer_element)
    return frame

def load_native_csv(csvfile):
    # load input
    frame = pd.read_csv(csvfile, index_col=0, converters={'match': eval}).reset_index(drop=True)

    # Process input
    topology_match = frame.topology_match.tolist()
    loops          = frame.loops.tolist()
    secstruct      = [t[-1] if t != '0' else 'L' for t in topology_match]
    layer          = [t[0]  if t != '0' else 'X' for t in topology_match]
    layer_element  = [int(t[1])  if t != '0' else int(l[0])
                      for t, l in zip(topology_match, loops)]
    frame = frame.assign(secstruct=secstruct, layer=layer, layer_element=layer_element)
    return frame

def _check_chain_breaks(frame):
    mat = frame[frame.atomtype == 'CA'][['x', 'y', 'z']].values
    con = np.diagonal( squareform(pdist(mat)), offset=1 )
    con = True if sum(con[con > 4.4]) > 0. else False
    return con

def _prepare_for_torch(naive, native):
    d = {}
    # Features
    residues = native[native.atomtype == 'CA']
    d['res_num']         = residues['id'].values
    d['res_type']        = residues['res1aa'].values
    d['layer']           = residues['layer'].values
    d['layer_element']   = residues['layer_element'].values
    d['topology_ss']     = residues['secstruct'].values
    d['dssp_ss']         = np.array([s.strip() for s in residues['struct'].values])
    d['dssp_ss_details'] = residues['structdetails'].values
    ### Native ###
    # Back-bone atoms
    d['native'] = {}
    for atom in ['N', 'CA', 'C', 'O', 'CB']:
        coords = torch.from_numpy(native[native.atomtype == atom][['x', 'y', 'z']].values)
        d['native'][atom] = { 'cartesian': coords }
    # Side-chain atoms
    for resi in native['id'].values:
        sidechain = native[ (native['id'] == resi) &
                            (~native['atomtype'].isin(['N', 'CA', 'C', 'O', 'CB'])) ]
        coords    = torch.from_numpy(sidechain[['x', 'y', 'z']].values)
        atomtypes = sidechain[['atomtype']].values.flatten()
        d['native'][resi] = { 'cartesian': coords, 'atomtype': atomtypes }
    ### Naive ###
    d['naive'] = {}
    for atom in ['N', 'CA', 'C', 'O', 'CB']:
        coords = torch.from_numpy(naive[naive.atomtype == atom][['x', 'y', 'z']].values)
        d['naive'][atom] = { 'cartesian': coords }

    return d

def _check_loop_lengths(df, max_length=13):
    """Checks the loop lengths, e.g. returns 1 or more if
    one loops or more are longer the the maxium allowed specified length.
    """
    l_lents = df[df['atomtype'] == 'CA'][
                ['id', 'loops']].groupby(['loops']).count().reset_index()
    l_lents = l_lents[l_lents['loops'] != '0']
    return sum([1 if l > max_length else 0 for l in l_lents['id'].values])

def _check_loop_contains_sse(df):
    """Checks if the loop regions assigned contain secondary structure elements.
    Returns the number of residues assigned to loops with secondary structure
    elements.
    """
    raw_sse = [s.strip() for s in df[df['atomtype'
                        ] == 'CA']['struct'].values]
    loops   = [s.strip() for s in df[df['atomtype'] == 'CA'
                        ]['loops'].values]
    return sum([1 if r in ['E', 'H'] and l.endswith('L')
                else 0 for r,l in zip(raw_sse,loops)])

def _check_naive_to_native_sse_match(naive, native):
    """Checks if all SSE between the naive and the native match their assignment.
    """
    native_sse_raw = native[native.atomtype == 'CA'].struct.str.strip().values
    native_sse_raw = np.array([s if s in ['E', 'H']
                               else 'L' for s in native_sse_raw])
    naive_sse      = naive[naive.atomtype == 'CA'].secstruct.values
    return sum([1 if e2 in ['E', 'H'] and e1 != e2
                else 0 for e1,e2 in zip(native_sse_raw, naive_sse)])

def _check_loop_percentage(df):
    """Checks the loop percentage with respect to the full structure.
    Returns the loop percentage.
    """
    n_resi_loops = sum([1 for l in df[df['atomtype'] == 'CA']['loops'] if l != '0' ])
    n_resi_sses  = sum([1 for l in df[df['atomtype'] == 'CA']['loops'] if l == '0' ])
    return (n_resi_loops / (n_resi_loops + n_resi_sses)) * 100

def _check_sse_percentage_in_loops(df):
    """Checks the SSE percentage within the loops.
    Returns the SSE within the loops percentage.
    """
    raw_sse = [s.strip() for s in df[df['atomtype'] == 'CA'
                                    ]['struct'].values]
    loops   = [s.strip() for s in df[df['atomtype'] == 'CA'
                                    ]['loops'].values]
    n_sse_in_loops = sum([1 if r in ['E', 'H'] and l.endswith('L') 
                            else 0 for r,l in zip(raw_sse,loops)])
    n_loops = sum([1 for l in loops if l != '0'])
    return (n_sse_in_loops / (n_sse_in_loops + n_loops)) * 100

def run_processor(top_folder):
    """
    """
    # process
    naives  = sorted(list(glob.iglob('./{}/final_csvs/*-naive.csv'.format(top_folder))))
    natives = sorted(list(glob.iglob('./{}/final_csvs/*-native.csv'.format(top_folder))))

    outcsv   = './{}/processed01_csvs'.format(top_folder)
    outpdb   = './{}/processed01_pdbs'.format(top_folder)
    outtorch = './{}/processed01_torch'.format(top_folder)
    os.makedirs(outcsv,   exist_ok=True)
    os.makedirs(outpdb,   exist_ok=True)
    os.makedirs(outtorch, exist_ok=True)

    #column_order = ['atomnum', 'atomtype', 'id', 'res1aa', 'res3aa', 'chain', 'struct', 'structdetails',
    #                'secstruct', 'layer', 'layer_element', 'x', 'y', 'z',]
    for nn, (naivefile, nativefile) in enumerate(zip(naives, natives)):
        if nn%100==0:
            print('{}/{}'.format(nn, len(naives)))
        try:
            ### NATIVE ###
            df2 = load_native_csv(nativefile)
            # Checks
            #chbreak = _check_chain_breaks(df2)
            #if chbreak == True:
            #    print('Chain break detected in {}'.format(
            #            os.path.basename(nativefile.replace('-native.csv', '')) ))
            #    continue
            #loop_perc = _check_loop_percentage(df2)
            #if loop_perc > 75.:
            #    print('Loops percentage too high for {} with {:.3f}'.format(
            #            os.path.basename(nativefile.replace('-native.csv', '')), loop_perc ))
            #    continue
            #sse_in_loops_perc = _check_sse_percentage_in_loops(df2)
            #if sse_in_loops_perc > 50.:
            #    print('Too many SSE in loops for {} with {:.3f}'.format(
            #            os.path.basename(nativefile.replace('-native.csv', '')), sse_in_loops_perc ))
            #    continue
            # Process missing Cbeta
            nativeBB   = df2[df2.atomtype.isin(['N', 'CA', 'C', 'O', 'CB'])
                            ].drop_duplicates(['id', 'atomtype'])
            #nativecB   = tgc.minitopobuilder.add_cbeta(nativeBB[nativeBB.res1aa == 'G'])
            cb_add_ids = [n for n in nativeBB['id'].drop_duplicates().tolist()
                            if not 'CB' in nativeBB[nativeBB.id == n]['atomtype'].tolist()]
            if cb_add_ids != []:
                nativecB   = tgc.minitopobuilder.add_cbeta(nativeBB[nativeBB.id.isin(cb_add_ids)])
                nativeBBcB = pd.concat( [df2[~df2.id.isin(cb_add_ids)], nativecB] )
                df2        = nativeBBcB.sort_values(['id', 'atomnum']).drop_duplicates([
                                                     'id', 'atomtype']).reset_index(drop=True)
            else:
                df2 = tgc.minitopobuilder.add_cbeta(df2)
            #df2   = df2[column_order]
            #d2    = _prepare_for_torch(df2)

            ### NAIVE ###
            df1   = load_naive_csv(naivefile)
            # Add loops
            df1   = tgc.minitopobuilder.add_loops(df1)
            # Add Cbeta
            df1   = tgc.minitopobuilder.add_cbeta(df1)
            #df1   = df1[column_order]
            #d1    = _prepare_for_torch(df1)
            # Check for mismatches
            naive_native_match_sse = _check_naive_to_native_sse_match(df1, df2)
            if naive_native_match_sse > 0:
                print('Naive to native SSE mismatch for {}'.format(
                        os.path.basename(nativefile.replace('-native.csv', '')) ))
                continue

            d =  _prepare_for_torch(df1, df2)

            ### SAVE ###
            df1.to_csv(outcsv + '/{}'.format(os.path.basename(naivefile)))
            df2.to_csv(outcsv + '/{}'.format(os.path.basename(nativefile)))
            tgc.utils.write_pdb( df1, action='write', outfile=outpdb + '/{}'.format(
                                            os.path.basename(naivefile.replace('.csv', '.pdb'))) )
            tgc.utils.write_pdb( df2, action='write', outfile=outpdb + '/{}'.format(
                                            os.path.basename(nativefile.replace('.csv', '.pdb'))) )

            torch.save(d, outtorch + '/{}.pt'.format(
                       '-'.join( os.path.basename(naivefile).split('-')[:2] )))
        except: continue


def prepare_slurm_script(top_folder):
    """
    """
    # process
    naives  = sorted(list(glob.iglob('./{}/final_csvs/*-naive.csv'.format(top_folder))))
    natives = sorted(list(glob.iglob('./{}/final_csvs/*-native.csv'.format(top_folder))))
    outcsv   = './{}/processed01_csvs'.format(top_folder)
    outpdb   = './{}/processed01_pdbs'.format(top_folder)
    outtorch = './{}/processed01_torch'.format(top_folder)
    os.makedirs(outcsv,   exist_ok=True)
    os.makedirs(outpdb,   exist_ok=True)
    os.makedirs(outtorch, exist_ok=True)
    script = textwrap.dedent("""\
# Standard Libraries
import os
import sys
import argparse
import glob
from itertools import groupby
# External Libraries
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import torch
# This library
sys.path.append('/work/upcorreia/users/hartevel/bin/topoGoCurvy/')
import topogocurvy as tgc
def load_naive_csv(csvfile):
    # load input
    frame = pd.read_csv(csvfile, index_col=0, converters={{'match': eval}}).reset_index(drop=True)
    # Process input
    topology = os.path.basename(csvfile).split('-')[0].split('.')
    topology_match = [[t[:3]] * 4 * int(t[3:]) for t in topology]
    topology_match = [item for sublist in topology_match for item in sublist]
    secstruct      = [t[-1] for t in topology_match]
    layer          = [t[0] for t in topology_match]
    layer_element  = [int(t[1]) for t in topology_match]
    frame = frame.assign(secstruct=secstruct, layer=layer, layer_element=layer_element)
    return frame
def load_native_csv(csvfile):
    # load input
    frame = pd.read_csv(csvfile, index_col=0, converters={{'match': eval}}).reset_index(drop=True)
    # Process input
    topology_match = frame.topology_match.tolist()
    loops          = frame.loops.tolist()
    secstruct      = [t[-1] if t != '0' else 'L' for t in topology_match]
    layer          = [t[0]  if t != '0' else 'X' for t in topology_match]
    layer_element  = [int(t[1])  if t != '0' else int(l[0])
                      for t, l in zip(topology_match, loops)]
    frame = frame.assign(secstruct=secstruct, layer=layer, layer_element=layer_element)
    return frame
def _check_chain_breaks(frame):
    mat = frame[frame.atomtype == 'CA'][['x', 'y', 'z']].values
    con = np.diagonal( squareform(pdist(mat)), offset=1 )
    con = True if sum(con[con > 4.4]) > 0. else False
    return con
def _check_loop_lengths(df, max_length=13):
    l_lents = df[df['atomtype'] == 'CA'][
                ['id', 'loops']].groupby(['loops']).count().reset_index()
    l_lents = l_lents[l_lents['loops'] != '0']
    return sum([1 if l > max_length else 0 for l in l_lents['id'].values])
def _check_loop_contains_sse(df):
    raw_sse = [s.strip() for s in native[native['atomtype'
                        ] == 'CA']['struct'].values]
    loops   = [s.strip() for s in native[native['atomtype'] == 'CA'
                        ]['loops'].values]
    return sum([1 if r in ['E', 'H'] and l.endswith('L')
                else 0 for r,l in zip(raw_sse,loops)])
def _check_naive_to_native_sse_match(naive, native):
    native_sse_raw = native[native.atomtype == 'CA'].struct.str.strip().values
    native_sse_raw = np.array([s if s in ['E', 'H']
                               else 'L' for s in native_sse_raw])
    naive_sse      = naive[naive.atomtype == 'CA'].secstruct.values
    return sum([1 if e2 in ['E', 'H'] and e1 != e2
                else 0 for e1,e2 in zip(native_sse_raw, naive_sse)])
def _check_loop_percentage(df):
    n_resi_loops = sum([1 for l in df[df['atomtype'] == 'CA']['loops'] if l != '0' ])
    n_resi_sses  = sum([1 for l in df[df['atomtype'] == 'CA']['loops'] if l == '0' ])
    return (n_resi_loops / (n_resi_loops + n_resi_sses)) * 100
def _check_sse_percentage_in_loops(df):
    raw_sse = [s.strip() for s in df[df['atomtype'] == 'CA'
                                    ]['struct'].values]
    loops   = [s.strip() for s in df[df['atomtype'] == 'CA'
                                    ]['loops'].values]
    n_sse_in_loops = sum([1 if r in ['E', 'H'] and l.endswith('L')
                            else 0 for r,l in zip(raw_sse,loops)])
    n_loops = sum([1 for l in loops if l != '0'])
    return (n_sse_in_loops / (n_sse_in_loops + n_loops)) * 100
def _prepare_for_torch(naive, native):
    d = {1}
    # Features
    residues = native[native.atomtype == 'CA']
    d['res_num']         = residues['id'].values
    d['res_type']        = residues['res1aa'].values
    d['layer']           = residues['layer'].values
    d['layer_element']   = residues['layer_element'].values
    d['topology_ss']     = residues['secstruct'].values
    d['dssp_ss']         = np.array([s.strip() for s in residues['struct'].values])
    d['dssp_ss_details'] = residues['structdetails'].values
    ### Native ###
    # Back-bone atoms
    d['native'] = {1}
    for atom in ['N', 'CA', 'C', 'O', 'CB']:
        coords = torch.from_numpy(native[native.atomtype == atom][['x', 'y', 'z']].values)
        d['native'][atom] = {{ 'cartesian': coords }}
    # Side-chain atoms
    for resi in native['id'].values:
        sidechain = native[ (native['id'] == resi) &
                            (~native['atomtype'].isin(['N', 'CA', 'C', 'O', 'CB'])) ]
        coords    = torch.from_numpy(sidechain[['x', 'y', 'z']].values)
        atomtypes = sidechain[['atomtype']].values.flatten()
        d['native'][resi] = {{ 'cartesian': coords, 'atomtype': atomtypes }}
    ### Naive ###
    d['naive'] = {1}
    for atom in ['N', 'CA', 'C', 'O', 'CB']:
        coords = torch.from_numpy(naive[naive.atomtype == atom][['x', 'y', 'z']].values)
        d['naive'][atom] = {{ 'cartesian': coords }}
    return d
starti  = int(sys.argv[1])
stopi   = int(sys.argv[2])
naives  = sorted(list(glob.iglob('./{0}/final_csvs/*-naive.csv')))
natives = sorted(list(glob.iglob('./{0}/final_csvs/*-native.csv')))
naives  = naives[starti:stopi]
natives = natives[starti:stopi]
outcsv = './{0}/processed01_csvs'
outpdb = './{0}/processed01_pdbs'
outtorch = './{0}/processed01_torch'
column_order = ['atomnum', 'atomtype', 'id', 'res1aa', 'res3aa', 'chain', 'secstruct', 'layer',
                'layer_element', 'x', 'y', 'z',]
for nn, (naivefile, nativefile) in enumerate(zip(naives, natives)):
    if nn%100==0:
        print('{1}/{1}'.format(nn, len(naives)))
    try:
        ### NATIVE ###
        df2 = load_native_csv(nativefile)
        # Checks
        #chbreak = _check_chain_breaks(df2)
        #if chbreak == True:
        #    print('Chain break detected in {1}'.format( os.path.basename(nativefile.replace('-native.csv', '')) ))
        #    continue
        loop_perc = _check_loop_percentage(df2)
        if loop_perc >= 75.:
            print('Loops percentage too high for {1} with {1}'.format( os.path.basename(nativefile.replace('-native.csv', '')), loop_perc ))
            continue
        sse_in_loops_perc = _check_sse_percentage_in_loops(df2)
        if sse_in_loops_perc >= 50.:
            print('Too many SSE in loops for {1} with {1}'.format( os.path.basename(nativefile.replace('-native.csv', '')), sse_in_loops_perc ))
            continue
        # Process missing Cbeta
        nativeBB   = df2[df2.atomtype.isin(['N', 'CA', 'C', 'O', 'CB'])].drop_duplicates(['id', 'atomtype'])
        #nativecB   = tgc.minitopobuilder.add_cbeta(nativeBB[nativeBB.res1aa == 'G'])
        cb_add_ids = [n for n in nativeBB['id'].drop_duplicates().tolist()
                        if not 'CB' in nativeBB[nativeBB.id == n]['atomtype'].tolist()]
        if cb_add_ids != []:
            nativecB   = tgc.minitopobuilder.add_cbeta(nativeBB[nativeBB.id.isin(cb_add_ids)])
            nativeBBcB = pd.concat( [df2[~df2.id.isin(cb_add_ids)], nativecB] )
            df2        = nativeBBcB.sort_values(['id', 'atomnum']).drop_duplicates([
                                                 'id', 'atomtype']).reset_index(drop=True)
        else:
            df2 = tgc.minitopobuilder.add_cbeta(df2)
        #df2   = df2[column_order]
        #d2    = _prepare_for_torch(df2)
        ### NAIVE ###
        df1   = load_naive_csv(naivefile)
        # Add loops
        df1   = tgc.minitopobuilder.add_loops(df1)
        # Add Cbeta
        df1   = tgc.minitopobuilder.add_cbeta(df1)
        #df1   = df1[column_order]
        #d1    = _prepare_for_torch(df1)
        # Check for mismatches
        naive_native_match_sse = _check_naive_to_native_sse_match(df1, df2)
        if naive_native_match_sse > 0:
            print('Naive to native SSE mismatch for {1}'.format( os.path.basename(nativefile.replace('-native.csv', '')) ))
            continue
        ### SAVE ###
        d =  _prepare_for_torch(df1, df2)
        df1.to_csv(outcsv + '/{1}'.format(os.path.basename(naivefile)))
        df2.to_csv(outcsv + '/{1}'.format(os.path.basename(nativefile)))
        tgc.utils.write_pdb( df1, action='write', outfile=outpdb + '/{1}'.format(
                                        os.path.basename(naivefile.replace('.csv', '.pdb'))) )
        tgc.utils.write_pdb( df2, action='write', outfile=outpdb + '/{1}'.format(
                                        os.path.basename(nativefile.replace('.csv', '.pdb'))) )
        torch.save(d, outtorch + '/{1}.pt'.format(
                   '-'.join( os.path.basename(naivefile).split('-')[:2] )))
    except: continue""".format(top_folder, "{}"))

    with open("./{}/_processing_script.py".format(top_folder), "w") as f:
        f.write(script)
    return len(naives)

def make_slurm_file(top_folder, n_arrays, n_per_batch, partits):
    """
    """
    _script = """#!/bin/bash\n#SBATCH --nodes 1
#SBATCH --partition={3}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4096
#SBATCH --time 10:00:00
#SBATCH --array=0-{1}\n\n
source /work/lpdi/users/hartevel/venv/pytorch_cpu_gcc54/bin/activate && export PYTHONPATH=''
echo ${{SLURM_ARRAY_TASK_ID}}
array_starter=$((SLURM_ARRAY_TASK_ID * {2}))
array_stoper=$(($array_starter + {2}))
echo $array_starter
echo $array_stoper
echo START
python -u ./{0}/_processing_script.py $array_starter $array_stoper
echo END""".format(top_folder, n_arrays, n_per_batch, partits)

    with open("./{}/_processing.slurm".format(top_folder), "w") as f:
        f.write(_script)

def control_slurm_file(slurm_file, main_id, partits, condition_file=None):
    """
    """
    _header = """#!/bin/bash\n#SBATCH --nodes 1
#SBATCH --partition={}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8000
#SBATCH --time 00:10:00\n\n""".format(partits)
    condition_file = "touch_control.{}".format(main_id)
    condition_file = Path().cwd().joinpath(condition_file)

    with open(slurm_file, 'w') as fd:
        fd.write(_header + '\n\n')
        fd.write('echo \'finished\' > {}\n'.format(condition_file.resolve()))
    return condition_file

def wait_for(condition_file):
    """
    """
    waiting_time = 0
    while not Path(condition_file).is_file():
        time.sleep(30)
        waiting_time += 1

def submit_nowait_slurm(slurm_file, dependency_mode=None, dependency_id=None):
    """
    """
    command = ['sbatch']
    if dependency_mode is not None and dependency_id is not None:
        command.append('--depend={}:{}'.format(dependency_mode, dependency_id))
    command.append('--parsable')
    command.append(str(slurm_file))
    p = subprocess.run(command, stdout=subprocess.PIPE)
    while not str(p.stdout.decode("utf-8")).strip().isdigit():
        print("submit error, trying again...")
        p = subprocess.run(command, stdout=subprocess.PIPE)
        time.sleep( 1 )
    else:
        return int(str(p.stdout.decode("utf-8")).strip())

def submit_slurm(slurm_file, top_folder, partits):
    """
    """
    # first submit all
    main_id = submit_nowait_slurm(slurm_file)
    print("./{}/slurm_control.{}.sh".format(top_folder, main_id))
    slurm_control_file = ( "./{}/slurm_control.{}.sh".format(top_folder, main_id) )
    condition_file = control_slurm_file(slurm_control_file, main_id, partits)
    submit_nowait_slurm(slurm_control_file, "afterany", main_id)
    wait_for(condition_file)
    os.unlink(str(condition_file))


def main():
    """
    Main execution point.
    """
    # Parse arguments
    args = parse_args(create_parser())
    topology = args.topology[0]
    architecture = args.architecture[0]
    slurm        = args.slurm
    partition    = args.partition[0]

    #wdir = os.getcwd()

    # Create top folder level
    top_folder = "./{}".format(architecture)
    os.makedirs(top_folder, exist_ok=True)

    # Check if checkpoint file exists
    chkpt = Path("./{}/process_sketches.chkpt".format(top_folder))
    if chkpt.is_file():
        print("Found checkpoint {}".format(chkpt))
        sys.exit()

    # run processor
    # No slurm here
    if slurm == False:
        run_processor(top_folder)
    else:
        shaper = prepare_slurm_script(top_folder)
        n_arrays    = 200
        n_per_batch = math.ceil( shaper / n_arrays )
        make_slurm_file(top_folder, n_arrays, n_per_batch, partition)
        submit_slurm("./{}/_processing.slurm".format(top_folder), top_folder, partition)

    # Create checkpoint file
    with open("./{}/process_sketches.chkpt".format(top_folder), 'w') as f:
        f.write("process_sketches DONE")


if __name__ == "__main__":
	main()
