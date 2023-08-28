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
import glob
import math
import re
import time
import shutil
import subprocess
import textwrap
import tempfile
from pathlib import Path
from collections import Counter, OrderedDict

# External Libraries
import pandas as pd

# This library
sys.path.append("/work/upcorreia/users/hartevel/bin/")
from genesis.minitopobuilder import build_forms, prepare_forms
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

def get_symbolic_sses(df, loops=True):
    """
    Finds the SSEs.
    """
    dfca = df[df.atomtype == 'CA']
    #helices = dfca.helices_match.values
    #strands = dfca.strands_match.values
    tpgs = dfca.topology_match.values

    if loops == True:
        loops = dfca.loops.values
        #return [sseE if sseE != '0' and sseE != 0
        #        else sseH if sseH != '0' and sseH != 0
        #        else lps for sseE, sseH, lps in zip(helices, strands, loops)]
        return [sseE if sseE != '0' and sseE != 0 else lps
                for sseE, lps in zip(tpgs, loops)]
    else:
        #return [sseE if sseE != '0' and sseE != 0 else sseH
        #for sseE, sseH in zip(helices, strands)]
        return [sseE  for sseE in tpgs if sseE != '0' and sseE != 0]

def get_symbolic_loops(df):
    """
    """
    #sse = [h if h != '0' and h != 0 else e if e != '0' and e != 0 else '0'
    #       for h, e in zip(df.helices_match, df.strands_match)]
    sse = [s if s != '0' and s != 0 else '0' for s in df.topology_match]
    loops = []
    prev_ = False
    count = 1
    for s in sse:
        if not 'E' in s and not 'H' in s and prev_ is False:
            loops.append('{}L'.format(count))
            prev_ = True
        elif not 'E' in s and not 'H' in s and prev_ is True:
            loops.append('{}L'.format(count))
        else:
            loops.append('0')
        if ('E' in s or 'H' in s) and prev_ is True:
            prev_ = False
            count += 1
    return loops

def _clean_termini_loops(df):
    """
    Remove loop parts at the beginning and the end.
    """
    l1 = df[(df.atomtype == 'CA') & (df.loops ==   df[df.loops   != '0'].loops.drop_duplicates().values[0])  ].index.values[0]
    e1 = df[(df.atomtype == 'CA') & (df.strands == df[df.strands != '0'].strands.drop_duplicates().values[0])].index.values[0]
    h1 = df[(df.atomtype == 'CA') & (df.helices == df[df.helices != '0'].helices.drop_duplicates().values[0])].index.values[0]

    ln = df[(df.atomtype == 'CA') & (df.loops ==   df[df.loops   != '0'].loops.drop_duplicates().values[-1])  ].index.values[0]
    en = df[(df.atomtype == 'CA') & (df.strands == df[df.strands != '0'].strands.drop_duplicates().values[-1])].index.values[0]
    hn = df[(df.atomtype == 'CA') & (df.helices == df[df.helices != '0'].helices.drop_duplicates().values[-1])].index.values[0]

    bag = []
    if l1 < e1 and l1 < h1:
        bag.append(df[df.loops != '0'].loops.drop_duplicates().values[0])
    if ln > en and ln > hn:
        bag.append(df[df.loops != '0'].loops.drop_duplicates().values[-1])
    return df[~df.loops.isin(bag)]

def _check_no_loop(sizes):
    """
    """
    elements = [(k, v) for k,v in sizes.items()]

    d = {}
    for i in range(len(elements)):
        if i < len(elements) - 1:
            ei, ej = elements[i], elements[i + 1]
            if ei[0][-1] in ['H', 'E'] and ej[0][-1] in ['H', 'E']:
                d[ei[0]] = ei[-1]
                d[ej[0][0] + 'L'] = 0
            else:
                d[ei[0]] = ei[-1]
        else:
            ei = elements[i]
            d[ei[0]] = ei[-1]
    return d

def run_builder(df, top_folder):
    """
    """
    d = { 'target': [],
          'query': [],
          'sizes': [],
          'rmsd_naive_natlik': [],
          'architecture': [],
          'topology': [],
          'linker': []
         }

    #pdbfile = './{}/clean_pdbs/{}_{}.pdb'
    os.makedirs('./{}/final_pdbs'.format(top_folder), exist_ok=True)
    os.makedirs('./{}/final_csvs'.format(top_folder), exist_ok=True)

    print('total sketches: {}'.format(len(df)))
    for i, row in df.reset_index(drop=True).iterrows():
        if i%100==0:
            print('@ {}/{}'.format(i, len(df)))
        #print('./clean_csvs/{}_{}.csv'.format(row.query, row.target))
        try:
            # Load
            try:
                pdb = pd.read_csv('./{}/clean_csvs/{}-{}.csv'.format(top_folder,
                                  row.query, row.target), index_col=0)
            except: continue
            lps = get_symbolic_loops(pdb)
            pdb = pdb.assign(loops=lps)

            # Clean termini
            #pdb = _clean_termini_loops(pdb)

            # SSE lenghts
            sses  = get_symbolic_sses(pdb)
            sizes = Counter(sses)
            #sizes = _check_no_loop(sizes)
            sse_elements = list(OrderedDict.fromkeys(sses))
            sizes_by_sequence = [sizes[e] for e in sse_elements if not e.endswith('L')]
            elements = [q[:3] for q in row.query.split('.')]
            #element_match = [ 0 if e in elements else 1
            #                for e in [k for k in sizes.keys() if not k.endswith('L')] ]

            if len(sizes_by_sequence) != len(elements):
                print('Missing SSEs to create sketch for ./{}/clean_csvs/{}-{}.csv'.format(
                                top_folder, row.query, row.target))
                continue

            #if sum(element_match) > 0:
            #    print('Non-matching SSEs to create sketch for ./{}/clean_csvs/{}_{}.csv'.format(
            #                    top_folder, row.query, row.target))
            #    continue

            # Prepare topology
            #cny  = '.'.join([e[:3] + str(s) for e, s in zip(row.query.split('.'), sizes_by_sequence)])
            cny  = '.'.join([e[:3] + str(s) for e, s in
                    zip([e for e in sse_elements if not e.endswith("L")], sizes_by_sequence)])
            tpg  = '.'.join(sorted( cny.split('.') ))
            lnks = [str(sizes[e]) for e in sse_elements if e.endswith('L')]

            start_loop, stop_loop = False, False
            if sse_elements[0].endswith('L'):
                start_loop = True
            if sse_elements[-1].endswith('L'):
                stop_loop  = True

            if start_loop is True and stop_loop is True:
                lnks = '.'.join( lnks )
            elif start_loop is True and stop_loop is False:
                lnks = '.'.join(['x'] + lnks)
            elif start_loop is False and stop_loop is True:
                lnks = '.'.join(lnks + ['x'])
            elif start_loop is False and stop_loop is False:
                lnks = '.'.join(['x'] + lnks + ['x'])
            else:
                print("Loops don't match")

            print("Current architecture: {}".format(tpg))
            print("Current topology: {}".format(cny))
            print("Current linkers: {}".format(lnks))

            # Build
            print('Building sketch for ./{}/clean_csvs/{}-{}'.format(
                        top_folder, row.query, row.target))
            try:
                container = build_forms(tpg, check_forms=False, connectivity=cny, links=lnks, verbose=1)
                sketches  = prepare_forms(container, two_way=True)
                sketches  = sketches[sketches.description == cny]
            except: continue

            native_CA = pdb[(pdb.loops == '0') & (pdb.atomtype == 'CA')]
            rmsds = []
            for j, rw in sketches.iterrows():
                naive_CA = rw.naive[rw.naive.atomtype == 'CA']
                rmsds.append( tgc.utils.superimpose(naive_CA, native_CA, return_alignment=False) )
            sketches = sketches.assign(rmsd=rmsds)
            sketch   = sketches.sort_values('rmsd').iloc[0]
            print('selected: {}, direction: {}'.format(sketch.description, sketch.direction)) 

            # write pdb
            #for sketch in sketches:
            tgc.utils.write_pdb(sketch.naive,  action='write', outfile='./{}/final_pdbs/{}-{}-naive.pdb'.format(top_folder, cny, row.target))
            tgc.utils.write_pdb(pdb, action='write', outfile='./{}/final_pdbs/{}-{}-native.pdb'.format(top_folder, cny, row.target))

            # write csv
            sketch['naive'].to_csv('./{}/final_csvs/{}-{}-naive.csv'.format(top_folder, cny, row.target))
            pdb.to_csv('./{}/final_csvs/{}-{}-native.csv'.format(top_folder, cny, row.target))

            d['query'].append(row.query)
            d['target'].append(row.target)
            d['sizes'].append(dict(sizes))
            d['rmsd_naive_natlik'].append(sketch.rmsd)
            d['architecture'].append(tpg)
            d['topology'].append(cny)
            d['linker'].append(lnks)
            #if i%3==0: break
        except: continue
    dff = pd.DataFrame(d)
    return dff

def prepare_slurm_script(top_folder):
    """
    """
    os.makedirs('./{}/final_pdbs'.format(top_folder), exist_ok=True)
    os.makedirs('./{}/final_csvs'.format(top_folder), exist_ok=True)
    script = textwrap.dedent("""\
# Standard Libraries
import os
import sys
import argparse
import glob
import math
import re
import time
import shutil
import subprocess
import textwrap
import tempfile
from pathlib import Path
from collections import Counter, OrderedDict
# External Libraries
import pandas as pd
# This library
sys.path.append("/work/upcorreia/users/hartevel/bin/")
from genesis.minitopobuilder import build_forms, prepare_forms
sys.path.append("/work/upcorreia/users/hartevel/bin/topoGoCurvy/")
import topogocurvy as tgc
def get_symbolic_sses(df, loops=True):
    dfca = df[df.atomtype == 'CA']
    #helices = dfca.helices_match.values
    #strands = dfca.strands_match.values
    tpgs = dfca.topology_match.values

    if loops == True:
        loops = dfca.loops.values
        #return [sseE if sseE != '0' and sseE != 0
        #        else sseH if sseH != '0' and sseH != 0
        #        else lps for sseE, sseH, lps in zip(helices, strands, loops)]
        return [sseE if sseE != '0' and sseE != 0 else lps
                for sseE, lps in zip(tpgs, loops)]
    else:
        #return [sseE if sseE != '0' and sseE != 0 else sseH
        #for sseE, sseH in zip(helices, strands)]
        return [sseE  for sseE in tpgs if sseE != '0' and sseE != 0]
def get_symbolic_loops(df):
    #sse = [h if h != '0' and h != 0 else e if e != '0' and e != 0 else '0'
    #       for h, e in zip(df.helices_match, df.strands_match)]
    sse = [s if s != '0' and s != 0 else '0' for s in df.topology_match]
    loops = []
    prev_ = False
    count = 1
    for s in sse:
        if not 'E' in s and not 'H' in s and prev_ is False:
            loops.append('{1}L'.format(count))
            prev_ = True
        elif not 'E' in s and not 'H' in s and prev_ is True:
            loops.append('{1}L'.format(count))
        else:
            loops.append('0')
        if ('E' in s or 'H' in s) and prev_ is True:
            prev_ = False
            count += 1
    return loops
def _clean_termini_loops(df):
    l1 = df[(df.atomtype == 'CA') & (df.loops ==   df[df.loops   != '0'].loops.drop_duplicates().values[0])  ].index.values[0]
    e1 = df[(df.atomtype == 'CA') & (df.strands == df[df.strands != '0'].strands.drop_duplicates().values[0])].index.values[0]
    h1 = df[(df.atomtype == 'CA') & (df.helices == df[df.helices != '0'].helices.drop_duplicates().values[0])].index.values[0]

    ln = df[(df.atomtype == 'CA') & (df.loops ==   df[df.loops   != '0'].loops.drop_duplicates().values[-1])  ].index.values[0]
    en = df[(df.atomtype == 'CA') & (df.strands == df[df.strands != '0'].strands.drop_duplicates().values[-1])].index.values[0]
    hn = df[(df.atomtype == 'CA') & (df.helices == df[df.helices != '0'].helices.drop_duplicates().values[-1])].index.values[0]

    bag = []
    if l1 < e1 and l1 < h1:
        bag.append(df[df.loops != '0'].loops.drop_duplicates().values[0])
    if ln > en and ln > hn:
        bag.append(df[df.loops != '0'].loops.drop_duplicates().values[-1])
    return df[~df.loops.isin(bag)]
def _check_no_loop(sizes):
    elements = [(k, v) for k,v in sizes.items()]

    d = {1}
    for i in range(len(elements)):
        if i < len(elements) - 1:
            ei, ej = elements[i], elements[i + 1]
            if ei[0][-1] in ['H', 'E'] and ej[0][-1] in ['H', 'E']:
                d[ei[0]] = ei[-1]
                d[ej[0][0] + 'L'] = 0
            else:
                d[ei[0]] = ei[-1]
        else:
            ei = elements[i]
            d[ei[0]] = ei[-1]
    return d
### main ####
starti = int(sys.argv[1])
stopi  = int(sys.argv[2])
df = pd.read_csv("./{0}/part_master.csv", index_col=0, converters={{"match": eval}}).reset_index(drop=True).reset_index(drop=True)
df = df.iloc[starti:stopi].reset_index(drop=True)
d = {{ 'target': [],
          'query': [],
          'sizes': [],
          'rmsd_naive_natlik': [],
          'architecture': [],
          'topology': [],
          'linker': []
         }}
print('total sketches: {1}'.format(len(df)))
for i, row in df.iterrows():
    if i%100==0:
        print('@ {1}/{1}'.format(i, len(df)))
    #print('./clean_csvs/{1}_{1}.csv'.format(row.query, row.target))
    try:
        # Load
        try:
            pdb = pd.read_csv('./{0}/clean_csvs/{1}-{1}.csv'.format(row.query, row.target), index_col=0)
        except: continue
        lps = get_symbolic_loops(pdb)
        pdb = pdb.assign(loops=lps)

        # Clean termini
        #pdb = _clean_termini_loops(pdb)

        # SSE lenghts
        sses  = get_symbolic_sses(pdb)
        sizes = Counter(sses)
        #sizes = _check_no_loop(sizes)
        sse_elements = list(OrderedDict.fromkeys(sses))
        sizes_by_sequence = [sizes[e] for e in sse_elements if not e.endswith('L')]
        elements = [q[:3] for q in row.query.split('.')]
        #element_match = [ 0 if e in elements else 1
        #                for e in [k for k in sizes.keys() if not k.endswith('L')] ]

        if len(sizes_by_sequence) != len(elements):
            print('Missing SSEs to create sketch for ./{0}/clean_csvs/{1}-{1}.csv'.format(row.query, row.target))
            continue
        #if sum(element_match) > 0:
        #    print('Non-matching SSEs to create sketch for ./{1}/clean_csvs/{1}_{1}.csv'.format(
        #                    top_folder, row.query, row.target))
        #    continue
        
        # Prepare topology
        #cny  = '.'.join([e[:3] + str(s) for e, s in zip(row.query.split('.'), sizes_by_sequence)])
        cny  = '.'.join([e[:3] + str(s) for e, s in
                zip([e for e in sse_elements if not e.endswith("L")], sizes_by_sequence)])
        tpg  = '.'.join(sorted( cny.split('.') ))
        lnks = [str(sizes[e]) for e in sse_elements if e.endswith('L')]
        
        start_loop, stop_loop = False, False
        if sse_elements[0].endswith('L'):
            start_loop = True
        if sse_elements[-1].endswith('L'):
            stop_loop  = True
        
        if start_loop is True and stop_loop is True:
            lnks = '.'.join( lnks )
        elif start_loop is True and stop_loop is False:
            lnks = '.'.join(['x'] + lnks)
        elif start_loop is False and stop_loop is True:
            lnks = '.'.join(lnks + ['x'])
        elif start_loop is False and stop_loop is False:
            lnks = '.'.join(['x'] + lnks + ['x'])
        else:
            print("Loops don't match")
        
        print("Current architecture: {1}".format(tpg))
        print("Current topology: {1}".format(cny))
        print("Current linkers: {1}".format(lnks))
        
        # Build
        print('Building sketch for ./{0}/clean_csvs/{1}-{1}'.format(
                    row.query, row.target))
        try:
            container = build_forms(tpg, check_forms=False, connectivity=cny, links=lnks, verbose=1)
            sketches  = prepare_forms(container, two_way=True)
            sketches  = sketches[sketches.description == cny]
        except: continue

        native_CA = pdb[(pdb.loops == '0') & (pdb.atomtype == 'CA')]
        rmsds = []
        for j, rw in sketches.iterrows():
            naive_CA = rw.naive[rw.naive.atomtype == 'CA']
            rmsds.append( tgc.utils.superimpose(naive_CA, native_CA, return_alignment=False) )
        sketches = sketches.assign(rmsd=rmsds)
        sketch   = sketches.sort_values('rmsd').iloc[0]
        print('selected: {1}, direction: {1}'.format(sketch.description, sketch.direction))

        # write pdb
        #for sketch in sketches:
        tgc.utils.write_pdb(sketch.naive,  action='write', outfile='./{0}/final_pdbs/{1}-{1}-naive.pdb'.format(cny, row.target))
        tgc.utils.write_pdb(pdb, action='write', outfile='./{0}/final_pdbs/{1}-{1}-native.pdb'.format(cny, row.target))

        # write csv
        sketch['naive'].to_csv('./{0}/final_csvs/{1}-{1}-naive.csv'.format(cny, row.target))
        pdb.to_csv('./{0}/final_csvs/{1}-{1}-native.csv'.format(cny, row.target))

        d['query'].append(row.query)
        d['target'].append(row.target)
        d['sizes'].append(dict(sizes))
        d['rmsd_naive_natlik'].append(sketch.rmsd)
        d['architecture'].append(tpg)
        d['topology'].append(cny)
        d['linker'].append(lnks)
        #if i%3==0: break
    except: continue
dff = pd.DataFrame(d)
dfft = pd.merge(df, dff, on=['target', 'query'], how='inner')
dfft.to_csv('./{0}/part_info_{1}_{1}.csv'.format(starti, stopi))
""".format(top_folder, "{}") )

    with open("./{}/_create_final_sketches.py".format(top_folder), "w") as f:
        f.write(script)

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
python -u ./{0}/_create_final_sketches.py $array_starter $array_stoper
echo END""".format(top_folder, n_arrays, n_per_batch, partits)

    with open("./{}/_create_final_sketches.slurm".format(top_folder), "w") as f:
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
        time.sleep( 120 )
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
    topology     = args.topology[0]
    architecture = args.architecture[0]
    slurm        = args.slurm
    partition    = args.partition[0]

    # Get current working directory
    #wdir = os.getcwd()

    # Create top folder level
    top_folder = "./{}".format(architecture)
    os.makedirs(top_folder, exist_ok=True)
    # ./part_master_sse.csv

    # Check if checkpoint file exists
    chkpt = Path("./{}/create_final_sketches.chkpt".format(top_folder))
    if chkpt.is_file():
        print("Found checkpoint {}".format(chkpt))
        sys.exit()

    # Load results
    df = pd.read_csv("./{}/part_master.csv".format(top_folder), index_col=0, converters={'match': eval}).reset_index(drop=True)

    # Make sketches
    # No slurm here
    if slurm == False: 
        dff  = run_builder(df, top_folder)
        dfft = pd.merge(df, dff, on=['target', 'query'], how='inner')
        dfft.to_csv('./{}/part_info.csv'.format(top_folder))
    else:
        n_arrays    = 200
        n_per_batch = math.ceil( df.shape[0] / n_arrays )
        prepare_slurm_script(top_folder)
        make_slurm_file(top_folder, n_arrays, n_per_batch, partition)
        submit_slurm("./{}/_create_final_sketches.slurm".format(top_folder), top_folder, partition)
        # Aggregate all results
        container = []
        for part in glob.iglob('./{}/part_info_*.csv'.format(top_folder)):
            container.append(pd.read_csv(part))
        dfft = pd.concat(container).reset_index(drop=True)
        dfft.to_csv('./{}/part_info.csv'.format(top_folder))

    # Cleaning up MASTER (takes lots of disk space and is not needed anymore)
    print('deleting okforms')
    os.system("rm -r ./{}/okforms/*_results".format(top_folder))

    # Create checkpoint file
    with open("./{}/create_final_sketches.chkpt".format(top_folder), 'w') as f:
        f.write("create_final_sketches DONE")

if __name__ == "__main__":
	main()
