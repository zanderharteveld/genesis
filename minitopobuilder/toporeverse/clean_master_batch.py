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
import re
import time
import shutil
import subprocess
import glob
import argparse
import textwrap
import tempfile
from pathlib import Path
from collections import OrderedDict

# External Libraries
import numpy as np
import pandas as pd

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
    parse.add_argument("--dssp", type=str, nargs=1, help="Path to DSSP executable.")
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

def read_master(masterfile):
    """
    """
    with open(masterfile, 'r') as f:
        lines = f.readlines()

    d = {
    'description': [],
    'query': [],
    'target': [],
    'rmsd': [],
    'match': []
    }

    for line in lines:
        splits = [l.strip() for l in line.split(' ') if l != '']
        rmsd, target = float(splits[0]), os.path.basename(splits[1]).replace('.pds', '')
        match = [ list(map(int, re.sub('[^\d.,]' , '', m.strip(',')).split(','))) for m in splits[2:] ]
        #match = [ re.sub('[^\d.,]' , '', m.strip(',')) for m in splits[2:] ]

        d['description'].append(os.path.basename(masterfile))
        d['query'].append(os.path.basename(masterfile).replace('.master', ''))
        d['target'].append(target)
        d['rmsd'].append(rmsd)
        d['match'].append(match)

    return pd.DataFrame( d )

def check_match_order(row):
    """
    """
    m = [int(i) for slist in row.match for i in slist]
    return sorted(m) == m

def loop_lengths(row):
    """
    """
    lnts = [ row.match[i + 1][0] - row.match[i][-1] for i in range(len(row.match) - 1) ]
    lnts = [0 if l < 21 and l > 0 else 1 for l in lnts]
    return sum(lnts)

def get_pdbs(df, top_folder):
    """
    """
    os.makedirs('./{}/raw_pdbs'.format(top_folder), exist_ok=True)
    for i, row in df.iterrows():
        targetfile = './{}/okforms/{}_results/{}/full1.pdb'.format(top_folder, row.query, row.target)
        try:
            shutil.copy(targetfile, './{}/raw_pdbs/{}-{}.pdb'.format(top_folder, row.query, row.target))
        except: continue

def _find_nearest(a, a0):
    """
    Element in nd array `a` closest to the scalar value `a0`
    """
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

def _expand_terminal_SSE(df, start, stop):
    """
    Given a protein frame, checks the terminal SSEs and expands them fully if needed.
    """
    strand_stop  = df[df.resnum == stop].strands_match.drop_duplicates().values[0]
    strand_start = df[df.resnum == start].strands_match.drop_duplicates().values[0]

    helix_start  = df[df.resnum == start].helices_match.drop_duplicates().values[0]
    helix_stop   = df[df.resnum == stop].helices_match.drop_duplicates().values[0]

    if strand_start != 0:
        start = min(df[df.strands == strand_start].resnum)
    if helix_start != 0:
        start = min(df[df.helices == helix_start].resnum)

    if strand_stop != 0:
        stop = max(df[df.strands == strand_stop].resnum)
    if helix_stop != 0:
        stop = max(df[df.helices == helix_stop].resnum)
    return start, stop

def _expand_match_region(t, start, stop):
    """
    """
    t_part = t[t.resnum.between(start, stop, inclusive=True)]
    h_part = [h for h in t_part.helices.drop_duplicates().tolist() if h != '0' and h != 0]
    e_part = [e for e in t_part.strands.drop_duplicates().tolist() if e != '0' and e != 0]

    if h_part != []:
        part = h_part
        bag = [int(r) for r, n in zip(t.resnum, t.helices) if n in h_part]
    elif e_part != []:
        part = e_part
        bag = [int(r) for r, n in zip(t.resnum, t.strands) if n in e_part]
    else:
        part = None
        bag = [r for r in t[t.resnum.between(start, stop, inclusive=True)].resnum]
    return bag, part

def get_targets(df, top_folder, dssp=None):
    """
    """
    os.makedirs('clean_pdbs', exist_ok=True)
    os.makedirs('clean_csvs', exist_ok=True)

    for i, row in df.reset_index(drop=True).iterrows():
        if i%100==0:
            print('@ {}'.format(i))
        try:
            # Load
            targetfile = './{}/raw_pdbs/{}-{}.pdb'.format(top_folder, row.query, row.target)
            try:
                t_dssp = tgc.utils.run_dssp(targetfile, exe=dssp)
            except:
                print('DSSP failed for {}'.format(targetfile))
                continue
            topology = row.query.split(".")
            t_pdb = tgc.utils.load_pdb(targetfile).drop_duplicates(['id', 'atomtype'])
            t_pdb = tgc.utils.renumber_pdb(t_pdb, start=0)
            t = pd.merge(t_pdb, t_dssp, on=['id', 'chain']).drop_duplicates('atomnum')
            if t.empty == True: # if only CA-trace
                continue
            helices, strands, sheets = tgc.utils.get_helices_and_sheets(t)
            t = t.assign(helices=helices, strands=strands, sheets=sheets)

            # Get machted region
            match_region, parts = [], []
            for m in row.match:
                start, stop = m[0] + 1, m[-1] + 1
                lt = t.resnum.drop_duplicates().values
                #start, stop  = _find_nearest(lt, start), _find_nearest(lt, stop)
                region, part = _expand_match_region(t, start, stop)
                match_region.extend( list(set(region)) )
                parts.append( part )
            _prev, h_match = None, []
            for ss, m in zip(t.helices, t.resnum):
                if m in match_region:
                    if ss != '0': #or ss != 0:
                        if not _prev: _prev = int(ss[:-1])
                        num = int(ss[:-1]) - _prev
                        h_match.append("{}{}".format(num + 1, ss[-1]))
                    else:
                        h_match.append('0')
                else:
                    h_match.append('0')
            _prev, e_match = None, []
            for ss, m in zip(t.strands, t.resnum):
                if m in match_region:
                    if ss != '0': #or ss != 0:
                        if not _prev: _prev = int(ss[:-1])
                        num = int(ss[:-1]) - _prev
                        e_match.append("{}{}".format(num + 1, ss[-1]))
                    else:
                        e_match.append('0')
                else:
                    e_match.append('0')
            t = t.assign(helices_match=h_match, strands_match=e_match)

            # Add the right topological elements
            sse = [h if h != '0' and h != 0 else e if e != '0' and e != 0 else '0'
                   for h, e in zip(t.helices_match, t.strands_match)]
            d_sse = dict( ((k,v) for k, v in zip( list(OrderedDict.fromkeys([s for s in sse if s != '0'])), topology)) )
            elements = [d_sse[ss][:3] if ss != '0' else '0' for ss in sse]
            t = t.assign(topology_match=elements)

            # Get match region
            sse_match   = [r for h, e, r in zip(t.helices_match, t.strands_match, t.resnum)
                           if h != '0' or e != '0']
            start, stop = min(sse_match), max(sse_match)

            try:
                ts = t[t.resnum.between(start, stop, inclusive=True)]
            except:
                print('Expansion failed for {}'.format(targetfile))
                continue

            # Save
            tgc.utils.write_pdb(ts, action='write', outfile='./{}/clean_pdbs/{}-{}.pdb'.format(top_folder, row.query, row.target))
            ts.to_csv('./{}/clean_csvs/{}-{}.csv'.format(top_folder, row.query, row.target))
        except: continue

def prepare_slurm_script(top_folder, dssp=None):
    """
    """
    os.makedirs('./{}/raw_pdbs'.format(top_folder), exist_ok=True)
    os.makedirs('./{}/clean_pdbs'.format(top_folder), exist_ok=True)
    os.makedirs('./{}/clean_csvs'.format(top_folder), exist_ok=True)
    script = textwrap.dedent("""\
import os
import sys
import math
import re
import time
import shutil
import subprocess
import glob
import argparse
import textwrap
import tempfile
from pathlib import Path
from collections import OrderedDict
# External Libraries
import numpy as np
import pandas as pd
# This library
sys.path.append('/work/upcorreia/users/hartevel/bin/topoGoCurvy/')
dssp = "{2}"
import topogocurvy as tgc
def _find_nearest(a, a0):
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]
def _expand_terminal_SSE(df, start, stop):
    strand_stop  = df[df.resnum == stop].strands_match.drop_duplicates().values[0]
    strand_start = df[df.resnum == start].strands_match.drop_duplicates().values[0]
    helix_start  = df[df.resnum == start].helices_match.drop_duplicates().values[0]
    helix_stop   = df[df.resnum == stop].helices_match.drop_duplicates().values[0]
    if strand_start != 0:
        start = min(df[df.strands == strand_start].resnum)
    if helix_start != 0:
        start = min(df[df.helices == helix_start].resnum)
    if strand_stop != 0:
        stop = max(df[df.strands == strand_stop].resnum)
    if helix_stop != 0:
        stop = max(df[df.helices == helix_stop].resnum)
    return start, stop
def _expand_match_region(t, start, stop):
    t_part = t[t.resnum.between(start, stop, inclusive=True)]
    h_part = [h for h in t_part.helices.drop_duplicates().tolist() if h != '0' and h != 0]
    e_part = [e for e in t_part.strands.drop_duplicates().tolist() if e != '0' and e != 0]
    if h_part != []:
        part = h_part
        bag = [int(r) for r, n in zip(t.resnum, t.helices) if n in h_part]
    elif e_part != []:
        part = e_part
        bag = [int(r) for r, n in zip(t.resnum, t.strands) if n in e_part]
    else:
        part = None
        bag = [r for r in t[t.resnum.between(start, stop, inclusive=True)].resnum]
    return bag, part
# Read data
starti = int(sys.argv[1])
stopi  = int(sys.argv[2])
df = pd.read_csv('./{0}/part_master.csv', index_col=0, converters={{'match': eval}})
df = df.iloc[starti:stopi]
for i, row in df.reset_index(drop=True).iterrows():
    targetfile = './{0}/okforms/{1}_results/{1}/full1.pdb'.format(row.query, row.target)
    try:
        shutil.copy(targetfile, './{0}/raw_pdbs/{1}-{1}.pdb'.format(row.query, row.target))
    except: continue
for i, row in df.reset_index(drop=True).iterrows():
    #if i%100==0:
    #    print('@ {1}'.format(i))
    try:
        # Load
        targetfile = './{0}/raw_pdbs/{1}-{1}.pdb'.format(row.query, row.target)
        try:
            t_dssp = tgc.utils.run_dssp(targetfile, exe=dssp)
        except:
            print('DSSP failed for {1}'.format(targetfile))
            continue
        topology = row.query.split(".")
        t_pdb = tgc.utils.load_pdb(targetfile).drop_duplicates(['id', 'atomtype'])
        t_pdb = tgc.utils.renumber_pdb(t_pdb, start=0)
        t = pd.merge(t_pdb, t_dssp, on=['id', 'chain']).drop_duplicates('atomnum')
        if t.empty == True: # if only CA-trace
            continue
        helices, strands, sheets = tgc.utils.get_helices_and_sheets(t)
        t = t.assign(helices=helices, strands=strands, sheets=sheets)
        # Get machted region
        match_region, parts = [], []
        for m in row.match:
            start, stop = m[0] + 1, m[-1] + 1
            lt = t.resnum.drop_duplicates().values
            #start, stop  = _find_nearest(lt, start), _find_nearest(lt, stop)
            region, part = _expand_match_region(t, start, stop)
            match_region.extend( list(set(region)) )
            parts.append( part )
        _prev, h_match = None, []
        for ss, m in zip(t.helices, t.resnum):
            if m in match_region:
                if ss != '0': #or ss != 0:
                    if not _prev: _prev = int(ss[:-1])
                    num = int(ss[:-1]) - _prev
                    h_match.append("{1}{1}".format(num + 1, ss[-1]))
                else:
                    h_match.append('0')
            else:
                h_match.append('0')
        _prev, e_match = None, []
        for ss, m in zip(t.strands, t.resnum):
            if m in match_region:
                if ss != '0': #or ss != 0:
                    if not _prev: _prev = int(ss[:-1])
                    num = int(ss[:-1]) - _prev
                    e_match.append("{1}{1}".format(num + 1, ss[-1]))
                else:
                    e_match.append('0')
            else:
                e_match.append('0')
        t = t.assign(helices_match=h_match, strands_match=e_match)
        # Add the right topological elements
        sse = [h if h != '0' and h != 0 else e if e != '0' and e != 0 else '0'
               for h, e in zip(t.helices_match, t.strands_match)]
        d_sse = dict( ((k,v) for k, v in zip( list(OrderedDict.fromkeys([s for s in sse if s != '0'])), topology)) )
        elements = [d_sse[ss][:3] if ss != '0' else '0' for ss in sse]
        t = t.assign(topology_match=elements)
        # Get match region
        sse_match   = [r for h, e, r in zip(t.helices_match, t.strands_match, t.resnum)
                       if h != '0' or e != '0']
        start, stop = min(sse_match), max(sse_match)
        try:
            ts = t[t.resnum.between(start, stop, inclusive=True)]
        except:
            print('Expansion failed for {1}'.format(targetfile))
            continue
        # Save
        print("Saving {1} {1}".format(row.query, row.target))
        tgc.utils.write_pdb(ts, action='write', outfile='./{0}/clean_pdbs/{1}-{1}.pdb'.format(row.query, row.target))
        ts.to_csv('./{0}/clean_csvs/{1}-{1}.csv'.format(row.query, row.target))
    except: continue""".format(top_folder, "{}", dssp) )

    with open("./{}/_cleaning_script.py".format(top_folder), "w") as f:
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
python -u ./{0}/_cleaning_script.py $array_starter $array_stoper
echo END""".format(top_folder, n_arrays, n_per_batch, partits)

    with open("./{}/_cleaning.slurm".format(top_folder), "w") as f:
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
    dssp         = args.dssp[0]
    partition    = args.partition[0]
    # Get current working directory
    #wdir = os.getcwd()

    # Create top folder level
    top_folder = "./{}".format(architecture)
    os.makedirs(top_folder, exist_ok=True)

    # Check if checkpoint file exists
    chkpt = Path("./{}/clean_master.chkpt".format(top_folder))
    if chkpt.is_file():
        print("Found checkpoint {}".format(chkpt))
        sys.exit()

    # Full master results
    container = []
    for masterfile in glob.iglob('./{}/*.master'.format(top_folder)):
        df = read_master(masterfile)
        df = df.sort_values('rmsd').drop_duplicates(['target', 'query'])
        container.append(df)
    df = pd.concat(container)
    df.to_csv('./{}/full_master.csv'.format(top_folder))
    print("Full Master file written...")    

    df['connectivity'] = df.apply(check_match_order, axis=1, result_type='expand')
    df['loop_lents']   = df.apply(loop_lengths, axis=1)
    df = df[(df.connectivity) & (df.loop_lents)]
    df.to_csv('./{}/part_master.csv'.format(top_folder))
    
    #df = pd.read_csv('./{}/part_master.csv'.format(top_folder),
    #index_col=0, converters={'match': eval}).reset_index(drop=True)
    print("Master matches for SSE cleansing: {}".format(df.shape[0]))

    # No slurm here
    if slurm == False:
        # Get the good sketches
        get_pdbs(df, top_folder)

        # Get matched parts
        get_targets(df, top_folder, dssp=dssp)
    else:
        n_arrays    = 1000
        n_per_batch = math.ceil( df.shape[0] / n_arrays )
        prepare_slurm_script(top_folder, dssp=dssp)
        make_slurm_file(top_folder, n_arrays, n_per_batch, partition)
        submit_slurm("./{}/_cleaning.slurm".format(top_folder), top_folder, partition)

    # Create checkpoint file
    with open("./{}/clean_master.chkpt".format(top_folder), 'w') as f:
        f.write("clean_master DONE")


if __name__ == "__main__":
	main()
