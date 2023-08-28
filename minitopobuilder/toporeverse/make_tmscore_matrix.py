"""
.. codeauthor:: Zander Harteveld <zandermilanh@gmail.com>

.. affiliation::
    Laboratory of Protein Design and Immunoengineering <lpdi.epfl.ch>
    Bruno Correia <bruno.correia@epfl.ch>
"""

# Standard Libraries
import os
import sys
import glob
import time
import shutil
import argparse
import subprocess
import textwrap
from pathlib import Path
#from subprocess import DEVNULL

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
    parse.add_argument("--tmalign",      "-e", type=str, nargs=1, help="Path to the TMalign executive.")
    parse.add_argument("--narrays",      "-n", type=str, nargs=1, default=200, help="Number of parallel slurm files (default: 200).")
    #parse.add_argument("--allvsall", action="store_true", help="Use slurm accelerated parallel system.")
    parse.add_argument("--partition", type=str, nargs=1, default=['serial'], help="Partition for slurm cluster system (default: serial).")
    return parse

def parse_args(parser):
	"""
	Parse the arguments of a parser object.
	:param parser: the parser object.
	:return: the specified arg0uments
	"""
	args = parser.parse_args()
	return args

def _get_pdbs(pdbfolder):
    ids = []
    for file in glob.iglob(pdbfolder + '/*.pdb'):
        ids.append( '-'.join(file.split('-')[:-1]) )
    structures = [i + '-native.pdb' for i in ids]
    bases      = [os.path.basename(i).replace('.pdb', '') for i in structures]
    return structures, bases

def make_exelogs(top_folder, all_files, tm_exe, n_slurmfiles):
    """
    """
    # Creating files
    files   = ['x'+str(n) for n in range(0,n_slurmfiles)]
    files2  = ['y'+str(n) for n in range(0,n_slurmfiles)]
    n_files = len(files)
    pdbs_per_file = int(np.ceil(float(len(all_files))/n_files))
    pdbs_in_file  = [all_files[i * pdbs_per_file:(i+1) * pdbs_per_file]
                        for i in range(n_files)]
    print('Currently {} pdbs per file'.format(pdbs_per_file))
    alignment_folder = top_folder + '/aligns/'
    os.makedirs(alignment_folder, exist_ok=True)
    cmd = '{} {} {} >> {}/{}_{}.align\n'
    os.makedirs(top_folder + '/exelogs/', exist_ok=True)
    for i, cmd_file in enumerate(files):
        with open(top_folder + '/exelogs/' + cmd_file, 'w+') as f:
            for j, pdb_file1 in enumerate(pdbs_in_file[i]):
                base1 = os.path.basename(pdb_file1).replace('.pdb', '')
                for k, pdb_file2 in enumerate(all_files):
                    base2 = os.path.basename(pdb_file2).replace('.pdb', '')
                    f.write(cmd.format(tm_exe, pdb_file1, pdb_file2, alignment_folder, base1, base2))
    for i, cmd_file2 in enumerate(files2):
        with open(top_folder + '/exelogs/' + cmd_file2, 'w') as f2:
            for j, pdb_file1 in enumerate(pdbs_in_file[i]):
                base1 = os.path.basename(pdb_file1).replace('.pdb', '')
                for k, pdb_file2 in enumerate(all_files):
                    base2 = os.path.basename(pdb_file2).replace('.pdb', '')
                    f2.write('{}_{}.align\n'.format(base1, base2))
    return alignment_folder

def make_slurm_file(top_folder, n_slurmfiles, partits):
    """
    """
    slurmscript = """#!/bin/bash\n#SBATCH --nodes 1
#SBATCH --partition={2}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 1024
#SBATCH --time 65:00:00
#SBATCH --array=0-{1}\n\n
#source {0}/exelogs/x${{SLURM_ARRAY_TASK_ID}}
python {0}/get_tm_align_matrix_per_batch.py {0}/exelogs/y${{SLURM_ARRAY_TASK_ID}} {0}/exelogs/x${{SLURM_ARRAY_TASK_ID}} ${{SLURM_ARRAY_TASK_ID}}""".format(top_folder, n_slurmfiles, partits)
    slurmfile = top_folder + '/exec.slurm'
    with open(slurmfile, 'w') as f:
        f.write(slurmscript)
    return slurmfile

def make_alignment_slurm_file(top_folder, script, partits):
    """
    """
    slurmscript = """#!/bin/bash\n#SBATCH --nodes 1
#SBATCH --partition={1}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 128G
#SBATCH --time 05:00:00\n\n
python -u "{0}"/get_all_alignments.py""".format(top_folder, partits)
    slurmfile = top_folder + '/exec2.slurm'
    with open(slurmfile, 'w') as f:
        f.write(slurmscript)
    return slurmfile

def make_matrix_script_per_batch(parent_folder, top_folder, alignments):
    """
    """
    script = textwrap.dedent("""\
import os, sys
import numpy as np
import pandas as pd
import glob
import shutil
input1 = sys.argv[1]
input2 = sys.argv[2]
ide    = sys.argv[3]
with open(input1, 'r') as f:
    lines1 = f.readlines()
lines1 = [line.strip() for line in lines1]
with open(input2, 'r') as f:
    lines2 = f.readlines()
lines2 = [line.strip() for line in lines2]
def _get_pdbs(pdbfolder):
    ids = []
    for file in glob.iglob(pdbfolder + '/*.pdb'):
        ids.append( '-'.join(file.split('-')[:-1]) )
    structures = [i + '-native.pdb' for i in ids]
    bases      = [os.path.basename(i).replace('.pdb', '') for i in structures]
    return structures, bases
def parse_TMalign(tmfile):
    base = os.path.basename(tmfile).replace('.align', '').split('-native_')
    id1, id2 = base[0].replace('-native', '') + '-native', base[1].replace('-native', '') + '-native'
    with open(tmfile, 'r') as f:
        lines = f.readlines()
    rmsd_done, tm1_done, tm2_done = None, None, None
    for line in lines:
        if 'RMSD' in line:
            rmsd = float( line.split(',')[1].split(' ')[-1] )
            rmsd_done = True
        if 'if normalized by length of Chain_1' in line:
            tm1 = float( line.split(' (')[0].split('=')[-1] )
            tm1_done = True
        if 'if normalized by length of Chain_2' in line:
            tm2 = float( line.split(' (')[0].split('=')[-1] )
            tm2_done = True
        if rmsd_done == True and tm1_done == True and tm2_done == True:
            break
    tm_max = max( tm1, tm2 )
    return (id1, id2, rmsd, tm_max)
info = []
for exeline,tmfile in zip(lines2,lines1):
    try:
        os.system(exeline)
        tmout = exeline.split('>>')[-1].strip()
        t = parse_TMalign(tmout)
        info.append(t)
        #shutil.rmtree(tmout)
        os.remove(tmout)
    except: continue
df_full = pd.DataFrame(info, columns=["description", "target", "rmsd", "tmscore"])
df_full.to_csv("{1}" + "/alignments_" + str(ide) + ".csv")""".format(parent_folder, top_folder, alignments) )
    scriptfile = top_folder + '/get_tm_align_matrix_per_batch.py'
    with open(scriptfile, 'w') as f:
        f.write(script)
    return scriptfile

def make_matrix_script(parent_folder, top_folder, alignments):
    """
    """
    script = textwrap.dedent("""\
import os, sys
import numpy as np
import pandas as pd
import glob
def _get_pdbs(pdbfolder):
    ids = []
    for file in glob.iglob(pdbfolder + '/*.pdb'):
        ids.append( '-'.join(file.split('-')[:-1]) )
    structures = [i + '-native.pdb' for i in ids]
    bases      = [os.path.basename(i).replace('.pdb', '') for i in structures]
    return structures, bases
def parse_TMalign(tmfile):
    base = os.path.basename(tmfile).replace('.align', '').split('-native_')
    id1, id2 = base[0].replace('-native', '') + '-native', base[1].replace('-native', '') + '-native'
    with open(tmfile, 'r') as f:
        lines = f.readlines()
    rmsd_done, tm1_done, tm2_done = None, None, None
    for line in lines:
        if 'RMSD' in line:
            rmsd = float( line.split(',')[1].split(' ')[-1] )
            rmsd_done = True
        if 'if normalized by length of Chain_1' in line:
            tm1 = float( line.split(' (')[0].split('=')[-1] )
            tm1_done = True
        if 'if normalized by length of Chain_2' in line:
            tm2 = float( line.split(' (')[0].split('=')[-1] )
            tm2_done = True
        if rmsd_done == True and tm1_done == True and tm2_done == True:
            break
    tm_max = max( tm1, tm2 )
    return (id1, id2, rmsd, tm_max)
info = []
for tmfile in glob.iglob("{2}" + "*.align"):
    try:
        t = parse_TMalign(tmfile)
        info.append(t)
    except: continue
df_full = pd.DataFrame(info, columns=["description", "target", "rmsd", "tmscore"])
df_full.to_csv("{1}" + "/alignments.csv")""".format(parent_folder, top_folder, alignments) )
    scriptfile = top_folder + '/get_tm_align_matrix.py'
    with open(scriptfile, 'w') as f:
        f.write(script)
    return scriptfile

def make_matrix_script2(top_folder, partits):
    """
    """
    script = textwrap.dedent("""\
import os
import sys
import pandas as pd
import glob
import shutil
container = []
count = 0
for ali in glob.iglob("{0}" + '/alignments_*.csv'):
    if count%10==0: print('@', count)
    try:
        df = pd.read_csv(ali, index_col=0)
        container.append(df)
    except: continue
    count += 1
df = pd.concat(container)
df.to_csv("{0}" + '/full_alignments.csv')
for ali in glob.iglob("{0}" + '/alignments_*.csv'):
    try:
        os.remove(ali)
    except: continue
print('done')""".format(top_folder))
    scriptfile = top_folder + '/get_all_alignments.py'
    with open(scriptfile, 'w') as f:
        f.write(script)
    return scriptfile

    slurmscript = """#!/bin/bash\n#SBATCH --nodes 1
#SBATCH --partition={}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 32000
#SBATCH --time 65:00:00\n\n
source /work/lpdi/users/hartevel/venv/pytorch_cpu_gcc54/bin/activate && export PYTHONPATH=''\n
python -u {}""".format(partits, script_file)
    slurmfile = top_folder + '/get_matrix.slurm'
    with open(slurmfile, 'w') as f:
        f.write(slurmscript)
    return slurmfile

def control_slurm_file(slurm_file, main_id, partition, condition_file=None):
    """
    """
    _header = """#!/bin/bash\n#SBATCH --nodes 1
#SBATCH --partition={}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 1024
#SBATCH --time 00:05:00\n\n""".format(partition)
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
        time.sleep( 500 )
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
        time.sleep( 25 )
    else:
        return int(str(p.stdout.decode("utf-8")).strip())

def submit_slurm(top_folder, slurm_file, partits):
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
    tmalign_exe  = args.tmalign[0]
    nfiles       = int(args.narrays[0])
    partition    = args.partition[0]

    #wdir = os.getcwd()

    # create top folder level
    top_folder = "./{}".format(architecture)
    os.makedirs(top_folder, exist_ok=True)

    # Check if checkpoint file exists
    chkpt = Path("./{}/make_tmscore_matrix.chkpt".format(top_folder))
    if chkpt.is_file():
        print("Found checkpoint {}".format(chkpt))
        sys.exit()

    # run TM align
    native_folder = "./{}/processed01_TMscore_matrix/native/".format(top_folder)
    os.makedirs(native_folder, exist_ok=True)

    # get pdbs
    structures, ids = _get_pdbs(top_folder + "/processed01_pdbs/")

    # prepare scripts and slurmfiles
    alignment_folder = make_exelogs(native_folder, structures, tmalign_exe, nfiles)
    alignment_folder = native_folder + '/aligns/'
    slurmfile1 = make_slurm_file(native_folder, nfiles, partition)

    ###script = make_matrix_script(top_folder, native_folder, alignment_folder)
    script = make_matrix_script_per_batch(top_folder, native_folder, alignment_folder)
    ###slurmfile2 = make_matrix_slurm_file(native_folder, script)

    # calculate tm scores and process data
    ###alignment_folder = make_exelogs(native_folder, structures, tmalign_exe, nfiles)
    ###slurmfile1 = make_slurm_file(native_folder, nfiles)
    submit_slurm(native_folder, slurmfile1, partition)

    # create all alignments file
    matrix_script = make_matrix_script2(native_folder, partition)
    slurmfile2 = make_alignment_slurm_file(native_folder, matrix_script, partition)
    submit_slurm(native_folder, slurmfile2, partition)

    # Create checkpoint file
    with open("./{}/make_tmscore_matrix.chkpt".format(top_folder), 'w') as f:
        f.write("make_tmscore_matrix DONE")

    # Remove folders with large data files
    shutil.rmtree(os.path.join(native_folder, 'exelogs'))
    shutil.rmtree(os.path.join(native_folder, 'aligns'))


if __name__ == "__main__":
	main()
