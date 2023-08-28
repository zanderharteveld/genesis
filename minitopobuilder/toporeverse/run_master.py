"""
.. codeauthor:: Zander Harteveld <zandermilanh@gmail.com>

.. affiliation::
    Laboratory of Protein Design and Immunoengineering <lpdi.epfl.ch>
    Bruno Correia <bruno.correia@epfl.ch>
"""

# Standard Libraries
import sys
import os
import math
import time
import glob
import argparse
import subprocess
import tempfile
from pathlib import Path

# External Libraries
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
    parse.add_argument("--pds",          "-l", type=str, nargs=1, help="List with path + PDS files to be searched.")
    parse.add_argument("--createpds",   type=str, nargs=1,   help="Path to createPDS executable.")
    parse.add_argument("--master",      type=str, nargs=1,   help="Path to MASTER executable.")
    parse.add_argument("--rmsd_cutoff", type=float, nargs=1, help="RMSD match cutoff (default = 2.5A).")
    parse.add_argument("--partition",   type=str, nargs=1, default=['serial'], help="Partition for slurm cluster (default: serial).")
    #parse.add_argument("--dssp", type=str, nargs=1, help="Path to DSSP executable.")
    return parse

def parse_args(parser):
	"""
	Parse the arguments of a parser object.
	:param parser: the parser object.
	:return: the specified arguments
	"""
	args = parser.parse_args()
	return args

def make_pds(pdbfile, exe='/Users/hartevel/bin/master-v1.6/bin/createPDS'):
    """
    """
    args    = [exe, '--type', 'query', '--pdb', pdbfile]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, universal_newlines=True)
    lines   = process.stdout.readlines()

def make_master(query, target, rmsd_cut=2.5, outfile=None, matches=None, exe=None):
    """
    """
    command = ["{}".format(exe), '--query', query, '--target', target,
               '--rmsdCut', str(rmsd_cut), '--topN', str(1)]
    if outfile:
        command.extend(['--matchOut', outfile])
    if matches:
        command.extend(['--structOut', matches, '--outType', 'full'])
    return ' '.join(command)

def make_slurm_files(top_folder, pds_list, partits, master=None, rmsd_cut=2.5):
    """
    """
    _header = """#!/bin/bash\n#SBATCH --nodes 1
#SBATCH --partition={}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4096
#SBATCH --time 10:00:00
#SBATCH --array=1-{}\n\n"""

    for query in glob.iglob("./{}/okforms/*.pds".format(top_folder)):
        qry = os.path.basename(query).replace(".pds", "")

        container = []
        for target in pds_list:
            trg = os.path.basename(target).replace(".pds", "")
            #os.makedirs('./okforms/{}_pdb/{}/'.format(qry, trg), exist_ok=True)
            # seems master can create 1 level folder structure by itself
            os.makedirs("./{}/okforms/{}_results/".format(top_folder, qry), exist_ok=True)
            cmd = make_master(query, target,
                              outfile="./{}/okforms/{}_results/_{}.master".format(top_folder, qry, trg),
                              matches="./{}/okforms/{}_results/{}/".format(top_folder, qry, trg),
                              exe="{}".format(master), rmsd_cut=rmsd_cut)
            container.append(cmd)

        with open("./{}/okforms/{}.exec".format(top_folder, qry), "w") as f:
            f.write("#!/bin/bash\n")
            f.write("SLURM_ARRAY_TASK_ID=$1\n")
            f.write("if (( ${SLURM_ARRAY_TASK_ID} == 1 )); then\n")
            taskid = 2
            for i, cmd in enumerate(container):
                if i%1000==0 and i != 0.:
                    f.write("fi\n")
                    f.write("if (( ${{SLURM_ARRAY_TASK_ID}} == {} )); then\n".format(taskid))
                    taskid += 1
                f.write(cmd + "\n")
            f.write("find ./{}/okforms/ -empty -type f -delete\n".format(top_folder) ) # remove all empty files
            f.write("find ./{}/okforms/ -empty -type d -delete\n".format(top_folder) ) # remove all empty folders
            f.write("fi\n")

        with open("./{}/{}.slurm".format(top_folder, qry), "w") as f:
            f.write(_header.format(partits,taskid))
            f.write("bash ./{}/okforms/{}.exec ${{SLURM_ARRAY_TASK_ID}}\n".format(top_folder, qry))

def control_slurm_file(slurm_file, main_id, count, partits, condition_file=None):
    """
    """
    _header = """#!/bin/bash\n#SBATCH --nodes 1
#SBATCH --partition={}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 2048
#SBATCH --time 00:05:00\n\n""".format(partits)
    condition_file = "touch_control.{}.{}".format(main_id, count)
    condition_file = Path().cwd().joinpath(condition_file)

    with open(slurm_file, 'w') as fd:
        fd.write(_header + '\n\n')
        fd.write('echo \'finished\' > {}\n'.format(condition_file.resolve()))
    return condition_file

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

def wait_for(condition_file):
    """
    """
    waiting_time = 0
    while not Path(condition_file).is_file():
        time.sleep(30)
        waiting_time += 1

def submit_slurm(top_folder, partits):
    """
    """
    # first submit all
    condition_files = []
    count = 0
    for slurm_file in glob.iglob("./{}/*.slurm".format(top_folder)):
        main_id = submit_nowait_slurm(slurm_file)
        print("./{}/slurm_control.{}.{}.sh".format(top_folder, main_id, count))
        slurm_control_file = ( "./{}/slurm_control.{}.{}.sh".format(top_folder, main_id, count) )
        condition_file = control_slurm_file(slurm_control_file, main_id, count, partits)
        condition_files.append(condition_file)

        submit_nowait_slurm(slurm_control_file, "afterany", main_id)
        count += 1

    # then check all
    for condition_file in condition_files:
        wait_for(condition_file)
        os.unlink(str(condition_file))

def submit_all(top_folder):
    """
    """
    for slurm_script in glob.iglob("./{}/*.slurm".format(top_folder)):
        print("Submit {}".format(slurm_script))
        submit_slurm(slurm_script)

def squeue():
    counter = 0
    args = ['squeue', '-u', os.getlogin()]
    out = subprocess.check_output(args, universal_newlines=True)
    for l in out.split("\n"):
        if os.getlogin() in l:
            if not "[" in l:
                counter += 1
            else:
                n = [int(x) for x in l.strip().split()[0].split("[")[1].split("]")[0].split("-")]
                try:
                    counter += ( (n[1] - n[0]) + 1 )
                except:
                    counter += 1
    return counter

def squeue_wait( max_jobs, wait_time ):
    sys.stdout.flush()
    while squeue() > max_jobs:
        time.sleep( wait_time )
    sys.stdout.write("Wait finished, submit next batch.\n")
    sys.stdout.flush()

def concat_all(top_folder):
    """
    """
    for folder in glob.iglob("./{}/okforms/*results".format(top_folder)):
        prefix = os.path.basename(folder).replace("_results", "")
        summary_file = "./{}/{}.master".format(top_folder, prefix)
        print("Generating summary file at {}".format(summary_file))
        with open(summary_file, 'w') as f:
            for filename in glob.iglob("{}/*.master".format(folder)):
                with open(filename, "r") as f2:
                    lines = f2.readlines()
                f.writelines(lines)


def main():
    """
    Main execution point.
    """
    # Parse arguments
    args = parse_args(create_parser())
    topology     = args.topology[0]
    architecture = args.architecture[0]
    pds          = args.pds[0]
    createpds    = args.createpds[0]
    master       = args.master[0]
    rmsd_cutoff  = args.rmsd_cutoff[0]
    #dssp         = args.dssp[0]
    partition    = args.partition[0]

    # Create top folder level
    #wdir = os.getcwd()

    # Create top folder level
    top_folder = "./{}".format(architecture)
    os.makedirs(top_folder, exist_ok=True)

    # Check if checkpoint file exists
    chkpt = Path("./{}/run_master.chkpt".format(top_folder))
    if chkpt.is_file():
        print("Found checkpoint {}".format(chkpt))
        sys.exit()

    # Create query pds files
    for sketch in glob.iglob("./{}/okforms/*.pdb".format(top_folder)):
        make_pds(sketch, exe=createpds)

    # Load targets
    with open("{}".format(pds), "r") as f:
        pds_list = f.readlines()
    pds_list = [line.strip() for line in pds_list]

    # Make slurm scripts
    make_slurm_files(top_folder, pds_list, partition, master=master, rmsd_cut=rmsd_cutoff)

    # Submission script
    submit_slurm(top_folder, partition)
    concat_all(top_folder)

    # Create checkpoint file
    with open("./{}/run_master.chkpt".format(top_folder), 'w') as f:
        f.write("run_master DONE")


if __name__ == "__main__":
	main()
