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
import argparse
from pathlib import Path
import subprocess
from subprocess import DEVNULL

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
    #parse.add_argument("--allvsall", action="store_true", help="Use slurm accelerated parallel system.")
    return parse

def parse_args(parser):
	"""
	Parse the arguments of a parser object.
	:param parser: the parser object.
	:return: the specified arguments
	"""
	args = parser.parse_args()
	return args

def _get_pdbs(pdbfolder):
    ids = []
    for file in glob.iglob(pdbfolder + '/*.pdb'):
        id_ = '-'.join(file.split('-')[:-1])
        ids.append(id_)
    naives  = [i + '-naive.pdb'  for i in ids]
    natives = [i + '-native.pdb' for i in ids]
    return naives, natives, ids

def _read_tmalign(filename):
    """
    Read TMalign file.
    :param filename: the path to the TMalign output file.
    :return: pandas DataFrame with info.
    """
    # Set frame
    d = {'description': [], 'target': [],
         'l_description': [], 'l_target': [],
         'tm_score': [], 'rmsd': [], 'seqid': [], 'n_aligned': [],
         'alignment': []}

    # Read
    with open(filename, 'r') as f:
       lines = f.readlines()

    # Find
    alignment_str = ''
    tmscore, _alignment = None, 0.
    for line in lines:
       # Note that chain 1 gets superimposed onto chain 2,
       # thus we save chain 1 under name2 and viceversa
       if line.startswith('Name of Chain_1'):
          name2 = [s.strip() for s in line.split()][-1]
          #name2 = name2.split('.')[0]
          #name2 = os.path.basename(name2).replace('.pdb', '')
       if line.startswith('Name of Chain_2'):
          name1 = [s.strip() for s in line.split()][-1]
          #name1 = name1.split('.')[0]
          #name1 = os.path.basename(name1).replace('.pdb', '')
       if line.startswith('Length of Chain_1'):
          l2 = int(line.split()[-2].strip())
       if line.startswith('Length of Chain_2'):
          l1 = int(line.split()[-2].strip())
       if line.startswith('Aligned'):
          infos = line.split('=')
          al    = int(infos[1].split(',')[0].strip())
          rmsd  = float(infos[2].split(',')[0].strip())
          seqid = float(infos[-1].strip())
       if not tmscore and line.startswith('TM-score'):
          tmscore = float(line.split()[1].strip())
       if tmscore and line.startswith('TM-score'):
          tmscore2 = float(line.split()[1].strip())
          if tmscore < tmscore2:
             tmscore = tmscore2
       if line.startswith('(":"'):
          _alignment += 1
          continue
       if _alignment > 0:
          if _alignment < 4:
             alignment_str += line.strip()
             alignment_str += '\n'
             _alignment += 1

    # Save
    d['description'].append(name1)
    d['target'].append(name2)
    d['l_description'].append(l1)
    d['l_target'].append(l2)
    d['tm_score'].append(tmscore)
    d['rmsd'].append(rmsd)
    d['seqid'].append(seqid)
    d['n_aligned'].append(al)
    d['alignment'].append(alignment_str)
    return pd.DataFrame(d)

def _get_basename(string):
    return os.path.basename(string).replace(".pdb", "")

def calculate_and_process_TMscore(pdbfolder, outputfolder, tmalign_exe):
    """
    """
    naives, natives, ids = _get_pdbs(pdbfolder)
    dfs = []
    for naive, native, i in zip(naives, natives, ids):
        try:
            print("alinging {} <-- {}".format(naive, native))
            tmalignfile = "{}/{}.tmalign".format(outputfolder, os.path.basename(i))
            with open(tmalignfile, 'w') as f:
                subprocess.call([tmalign_exe, naive, native], stdout=f)
            df = _read_tmalign(tmalignfile)
            df = df.drop(columns=['description', 'target'])
            df = df.assign(description=_get_basename(naive), target=_get_basename(native))
            dfs.append(df)
        except: continue
    return pd.concat( dfs )

def process_tmalign_files(filenames):
    """
    """
    dfs = []
    for tmfile in filenames:
        df = _read_tmalign(tmfile)
        dfs.append(df)
    return pd.concat( dfs )


def main():
    """
    Main execution point.
    """
    # Parse arguments
    args = parse_args(create_parser())
    topology     = args.topology[0]
    architecture = args.architecture[0]
    tmalign_exe  = args.tmalign[0]

    #wdir = os.getcwd()

    # create top folder level
    top_folder = "./{}".format(architecture)
    os.makedirs(top_folder, exist_ok=True)

    # Check if checkpoint file exists
    chkpt = Path("./{}/calculate_tmscore.chkpt".format(top_folder))
    if chkpt.is_file():
        print("Found checkpoint {}".format(chkpt))
        sys.exit()

    # run TM align
    tmalign_folder = "./{}/processed01_TMalignments/".format(top_folder)
    os.makedirs(tmalign_folder, exist_ok=True)
    tm_frame  = calculate_and_process_TMscore("./{}/processed01_pdbs/".format(top_folder), tmalign_folder, tmalign_exe)
    #tm_frame = process_tmalign_files(tmfiles)

    # save final output
    tm_frame.to_csv(tmalign_folder + "full_tmalignments.csv")

    # Create checkpoint file
    with open("./{}/calculate_tmscore.chkpt".format(top_folder), 'w') as f:
        f.write("calculate_tmscore DONE")


if __name__ == "__main__":
	main()
