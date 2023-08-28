"""
.. codeauthor:: Zander Harteveld <zandermilanh@gmail.com>

.. affiliation::
    Laboratory of Protein Design and Immunoengineering <lpdi.epfl.ch>
    Bruno Correia <bruno.correia@epfl.ch>
"""

# Standard Libraries
import os
import sys
import warnings
import argparse
from pathlib import Path

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
    return parse

def parse_args(parser):
	"""
	Parse the arguments of a parser object.
	:param parser: the parser object.
	:return: the specified arguments
	"""
	args = parser.parse_args()
	return args


def main():
    """
    Main execution point.
    """
    # Parse arguments
    args = parse_args(create_parser())
    topology = args.topology[0]
    architecture = args.architecture[0]
    #wdir = os.getcwd()

    # Create top folder level
    top_folder = "./{}".format(architecture)
    os.makedirs(top_folder, exist_ok=True)

    # Check if checkpoint file exists
    chkpt = Path("./{}/make_sketches.chkpt".format(top_folder))
    if chkpt.is_file():
        print("Found checkpoint {}".format(chkpt))
        sys.exit()

    # Make topologies
    # Increase link distance
    # We do not care about it in the miniform search
    container = build_forms(architecture, check_forms=True,
                            link_distance_add=5., verbose=0)
    sketches  = prepare_forms(container, two_way=False)

    if sketches[sketches.description == topology].empty:
        #raise RuntimeError("DataFrame empty. Topology not viable.")
        warnings.warn("DataFrame empty. Topology not viable.")

    os.makedirs("./{}/okforms".format(top_folder), exist_ok=True)
    for i, row in sketches.iterrows():
        naive, desc = row.naive, row.description
        tgc.utils.write_pdb(naive, action="write", outfile="./{}/okforms/{}.pdb".format(top_folder, desc))

    # Create checkpoint file
    with open("./{}/make_sketches.chkpt".format(top_folder), 'w') as f:
        f.write("make_sketches DONE")


if __name__ == "__main__":
	main()
