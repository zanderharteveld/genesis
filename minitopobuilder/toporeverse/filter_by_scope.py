"""
.. codeauthor:: Zander Harteveld <zandermilanh@gmail.com>

.. affiliation::
    Laboratory of Protein Design and Immunoengineering <lpdi.epfl.ch>
    Bruno Correia <bruno.correia@epfl.ch>
"""

import os
import argparse
import re

import pandas as pd
import rstoolbox as rs


def cli_parser():
    """
    Create a CLI parser.
    :return: the parser object.
    """
    parser = argparse.ArgumentParser(description="Clean database with scope class identifiers.",
                                     epilog="Takes the classes (at all levels) and returns a"
                                            " clean file with the pdbs of the wanted classes.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--prefix', '-p', type=str, nargs=1, help="Prefix for output file.")
    parser.add_argument('--pds_file', '-m', type=str, nargs=1, help="Path to pds list for the maps.")
    parser.add_argument('--scope_file', '-s', type=str, nargs=1, default=["/work/lpdi/databases/master_scope2020/dir.des.scope.2.06-stable.txt"], help="Path to inital scope file.")
    parser.add_argument('--max_len', '-xmax', type=int, nargs=1, default=[None], help="Maximum domain length in num. of residues.")
    parser.add_argument('--min_len', '-xmin', type=int, nargs=1, default=[None], help="Minimum domain length in num. of residues.")
    parser.add_argument('--scope_classes', '-l', type=str, nargs="*", help="List of wanted scope classes.")
    return parser


def parse_args(parser):
    """
    Parse the arguments of a parser object.
    :param parser: the parser object.
    :return: the specified arguments
    """
    args = parser.parse_args()
    return args


def parse_pds_file(pds_file):
    """
    """
    with open(pds_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    d = {'path': [], 'scope_id': [],}
    for line in lines:
        line = line.strip()
        path   = os.path.dirname(line)
        scope_id = os.path.basename(line).replace(".pdb", "").replace(".pds", "")
        #chain  = os.path.basename(line).split('_')[-1].replace(".pdb", "").replace(".pds", "")

        d["path"].append(path)
        d["scope_id"].append(scope_id)
        #d["chain"].append(chain)
    return pd.DataFrame(d)


def parse_scope_file(scope_file):
    """
    """
    with open(scope_file, 'r') as f:
        lines = f.readlines()

    d = {'scope_nr': [], 'scope_id': [], 'pdb_id': [], 'chain': [], 'reschop': [], 'scope_class': []}
    for line in lines:
        if line.startswith('#') or line.startswith('\n'):
            continue

        line = re.split('\t|:|\s', line.strip())
        if line[3] == '-' or len(line) != 7:
            continue

        d['scope_nr'].append(line[0])
        d['scope_id'].append(line[3])
        d['pdb_id'].append(line[4])
        d['chain'].append(line[5])
        d['reschop'].append(line[6])
        d['scope_class'].append(line[2])
    return pd.DataFrame(d)


def parse_dssp_file(dssp_file):
    """
    """
    df_dssp = rs.io.read_fasta(dssp_file).rename(columns={'sequence_A': 'dssp'})
    series = df_dssp.description.str.split('_')
    df_dssp = df_dssp.assign(pdb_id=series.str[0], chain=series.str[1], sse_length=df_dssp.dssp.str.len())
    return df_dssp


def write_master_file( df, prefix="default" ):
    """
    """
    print("writing file of shape {}".format(df.shape))
    with open("{}.xfilter".format(prefix), "w") as f:
        for i, row in df.iterrows():
            f.write("{}/{}.pds\n".format(row.path, row.scope_id))


def main():
    """
    Main entry point.
    """
    # Arguments
    args = parse_args(cli_parser())
    prefix = args.prefix[0]
    max_len = args.max_len[0]
    min_len = args.min_len[0]
    pds_file = args.pds_file[0]
    scope_file  = args.scope_file[0]
    scope_classes = args.scope_classes

    # Get current working directory
    wdir = os.getcwd()

    # Create top folder level
    top_folder = "{}".format(wdir)

    # Parse
    df_scope  = parse_scope_file( scope_file )
    df_pds    = parse_pds_file( pds_file )
    if df_scope.empty or df_pds.empty:
        raise AssertionError("Empty input files")

    # Get common set
    #df = pd.merge(df_pds, df_scope[["pdb_id", "scope_class", "chain"]], on=["pdb_id", "chain"], how="inner")
    df = pd.merge(df_pds, df_scope[["scope_id", "scope_class"]], on=["scope_id"], how="outer")
    df = df.drop_duplicates(['scope_id'])
    df_filtered = df[df.scope_class.str.startswith(tuple(scope_classes))]
    if df_filtered.empty:
        raise AssertionError("Empty filtered list. Try adding more classes...")
    # Write
    write_master_file(df_filtered, prefix=top_folder + "/" + prefix)


if __name__ == '__main__':
    main()
