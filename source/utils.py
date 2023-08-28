# coding: utf-8
# standard libraries
import os
import sys
import re
import glob
import json
import time
import random
import tempfile
import numpy as np
import pandas as pd

# nn libraries
import torch
from torch import nn

# pyrosetta related
from pyrosetta import *
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

# trDesign
from .utils_trDesign import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
################################### Helpers ####################################
################################################################################


def atof(text):
    """ """
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    """
    return [atof(c) for c in re.split(r"[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)", text)]


def apply_masks(x, mask):
    return [xm * mask.unsqueeze(-1).expand(xm.shape) for xm in x]


def write_pdb(
    df,
    action="print",
    outfile="pdb_0001.pdb",
    xcoord="x",
    ycoord="y",
    zcoord="z",
    chain_id="A",
    atomtype="atomtype",
    residuetype="res3aa",
    residuenum="id",
    bb_atoms=None,
):
    """Print or write PDB from information a DataFrame."""
    _PDB = "{:6s}{:5d}  {:3s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}"
    # df = df.sort_values(['id', 'atomtype'])
    if bb_atoms:
        # bb_atoms.sort()
        if "atomtype" not in df.columns:
            atomtypes = bb_atoms * int(len(df) / len(bb_atoms))
        else:
            atomtypes = df[atomtype].values

        if "id" not in df.columns:
            residuenums = np.repeat(
                np.array([i for i in range(1, int(len(df) / len(bb_atoms)) + 1)]),
                len(bb_atoms),
            )
        else:
            residuenums = df[residuenum].values
    else:
        atomtypes = df[atomtype].values
        residuenums = df[residuenum].values
    if action == "print":
        for i in range(len(df)):
            print(
                _PDB.format(
                    "ATOM",
                    int(i + 1),
                    atomtypes[i],  # df.iloc[i][atomtype],
                    "",
                    df.iloc[i][residuetype],
                    chain_id,
                    int(residuenums[i]),  # int(df.iloc[i][residuenum]),
                    "",
                    float(df.iloc[i][xcoord]),
                    float(df.iloc[i][ycoord]),
                    float(df.iloc[i][zcoord]),
                    1.0,
                    0.0,
                    "",
                    "",
                )
            )
    if action == "write":
        with open(outfile, "w") as f:
            for i in range(len(df)):
                f.write(
                    _PDB.format(
                        "ATOM",
                        int(i + 1),
                        atomtypes[i],  # df.iloc[i][atomtype],
                        "",
                        df.iloc[i][residuetype],
                        chain_id,
                        int(residuenums[i]),  # int(df.iloc[i][residuenum]),
                        "",
                        float(df.iloc[i][xcoord]),
                        float(df.iloc[i][ycoord]),
                        float(df.iloc[i][zcoord]),
                        1.0,
                        0.0,
                        "",
                        "",
                    )
                )
                f.write("\n")


################################################################################
############################### PRE-processing #################################
################################################################################


def sse_to_1hot(sse):
    # total 3 sse types
    sse1hot = {"H": [1.0, 0.0, 0.0], "E": [0.0, 1.0, 0.0]}
    return (
        torch.tensor([sse1hot[s] if s in ["H", "E"] else [0.0, 0.0, 1.0] for s in sse])
        .float()
        .to("cpu")
    )


def pad1d_map(xd, max_length=128):
    p1d = (0, 0, 0, max_length - xd.shape[0])
    xd_pad = nn.functional.pad(xd, p1d, "constant", 0)
    return xd_pad.float().to("cpu")


def pretrain_preprocess(data_dir, save_dir, testing, pad_size=128, full_match=True):
    """Preprocesses and saves the scope data."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        data_processed_train = {
            "topology": [],
            "description": [],
            "dim": [],
            "x_ca": [],
            "x_c": [],
            "x_n": [],
            "x_cb": [],
            "y_ca": [],
            "y_c": [],
            "y_n": [],
            "y_cb": [],
        }
        data_processed_test = {
            "topology": [],
            "description": [],
            "dim": [],
            "x_ca": [],
            "x_c": [],
            "x_n": [],
            "x_cb": [],
            "y_ca": [],
            "y_c": [],
            "y_n": [],
            "y_cb": [],
        }
        for i, datafile in enumerate(glob.iglob(data_dir + "/*.pt")):
            if (i + 1) % 200 == 0:
                print("@", i)

            basename = os.path.basename(datafile)
            topology = basename.split("-")[0]
            if full_match == True:
                split = (
                    "test"
                    if basename.split("-")[1].replace(".pt", "") in testing
                    else "train"
                )
            else:
                split = (
                    "test"
                    if any(
                        [
                            basename.split("-")[1].replace(".pt", "").startswith(item)
                            for item in testing
                        ]
                    )
                    else "train"
                )

            # load input
            data = torch.load(os.path.join(datafile), map_location=torch.device("cpu"))

            # drop large domains
            tpg_sse = sse_to_1hot(data["topology_ss"])
            if len(tpg_sse) > pad_size:
                continue
            tpg_sse_pad = pad1d_map(tpg_sse, max_length=pad_size)

            # load coordinates
            xatom_ca, yatom_ca = (
                data["naive"]["CA"]["cartesian"],
                data["native"]["CA"]["cartesian"],
            )
            xatom_c, yatom_c = (
                data["naive"]["C"]["cartesian"],
                data["native"]["C"]["cartesian"],
            )
            xatom_n, yatom_n = (
                data["naive"]["N"]["cartesian"],
                data["native"]["N"]["cartesian"],
            )
            xatom_cb, yatom_cb = (
                data["naive"]["CB"]["cartesian"],
                data["native"]["CB"]["cartesian"],
            )

            # if len(xatom_ca) != len(yatom_ca): print('input and target shapes do not align'); continue
            if len(xatom_ca) != len(yatom_ca):
                continue  # remove non-aligned structures
            if len(xatom_c) != len(yatom_c):
                continue  # remove non-aligned structures
            if len(xatom_n) != len(yatom_n):
                continue  # remove non-aligned structures
            if len(xatom_ca) < 40:
                continue  # remove very small structures

            xatom_ca_pad, yatom_ca_pad = pad1d_map(
                xatom_ca, max_length=pad_size
            ), pad1d_map(yatom_ca, max_length=pad_size)
            xatom_c_pad, yatom_c_pad = pad1d_map(
                xatom_c, max_length=pad_size
            ), pad1d_map(yatom_c, max_length=pad_size)
            xatom_n_pad, yatom_n_pad = pad1d_map(
                xatom_n, max_length=pad_size
            ), pad1d_map(yatom_n, max_length=pad_size)
            xatom_cb_pad, yatom_cb_pad = pad1d_map(
                xatom_cb, max_length=pad_size
            ), pad1d_map(yatom_cb, max_length=pad_size)
            if torch.sum(torch.isnan(tpg_sse_pad)) > 0.0:
                print("tpg_sse_pad  nan")
                continue
            if torch.sum(torch.isnan(xatom_ca_pad)) > 0.0:
                print("xatom_ca_pad nan")
                continue
            if torch.sum(torch.isnan(xatom_ca_pad)) > 0.0:
                print("xatom_c_pad  nan")
                continue
            if torch.sum(torch.isnan(xatom_c_pad)) > 0.0:
                print("xatom_c_pad  nan")
                continue
            if torch.sum(torch.isnan(xatom_n_pad)) > 0.0:
                print("xatom_n_pad  nan")
                continue
            if torch.sum(torch.isnan(yatom_ca_pad)) > 0.0:
                print("yatom_ca_pad nan")
                continue
            if torch.sum(torch.isnan(yatom_c_pad)) > 0.0:
                print("yatom_c_pad  nan")
                continue
            if torch.sum(torch.isnan(yatom_n_pad)) > 0.0:
                print("yatom_n_pad  nan")
                continue
            if torch.sum(torch.isnan(yatom_cb_pad)) > 0.0:
                print("yatom_cb_pad nan")
                continue

            if split == "train":
                data_processed_train["description"].append(basename.replace(".pt", ""))
                data_processed_train["topology"].append(tpg_sse_pad)
                data_processed_train["dim"].append(len(tpg_sse))
                data_processed_train["x_ca"].append(xatom_ca_pad)
                data_processed_train["x_c"].append(xatom_c_pad)
                data_processed_train["x_n"].append(xatom_n_pad)
                data_processed_train["x_cb"].append(xatom_n_pad)
                data_processed_train["y_ca"].append(yatom_ca_pad)
                data_processed_train["y_c"].append(yatom_c_pad)
                data_processed_train["y_n"].append(yatom_n_pad)
                data_processed_train["y_cb"].append(yatom_cb_pad)
            else:
                data_processed_test["description"].append(basename.replace(".pt", ""))
                data_processed_test["topology"].append(tpg_sse_pad)
                data_processed_test["dim"].append(len(tpg_sse))
                data_processed_test["x_ca"].append(xatom_ca_pad)
                data_processed_test["x_c"].append(xatom_c_pad)
                data_processed_test["x_n"].append(xatom_n_pad)
                data_processed_test["x_cb"].append(xatom_n_pad)
                data_processed_test["y_ca"].append(yatom_ca_pad)
                data_processed_test["y_c"].append(yatom_c_pad)
                data_processed_test["y_n"].append(yatom_n_pad)
                data_processed_test["y_cb"].append(yatom_cb_pad)

        print("Dump train, {}".format(len(data_processed_train["dim"])))
        torch.save(data_processed_train, save_dir + "/train.pt")
        print("Dump train csv, {}".format(len(data_processed_train["dim"])))
        pd.DataFrame(data_processed_train)[["description"]].to_csv(
            save_dir + "/train.csv"
        )
        print("Dump test,  {}".format(len(data_processed_test["dim"])))
        torch.save(data_processed_test, save_dir + "/test.pt")
        print("Dump test csv, {}".format(len(data_processed_test["dim"])))
        pd.DataFrame(data_processed_test)[["description"]].to_csv(
            save_dir + "/test.csv"
        )


class ProteinSketchPretrain(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        filename = os.path.join(root, "{}.pt".format(split, split))
        print("Loading {}".format(filename))
        self.data = torch.load(filename, map_location=torch.device(device))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data["dim"])

    def __getitem__(self, idx):
        topology = self.data["topology"][idx]
        desc = self.data["description"][idx]
        dim = self.data["dim"][idx]
        xca, xc, xn, xcb = (
            self.data["x_ca"][idx],
            self.data["x_c"][idx],
            self.data["x_n"][idx],
            self.data["x_cb"][idx],
        )
        yca, yc, yn, ycb = (
            self.data["y_ca"][idx],
            self.data["y_c"][idx],
            self.data["y_n"][idx],
            self.data["y_cb"][idx],
        )
        return (topology, dim, desc), (xca, xc, xn, xcb), (yca, yc, yn, ycb)


def encode_on_gpu(data, pad_size=128):
    dims = data[0][1].to(device)

    xca, xc, xn = data[1][0].to(device), data[1][1].to(device), data[1][2].to(device)
    xcb = extend_torch(xc, xn, xca, 1.522, 1.927, -2.143)
    yca, yc, yn = data[2][0].to(device), data[2][1].to(device), data[2][2].to(device)
    ycb = extend_torch(yc, yn, yca, 1.522, 1.927, -2.143)

    # featurize
    xdist, ydist = to_len_torch(xcb[:, :, None], xcb[:, None, :]), to_len_torch(
        ycb[:, :, None], ycb[:, None, :]
    )
    xomega, yomega = to_dih_torch(
        xca[:, :, None], xcb[:, :, None], xcb[:, None, :], xca[:, None, :]
    ), to_dih_torch(yca[:, :, None], ycb[:, :, None], ycb[:, None, :], yca[:, None, :])
    xtheta, ytheta = to_dih_torch(
        xn[:, :, None], xca[:, :, None], xcb[:, :, None], xcb[:, None, :]
    ), to_dih_torch(yn[:, :, None], yca[:, :, None], ycb[:, :, None], ycb[:, None, :])
    xphi, yphi = to_ang_torch(
        xca[:, :, None], xcb[:, :, None], xcb[:, None, :]
    ), to_ang_torch(yca[:, :, None], ycb[:, :, None], ycb[:, None, :])

    # bin
    yp_dist = mtx2bins_torch(ydist, 2, 20, 37, mask=(ydist > 20))
    yp_omega = mtx2bins_torch(
        yomega, -np.pi, np.pi, 25, mask=(yp_dist[..., 0] == 1)
    )  # 25
    yp_theta = mtx2bins_torch(
        ytheta, -np.pi, np.pi, 25, mask=(yp_dist[..., 0] == 1)
    )  # 25
    yp_phi = mtx2bins_torch(yphi, 0, np.pi, 13, mask=(yp_dist[..., 0] == 1))  # 13

    # apply mask to padded dim
    yp_dist[(yp_dist.sum(-2) == pad_size).unsqueeze(-2).expand(yp_dist.shape)] = 0
    yp_omega[(yp_omega.sum(-2) == pad_size).unsqueeze(-2).expand(yp_omega.shape)] = 0
    yp_theta[(yp_theta.sum(-2) == pad_size).unsqueeze(-2).expand(yp_theta.shape)] = 0
    yp_phi[(yp_phi.sum(-2) == pad_size).unsqueeze(-2).expand(yp_phi.shape)] = 0

    # clean
    # efficient solution: make one mask and expand into needed shape!
    mask = create2d_masks(dims)
    yp_dist *= mask.unsqueeze(-1).expand(yp_dist.shape)
    yp_omega *= mask.unsqueeze(-1).expand(yp_omega.shape)
    yp_theta *= mask.unsqueeze(-1).expand(yp_theta.shape)
    yp_phi *= mask.unsqueeze(-1).expand(yp_phi.shape)

    xomega[xdist >= 20] = 0.0
    xtheta[xdist >= 20] = 0.0
    xphi[xdist >= 20] = 0.0
    xdist[xdist >= 20] = 0.0

    xdist *= mask
    xomega *= mask
    xtheta *= mask
    xphi *= mask

    # stack channels
    x = torch.stack([xdist, xomega, xtheta, xphi], dim=1)

    return (
        x.float().detach(),
        (
            yp_dist.float().detach(),
            yp_omega.float().detach(),
            yp_theta.float().detach(),
            yp_phi.float().detach(),
        ),
        mask.float().detach(),
        dims.float().detach(),
    )


################################################################################
############################### Processing #####################################
################################################################################


def get_loops(s):
    llen, linside = [], []
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            linside.extend(s[i])
        if s[i] != s[i + 1]:
            linside.extend(s[i])
            llen.append(linside)
            linside = []
    return "x." + ".".join([str(len(e)) for e in llen if e[0] == "L"]) + ".x"


def preprocess(save_dir, splits, pad_size=128):
    """Preprocesses and saves the data."""
    for split in splits.keys():
        print(split)
        outdir = save_dir + "/{}".format(split)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            data_processed = {
                "topology": [],
                "description": [],
                "dim": [],
                "loops": [],
                "x_ca": [],
                "x_c": [],
                "x_n": [],
                "x_cb": [],
                "y_ca": [],
                "y_c": [],
                "y_n": [],
                "y_cb": [],
            }
            for i, datafile in enumerate(splits[split]):
                if (i + 1) % 200 == 0:
                    print("@", i)

                basename = os.path.basename(datafile)
                topology = basename.split("-")[0]

                # load input
                # data  = torch.load(os.path.join(root_path, datafile), map_location=torch.device('cpu'))
                try:
                    data = torch.load(datafile, map_location=torch.device("cpu"))
                except:
                    print("skipping {}".format(datafile))
                    continue
                loops = get_loops(data["topology_ss"])

                # drop large domains
                tpg_sse = sse_to_1hot(data["topology_ss"])
                if len(tpg_sse) > pad_size:
                    continue
                tpg_sse_pad = pad1d_map(tpg_sse, max_length=pad_size)

                # load coordinates
                xatom_ca, yatom_ca = (
                    data["naive"]["CA"]["cartesian"],
                    data["native"]["CA"]["cartesian"],
                )
                xatom_c, yatom_c = (
                    data["naive"]["C"]["cartesian"],
                    data["native"]["C"]["cartesian"],
                )
                xatom_n, yatom_n = (
                    data["naive"]["N"]["cartesian"],
                    data["native"]["N"]["cartesian"],
                )
                xatom_cb, yatom_cb = (
                    data["naive"]["CB"]["cartesian"],
                    data["native"]["CB"]["cartesian"],
                )
                if len(xatom_ca) != len(yatom_ca):
                    print("input and target shapes do not align")
                    continue

                xatom_ca_pad, yatom_ca_pad = pad1d_map(
                    xatom_ca, max_length=pad_size
                ), pad1d_map(yatom_ca, max_length=pad_size)
                xatom_c_pad, yatom_c_pad = pad1d_map(
                    xatom_c, max_length=pad_size
                ), pad1d_map(yatom_c, max_length=pad_size)
                xatom_n_pad, yatom_n_pad = pad1d_map(
                    xatom_n, max_length=pad_size
                ), pad1d_map(yatom_n, max_length=pad_size)
                xatom_cb_pad, yatom_cb_pad = pad1d_map(
                    xatom_cb, max_length=pad_size
                ), pad1d_map(yatom_cb, max_length=pad_size)
                if torch.sum(torch.isnan(tpg_sse_pad)) > 0.0:
                    print("tpg_sse_pad  nan")
                    continue
                if torch.sum(torch.isnan(xatom_ca_pad)) > 0.0:
                    print("xatom_ca_pad nan")
                    continue
                if torch.sum(torch.isnan(xatom_ca_pad)) > 0.0:
                    print("xatom_c_pad  nan")
                    continue
                if torch.sum(torch.isnan(xatom_c_pad)) > 0.0:
                    print("xatom_c_pad  nan")
                    continue
                if torch.sum(torch.isnan(xatom_n_pad)) > 0.0:
                    print("xatom_n_pad  nan")
                    continue
                if torch.sum(torch.isnan(yatom_ca_pad)) > 0.0:
                    print("yatom_ca_pad nan")
                    continue
                if torch.sum(torch.isnan(yatom_c_pad)) > 0.0:
                    print("yatom_c_pad  nan")
                    continue
                if torch.sum(torch.isnan(yatom_n_pad)) > 0.0:
                    print("yatom_n_pad  nan")
                    continue
                if torch.sum(torch.isnan(yatom_cb_pad)) > 0.0:
                    print("yatom_cb_pad nan")
                    continue

                data_processed["description"].append(basename.replace(".pt", ""))
                data_processed["topology"].append(tpg_sse_pad)
                data_processed["loops"].append(loops)
                data_processed["dim"].append(len(tpg_sse))
                data_processed["x_ca"].append(xatom_ca_pad)
                data_processed["x_c"].append(xatom_c_pad)
                data_processed["x_n"].append(xatom_n_pad)
                data_processed["x_cb"].append(xatom_n_pad)
                data_processed["y_ca"].append(yatom_ca_pad)
                data_processed["y_c"].append(yatom_c_pad)
                data_processed["y_n"].append(yatom_n_pad)
                data_processed["y_cb"].append(yatom_cb_pad)

            print("Dump {}, {}".format(split, len(data_processed["dim"])))
            torch.save(data_processed, outdir + "/{}.pt".format(split))
            pd.DataFrame(data_processed)[["description"]].to_csv(
                outdir + "/{}.csv".format(split)
            )


class ProteinSketch(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        filename = os.path.join(root, "{}/{}.pt".format(split, split))
        print("Loading {}".format(filename))
        self.data = torch.load(filename, map_location=torch.device(device))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data["dim"])

    def __getitem__(self, idx):
        topology = self.data["topology"][idx]
        dim = self.data["dim"][idx]
        desc = self.data["description"][idx]
        loops = self.data["loops"][idx]
        xca, xc, xn, xcb = (
            self.data["x_ca"][idx],
            self.data["x_c"][idx],
            self.data["x_n"][idx],
            self.data["x_cb"][idx],
        )
        yca, yc, yn, ycb = (
            self.data["y_ca"][idx],
            self.data["y_c"][idx],
            self.data["y_n"][idx],
            self.data["y_cb"][idx],
        )
        return (topology, dim, desc, loops), (xca, xc, xn, xcb), (yca, yc, yn, ycb)


################################################################################
################################## Encoding ####################################
################################################################################


def pad1d_map(xd, max_length=128):
    p1d = (0, 0, 0, max_length - xd.shape[0])
    xd_pad = nn.functional.pad(xd, p1d, "constant", 0)
    return xd_pad.float().to(device)


def pad2d_map(xd, max_length=128):
    p2d = (0, max_length - xd.shape[1], 0, max_length - xd.shape[1])
    xd_pad = nn.functional.pad(xd, p2d, "constant", 0)
    # pad_mask = nn.functional.pad(torch.ones_like(xd), p2d, "constant", 0)
    return xd_pad.float().to(device)


def pad3d_map(xd, max_length=128):
    p3d = (0, 0, 0, max_length - xd.shape[1], 0, max_length - xd.shape[1])
    xd_pad = nn.functional.pad(xd, p3d, "constant", 0)
    # pad_mask = nn.functional.pad(torch.ones_like(xd), p3d, "constant", 0)
    return xd_pad.float().to(device)


def create2d_masks(dims):
    return torch.stack(
        [pad2d_map(torch.ones((dim, dim), device=device)) for dim in dims], dim=0
    )


def create3d_masks(dims, pdim=37):
    return torch.stack(
        [pad3d_map(torch.ones((dim, dim, pdim), device=device)) for dim in dims], dim=0
    )


def make_contact_range(t, lower=1.0, higher=25.0):
    t = t.float()
    return torch.where((lower < t) & (t < higher), t, torch.tensor(0.0, device=device))


def mtx2bins_torch(x_ref, start, end, nbins, mask=None):
    bins = torch.linspace(start, end, nbins, device=device)
    x_true = torch.bucketize(x_ref, bins)
    if mask is not None:
        x_true[mask] = 0
    return torch.eye(nbins + 1, device=device)[x_true][..., :-1]


def bins2real(x_ref, start, end, nbins, mask=None, add_bin=False):
    """ """
    bins = torch.linspace(start, end, nbins, device=device)
    if add_bin == True:
        bins = torch.cat([torch.tensor([0], device=device), bins])
    # m = x_ref.argmax(-1)
    # m[m>0] = 1
    x_true = bins[x_ref.argmax(-1)] * mask
    return x_true


def extend_torch(a, b, c, L, A, D):
    L, A, D = (
        torch.tensor(L, device=device),
        torch.tensor(A, device=device),
        torch.tensor(D, device=device),
    )
    N = lambda x: x / torch.sqrt((x**2).sum(-1, keepdims=True) + 1e-8)
    bc = N(b - c)
    n = N(torch.cross(b - a, bc))
    m = [bc, torch.cross(n, bc), n]
    d = [
        L * torch.cos(A),
        L * torch.sin(A) * torch.cos(D),
        -L * torch.sin(A) * torch.sin(D),
    ]
    return c + sum([m * d for m, d in zip(m, d)])


def to_len_torch(a, b):
    return torch.sqrt(torch.sum((a - b) ** 2, dim=-1))


def to_ang_torch(a, b, c):
    D = lambda x, y: torch.sum(x * y, dim=-1)
    N = lambda x: x / torch.sqrt((x**2).sum(-1, keepdims=True) + 1e-8)
    return torch.acos(D(N(b - a), N(b - c)))


def to_dih_torch(a, b, c, d):
    D = lambda x, y: torch.sum(x * y, dim=-1)
    N = lambda x: x / torch.sqrt((x**2).sum(-1, keepdims=True) + 1e-8)
    bc = N(b - c)
    ab = N(a - b)
    cd = N(c - d)
    if len(bc.shape) == 4:
        batch, l1, l2, coords = bc.shape
        bc, ab, cd = (
            bc.expand(batch, l1, l1, coords),
            ab.expand(batch, l1, l1, coords),
            cd.expand(batch, l1, l1, coords),
        )
    else:
        batch, examples, l1, l2, coords = bc.shape
        bc, ab, cd = (
            bc.expand(batch, examples, l1, l1, coords),
            ab.expand(batch, examples, l1, l1, coords),
            cd.expand(batch, examples, l1, l1, coords),
        )
    n1 = torch.cross(ab, bc)
    n2 = torch.cross(bc, cd)
    n1bc = torch.cross(n1, bc)
    n1bcn2, n1n2 = D(n1bc, n2), D(n2, n1)
    return torch.atan2(n1bcn2, n1n2)


def encode_x_on_gpu(data, pad_size=128):
    dims = data[0].to(device)
    xca, xc, xn = data[1][0].to(device), data[1][1].to(device), data[1][2].to(device)
    xcb = extend_torch(xc, xn, xca, 1.522, 1.927, -2.143)

    # featurize
    xdist = to_len_torch(xcb[:, :, None], xcb[:, None, :])
    xomega = to_dih_torch(
        xca[:, :, None], xcb[:, :, None], xcb[:, None, :], xca[:, None, :]
    )
    xtheta = to_dih_torch(
        xn[:, :, None], xca[:, :, None], xcb[:, :, None], xcb[:, None, :]
    )
    xphi = to_ang_torch(xca[:, :, None], xcb[:, :, None], xcb[:, None, :])

    xomega[xdist >= 20] = 0.0
    xtheta[xdist >= 20] = 0.0
    xphi[xdist >= 20] = 0.0
    xdist[xdist >= 20] = 0.0

    mask = create2d_masks(dims)
    xdist *= mask
    xomega *= mask
    xtheta *= mask
    xphi *= mask

    # stack channels
    x = torch.stack([xdist, xomega, xtheta, xphi], dim=1)
    return x.float().detach(), mask.float().detach(), dims.float().detach()


def symmetrize(x):
    return 0.5 * (x + x.permute((1, 0, 2)))


def discretize_torch(x, nbins):
    return torch.eye(nbins + 1)[x][..., :-1]


################################################################################
############################ Training and Loss #################################
################################################################################


def cdf_loss(a, b, p=1):
    """
    last-dimension is weight distribution
    p is the norm of the distance, p=1 --> First Wasserstein Distance
    normalize distribution, add 1e-14 to divisor to avoid 0/0
    """
    a = a / (np.sum(a, axis=-1, keepdims=True) + 1e-14)
    b = b / (np.sum(b, axis=-1, keepdims=True) + 1e-14)
    # make cdf with cumsum
    cdf_a = np.cumsum(a, axis=-1)
    cdf_b = np.cumsum(b, axis=-1)
    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = np.sum(np.abs((cdf_a - cdf_b)), axis=-1)
    elif p == 2:
        cdf_distance = np.sqrt(np.sum(np.power((cdf_a - cdf_b), 2), axis=-1))
    else:
        cdf_distance = np.power(
            np.sum(np.power(np.abs(cdf_a - cdf_b), p), axis=-1), 1 / p
        )
    return cdf_distance


def wasserstein_loss(a, b):
    """Compute the first Wasserstein distance between two 1D distributions."""
    return cdf_loss(a, b, p=1)


def torch_cdf_loss(tensor_a, tensor_b, p=1):
    """
    last-dimension is weight distribution
    p is the norm of the distance, p=1 --> First Wasserstein Distance
    normalize distribution, add 1e-14 to divisor to avoid 0/0
    """
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a, dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b, dim=-1)
    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a - cdf_tensor_b)), dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(
            torch.sum(torch.pow((cdf_tensor_a - cdf_tensor_b), 2), dim=-1)
        )
    else:
        cdf_distance = torch.pow(
            torch.sum(torch.pow(torch.abs(cdf_tensor_a - cdf_tensor_b), p), dim=-1),
            1 / p,
        )
    return cdf_distance


def torch_wasserstein_loss(tensor_a, tensor_b):
    """Compute the first Wasserstein distance between two 1D distributions."""
    return torch_cdf_loss(tensor_a, tensor_b, p=1)


def loss_function(x, x_hat, mask, dim, means, log_vars):
    """
    Overall loss function used. Contains the KL divergence on the latent space +
    the 1st Wasserstein losses on each of the feature maps (we aggregate `pixel`-wise)
    """
    yd, yo, yt, yp = x
    xd, xo, xt, xp = apply_masks(x_hat, mask)
    wasser_dist_loss = torch_wasserstein_loss(xd, yd).sum() / dim.sum()
    wasser_omega_loss = torch_wasserstein_loss(xo, yo).sum() / dim.sum()
    wasser_theta_loss = torch_wasserstein_loss(xt, yt).sum() / dim.sum()
    wasser_phi_loss = torch_wasserstein_loss(xp, yp).sum() / dim.sum()
    reprod_loss = (
        wasser_dist_loss + wasser_omega_loss + wasser_theta_loss + wasser_phi_loss
    )

    KLD = (
        torch.sum(-1 - log_vars + means.pow(2) + log_vars.exp(), dim=-1).sum()
        / dim.sum()
    )

    return reprod_loss + KLD, reprod_loss


def loss_function2(x, x_hat, mask, dim, means, log_vars):
    """
    Overall loss function used. Contains the KL divergence on the latent space +
    the 1st Wasserstein losses on each of the feature maps (we aggregate `pixel`-wise)
    """
    yd, yo, yt, yp = x
    xd, xo, xt, xp = apply_masks(x_hat, mask)
    wasser_dist_loss = torch_wasserstein_loss(xd, yd).sum() / dim.sum()
    wasser_omega_loss = torch_wasserstein_loss(xo, yo).sum() / dim.sum()
    wasser_theta_loss = torch_wasserstein_loss(xt, yt).sum() / dim.sum()
    wasser_phi_loss = torch_wasserstein_loss(xp, yp).sum() / dim.sum()
    reprod_loss = (
        wasser_dist_loss + wasser_omega_loss + wasser_theta_loss + wasser_phi_loss
    )

    KLD = (
        torch.sum(-1 - log_vars + means.pow(2) + log_vars.exp(), dim=-1).sum()
        / dim.sum()
    )

    return reprod_loss + (KLD * 100), reprod_loss, KLD


def train(model, optimizer, loss_fn, loader, pad_size=128):
    """ """
    model.train()
    losses, reprod_losses = [], []
    for batch_idx, data in enumerate(loader):
        with torch.no_grad():
            data = encode_on_gpu(data, pad_size=pad_size)  # pushes to device iternally
            x, mask, dim = data[0], data[2], data[3]
            yd, yo, yt, yp = data[1][0], data[1][1], data[1][2], data[1][3]
            if torch.sum(torch.isnan(x)) > 0:
                continue
        optimizer.zero_grad()

        x_hat, means, log_vars = model(x)
        loss, reprod_loss = loss_fn((yd, yo, yt, yp), x_hat, mask, dim, means, log_vars)

        losses.append(loss.item())
        reprod_losses.append(reprod_loss.item())
        loss.backward()

        # Clip grads
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)

        optimizer.step()
        # scheduler.step()

    batch_size = x.shape[0]
    losses, reprod_losses = torch.tensor(losses), torch.tensor(reprod_losses)
    return losses.mean(), losses.std(), reprod_losses.mean()


def train2(model, optimizer, loss_fn, loader, pad_size=128):
    """ """
    model.train()
    losses, reprod_losses, kld_losses = [], [], []
    for batch_idx, data in enumerate(loader):
        with torch.no_grad():
            data = encode_on_gpu(data, pad_size=pad_size)  # pushes to device iternally
            x, mask, dim = data[0], data[2], data[3]
            yd, yo, yt, yp = data[1][0], data[1][1], data[1][2], data[1][3]
            if torch.sum(torch.isnan(x)) > 0:
                continue
        optimizer.zero_grad()

        x_hat, means, log_vars = model(x)
        loss, reprod_loss, kld = loss_fn(
            (yd, yo, yt, yp), x_hat, mask, dim, means, log_vars
        )

        losses.append(loss.item())
        reprod_losses.append(reprod_loss.item())
        kld_losses.append(kld.item())
        loss.backward()

        # Clip grads
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)

        optimizer.step()
        # scheduler.step()

    batch_size = x.shape[0]
    losses, reprod_losses, kld_losses = (
        torch.tensor(losses),
        torch.tensor(reprod_losses),
        torch.tensor(kld_losses),
    )
    return losses.mean(), losses.std(), reprod_losses.mean(), kld_losses.mean()


def evaluate_loss(model, loss_fn, loader, pad_size=128):
    """ """
    model.eval()
    losses, reprod_losses = [], []
    for batch_idx, data in enumerate(loader):
        with torch.no_grad():
            data = encode_on_gpu(data, pad_size=pad_size)
            x, mask, dim = data[0], data[2], data[3]
            yd, yo, yt, yp = data[1][0], data[1][1], data[1][2], data[1][3]
            x_hat, means, log_vars = model(x)
            loss, reprod_loss = loss_fn(
                (yd, yo, yt, yp), x_hat, mask, dim, means, log_vars
            )
            losses.append(loss.item())
            reprod_losses.append(reprod_loss.item())
    batch_size = x.shape[0]
    losses, reprod_losses = torch.tensor(losses), torch.tensor(reprod_losses)
    return losses.mean(), losses.std(), reprod_losses.mean()


def evaluate_loss2(model, loss_fn, loader, pad_size=128):
    """ """
    model.eval()
    losses, reprod_losses, kld_losses = [], [], []
    for batch_idx, data in enumerate(loader):
        with torch.no_grad():
            data = encode_on_gpu(data, pad_size=pad_size)
            x, mask, dim = data[0], data[2], data[3]
            yd, yo, yt, yp = data[1][0], data[1][1], data[1][2], data[1][3]
            x_hat, means, log_vars = model(x)
            loss, reprod_loss, kld = loss_fn(
                (yd, yo, yt, yp), x_hat, mask, dim, means, log_vars
            )
            losses.append(loss.item())
            reprod_losses.append(reprod_loss.item())
            kld_losses.append(kld)
    batch_size = x.shape[0]
    losses, reprod_losses, kld_losses = (
        torch.tensor(losses),
        torch.tensor(reprod_losses),
        torch.tensor(kld_losses),
    )
    return losses.mean(), losses.std(), reprod_losses.mean(), kld_losses.mean()


def get_lr(optimizer):
    """ """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


################################################################################
############################# trRosetta design #################################
################################################################################


def get_sse(topology, loops):
    """ """
    from itertools import chain, zip_longest

    def interleave(l1, l2):
        return [x for x in chain(*zip_longest(l1, l2)) if x is not None]

    ssei = interleave(loops.split("."), topology.split("."))
    sse = []
    for e in ssei:
        if e == "x":
            continue
        if len(e) > 2:
            sse.extend([e[2] * int(e[3:])])
        else:
            sse.extend(["L" * int(e)])
    return "".join(sse)


def get_ss_pairs(topology):
    """ """
    ss_pairs = []
    total_layers = list(set([tp[0] for tp in topology.split(".")]))

    for layer in total_layers:
        c1, c2 = 0, 0
        for c1, tp1 in enumerate(topology.split(".")):
            lyr1, ele1, sse1, len1 = tp1[0], int(tp1[1]), tp1[2], int(tp1[3:])
            if sse1 != "E":
                continue
            if lyr1 != layer:
                continue
            c1 += 1

            for c2, tp2 in enumerate(topology.split(".")):
                lyr2, ele2, sse2, len2 = tp2[0], int(tp2[1]), tp2[2], int(tp2[3:])
                if sse2 != "E":
                    continue
                if lyr2 != layer:
                    continue
                c2 += 1

                # check if element is consecutive
                if ele1 + 1 == ele2:
                    # Get direction
                    # dir1 = (ele1 + 1 % 2) == 0
                    # dir2 = (ele2 % 2) == 0
                    dir1 = (c1 % 2) == 0
                    dir2 = (c2 % 2) == 0
                    gdir = "A" if dir1 != dir2 else "P"

                    # print('{}-{}.{}.99'.format(c1, c2, gdir))
                    if c1 < c2:
                        ss_pairs.append("{}-{}.{}.99".format(c1, c2, gdir))
                    else:
                        ss_pairs.append("{}-{}.{}.99".format(c2, c1, gdir))
    return ss_pairs


def get_hh_pairs(topology):
    """ """
    hh_pairs = []
    total_layers = list(set([tp[0] for tp in topology.split(".")]))

    for layer in total_layers:
        c1, c2 = 0, 0
        for c1, tp1 in enumerate(topology.split(".")):
            lyr1, ele1, sse1, len1 = tp1[0], int(tp1[1]), tp1[2], int(tp1[3:])
            if sse1 != "H":
                continue
            if lyr1 != layer:
                continue
            c1 += 1

            for c2, tp2 in enumerate(topology.split(".")):
                lyr2, ele2, sse2, len2 = tp2[0], int(tp2[1]), tp2[2], int(tp2[3:])
                if sse2 != "H":
                    continue
                if lyr2 != layer:
                    continue
                c2 += 1

                # check if element is consecutive
                if ele1 + 1 == ele2:
                    # Get direction
                    dir1 = (c1 % 2) == 0
                    dir2 = (c2 % 2) == 0
                    gdir = "A" if dir1 != dir2 else "P"

                    # print('{}-{}.{}.99'.format(c1, c2, gdir))
                    if c1 < c2:
                        hh_pairs.append("{}-{}.{}".format(c1, c2, gdir))
                    else:
                        hh_pairs.append("{}-{}.{}".format(c2, c1, gdir))
    return hh_pairs


def get_hss_triplets(topology):
    """
    Gets the Helix - beta pair triplets.

    Examples:
    #topology = 'A1H1.B1E2.B2E3.B3E4.B4E5.B5E6.C1H7.C2H8.C3H9.D1H10'
    # 1,2-3 1,3-4, 1,4-5
    # 7,2-3 8,3-4, 9,4-5

    #topology = 'A1H1.B1H2.C1E3.C2E4.C3E5.D1H6.D2H7'
    # 2,3-4 2,4-5
    # 6,3-4 7,4-5

    topology = 'A1H1.B1H2.C1E3.C2E4.C3E5.C4E6.D1H7.D2H8'
    # 2,3-4 2,4-5 2,5-6
    # 7,3-4 7,4-5 8,4-5 8,5-6
    """
    hss_triplets = []
    total_layers = list(set([tp[0] for tp in topology.split(".")]))
    d = {}
    for l in total_layers:
        d[l] = []

    for c, tp in enumerate(topology.split(".")):
        lyr, ele, sse, lent = tp[0], int(tp[1]), tp[2], int(tp[3:])
        d[lyr].append((ele, sse, c + 1))
        # d[lyr][sse] = []

    hss_triplets = []
    for i in range(len(sorted(total_layers)) - 1):
        lyr1, lyr2 = sorted(total_layers)[i], sorted(total_layers)[i + 1]

        set1, set2 = sorted(d[lyr1]), sorted(d[lyr2])
        if set1[0][1] == set2[0][1]:
            # print('{}<->{} same SSEs, skipping'.format(lyr1, lyr2))
            continue  # same SSEs
        if set1[0][1] == "H":
            setH = set1
            setE = set2
        else:
            setH = set2
            setE = set1

        # if more helices than strands, we skip
        # because idk what this is then
        if len(setH) >= len(setE):
            print("more strands than helices, skipping...")
        else:
            # chunk the sheet into pairs
            chunk2 = [setE[x : x + 2] for x in range(0, len(setE), 1)]
            if len(chunk2[-2]) != len(chunk2[-1]):
                chunk2 = chunk2[:-1]
            # we need to assign the beta pairs to the helices
            # example 1:
            # 5 strands (E1, ..., E5), 2 helices (H1, H2)
            # H1,E1-E2 H1,E2-E3
            # H2,E3-E4 H2,E4-E5
            #
            # example 2:
            # 6 strands (E1, ..., E6), 2 helices (H1, H2)
            # H1,E1-E2 H1,E2-E3 H1,E3-E4
            # H2,E3-E4 H2,E4-E5 H2,E5-E6

            if (
                len(chunk2) % len(setH) == 0
            ):  # we can divide the chunks with the number of helices
                # print('divisible')
                repeatsH = np.repeat(setH, int(len(chunk2) / len(setH)), axis=0)
                for sH, sE in zip(repeatsH, chunk2):
                    hss_triplet = "{},{}-{}".format(sH[-1], sE[0][-1], sE[1][-1])
                    hss_triplets.append(hss_triplet)
            else:
                # print('non-divisible')
                # chunk2 = chunk2[:-1]
                n = int(np.floor(len(chunk2) / 2))
                chunk2 = chunk2[:n] + [chunk2[n]] + chunk2[n:]
                repeatsH = np.repeat(
                    setH, int(np.ceil(len(chunk2) / len(setH))), axis=0
                )
                for sH, sE in zip(repeatsH, chunk2):
                    hss_triplet = "{},{}-{}".format(sH[-1], sE[0][-1], sE[1][-1])
                    hss_triplets.append(hss_triplet)
    return hss_triplets


def make_blueprint(seq, sse, ss_pairs, hh_pairs, outfile):
    """ """
    with open(outfile, "w") as f:
        if ss_pairs != []:
            # print('SSPAIR {}'.format(';'.join(ss_pairs)))
            f.write("SSPAIR {}\n".format(";".join(ss_pairs)))
        if hh_pairs != []:
            # print('HHPAIR {}'.format(';'.join(HH_pairs)))
            f.write("HHPAIR {}\n".format(";".join(HH_pairs)))

        f.write("\n")
        for i, (sq, ss) in enumerate(zip(seq, sse)):
            # print(i + 1, sq, ss)
            f.write("{} {} {}\n".format(i + 1, sq, ss))


def N_to_AA(x):
    # [[0,1,2,3]] -> ["ARND"]
    x = np.array(x)
    if x.ndim == 1:
        x = x[None]
    return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]


def set_random_dihedral(pose):
    nres = pose.total_residue()
    for i in range(1, nres):
        phi, psi = random_dihedral()
        pose.set_phi(i, phi)
        pose.set_psi(i, psi)
        pose.set_omega(i, 180)

    return pose


def remove_clash(scorefxn, mover, pose):
    for _ in range(0, 5):
        if float(scorefxn(pose)) < 10:
            break
        mover.apply(pose)


def add_rst(pose, rst, sep1, sep2, params, nogly=False):
    pcut = params["PCUT"]
    seq = params["seq"]

    array = []

    if nogly == True:
        array += [
            line
            for a, b, p, line in rst["dist"]
            if abs(a - b) >= sep1
            and abs(a - b) < sep2
            and seq[a] != "G"
            and seq[b] != "G"
            and p >= pcut
        ]
        if params["USE_ORIENT"] == True:
            array += [
                line
                for a, b, p, line in rst["omega"]
                if abs(a - b) >= sep1
                and abs(a - b) < sep2
                and seq[a] != "G"
                and seq[b] != "G"
                and p >= pcut + 0.5
            ]  # 0.5
            array += [
                line
                for a, b, p, line in rst["theta"]
                if abs(a - b) >= sep1
                and abs(a - b) < sep2
                and seq[a] != "G"
                and seq[b] != "G"
                and p >= pcut + 0.5
            ]  # 0.5
            array += [
                line
                for a, b, p, line in rst["phi"]
                if abs(a - b) >= sep1
                and abs(a - b) < sep2
                and seq[a] != "G"
                and seq[b] != "G"
                and p >= pcut + 0.6
            ]  # 0.6
    else:
        array += [
            line
            for a, b, p, line in rst["dist"]
            if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut
        ]
        if params["USE_ORIENT"] == True:
            array += [
                line
                for a, b, p, line in rst["omega"]
                if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut + 0.5
            ]
            array += [
                line
                for a, b, p, line in rst["theta"]
                if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut + 0.5
            ]
            array += [
                line
                for a, b, p, line in rst["phi"]
                if abs(a - b) >= sep1 and abs(a - b) < sep2 and p >= pcut + 0.6
            ]  # 0.6

    if len(array) < 1:
        return

    random.shuffle(array)

    # save to file
    tmpname = params["TDIR"] + "/minimize.cst"
    with open(tmpname, "w") as f:
        for line in array:
            f.write(line + "\n")
        f.close()

    # add to pose
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_file(tmpname)
    constraints.add_constraints(True)
    constraints.apply(pose)

    os.remove(tmpname)


def random_dihedral():
    phi = 0
    psi = 0
    r = random.random()
    if r <= 0.135:
        phi = -140
        psi = 153
    elif r > 0.135 and r <= 0.29:
        phi = -72
        psi = 145
    elif r > 0.29 and r <= 0.363:
        phi = -122
        psi = 117
    elif r > 0.363 and r <= 0.485:
        phi = -82
        psi = -14
    elif r > 0.485 and r <= 0.982:
        phi = -61
        psi = -41
    else:
        phi = 57
        psi = 39
    return (phi, psi)


def gen_rst(npz, tmpdir, params):
    dist, omega, theta, phi = npz["dist"], npz["omega"], npz["theta"], npz["phi"]

    # dictionary to store Rosetta restraints
    # rst = {'dist' : [], 'omega' : [], 'theta' : [], 'phi' : [], 'rep' : []}
    rst = {"dist": [], "omega": [], "theta": [], "phi": []}

    ########################################################
    # assign parameters
    ########################################################
    PCUT = 0.05  # params['PCUT']
    PCUT1 = params["PCUT1"]
    EBASE = params["EBASE"]
    EREP = params["EREP"]
    DREP = params["DREP"]
    PREP = params["PREP"]
    SIGD = params["SIGD"]
    SIGM = params["SIGM"]
    MEFF = params["MEFF"]
    DCUT = params["DCUT"]
    ALPHA = params["ALPHA"]

    DSTEP = params["DSTEP"]
    ASTEP = np.deg2rad(params["ASTEP"])

    seq = params["seq"]

    ########################################################
    # dist: 0..20A
    ########################################################
    nres = dist.shape[0]
    bins = np.array([4.25 + DSTEP * i for i in range(32)])
    prob = np.sum(dist[:, :, 5:], axis=-1)
    bkgr = np.array((bins / DCUT) ** ALPHA)
    attr = (
        -np.log(
            (dist[:, :, 5:] + MEFF) / (dist[:, :, -1][:, :, None] * bkgr[None, None, :])
        )
        + EBASE
    )
    repul = (
        np.maximum(attr[:, :, 0], np.zeros((nres, nres)))[:, :, None]
        + np.array(EREP)[None, None, :]
    )
    dist = np.concatenate([repul, attr], axis=-1)
    bins = np.concatenate([DREP, bins])
    i, j = np.where(prob > PCUT)
    prob = prob[i, j]
    nbins = 35
    step = 0.5
    for a, b, p in zip(i, j, prob):
        if b > a:
            name = tmpdir.name + "/%d.%d.txt" % (a + 1, b + 1)
            with open(name, "w") as f:
                f.write("x_axis" + "\t%.3f" * nbins % tuple(bins) + "\n")
                f.write("y_axis" + "\t%.3f" * nbins % tuple(dist[a, b]) + "\n")
                f.close()
            rst_line = "AtomPair %s %d %s %d SPLINE TAG %s 1.0 %.3f %.5f" % (
                "CB",
                a + 1,
                "CB",
                b + 1,
                name,
                1.0,
                step,
            )
            rst["dist"].append([a, b, p, rst_line])
    print("dist restraints:  %d" % (len(rst["dist"])))

    ########################################################
    # omega: -pi..pi
    ########################################################
    nbins = omega.shape[2] - 1 + 4
    bins = np.linspace(-np.pi - 1.5 * ASTEP, np.pi + 1.5 * ASTEP, nbins)
    prob = np.sum(omega[:, :, 1:], axis=-1)
    i, j = np.where(prob > PCUT)
    prob = prob[i, j]
    omega = -np.log((omega + MEFF) / (omega[:, :, -1] + MEFF)[:, :, None])
    omega = np.concatenate(
        [omega[:, :, -2:], omega[:, :, 1:], omega[:, :, 1:3]], axis=-1
    )
    for a, b, p in zip(i, j, prob):
        if b > a:
            name = tmpdir.name + "/%d.%d_omega.txt" % (a + 1, b + 1)
            with open(name, "w") as f:
                f.write("x_axis" + "\t%.5f" * nbins % tuple(bins) + "\n")
                f.write("y_axis" + "\t%.5f" * nbins % tuple(omega[a, b]) + "\n")
                f.close()
            rst_line = (
                "Dihedral CA %d CB %d CB %d CA %d SPLINE TAG %s 1.0 %.3f %.5f"
                % (a + 1, a + 1, b + 1, b + 1, name, 1.0, ASTEP)
            )
            rst["omega"].append([a, b, p, rst_line])
    print("omega restraints: %d" % (len(rst["omega"])))

    ########################################################
    # theta: -pi..pi
    ########################################################
    prob = np.sum(theta[:, :, 1:], axis=-1)
    i, j = np.where(prob > PCUT)
    prob = prob[i, j]
    theta = -np.log((theta + MEFF) / (theta[:, :, -1] + MEFF)[:, :, None])
    theta = np.concatenate(
        [theta[:, :, -2:], theta[:, :, 1:], theta[:, :, 1:3]], axis=-1
    )
    for a, b, p in zip(i, j, prob):
        if b != a:
            name = tmpdir.name + "/%d.%d_theta.txt" % (a + 1, b + 1)
            with open(name, "w") as f:
                f.write("x_axis" + "\t%.3f" * nbins % tuple(bins) + "\n")
                f.write("y_axis" + "\t%.3f" * nbins % tuple(theta[a, b]) + "\n")
                f.close()
            rst_line = "Dihedral N %d CA %d CB %d CB %d SPLINE TAG %s 1.0 %.3f %.5f" % (
                a + 1,
                a + 1,
                a + 1,
                b + 1,
                name,
                1.0,
                ASTEP,
            )
            rst["theta"].append([a, b, p, rst_line])
            # if a==0 and b==9:
            #    with open(name,'r') as f:
            #        print(f.read())
    print("theta restraints: %d" % (len(rst["theta"])))

    ########################################################
    # phi: 0..pi
    ########################################################
    nbins = phi.shape[2] - 1 + 4
    bins = np.linspace(-1.5 * ASTEP, np.pi + 1.5 * ASTEP, nbins)
    prob = np.sum(phi[:, :, 1:], axis=-1)
    i, j = np.where(prob > PCUT)
    prob = prob[i, j]
    phi = -np.log((phi + MEFF) / (phi[:, :, -1] + MEFF)[:, :, None])
    phi = np.concatenate(
        [
            np.flip(phi[:, :, 1:3], axis=-1),
            phi[:, :, 1:],
            np.flip(phi[:, :, -2:], axis=-1),
        ],
        axis=-1,
    )
    for a, b, p in zip(i, j, prob):
        if b != a:
            name = tmpdir.name + "/%d.%d_phi.txt" % (a + 1, b + 1)
            with open(name, "w") as f:
                f.write("x_axis" + "\t%.3f" * nbins % tuple(bins) + "\n")
                f.write("y_axis" + "\t%.3f" * nbins % tuple(phi[a, b]) + "\n")
                f.close()
            rst_line = "Angle CA %d CB %d CB %d SPLINE TAG %s 1.0 %.3f %.5f" % (
                a + 1,
                a + 1,
                b + 1,
                name,
                1.0,
                ASTEP,
            )
            rst["phi"].append([a, b, p, rst_line])
            # if a==0 and b==9:
            #    with open(name,'r') as f:
            #        print(f.read())

    print("phi restraints:   %d" % (len(rst["phi"])))
    return rst


def final_design(sse, sspairs, hhpairs, hsstriplets):
    """ """
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

    if sspairs != "" and hhpairs == "":
        print("only sspairs")
        protocol = """
        <ROSETTASCRIPTS>
            <SCOREFXNS>
                <ScoreFunction name="sfxn_fa" weights="ref2015">
                    <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                    <Reweight scoretype="angle_constraint" weight="1.0"/>
                    <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                </ScoreFunction>
                <ScoreFunction name="sfxn_fa_cart" weights="ref2015_cart">
                    <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                    <Reweight scoretype="angle_constraint" weight="1.0"/>
                    <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                </ScoreFunction>
                <ScoreFunction name="bb_only" weights="empty.wts">
                    <Reweight scoretype="fa_rep" weight="0.1"/>
                    <Reweight scoretype="fa_atr" weight="0.2"/>
                    <Reweight scoretype="hbond_sr_bb" weight="2.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                    <Reweight scoretype="rama_prepro" weight="0.45"/>
                    <Reweight scoretype="omega" weight="0.4"/>
                    <Reweight scoretype="p_aa_pp" weight="0.6"/>
                </ScoreFunction>
            </SCOREFXNS>
            <RESIDUE_SELECTORS>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="sse_cstdes" overlap="0" pose_secstruct="{0}" ss="HE" use_dssp="0"/>
                <Layer name="surface" select_boundary="0" select_core="0" select_surface="1" use_sidechain_neighbors="1"/>
                <Layer name="boundary" select_boundary="1" select_core="0" select_surface="0" use_sidechain_neighbors="1"/>
                <Layer name="core" select_boundary="0" select_core="1" select_surface="0" use_sidechain_neighbors="1"/>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="sheet" overlap="0" pose_secstruct="{0}" ss="E" use_dssp="0"/>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="entire_helix" overlap="0" pose_secstruct="{0}" ss="H" use_dssp="0"/>
                <SecondaryStructure include_terminal_loops="1" minE="1" minH="1" name="entire_loop" overlap="0" pose_secstruct="{0}" ss="L" use_dssp="0"/>
                <And name="helix_cap" selectors="entire_loop">
                    <PrimarySequenceNeighborhood lower="1" selector="entire_helix" upper="0"/>
                </And>
                <And name="helix_start" selectors="entire_helix">
                    <PrimarySequenceNeighborhood lower="0" selector="helix_cap" upper="1"/>
                </And>
                <And name="helix" selectors="entire_helix">
                    <Not selector="helix_start"/>
                </And>
                <And name="loop" selectors="entire_loop">
                    <Not selector="helix_cap"/>
                </And>
                <Layer name="surface_cstdes" select_boundary="0" select_core="0" select_surface="1" use_sidechain_neighbors="1"/>
                <Layer name="boundary_cstdes" select_boundary="1" select_core="0" select_surface="0" use_sidechain_neighbors="1"/>
                <Layer name="core_cstdes" select_boundary="0" select_core="1" select_surface="0" use_sidechain_neighbors="1"/>
            </RESIDUE_SELECTORS>
            <TASKOPERATIONS>
                <DesignRestrictions name="layer_design">
                   <Action aas="DEHKPQR" selector_logic="surface AND helix_start"/>
                   <Action aas="EHKQR" selector_logic="surface AND helix"/>
                   <Action aas="EHKNQRST" selector_logic="surface AND sheet"/>
                   <Action aas="DEGHKNPQRST" selector_logic="surface AND loop"/>
                   <Action aas="ADEHIKLMNPQRSTVWY" selector_logic="boundary AND helix_start"/>
                   <Action aas="ADEHIKLMNQRSTVWY" selector_logic="boundary AND helix"/>
                   <Action aas="DEFHIKLMNQRSTVWY" selector_logic="boundary AND sheet"/>
                   <Action aas="ADEFGHIKLMNPQRSTVWY" selector_logic="boundary AND loop"/>
                   <Action aas="AFILMPVWY" selector_logic="core AND helix_start"/>
                   <Action aas="AFILMVWY" selector_logic="core AND helix"/>
                   <Action aas="FILMVWY" selector_logic="core AND sheet"/>
                   <Action aas="AFGILMPVWY" selector_logic="core AND loop"/>
                   <Action aas="DNST" selector_logic="helix_cap"/>
                </DesignRestrictions>
            </TASKOPERATIONS>
            <FILTERS>
            </FILTERS>
            <MOVERS>
                <SetSecStructEnergies name="ssse_cstdes" natbias_ss="5.0" scorefxn="sfxn_fa" secstruct="{0}" ss_pair="{1}" use_dssp="0"/>
                <SetSecStructEnergies name="ssse_cstdes_cart" natbias_ss="5.0" scorefxn="sfxn_fa_cart" secstruct="{0}" ss_pair="{1}" use_dssp="0"/>
                <FastDesign dualspace="true" name="design" ramp_down_constraints="false" relaxscript="MonomerDesign2019" repeats="5" scorefxn="sfxn_fa_cart" task_operations="layer_design"/>
                <FastRelax  cartesian="true" name="relax" repeats="5" scorefxn="sfxn_fa_cart"/>
            </MOVERS>
            <PROTOCOLS>
                 <!--Add mover="ssse_cstdes"/-->
                 <Add mover="ssse_cstdes_cart"/>
                 <!--Add mover="design"/-->
                 <Add mover="relax"/>
            </PROTOCOLS>
        </ROSETTASCRIPTS>
        """.format(
            sse, sspairs
        )
    elif sspairs == "" and hhpairs != "":
        print("only hhpairs")
        protocol = """
        <ROSETTASCRIPTS>
            <SCOREFXNS>
                <ScoreFunction name="sfxn_fa" weights="ref2015">
                    <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                    <Reweight scoretype="angle_constraint" weight="1.0"/>
                    <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                </ScoreFunction>
                <ScoreFunction name="sfxn_fa_cart" weights="ref2015_cart">
                    <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                    <Reweight scoretype="angle_constraint" weight="1.0"/>
                    <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                </ScoreFunction>
                <ScoreFunction name="bb_only" weights="empty.wts">
                    <Reweight scoretype="fa_rep" weight="0.1"/>
                    <Reweight scoretype="fa_atr" weight="0.2"/>
                    <Reweight scoretype="hbond_sr_bb" weight="2.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                    <Reweight scoretype="rama_prepro" weight="0.45"/>
                    <Reweight scoretype="omega" weight="0.4"/>
                    <Reweight scoretype="p_aa_pp" weight="0.6"/>
                </ScoreFunction>
            </SCOREFXNS>
            <RESIDUE_SELECTORS>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="sse_cstdes" overlap="0" pose_secstruct="{0}" ss="HE" use_dssp="0"/>
                <Layer name="surface" select_boundary="0" select_core="0" select_surface="1" use_sidechain_neighbors="1"/>
                <Layer name="boundary" select_boundary="1" select_core="0" select_surface="0" use_sidechain_neighbors="1"/>
                <Layer name="core" select_boundary="0" select_core="1" select_surface="0" use_sidechain_neighbors="1"/>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="sheet" overlap="0" pose_secstruct="{0}" ss="E" use_dssp="0"/>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="entire_helix" overlap="0" pose_secstruct="{0}" ss="H" use_dssp="0"/>
                <SecondaryStructure include_terminal_loops="1" minE="1" minH="1" name="entire_loop" overlap="0" pose_secstruct="{0}" ss="L" use_dssp="0"/>
                <And name="helix_cap" selectors="entire_loop">
                    <PrimarySequenceNeighborhood lower="1" selector="entire_helix" upper="0"/>
                </And>
                <And name="helix_start" selectors="entire_helix">
                    <PrimarySequenceNeighborhood lower="0" selector="helix_cap" upper="1"/>
                </And>
                <And name="helix" selectors="entire_helix">
                    <Not selector="helix_start"/>
                </And>
                <And name="loop" selectors="entire_loop">
                    <Not selector="helix_cap"/>
                </And>
                <Layer name="surface_cstdes" select_boundary="0" select_core="0" select_surface="1" use_sidechain_neighbors="1"/>
                <Layer name="boundary_cstdes" select_boundary="1" select_core="0" select_surface="0" use_sidechain_neighbors="1"/>
                <Layer name="core_cstdes" select_boundary="0" select_core="1" select_surface="0" use_sidechain_neighbors="1"/>
            </RESIDUE_SELECTORS>
            <TASKOPERATIONS>
                <DesignRestrictions name="layer_design">
                   <Action aas="DEHKPQR" selector_logic="surface AND helix_start"/>
                   <Action aas="EHKQR" selector_logic="surface AND helix"/>
                   <Action aas="EHKNQRST" selector_logic="surface AND sheet"/>
                   <Action aas="DEGHKNPQRST" selector_logic="surface AND loop"/>
                   <Action aas="ADEHIKLMNPQRSTVWY" selector_logic="boundary AND helix_start"/>
                   <Action aas="ADEHIKLMNQRSTVWY" selector_logic="boundary AND helix"/>
                   <Action aas="DEFHIKLMNQRSTVWY" selector_logic="boundary AND sheet"/>
                   <Action aas="ADEFGHIKLMNPQRSTVWY" selector_logic="boundary AND loop"/>
                   <Action aas="AFILMPVWY" selector_logic="core AND helix_start"/>
                   <Action aas="AFILMVWY" selector_logic="core AND helix"/>
                   <Action aas="FILMVWY" selector_logic="core AND sheet"/>
                   <Action aas="AFGILMPVWY" selector_logic="core AND loop"/>
                   <Action aas="DNST" selector_logic="helix_cap"/>
                </DesignRestrictions>
            </TASKOPERATIONS>
            <FILTERS>
            </FILTERS>
            <MOVERS>
                <SetSecStructEnergies name="ssse_cstdes" natbias_ss="5.0" scorefxn="sfxn_fa" secstruct="{0}" hh_pair="{1}" use_dssp="0"/>
                <SetSecStructEnergies name="ssse_cstdes_cart" natbias_ss="5.0" scorefxn="sfxn_fa_cart" secstruct="{0}" hh_pair="{1}" use_dssp="0"/>
                <FastDesign dualspace="true" name="design" ramp_down_constraints="false" relaxscript="MonomerDesign2019" repeats="5" scorefxn="sfxn_fa_cart" task_operations="layer_design"/>
                <FastRelax  cartesian="true" name="relax" repeats="5" scorefxn="sfxn_fa_cart"/>
            </MOVERS>
            <PROTOCOLS>
                 <Add mover="ssse_cstdes"/>
                 <Add mover="ssse_cstdes_cart"/>
                 <!--Add mover="design"/-->
                 <Add mover="relax"/>
            </PROTOCOLS>
        </ROSETTASCRIPTS>
        """.format(
            sse, hhpairs
        )
    else:
        print("sspairs, hhpairs, hsstriplets")
        protocol = """
        <ROSETTASCRIPTS>
            <SCOREFXNS>
                <ScoreFunction name="sfxn_fa" weights="ref2015">
                    <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                    <Reweight scoretype="angle_constraint" weight="1.0"/>
                    <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                </ScoreFunction>
                <ScoreFunction name="sfxn_fa_cart" weights="ref2015_cart">
                    <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                    <Reweight scoretype="angle_constraint" weight="1.0"/>
                    <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                </ScoreFunction>
                <ScoreFunction name="bb_only" weights="empty.wts">
                    <Reweight scoretype="fa_rep" weight="0.1"/>
                    <Reweight scoretype="fa_atr" weight="0.2"/>
                    <Reweight scoretype="hbond_sr_bb" weight="2.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                    <Reweight scoretype="rama_prepro" weight="0.45"/>
                    <Reweight scoretype="omega" weight="0.4"/>
                    <Reweight scoretype="p_aa_pp" weight="0.6"/>
                </ScoreFunction>
            </SCOREFXNS>
            <RESIDUE_SELECTORS>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="sse_cstdes" overlap="0" pose_secstruct="{0}" ss="HE" use_dssp="0"/>
                <Layer name="surface" select_boundary="0" select_core="0" select_surface="1" use_sidechain_neighbors="1"/>
                <Layer name="boundary" select_boundary="1" select_core="0" select_surface="0" use_sidechain_neighbors="1"/>
                <Layer name="core" select_boundary="0" select_core="1" select_surface="0" use_sidechain_neighbors="1"/>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="sheet" overlap="0" pose_secstruct="{0}" ss="E" use_dssp="0"/>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="entire_helix" overlap="0" pose_secstruct="{0}" ss="H" use_dssp="0"/>
                <SecondaryStructure include_terminal_loops="1" minE="1" minH="1" name="entire_loop" overlap="0" pose_secstruct="{0}" ss="L" use_dssp="0"/>
                <And name="helix_cap" selectors="entire_loop">
                    <PrimarySequenceNeighborhood lower="1" selector="entire_helix" upper="0"/>
                </And>
                <And name="helix_start" selectors="entire_helix">
                    <PrimarySequenceNeighborhood lower="0" selector="helix_cap" upper="1"/>
                </And>
                <And name="helix" selectors="entire_helix">
                    <Not selector="helix_start"/>
                </And>
                <And name="loop" selectors="entire_loop">
                    <Not selector="helix_cap"/>
                </And>
                <Layer name="surface_cstdes" select_boundary="0" select_core="0" select_surface="1" use_sidechain_neighbors="1"/>
                <Layer name="boundary_cstdes" select_boundary="1" select_core="0" select_surface="0" use_sidechain_neighbors="1"/>
                <Layer name="core_cstdes" select_boundary="0" select_core="1" select_surface="0" use_sidechain_neighbors="1"/>
            </RESIDUE_SELECTORS>
            <TASKOPERATIONS>
                <DesignRestrictions name="layer_design">
                   <Action aas="DEHKPQR" selector_logic="surface AND helix_start"/>
                   <Action aas="EHKQR" selector_logic="surface AND helix"/>
                   <Action aas="EHKNQRST" selector_logic="surface AND sheet"/>
                   <Action aas="DEGHKNPQRST" selector_logic="surface AND loop"/>
                   <Action aas="ADEHIKLMNPQRSTVWY" selector_logic="boundary AND helix_start"/>
                   <Action aas="ADEHIKLMNQRSTVWY" selector_logic="boundary AND helix"/>
                   <Action aas="DEFHIKLMNQRSTVWY" selector_logic="boundary AND sheet"/>
                   <Action aas="ADEFGHIKLMNPQRSTVWY" selector_logic="boundary AND loop"/>
                   <Action aas="AFILMPVWY" selector_logic="core AND helix_start"/>
                   <Action aas="AFILMVWY" selector_logic="core AND helix"/>
                   <Action aas="FILMVWY" selector_logic="core AND sheet"/>
                   <Action aas="AFGILMPVWY" selector_logic="core AND loop"/>
                   <Action aas="DNST" selector_logic="helix_cap"/>
                </DesignRestrictions>
            </TASKOPERATIONS>
            <FILTERS>
            </FILTERS>
            <MOVERS>
                <SetSecStructEnergies name="ssse_cstdes" natbias_ss="5.0" scorefxn="sfxn_fa" secstruct="{0}" hh_pair="{2}" ss_pair="{1}" hss_triplets="{3}" use_dssp="0"/>
                <SetSecStructEnergies name="ssse_cstdes_cart" natbias_ss="5.0" scorefxn="sfxn_fa_cart" secstruct="{0}" hh_pair="{2}" ss_pair="{1}" hss_triplets="{3}" use_dssp="0"/>
                <FastDesign dualspace="true" name="design" ramp_down_constraints="false" relaxscript="MonomerDesign2019" repeats="5" scorefxn="sfxn_fa_cart" task_operations="layer_design"/>
                <FastRelax  cartesian="true" name="relax" repeats="5" scorefxn="sfxn_fa_cart"/>
            </MOVERS>
            <PROTOCOLS>
                 <Add mover="ssse_cstdes"/>
                 <Add mover="ssse_cstdes_cart"/>
                 <!--Add mover="design"/-->
                 <Add mover="relax"/>
            </PROTOCOLS>
        </ROSETTASCRIPTS>
        """.format(
            sse, sspairs, hhpairs, hsstriplets
        )
    xml = XmlObjects.create_from_string(protocol).get_mover("ParsedProtocol")
    return xml, protocol


def design_with_pssm(sse, sspairs, hhpairs, hsstriplets, pssm):
    """ """
    from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

    if sspairs == "" and hhpairs == "" and sse == "":
        print("No SSE energy bonus. No SSE, using DSSP.")
        protocol = """
        <ROSETTASCRIPTS>
            <SCOREFXNS>
                <ScoreFunction name="sfxn_fa" weights="ref2015">
                    <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                    <Reweight scoretype="angle_constraint" weight="1.0"/>
                    <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                </ScoreFunction>
                <ScoreFunction name="sfxn_fa_cart" weights="ref2015_cart">
                    <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                    <Reweight scoretype="angle_constraint" weight="1.0"/>
                    <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                </ScoreFunction>
                <ScoreFunction name="bb_only" weights="empty.wts">
                    <Reweight scoretype="fa_rep" weight="0.1"/>
                    <Reweight scoretype="fa_atr" weight="0.2"/>
                    <Reweight scoretype="hbond_sr_bb" weight="2.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                    <Reweight scoretype="rama_prepro" weight="0.45"/>
                    <Reweight scoretype="omega" weight="0.4"/>
                    <Reweight scoretype="p_aa_pp" weight="0.6"/>
                </ScoreFunction>
            </SCOREFXNS>
            <RESIDUE_SELECTORS>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="sse_cstdes" overlap="0" ss="HE" use_dssp="1"/>
                <Layer name="surface" select_boundary="0" select_core="0" select_surface="1" use_sidechain_neighbors="1"/>
                <Layer name="boundary" select_boundary="1" select_core="0" select_surface="0" use_sidechain_neighbors="1"/>
                <Layer name="core" select_boundary="0" select_core="1" select_surface="0" use_sidechain_neighbors="1"/>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="sheet" overlap="0" ss="E" use_dssp="1"/>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="entire_helix" overlap="0" ss="H" use_dssp="1"/>
                <SecondaryStructure include_terminal_loops="1" minE="1" minH="1" name="entire_loop" overlap="0" ss="L" use_dssp="1"/>
                <And name="helix_cap" selectors="entire_loop">
                    <PrimarySequenceNeighborhood lower="1" selector="entire_helix" upper="0"/>
                </And>
                <And name="helix_start" selectors="entire_helix">
                    <PrimarySequenceNeighborhood lower="0" selector="helix_cap" upper="1"/>
                </And>
                <And name="helix" selectors="entire_helix">
                    <Not selector="helix_start"/>
                </And>
                <And name="loop" selectors="entire_loop">
                    <Not selector="helix_cap"/>
                </And>
                <Layer name="surface_cstdes" select_boundary="0" select_core="0" select_surface="1" use_sidechain_neighbors="1"/>
                <Layer name="boundary_cstdes" select_boundary="1" select_core="0" select_surface="0" use_sidechain_neighbors="1"/>
                <Layer name="core_cstdes" select_boundary="0" select_core="1" select_surface="0" use_sidechain_neighbors="1"/>
            </RESIDUE_SELECTORS>
            <TASKOPERATIONS>
                <DesignRestrictions name="layer_design">
                   <Action aas="DEHKPQR" selector_logic="surface AND helix_start"/>
                   <Action aas="EHKQR" selector_logic="surface AND helix"/>
                   <Action aas="EHKNQRST" selector_logic="surface AND sheet"/>
                   <Action aas="DEGHKNPQRST" selector_logic="surface AND loop"/>
                   <Action aas="ADEHIKLNPQRSTVWY" selector_logic="boundary AND helix_start"/> # M
                   <Action aas="ADEHIKLNQRSTVWY" selector_logic="boundary AND helix"/> # M
                   <Action aas="DEFHIKLNQRSTVWY" selector_logic="boundary AND sheet"/> # M
                   <Action aas="ADEFGHIKLNPQRSTVWY" selector_logic="boundary AND loop"/> # M
                   <Action aas="AFILPVWY" selector_logic="core AND helix_start"/> # M
                   <Action aas="AFILVWY" selector_logic="core AND helix"/> # M
                   <Action aas="FILVWY" selector_logic="core AND sheet"/> # M
                   <Action aas="AFGILPVWY" selector_logic="core AND loop"/> # M
                   <Action aas="DNST" selector_logic="helix_cap"/>
                </DesignRestrictions>
            </TASKOPERATIONS>
            <FILTERS>
            </FILTERS>
            <MOVERS>
                <!--SetSecStructEnergies name="ssse_cstdes" natbias_ss="5.0" scorefxn="sfxn_fa" ss_pair="{1}" use_dssp="1"/-->
                <SetSecStructEnergies name="ssse_cstdes_cart" natbias_ss="5.0" scorefxn="sfxn_fa_cart" use_dssp="1"/>
                <FavorSequenceProfile name="pssm_cst" weight="0.4" scorefxns="sfxn_fa_cart" pssm="{2}" scaling="prob" matrix="BLOSUM62" />
                <FastDesign dualspace="true" name="design" ramp_down_constraints="false" relaxscript="MonomerDesign2019" repeats="5" scorefxn="sfxn_fa_cart" task_operations="layer_design"/>
                <FastRelax  cartesian="true" name="relax" repeats="5" scorefxn="sfxn_fa_cart"/>
            </MOVERS>
            <PROTOCOLS>
                 <!--Add mover="ssse_cstdes"/-->
                 <Add mover="ssse_cstdes_cart"/>
                 <Add mover="pssm_cst"/>
                 <Add mover="design"/>
                 <!--Add mover="relax"/-->
            </PROTOCOLS>
        </ROSETTASCRIPTS>
        """.format(
            sse, sspairs, pssm
        )
    elif sspairs != "" and hhpairs == "":
        print("only sspairs")
        protocol = """
        <ROSETTASCRIPTS>
            <SCOREFXNS>
                <ScoreFunction name="sfxn_fa" weights="ref2015">
                    <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                    <Reweight scoretype="angle_constraint" weight="1.0"/>
                    <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                </ScoreFunction>
                <ScoreFunction name="sfxn_fa_cart" weights="ref2015_cart">
                    <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                    <Reweight scoretype="angle_constraint" weight="1.0"/>
                    <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                </ScoreFunction>
                <ScoreFunction name="bb_only" weights="empty.wts">
                    <Reweight scoretype="fa_rep" weight="0.1"/>
                    <Reweight scoretype="fa_atr" weight="0.2"/>
                    <Reweight scoretype="hbond_sr_bb" weight="2.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                    <Reweight scoretype="rama_prepro" weight="0.45"/>
                    <Reweight scoretype="omega" weight="0.4"/>
                    <Reweight scoretype="p_aa_pp" weight="0.6"/>
                </ScoreFunction>
            </SCOREFXNS>
            <RESIDUE_SELECTORS>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="sse_cstdes" overlap="0" pose_secstruct="{0}" ss="HE" use_dssp="0"/>
                <Layer name="surface" select_boundary="0" select_core="0" select_surface="1" use_sidechain_neighbors="1"/>
                <Layer name="boundary" select_boundary="1" select_core="0" select_surface="0" use_sidechain_neighbors="1"/>
                <Layer name="core" select_boundary="0" select_core="1" select_surface="0" use_sidechain_neighbors="1"/>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="sheet" overlap="0" pose_secstruct="{0}" ss="E" use_dssp="0"/>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="entire_helix" overlap="0" pose_secstruct="{0}" ss="H" use_dssp="0"/>
                <SecondaryStructure include_terminal_loops="1" minE="1" minH="1" name="entire_loop" overlap="0" pose_secstruct="{0}" ss="L" use_dssp="0"/>
                <And name="helix_cap" selectors="entire_loop">
                    <PrimarySequenceNeighborhood lower="1" selector="entire_helix" upper="0"/>
                </And>
                <And name="helix_start" selectors="entire_helix">
                    <PrimarySequenceNeighborhood lower="0" selector="helix_cap" upper="1"/>
                </And>
                <And name="helix" selectors="entire_helix">
                    <Not selector="helix_start"/>
                </And>
                <And name="loop" selectors="entire_loop">
                    <Not selector="helix_cap"/>
                </And>
                <Layer name="surface_cstdes" select_boundary="0" select_core="0" select_surface="1" use_sidechain_neighbors="1"/>
                <Layer name="boundary_cstdes" select_boundary="1" select_core="0" select_surface="0" use_sidechain_neighbors="1"/>
                <Layer name="core_cstdes" select_boundary="0" select_core="1" select_surface="0" use_sidechain_neighbors="1"/>
            </RESIDUE_SELECTORS>
            <TASKOPERATIONS>
                <DesignRestrictions name="layer_design">
                   <Action aas="DEHKPQR" selector_logic="surface AND helix_start"/>
                   <Action aas="EHKQR" selector_logic="surface AND helix"/>
                   <Action aas="EHKNQRST" selector_logic="surface AND sheet"/>
                   <Action aas="DEGHKNPQRST" selector_logic="surface AND loop"/>
                   <Action aas="ADEHIKLNPQRSTVWY" selector_logic="boundary AND helix_start"/> # M
                   <Action aas="ADEHIKLNQRSTVWY" selector_logic="boundary AND helix"/> # M
                   <Action aas="DEFHIKLNQRSTVWY" selector_logic="boundary AND sheet"/> # M
                   <Action aas="ADEFGHIKLNPQRSTVWY" selector_logic="boundary AND loop"/> # M
                   <Action aas="AFILPVWY" selector_logic="core AND helix_start"/> # M
                   <Action aas="AFILVWY" selector_logic="core AND helix"/> # M
                   <Action aas="FILVWY" selector_logic="core AND sheet"/> # M
                   <Action aas="AFGILPVWY" selector_logic="core AND loop"/> # M
                   <Action aas="DNST" selector_logic="helix_cap"/>
                </DesignRestrictions>
            </TASKOPERATIONS>
            <FILTERS>
            </FILTERS>
            <MOVERS>
                <SetSecStructEnergies name="ssse_cstdes" natbias_ss="5.0" scorefxn="sfxn_fa" secstruct="{0}" ss_pair="{1}" use_dssp="0"/>
                <SetSecStructEnergies name="ssse_cstdes_cart" natbias_ss="5.0" scorefxn="sfxn_fa_cart" secstruct="{0}" ss_pair="{1}" use_dssp="0"/>
                <FavorSequenceProfile name="pssm_cst" weight="0.4" scorefxns="sfxn_fa_cart" pssm="{2}" scaling="prob" matrix="BLOSUM62" />
                <FastDesign dualspace="true" name="design" ramp_down_constraints="false" relaxscript="MonomerDesign2019" repeats="5" scorefxn="sfxn_fa_cart" task_operations="layer_design"/>
                <FastRelax  cartesian="true" name="relax" repeats="5" scorefxn="sfxn_fa_cart"/>
            </MOVERS>
            <PROTOCOLS>
                 <!--Add mover="ssse_cstdes"/-->
                 <Add mover="ssse_cstdes_cart"/>
                 <Add mover="pssm_cst"/>
                 <Add mover="design"/>
                 <!--Add mover="relax"/-->
            </PROTOCOLS>
        </ROSETTASCRIPTS>
        """.format(
            sse, sspairs, pssm
        )
    elif sspairs == "" and hhpairs != "":
        print("only hhpairs")
        protocol = """
        <ROSETTASCRIPTS>
            <SCOREFXNS>
                <ScoreFunction name="sfxn_fa" weights="ref2015">
                    <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                    <Reweight scoretype="angle_constraint" weight="1.0"/>
                    <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                </ScoreFunction>
                <ScoreFunction name="sfxn_fa_cart" weights="ref2015_cart">
                    <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                    <Reweight scoretype="angle_constraint" weight="1.0"/>
                    <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                </ScoreFunction>
                <ScoreFunction name="bb_only" weights="empty.wts">
                    <Reweight scoretype="fa_rep" weight="0.1"/>
                    <Reweight scoretype="fa_atr" weight="0.2"/>
                    <Reweight scoretype="hbond_sr_bb" weight="2.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                    <Reweight scoretype="rama_prepro" weight="0.45"/>
                    <Reweight scoretype="omega" weight="0.4"/>
                    <Reweight scoretype="p_aa_pp" weight="0.6"/>
                </ScoreFunction>
            </SCOREFXNS>
            <RESIDUE_SELECTORS>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="sse_cstdes" overlap="0" pose_secstruct="{0}" ss="HE" use_dssp="0"/>
                <Layer name="surface" select_boundary="0" select_core="0" select_surface="1" use_sidechain_neighbors="1"/>
                <Layer name="boundary" select_boundary="1" select_core="0" select_surface="0" use_sidechain_neighbors="1"/>
                <Layer name="core" select_boundary="0" select_core="1" select_surface="0" use_sidechain_neighbors="1"/>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="sheet" overlap="0" pose_secstruct="{0}" ss="E" use_dssp="0"/>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="entire_helix" overlap="0" pose_secstruct="{0}" ss="H" use_dssp="0"/>
                <SecondaryStructure include_terminal_loops="1" minE="1" minH="1" name="entire_loop" overlap="0" pose_secstruct="{0}" ss="L" use_dssp="0"/>
                <And name="helix_cap" selectors="entire_loop">
                    <PrimarySequenceNeighborhood lower="1" selector="entire_helix" upper="0"/>
                </And>
                <And name="helix_start" selectors="entire_helix">
                    <PrimarySequenceNeighborhood lower="0" selector="helix_cap" upper="1"/>
                </And>
                <And name="helix" selectors="entire_helix">
                    <Not selector="helix_start"/>
                </And>
                <And name="loop" selectors="entire_loop">
                    <Not selector="helix_cap"/>
                </And>
                <Layer name="surface_cstdes" select_boundary="0" select_core="0" select_surface="1" use_sidechain_neighbors="1"/>
                <Layer name="boundary_cstdes" select_boundary="1" select_core="0" select_surface="0" use_sidechain_neighbors="1"/>
                <Layer name="core_cstdes" select_boundary="0" select_core="1" select_surface="0" use_sidechain_neighbors="1"/>
            </RESIDUE_SELECTORS>
            <TASKOPERATIONS>
                <DesignRestrictions name="layer_design">
                   <Action aas="DEHKPQR" selector_logic="surface AND helix_start"/>
                   <Action aas="EHKQR" selector_logic="surface AND helix"/>
                   <Action aas="EHKNQRST" selector_logic="surface AND sheet"/>
                   <Action aas="DEGHKNPQRST" selector_logic="surface AND loop"/>
                   <Action aas="ADEHIKLNPQRSTVWY" selector_logic="boundary AND helix_start"/> # M
                   <Action aas="ADEHIKLNQRSTVWY" selector_logic="boundary AND helix"/> # M
                   <Action aas="DEFHIKLNQRSTVWY" selector_logic="boundary AND sheet"/> # M
                   <Action aas="ADEFGHIKLNPQRSTVWY" selector_logic="boundary AND loop"/> # M
                   <Action aas="AFILPVWY" selector_logic="core AND helix_start"/> # M
                   <Action aas="AFILVWY" selector_logic="core AND helix"/> # M
                   <Action aas="FILVWY" selector_logic="core AND sheet"/> # M
                   <Action aas="AFGILPVWY" selector_logic="core AND loop"/> # M
                   <Action aas="DNST" selector_logic="helix_cap"/>
                </DesignRestrictions>
            </TASKOPERATIONS>
            <FILTERS>
            </FILTERS>
            <MOVERS>
                <SetSecStructEnergies name="ssse_cstdes" natbias_ss="5.0" scorefxn="sfxn_fa" secstruct="{0}" hh_pair="{1}" use_dssp="0"/>
                <SetSecStructEnergies name="ssse_cstdes_cart" natbias_ss="5.0" scorefxn="sfxn_fa_cart" secstruct="{0}" hh_pair="{1}" use_dssp="0"/>
                <FavorSequenceProfile name="pssm_cst" weight="0.4" scorefxns="sfxn_fa_cart" pssm="{2}" scaling="prob" matrix="BLOSUM62" />
                <FastDesign dualspace="true" name="design" ramp_down_constraints="false" relaxscript="MonomerDesign2019" repeats="5" scorefxn="sfxn_fa_cart" task_operations="layer_design"/>
                <FastRelax  cartesian="true" name="relax" repeats="5" scorefxn="sfxn_fa_cart"/>
            </MOVERS>
            <PROTOCOLS>
                <!--Add mover="ssse_cstdes_cart"/-->
                <Add mover="pssm_cst"/>
                <Add mover="design"/>
            </PROTOCOLS>
        </ROSETTASCRIPTS>
        """.format(
            sse, hhpairs, pssm
        )
    else:
        print("sspairs, hhpairs, hsstriplets")
        protocol = """
        <ROSETTASCRIPTS>
            <SCOREFXNS>
                <ScoreFunction name="sfxn_fa" weights="ref2015">
                    <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                    <Reweight scoretype="angle_constraint" weight="1.0"/>
                    <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                </ScoreFunction>
                <ScoreFunction name="sfxn_fa_cart" weights="ref2015_cart">
                    <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                    <Reweight scoretype="angle_constraint" weight="1.0"/>
                    <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                </ScoreFunction>
                <ScoreFunction name="bb_only" weights="empty.wts">
                    <Reweight scoretype="fa_rep" weight="0.1"/>
                    <Reweight scoretype="fa_atr" weight="0.2"/>
                    <Reweight scoretype="hbond_sr_bb" weight="2.0"/>
                    <Reweight scoretype="hbond_lr_bb" weight="2.0"/>
                    <Reweight scoretype="rama_prepro" weight="0.45"/>
                    <Reweight scoretype="omega" weight="0.4"/>
                    <Reweight scoretype="p_aa_pp" weight="0.6"/>
                </ScoreFunction>
            </SCOREFXNS>
            <RESIDUE_SELECTORS>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="sse_cstdes" overlap="0" pose_secstruct="{0}" ss="HE" use_dssp="0"/>
                <Layer name="surface" select_boundary="0" select_core="0" select_surface="1" use_sidechain_neighbors="1"/>
                <Layer name="boundary" select_boundary="1" select_core="0" select_surface="0" use_sidechain_neighbors="1"/>
                <Layer name="core" select_boundary="0" select_core="1" select_surface="0" use_sidechain_neighbors="1"/>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="sheet" overlap="0" pose_secstruct="{0}" ss="E" use_dssp="0"/>
                <SecondaryStructure include_terminal_loops="0" minE="1" minH="1" name="entire_helix" overlap="0" pose_secstruct="{0}" ss="H" use_dssp="0"/>
                <SecondaryStructure include_terminal_loops="1" minE="1" minH="1" name="entire_loop" overlap="0" pose_secstruct="{0}" ss="L" use_dssp="0"/>
                <And name="helix_cap" selectors="entire_loop">
                    <PrimarySequenceNeighborhood lower="1" selector="entire_helix" upper="0"/>
                </And>
                <And name="helix_start" selectors="entire_helix">
                    <PrimarySequenceNeighborhood lower="0" selector="helix_cap" upper="1"/>
                </And>
                <And name="helix" selectors="entire_helix">
                    <Not selector="helix_start"/>
                </And>
                <And name="loop" selectors="entire_loop">
                    <Not selector="helix_cap"/>
                </And>
                <Layer name="surface_cstdes" select_boundary="0" select_core="0" select_surface="1" use_sidechain_neighbors="1"/>
                <Layer name="boundary_cstdes" select_boundary="1" select_core="0" select_surface="0" use_sidechain_neighbors="1"/>
                <Layer name="core_cstdes" select_boundary="0" select_core="1" select_surface="0" use_sidechain_neighbors="1"/>
            </RESIDUE_SELECTORS>
            <TASKOPERATIONS>
                <DesignRestrictions name="layer_design">
                   <Action aas="DEHKPQR" selector_logic="surface AND helix_start"/>
                   <Action aas="EHKQR" selector_logic="surface AND helix"/>
                   <Action aas="EHKNQRST" selector_logic="surface AND sheet"/>
                   <Action aas="DEGHKNPQRST" selector_logic="surface AND loop"/>
                   <Action aas="ADEHIKLNPQRSTVWY" selector_logic="boundary AND helix_start"/> # M
                   <Action aas="ADEHIKLNQRSTVWY" selector_logic="boundary AND helix"/> # M
                   <Action aas="DEFHIKLNQRSTVWY" selector_logic="boundary AND sheet"/> # M
                   <Action aas="ADEFGHIKLNPQRSTVWY" selector_logic="boundary AND loop"/> # M
                   <Action aas="AFILPVWY" selector_logic="core AND helix_start"/> # M
                   <Action aas="AFILVWY" selector_logic="core AND helix"/> # M
                   <Action aas="FILVWY" selector_logic="core AND sheet"/> # M
                   <Action aas="AFGILPVWY" selector_logic="core AND loop"/> # M
                   <Action aas="DNST" selector_logic="helix_cap"/>
                </DesignRestrictions>
            </TASKOPERATIONS>
            <FILTERS>
            </FILTERS>
            <MOVERS>
                <SetSecStructEnergies name="ssse_cstdes" natbias_ss="5.0" scorefxn="sfxn_fa" secstruct="{0}" hh_pair="{2}" ss_pair="{1}" hss_triplets="{3}" use_dssp="0"/>
                <SetSecStructEnergies name="ssse_cstdes_cart" natbias_ss="5.0" scorefxn="sfxn_fa_cart" secstruct="{0}" hh_pair="{2}" ss_pair="{1}" hss_triplets="{3}" use_dssp="0"/>
                <FavorSequenceProfile name="pssm_cst" weight="0.4" scorefxns="sfxn_fa_cart" pssm="{4}" scaling="prob" matrix="BLOSUM62" />
                <FastDesign dualspace="true" name="design" ramp_down_constraints="false" relaxscript="MonomerDesign2019" repeats="5" scorefxn="sfxn_fa_cart" task_operations="layer_design"/>
                <FastRelax  cartesian="true" name="relax" repeats="5" scorefxn="sfxn_fa_cart"/>
            </MOVERS>
            <PROTOCOLS>
                 <Add mover="ssse_cstdes_cart"/>
                 <Add mover="pssm_cst"/>
                 <Add mover="design"/>
            </PROTOCOLS>
        </ROSETTASCRIPTS>
        """.format(
            sse, sspairs, hhpairs, hsstriplets, pssm
        )
    xml = XmlObjects.create_from_string(protocol).get_mover("ParsedProtocol")
    return xml, protocol


def pyrosetta_design(
    npz,
    outfile,
    sse=None,
    ss_pairs="",
    hh_pairs="",
    hss_triplets="",
    add_free_relax=False,
    pssm=None,
):
    params = {
        "WDIR": "/dev/shm",
        "PCUT": 0.05,
        "PCUT1": 0.5,
        "EBASE": -0.5,
        "EREP": [10.0, 3.0, 0.5],
        "DREP": [0.0, 2.0, 3.5],
        "PREP": 0.1,
        "SIGD": 10.0,
        "SIGM": 1.0,
        "MEFF": 0.0001,
        "DCUT": 19.5,
        "ALPHA": 1.57,
        "DSTEP": 0.5,
        "ASTEP": 15.0,
    }

    # init PyRosetta
    # init('-hb_cen_soft -relax:default_repeats 5 -default_max_cycles 200 -out:level 100')
    init("-ignore_unrecognized_res 1")

    # Create temp folder to store all the restraints
    tmpdir = tempfile.TemporaryDirectory(prefix="./")
    params["TDIR"] = tmpdir.name
    print("temp folder:     ", tmpdir.name)
    try:
        seq = N_to_AA(npz["I"][0].argmax(-1))[0]
    except:
        seq = npz["I"]
    L = len(seq)
    params["seq"] = seq
    params["USE_ORIENT"] = True

    # params['PCUT'] = 0.15
    rst = gen_rst(npz["feat2"], tmpdir, params)
    # seq_polyala = 'A'*len(seq)

    ########################################################
    # Scoring functions and movers
    ########################################################
    top_folder = os.path.dirname(os.path.dirname(__file__))
    sf = ScoreFunction()
    sf.add_weights_from_file(
        os.path.join(top_folder, "data/data_trRosetta/scorefxn.wts")
    )

    sf1 = ScoreFunction()
    sf1.add_weights_from_file(
        os.path.join(top_folder, "data/data_trRosetta/scorefxn1.wts")
    )

    sf_vdw = ScoreFunction()
    sf_vdw.add_weights_from_file(
        os.path.join(top_folder, "data/data_trRosetta/scorefxn_vdw.wts")
    )

    sf_cart = ScoreFunction()
    sf_cart.add_weights_from_file(
        os.path.join(top_folder, "data/data_trRosetta/scorefxn_cart.wts")
    )

    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(True)

    print("minimizing")
    min_mover = MinMover(mmap, sf, "lbfgs_armijo_nonmonotone", 0.0001, True)
    min_mover.max_iter(1000)

    min_mover1 = MinMover(mmap, sf1, "lbfgs_armijo_nonmonotone", 0.0001, True)
    min_mover1.max_iter(1000)

    min_mover_vdw = MinMover(mmap, sf_vdw, "lbfgs_armijo_nonmonotone", 0.0001, True)
    min_mover_vdw.max_iter(500)

    min_mover_cart = MinMover(mmap, sf_cart, "lbfgs_armijo_nonmonotone", 0.0001, True)
    min_mover_cart.max_iter(1000)
    min_mover_cart.cartesian(True)

    repeat_mover = RepeatMover(min_mover, 3)

    ########################################################
    # initialize pose
    ########################################################
    pose = pose_from_sequence(seq, "centroid")

    # mutate GLY to ALA
    for i, a in enumerate(seq):
        if a == "G":
            mutator = rosetta.protocols.simple_moves.MutateResidue(i + 1, "ALA")
            mutator.apply(pose)
            print("mutation: G%dA" % (i + 1))

    set_random_dihedral(pose)
    remove_clash(sf_vdw, min_mover_vdw, pose)

    # short + medium + long
    print("short + medium + long")
    add_rst(pose, rst, 1, len(seq), params)
    repeat_mover.apply(pose)
    min_mover_cart.apply(pose)
    remove_clash(sf_vdw, min_mover1, pose)

    # mutate ALA back to GLY
    for i, a in enumerate(seq):
        if a == "G":
            mutator = rosetta.protocols.simple_moves.MutateResidue(i + 1, "GLY")
            mutator.apply(pose)
            print("mutation: A%dG" % (i + 1))

    sf_fa = create_score_function("ref2015")
    sf_fa2 = create_score_function("ref2015")
    sf_fa2.set_weight(rosetta.core.scoring.atom_pair_constraint, 5)
    sf_fa2.set_weight(rosetta.core.scoring.dihedral_constraint, 1)
    sf_fa2.set_weight(rosetta.core.scoring.angle_constraint, 1)
    # secstr = rosetta.protocols.fldsgn.potentials.SetSecStructEnergies(sf_fa, './trial.blueprint', True)

    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)

    relax = rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(sf_fa2)
    relax.max_iter(200)
    relax.dualspace(True)
    relax.ramp_down_constraints(True)
    relax.set_movemap(mmap)

    pose.remove_constraints()
    switch = SwitchResidueTypeSetMover("fa_standard")
    switch.apply(pose)
    # secstr.apply(pose)
    pose.dump_pdb(outfile.replace(".pdb", "_onlyMin.pdb"))  # dump intermediate pose

    # print('relax...')
    # params['PCUT'] = 0.15
    # add_rst(pose, rst, 1, len(seq), params, True)
    # relax.apply(pose)

    if add_free_relax == True:
        print("free relax...")
        xml, protocol = final_design(
            sse, ";".join(ss_pairs), ";".join(hh_pairs), ";".join(hss_triplets)
        )
    if pssm:
        print("pssm design...")
        xml, protocol = design_with_pssm(
            sse, ";".join(ss_pairs), ";".join(hh_pairs), ";".join(hss_triplets), pssm
        )

    with open(outfile.replace(".pdb", ".xml"), "w") as f:  # dump design script
        f.writelines(protocol)
    xml.apply(pose)

    print("total score", sf_fa(pose))
    pose.dump_pdb(outfile)  # dump final pose
    return pose
