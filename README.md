# Genesis
Automated protein backbone refinement from a protein sketch as described in:

(1) Zander Harteveld, Joshua Southern, Michaël Defferrard, Andreas Loukas, Pierre Vandergheynst, Micheal M. Bronstein, Bruno E. Correia. *Deep sharpening of topolgical features for de novo protein design*. ICLR MLDD (2022).

(2) Zander Harteveld, Alexandra Van Hall-Beauvais, Irina Morozova, Joshua Southern, Casper Goverde, Sandrine Georgeon, Stéphane Rosset, Michëal Defferrard, Andreas Loukas, Pierre Vandergheynst, Michael M. Bronstein, and Bruno E. Correia. *Exploring "dark matter" protein folds using deep learning*. (2023).


Installation
------------

1. Install libraries from the requirements file:

```
pip install -r REQUIREMENTS
```

2. Install [PyRosetta](https://www.pyrosetta.org/) for the structure modelling part.

Usage
-----

Sample freely from a given FORM (a string specifying how the sketch or protein backbone should look) with set loop lengths via:

```
python -u sample.py \
--form A1E7.B1E7.A3E7.B3E7.B4E7.A4E7.B2E7.A2E7 \
--loops x.4.4.3.5.2.1.3.x \
--num_decoys 5 \
--wts ./data/finetune_checkpoint_500 \
--prefix designedSeq \
--out_dir ./A1E7.B1E7.A3E7.B3E7.B4E7.A4E7.B2E7.A2E7_x.4.4.3.5.2.1.3.x \
--optimizer ADAM \
--opt_iterations 101 \
--num_recycling 0 \
--polyVAL_seq False \
--pssm_design
```
