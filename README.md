RImPep: Reactive Immunopeptidome prediction pipeline
====

The RImPep pipeline is used to predict peptides that are likely to have CD8 T-cell reactivity. It has three main steps, which are each executed by three components:
1. P-model: Deep-learning model for predicting the specific proteins (P-set) which will be processed and presented in the immunopeptidome (P-set).
2. TinyHLAnet: Deep-learning model for predicting the specific peptides (PT-set) that will be derived from the P-set.
3. ThymE: Knowledge-based extraction of peptides that are not covered by thymic education (PTS-set).

## Setting up dependencies

This repository has scripts that will help setup the dependencies required for running RImPep. The only assumptions made by the script is that the user has access to a UNIX-like environment with `bash`, `R`, `python`, and `git`.

### R dependencies
We assume that the user has access to a reasonably recent R environment (at least version 4). Only three packages are required for R: `getopt`, `R.utils`, and `data.table`, with no specific requirements on which version they should be. They can be installed from within an R environment (either in the terminal or through RStudio) as follows:
```
install.packages(c("getopt", "R.utils", "data.table"))
```

Alternatively you can also install the required packages from Bash as follows:
```
Rscript prereq/r-reqs.R
```

### TinyHLAnet/Python
P-model and TinyHLAnet are deep-learning models that require the installation of specific Python packages. Development of P-model and TinyHLAnet was done in concert, so the same set of packages can be shared across both. We provide a script in the repository to download TinyHLAnet and install a conda environment with the required packages. This can be done by running the following command in bash:

```
bash prereq/download-tinyhlanet-conda.sh
```

This should download TinyHLAnet into the folder `bin/tinyhlanet` and also have installed the associated conda environment `tinyhlanet`. Please check whether you are able to access the conda environment by activating it with the command:

```
conda env activate tinyhlanet
```

If this command works, your bash prompt should have changed to begin with `(tinyhlanet)`, and you're all good to test out RImPep!

If you want to setup the environment yourself using venv, you can check the `requirements.txt` file given in `bin/TinyHLAnet/prereq` directory instead.

## Quick start guide

The `run.sh` script is the main interface provided for running RImPep. The input to the `run.sh` script is a folder with the following files:
1. `tpm.tsv`: Tab-separated file with two columns: `gene` and `tpm`, which have the ENSEMBL gene ID and TPM values respectively.
2. `alleles.txt`: Text file containing the list of alleles corresponding to the sample, with each allele listed in a separated line. If both alleles of a gene is the same (example, if both HLA-A alleles of a person are HLA-A\*02:01), it is recommended that the user lists the same allele twice in this file.

The files provided in `example/sample-input` can be used as reference for how the input files should look like.

Optionally, the user can also provide a `mutated.fa` file with the sequence of the mutated genes (typically obtained from ENSEMBL's VEP tool).


Running RImPep is as simple as providing this folder to `run.sh` as follows (make sure that the `tinyhlanet` conda env is active):

```
bash run.sh example/sample-input mtechi
```

The predictions from this run will be placed in the same input folder. The two main outputs you would need to pay attention to are `potentially-reactive-eptiopes.tsv` and `epitopes.tsv`. These files contain the peptides from the reactive immunopeptidome and the full immunopeptidome respectively, and can be used for further analysis.


## Repository organization
This repository contains the code and data required to train and benchmark the `P-model` and also for executing the full RImPep pipeline. The files/directories that come with this repository include:

```
RImPep
|
|-bin              : Contains wrapper scripts used during model benchmarking
|-promiselib       : Python library that contains the code used to built P-model
|-prereq           : Contains preprocessed data that will be used to kickstart the manuscript-related analysis for P-model.
|                    Also contains scripts to help setup the required packages for running RImPep
|-scripts          : Contains the scripts required to reproduce the manuscript-related analysis & data from scratch.
|-tools            : Contains wrapper scripts used for running RImPep
|
|
|-reproduce.sh     : One shot script for reproducing the manuscript-related analysis for P-model.
|-run.sh           : User interface for running RImPep
```

