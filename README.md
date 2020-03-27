**J**oint **E**pitope **S**election and **S**pacer Design for String-of-Beads (or **EV**) Vaccines
=====

JessEV is a framework for simultaneous selection of epitopes and design of spacers for string-of-beads vaccines based on mixed integer linear programming. The linear program maximizes the immunogenicity of the epitopes selected, while respecting constraints related to their pathogen/HLA coverage and conservation, as well as cleavage scores at certain critical locations of the vaccine.

# Installation
The required dependencies can be installed via conda:

```
conda create -n jessev --file packages.txt -c conda-forge
```

Additionally, you should install one of the solvers that are supported by [pyomo](https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html#supported-solvers). The default solver used by this project is [gurobi](https://www.gurobi.com/), which is free for academic usage.

# Usage
The command line interface is contained in `design.py`, whose usage can be seen via `python design.py --help`. As input it requires a CSV file with the epitopes to consider, the output file for the vaccine, and a series of options to specify constraints on the vaccine:

```
Usage: design.py [OPTIONS] INPUT_EPITOPES OUTPUT_VACCINE

Options:
  -s, --min-spacer-length INTEGER
                                  Minimum length of the spacer to be designed
  -S, --max-spacer-length INTEGER
                                  Maximum length of the spacer to be designed
  -e, --num-epitopes INTEGER      Number of epitopes in the vaccine
  --top-immunogen FLOAT           Only consider the top epitopes by
                                  immunogenicity

  --top-proteins FLOAT            Only consider the top epitopes by protein
                                  coverage

  --top-alleles FLOAT             Only consider the top epitopes by allele
                                  coverage

  --min-alleles FLOAT             Vaccine must cover at least this many
                                  alleles

  --min-proteins FLOAT            Vaccine must cover at least this many
                                  proteins

  --min-avg-prot-conservation FLOAT
                                  On average, epitopes in the vaccine must
                                  cover at least this many proteins

  --min-avg-alle-conservation FLOAT
                                  On average, epitopes in the vaccine must
                                  cover at least this many alleles

  -g, --min-nterminus-gap FLOAT   Minimum cleavage gap
  -n, --min-nterminus-cleavage FLOAT
                                  Minimum cleavage at the n-terminus
  -ct, --min-cterminus-cleavage FLOAT
                                  Minimum cleavage at the n-terminus
  -c, --min-spacer-cleavage FLOAT
                                  Minimum cleavage inside the spacers
  -C, --max-spacer-cleavage FLOAT
                                  Maximum cleavage inside the spacers
  -E, --max-epitope-cleavage FLOAT
                                  Maximum cleavage inside epitopes
  -i, --epitope-cleavage-ignore-first INTEGER
                                  Ignore first amino acids for epitope
                                  cleavage

  --log-file PATH                 Where to save the logs
  --verbose                       Print debug messages
  --solver-type TEXT              Which linear programming solver to use
  --help                          Show this message and exit.
```

The input epitopes must be in a CSV file with the following columns:

 - `immunogen`: the immunogenicity of the epitope.
 - `alleles`: a list of alleles to which the epitope binds separated by `;`, e.g. `HLA-B*40:06;HLA-A*01:01;HLA-B*40:01`.
 - `proteins`: a list of numerical IDs of the proteins that contain the epitope, e.g. `53;63;2`.
 - `epitope`: the epitope sequence, e.g. `MGNKWSKSI`.

# Reproducibility
The experiments performed in the paper can be run with the bash scripts in the `experiments` directory. The necessary input data is located in the `dev` directory, where results and log files will be placed. These results can be analyzed by running `plots.py`, which will create the paper's figures in the `dev` directory and print the results of the analyses mentioned in the paper.

Note: the sequential approach requires [FRED-2](https://github.com/FRED-2/Fred2), which does not officially support python 3 yet. Unfortunately, some fiddling is required on your part.