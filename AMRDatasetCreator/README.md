# Generating Aligned RDF-AMR graphs for downstream applications

1. Run the `install.py` script (needed because we run on a cluster).
2. Run the `dataset_to_amr.py` script to generate AMR graphs, you need to change the path the the AMRLib model in the python source code! You run it once per dataset which is passed as an argument.

If you run on a cluster with LFS you can also use the `run_lsf_cluster.sh` to run your scripts.

