# Generating Aligned RDF-AMR graphs for downstream applications

This project uses pipenv. Use `pipenv install` to create the environment and `pipenv run python <script_name>.py` to run your code in the environment.

1. Run the `install.py` script (needed because we run on a cluster).
2. Run the `dataset_to_amr.py` script to generate AMR graphs, you need to change the path the the AMRLib model in the python source code! You run it once per dataset which is passed as an argument.
3. To generate RDF graphs from the generated AMR graphs you can use the `amr_to_rdf.py` file to generate the needed files for the SIFT model. Alternative you can also run the `all_amr_to_rdf.py` script which finds all generated files from `dataset_to_amr.py` in the current folder and generates all needed files for the SIFT model.

If you run on a cluster with LFS you can also use the `run_lsf_cluster.sh` to run your scripts.


If you have any questions feel free to reach out to us!