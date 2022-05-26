# Running on ETH's Euler Cluster

## Link Collection

#### "summary"
- https://scicomp.ethz.ch/wiki/LSF_mini_reference

####  rest
- https://scicomp.ethz.ch/wiki/Python
- https://scicomp.ethz.ch/wiki/Getting_started_with_clusters



## How to run
```
bsub -n 20 -W 20:00 -R "rusage[mem=4500,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" ./my_cuda_program
```
-n 20 -> 20 processor cores
-W 20:00 -> set the runtime to max 20 hours !!! HAS TO BE SET ELSE 4h max!!!
-R "rusage[mem=4500]" -> set memory requirement to 4500MB !!!PER CORE!!!
-R "rusage[ngpus_excl_p=8]" -> set the number of gpus to 1
-R "select[gpu_model0==GeForceGTX1080Ti]" -> selects the GPU model 
    -> see available GPUs here: https://scicomp.ethz.ch/wiki/Using_the_batch_system#GPU
-R "rusage[scratch=YYY]" -> if we want to use scratch space on exec node, need to request it too

put files in  /cluster/scratch -> scratch space is not backed up and auto delete is after 2 weeks, max 2.5 TB

#### need to locally install all the additional python files
pip install --user package

#### then also load the installed python_gpu version with the corresponding gcc
module load gcc/8.2.0 python_gpu/3.9.9

