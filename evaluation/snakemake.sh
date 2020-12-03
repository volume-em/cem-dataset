#!/bin/bash

#this file will run the snakemake job across multiple nodes and store results
#in some local directories as well as mlflow

#get a partition with a k80 for model training
#make sure that the wall time is long enough
sbcmd="sbatch --partition={cluster.partition} --gres={cluster.gres} --mem={cluster.mem} \
--cpus-per-task={cluster.cpus} --time={cluster.time}"

#spawn the jobs, change the number of jobs for potentially faster execution
snakemake -pr --jobs 7 --cluster-config cluster.json --cluster "$sbcmd" --latency-wait 120 all
