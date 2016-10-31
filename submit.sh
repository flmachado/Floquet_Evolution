#!/bin/bash 
#SBATCH -J Evolution # A single job name for the array 
#SBATCH -n 1 # Number of cores 
#SBATCH -N 1 # All cores on one machine 
#SBATCH -p interact # Partition 
#SBATCH --mem 500 # Memory request (4Gb) 
#SBATCH -t 0-00:05 # Maximum execution time (D-HH:MM) 
#SBATCH -o Evolution_%A_%a.out # Standard output 
#SBATCH -e Evolution_%A_%a.err # Standard error

output_Folder="test/"
mkdir $output_Folder

params=("L 4 logEvo 0 Nsteps 100 hz 0.1" \
        "L 4 logEvo 0 Nsteps 100 hz 0.12" \
        "L 4 logEvo 0 Nsteps 100 hz 0.15"
)

echo "Running params"
echo ${params[${SLURM_ARRAY_TASK_ID}]} 
python Evolution.py -o $output_Folder ${params[${SLURM_ARRAY_TASK_ID}]} 
