#!/bin/bash 
#SBATCH -J Test_Benchmark # A single job name for the array 
#SBATCH -n 1 # Number of cores 
#SBATCH -N 1 # All cores on one machine 
#SBATCH -p general # Partition 
#SBATCH --mem 32000 # Memory request 
#SBATCH -t 3-00:00 # Maximum execution time (D-HH:MM) 
#SBATCH -o Benchmark_%A_%a.out # Standard output 
#SBATCH -e Benchmark_%A_%a.err # Standard error

output_Folder="Nov_11/"
mkdir $output_Folder

params=(
#"L 10 logEvo 1 Nsteps 1e10 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 2"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.6"
#"L 12 hz 0.0 hx 0.31 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 12 hz 0.0 hx 0.67 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 12 hz 0.0 hx 1.19 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 12 hz 0.0 hx 2.19 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 1.1"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 2.3"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 4.2 Omega 0.4"
"L 14 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
"L 14 hz 0.0 hx 0.31 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
"L 14 hz 0.0 hx 0.67 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
"L 14 hz 0.0 hx 1.19 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
"L 14 hz 0.0 hx 2.19 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
"L 14 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 1.1"
"L 14 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 2.3"
"L 14 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 4.2 Omega 0.4"
)

echo "Running params"
echo ${params[${SLURM_ARRAY_TASK_ID}]} 
time python Evolution.py -o $output_Folder ${params[${SLURM_ARRAY_TASK_ID}]} 
