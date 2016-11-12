#!/bin/bash 
#SBATCH -J Test_Benchmark # A single job name for the array 
#SBATCH -n 1 # Number of cores 
#SBATCH -N 1 # All cores on one machine 
#SBATCH -p general # Partition 
#SBATCH --mem 32000 # Memory request 
#SBATCH -t 0-12:00 # Maximum execution time (D-HH:MM) 
#SBATCH -o Benchmark_%A_%a.out # Standard output 
#SBATCH -e Benchmark_%A_%a.err # Standard error

output_Folder="Nov_7/Fixed/"
mkdir $output_Folder


# params=(
# "L 10 logEvo 1 Nsteps 1e10 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 2"
# "L 10 logEvo 1 Nsteps 1e10 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 1"
# "L 10 logEvo 1 Nsteps 1e10 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
# "L 10 logEvo 1 Nsteps 1e10 hz 0.0 hx 3.79 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 2"
# "L 10 logEvo 1 Nsteps 1e10 hz 0.0 hx 3.79 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 1"
# "L 10 logEvo 1 Nsteps 1e10 hz 0.0 hx 3.79 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
# "L 12 logEvo 1 Nsteps 1e10 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
# "L 12 logEvo 1 Nsteps 1e10 hz 0.0 hx 3.79 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
# "L 10 logEvo 1 Nsteps 1e10 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 10 Omega 0.4 MAX_COUNTER 2"
# "L 10 logEvo 1 Nsteps 1e10 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 10 Omega 0.4 MAX_COUNTER 1"
# "L 10 logEvo 1 Nsteps 1e10 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 10 Omega 0.4 MAX_COUNTER 0"
# "L 10 logEvo 1 Nsteps 1e10 hz 0.0 hx 3.79 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 10 Omega 0.4 MAX_COUNTER 2"
# "L 10 logEvo 1 Nsteps 1e10 hz 0.0 hx 3.79 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 10 Omega 0.4 MAX_COUNTER 1"
# "L 10 logEvo 1 Nsteps 1e10 hz 0.0 hx 3.79 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 10 Omega 0.4 MAX_COUNTER 0"
# "L 12 logEvo 1 Nsteps 1e10 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 10 Omega 0.4 MAX_COUNTER 0"
# "L 12 logEvo 1 Nsteps 1e10 hz 0.0 hx 3.79 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 10 Omega 0.4 MAX_COUNTER 0"
# )


params=(
"L 14 logEvo 1 Nsteps 1e1 hz 0.0 hx 3.79 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
"L 14 logEvo 1 Nsteps 1e3 hz 0.0 hx 3.79 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
"L 14 logEvo 1 Nsteps 1e5 hz 0.0 hx 3.79 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
"L 14 logEvo 1 Nsteps 1e7 hz 0.0 hx 3.79 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
"L 14 logEvo 1 Nsteps 1e9 hz 0.0 hx 3.79 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
"L 14 logEvo 1 Nsteps 1e11 hz 0.0 hx 3.79 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
"L 14 logEvo 1 Nsteps 1e1 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
"L 14 logEvo 1 Nsteps 1e3 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
"L 14 logEvo 1 Nsteps 1e5 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
"L 14 logEvo 1 Nsteps 1e7 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
"L 14 logEvo 1 Nsteps 1e9 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
"L 14 logEvo 1 Nsteps 1e11 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 0"
)

echo "Running params"
echo ${params[${SLURM_ARRAY_TASK_ID}]} 
time python Evolution.py -o $output_Folder ${params[${SLURM_ARRAY_TASK_ID}]} 
