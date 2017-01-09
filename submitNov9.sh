#!/bin/bash 
#SBATCH -J L14Jx # A single job name for the array 
#SBATCH -n 1 # Number of cores 
#SBATCH -N 1 # All cores on one machine 
#SBATCH -p general # Partition 
#SBATCH --mem 10000 # Memory request 
#SBATCH -t 0-2:00 # Maximum execution time (D-HH:MM) 
#SBATCH -o L14Jx_%A_%a.out # Standard output 
#SBATCH -e L14Jx_%A_%a.err # Standard error

output_Folder="Dec_15/"
mkdir $output_Folder

params=(
"L 12 hz 0.03 hx 0.13 hy 0.07 J 1.00 Jx 0 epsilon 0 alpha 1.13 Omega 0.4"
"L 12 hz 0.03 hx 0.13 hy 0.07 J 1.00 Jx 0 epsilon 0 alpha 1.13 Omega 1"
"L 12 hz 0.03 hx 0.13 hy 0.07 J 1.00 Jx 0 epsilon 0 alpha 1.13 Omega 2"
"L 12 hz 0.03 hx 0.13 hy 0.07 J 1.00 Jx 0 epsilon 0 alpha 1.13 Omega 4"
"L 12 hz 0.03 hx 0.13 hy 0.07 J 1.00 Jx 0 epsilon 0 alpha 1.13 Omega 6"
"L 12 hz 0.03 hx 0.13 hy 0.07 J 1.00 Jx 0 epsilon 0 alpha 1.13 Omega 8"
"L 12 hz 0.03 hx 0.13 hy 0.07 J 1.00 Jx 0 epsilon 0 alpha 1.13 Omega 10"
"L 12 hz 0.03 hx 0.13 hy 0.07 J 1.00 Jx 0 epsilon 0 alpha 1.13 Omega 15"
"L 12 hz 0.03 hx 0.13 hy 0.07 J 1.00 Jx 0 epsilon 0 alpha 1.13 Omega 20"
"L 12 hz 0.03 hx 0.13 hy 0.07 J 1.00 Jx 0 epsilon 0 alpha 1.13 Omega 25"
"L 12 hz 0.03 hx 0.13 hy 0.07 J 1.00 Jx 0 epsilon 0 alpha 1.13 Omega 30"
"L 12 hz 0.03 hx 0.13 hy 0.07 J 1.00 Jx 0 epsilon 0 alpha 1.13 Omega 35"
"L 12 hz 0.03 hx 0.13 hy 0.07 J 1.00 Jx 0 epsilon 0 alpha 1.13 Omega 40"
"L 12 hz 0.03 hx 0.13 hy 0.07 J 1.00 Jx 0 epsilon 0 alpha 1.13 Omega 45"
"L 12 hz 0.03 hx 0.13 hy 0.07 J 1.00 Jx 0 epsilon 0 alpha 1.13 Omega 50"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.5"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.6"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.7"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.8"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.9"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 1.0"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 1.2"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 1.4"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 1.6"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 1.8"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 2.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 5.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 10.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 15.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 20.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 30.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 50.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 100.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 5.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 10.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 15.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 20.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 30.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 50.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 100.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 4.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 6.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 8.0"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 10.0"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 0.4"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 0.5"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 0.6"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 0.7"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 0.8"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 0.9"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 1.0"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 1.2"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 1.4"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 1.6"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 1.8"
# "L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10.0 Omega 2.0"
#"L 10 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10 Omega 0.4"
#"L 14 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 14 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10 Omega 0.4"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10 Omega 0.4"
#"L 10 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10 Omega 0.4"
#"L 12 hz 0.03 hx 0.13 hy 0.07 J 1 Jx 0.0 epsilon 0 alpha 10 Omega 0.4"
#"L 10 logEvo 1 Nsteps 1e10 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4 MAX_COUNTER 2"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.00 epsilon 0 alpha 1.13 Omega 0.4"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.17 epsilon 0 alpha 1.13 Omega 0.4"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.00 epsilon 0 alpha 1.13 Omega 0.3"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.17 epsilon 0 alpha 1.13 Omega 0.3"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.00 epsilon 0 alpha 1.13 Omega 0.2"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.17 epsilon 0 alpha 1.13 Omega 0.2"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.00 epsilon 0 alpha 1.13 Omega 1.13"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.00 epsilon 0 alpha 2.27 Omega 0.3"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.17 epsilon 0 alpha 1.13 Omega 0.4"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.17 epsilon 0 alpha 2.27 Omega 0.4"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.00 epsilon 0 alpha 0.73 Omega 0.4"
#"L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.17 epsilon 0 alpha 0.73 Omega 0.4"
# "L 12 hz 0.0 hx 0.31 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
# "L 12 hz 0.0 hx 0.67 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
# "L 12 hz 0.0 hx 1.19 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
# "L 12 hz 0.0 hx 2.19 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
# "L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 1.1"
# "L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 2.3"
# "L 12 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 4.2 Omega 0.4"
#"L 13 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 13 hz 0.0 hx 0.31 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 13 hz 0.0 hx 0.67 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 13 hz 0.0 hx 1.19 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 13 hz 0.0 hx 2.19 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 13 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 1.1"
#"L 13 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 2.3"
#"L 13 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 4.2 Omega 0.4"
#"L 14 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 14 hz 0.0 hx 0.31 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 14 hz 0.0 hx 0.67 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 14 hz 0.0 hx 1.19 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 14 hz 0.0 hx 2.19 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.4"
#"L 14 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 1.1"
#"L 14 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 2.3"
#"L 14 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 4.2 Omega 0.4"
)

echo "Running params"
echo ${params[${SLURM_ARRAY_TASK_ID}]} 
time python2.7 Evolution.py -o $output_Folder ${params[${SLURM_ARRAY_TASK_ID}]} 
