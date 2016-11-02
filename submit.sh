#!/bin/bash 
#SBATCH -J Evolution # A single job name for the array 
#SBATCH -n 1 # Number of cores 
#SBATCH -N 1 # All cores on one machine 
#SBATCH -p general # Partition 
#SBATCH --mem 2000 # Memory request (4Gb) 
#SBATCH -t 0-04:00 # Maximum execution time (D-HH:MM) 
#SBATCH -o Evolution_%A_%a.out # Standard output 
#SBATCH -e Evolution_%A_%a.err # Standard error

output_Folder="Oct_31/"
mkdir $output_Folder

# params=("L 10 logEvo 0 Nsteps 200 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 10 T 0.124" \
#             "L 10 logEvo 0 Nsteps 200 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 10 T 0.628" \
#             "L 10 logEvo 0 Nsteps 200 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 10 T 6.283" \
#             "L 10 logEvo 0 Nsteps 200 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 10 T 62.830"\
#             "L 10 logEvo 0 Nsteps 200 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 10 T 628.30" \
#             "L 10 logEvo 0 Nsteps 200 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 T 0.124" \
#             "L 10 logEvo 0 Nsteps 200 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 T 0.628" \
#             "L 10 logEvo 0 Nsteps 200 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 T 6.283" \
#             "L 10 logEvo 0 Nsteps 200 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 T 62.830"\
# )


# params=("L 10 logEvo 0 Nsteps 500 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 T 1.2566" \
#         "L 10 logEvo 0 Nsteps 500 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 T 4.488" \
#         "L 10 logEvo 0 Nsteps 500 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 T 6.283" \
#         "L 10 logEvo 0 Nsteps 500 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 T 7.85375" \
#         "L 10 logEvo 0 Nsteps 500 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 T 10.472"\
#         "L 10 logEvo 0 Nsteps 500 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 10 T 1.2566" \
#         "L 10 logEvo 0 Nsteps 500 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 10 T 4.488" \
#         "L 10 logEvo 0 Nsteps 500 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 10 T 6.283" \
#         "L 10 logEvo 0 Nsteps 500 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 10 T 7.85375" \
#         "L 10 logEvo 0 Nsteps 500 hz 0 hx 0.17 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 10 T 10.472"
# )

# params=("L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 50" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.01" 
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.0 hy 0.00 J 1 Jx 1 epsilon 0 alpha 10 Omega 50" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.0 hy 0.00 J 1 Jx 1 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.0 hy 0.00 J 1 Jx 1 epsilon 0 alpha 10 Omega 1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.0 hy 0.00 J 1 Jx 1 epsilon 0 alpha 10 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.0 hy 0.00 J 1 Jx 1 epsilon 0 alpha 10 Omega 0.01" 
# )

# params=("L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 50" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.1 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 50" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.2 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.3 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.4 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.5 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.01" \
# )

# params=("L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.1 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.1 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.1 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.1 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.01" \
# )

# params=("L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.001" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.0001" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 100" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.0 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.0001" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.001" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 100" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.0 hy 0.00 J -1 Jx 1 epsilon 0 alpha 10 Omega 1000" \
# )

# params=("L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.0001" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.001" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 1 epsilon 0 alpha 10 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 1 epsilon 0 alpha 10 Omega 1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 1 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 1 epsilon 0 alpha 10 Omega 100" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 1 epsilon 0 alpha 10 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 0.0001" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 0.001" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 100" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 1000" \
# )

# params=("L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 0.001" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 0.002" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 0.003" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 0.003" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 0.0008" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 0.0006" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 0.0004" \
# )

# params=("L 10 logEvo 0 Nsteps 500 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 0.001")

# params=("L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.13 hy 0.01 J -1 Jx 1 epsilon 0 alpha 10 Omega 100" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.13 hy 0.01 J -1 Jx 1 epsilon 0 alpha 10 Omega 50" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.13 hy 0.01 J -1 Jx 1 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.13 hy 0.01 J -1 Jx 1 epsilon 0 alpha 10 Omega 1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 100" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 50" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 1" 
# )

# params=("L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.7 epsilon 0 alpha 10 Omega 100" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.7 epsilon 0 alpha 10 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.7 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.7 epsilon 0 alpha 10 Omega 1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.7 epsilon 0 alpha 10 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.7 epsilon 0 alpha 10 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.7 epsilon 0 alpha 10 Omega 0.001"\
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.3 epsilon 0 alpha 10 Omega 100" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.3 epsilon 0 alpha 10 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.3 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.3 epsilon 0 alpha 10 Omega 1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.3 epsilon 0 alpha 10 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.3 epsilon 0 alpha 10 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.3 epsilon 0 alpha 10 Omega 0.001"\
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.1 epsilon 0 alpha 10 Omega 100" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.1 epsilon 0 alpha 10 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.1 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.1 epsilon 0 alpha 10 Omega 1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.1 epsilon 0 alpha 10 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.1 epsilon 0 alpha 10 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.1 epsilon 0 alpha 10 Omega 0.001"\
# )


# params=("L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 0.1" \
# "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 0.01" \
# "L 10 logEvo 0 Nsteps 150 hz 0.5 hx 0.13 hy 0.01 J -1 Jx 0 epsilon 0 alpha 10 Omega 0.001"
# )

# params=("L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.7 epsilon 0 alpha 1.13 Omega 100" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.7 epsilon 0 alpha 1.13 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.7 epsilon 0 alpha 1.13 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.7 epsilon 0 alpha 1.13 Omega 1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.7 epsilon 0 alpha 1.13 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.7 epsilon 0 alpha 1.13 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.7 epsilon 0 alpha 1.13 Omega 0.001"\
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.3 epsilon 0 alpha 1.13 Omega 100" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.3 epsilon 0 alpha 1.13 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.3 epsilon 0 alpha 1.13 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.3 epsilon 0 alpha 1.13 Omega 1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.3 epsilon 0 alpha 1.13 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.3 epsilon 0 alpha 1.13 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.3 epsilon 0 alpha 1.13 Omega 0.001"\
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.1 epsilon 0 alpha 1.13 Omega 100" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.1 epsilon 0 alpha 1.13 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.1 epsilon 0 alpha 1.13 Omega 10" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.1 epsilon 0 alpha 1.13 Omega 1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.1 epsilon 0 alpha 1.13 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.1 epsilon 0 alpha 1.13 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 150 hz 0.1 hx 0.13 hy 0.01 J -1 Jx 0.1 epsilon 0 alpha 1.13 Omega 0.001"\
# )

# params=("L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 10 Omega 100" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 10 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 10 Omega 1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 10 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 10 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 10 Omega 0.001"\
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 Omega 100" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 Omega 1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 10 Omega 0.001"\
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 10 Omega 100" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 10 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 10 Omega 10" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 10 Omega 1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 10 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 10 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 10 Omega 0.001"
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 1.13 Omega 100" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 1.13 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 1.13 Omega 10" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 1.13 Omega 1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 1.13 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 1.13 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 1.13 Omega 0.001"\
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 1.13 Omega 100" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 1.13 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 1.13 Omega 10" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 1.13 Omega 1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 1.13 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 1.13 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 1.13 Omega 0.001"\
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 1.13 Omega 100" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 1.13 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 1.13 Omega 10" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 1.13 Omega 1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 1.13 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 1.13 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 1.13 Omega 0.001"\
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 1.5 Omega 100" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 1.5 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 1.5 Omega 10" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 1.5 Omega 1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 1.5 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 1.5 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.7 epsilon 0 alpha 1.5 Omega 0.001"\
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 1.5 Omega 100" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 1.5 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 1.5 Omega 10" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 1.5 Omega 1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 1.5 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 1.5 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.3 epsilon 0 alpha 1.5 Omega 0.001"\
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 1.5 Omega 100" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 1.5 Omega 1000" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 1.5 Omega 10" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 1.5 Omega 1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 1.5 Omega 0.1" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 1.5 Omega 0.01" \
#         "L 10 logEvo 0 Nsteps 1000 hz 0.1 hx 0.13 hy 0.01 J 1 Jx 0.1 epsilon 0 alpha 1.5 Omega 0.001"\
# )

# params=("L 8 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.00001"\
#         "L 8 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.0001"\
#         "L 8 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.001"\
#         "L 8 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.01"\
#         "L 8 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.1"\
#         "L 8 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 1"\
#         "L 8 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 10"\
#         "L 8 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 100"\
#         "L 8 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 1000"\
#         "L 8 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 10000"
# )

params=("L 10 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.00001"\
        "L 10 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.0001"\
        "L 10 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.001"\
        "L 10 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.01"\
        "L 10 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 0.1"\
        "L 10 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 1"\
        "L 10 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 10"\
        "L 10 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 100"\
        "L 10 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 1000"\
        "L 10 logEvo 0 Nsteps 1000 hz 0.0 hx 0.13 hy 0.01 J 1 Jx 0.0 epsilon 0 alpha 1.13 Omega 10000"
)

echo "Running params"
echo ${params[${SLURM_ARRAY_TASK_ID}]} 
time python Evolution.py -o $output_Folder ${params[${SLURM_ARRAY_TASK_ID}]} 
