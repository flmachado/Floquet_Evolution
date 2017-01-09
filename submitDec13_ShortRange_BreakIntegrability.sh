#!/bin/bash 
#SBATCH -J StatesL13_Nov23 # A single job name for the array 
#SBATCH -n 1 # Number of cores 
#SBATCH -N 1 # All cores on one machine 
#SBATCH -p general # Partition 
#SBATCH --mem 8000 # Memory request 
#SBATCH -t 0-01:30 # Maximum execution time (D-HH:MM) 
#SBATCH -o StatesL13_Nov23_%A_%a.out # Standard output 
#SBATCH -e StatesL13_Nov23_%A_%a.err # Standard error

output_Folder="Dec_13/"
PreComp=(
"Dec_13/PreComp_GLRF_L12_T0.251_alpha10.000_J4.3700_Jx0.170_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_13/PreComp_GLRF_L12_T0.314_alpha10.000_J4.3700_Jx0.170_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_13/PreComp_GLRF_L12_T0.419_alpha10.000_J4.3700_Jx0.170_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_13/PreComp_GLRF_L12_T0.628_alpha10.000_J4.3700_Jx0.170_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_13/PreComp_GLRF_L12_T1.257_alpha10.000_J4.3700_Jx0.170_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_13/PreComp_GLRF_L12_T15.708_alpha10.000_J4.3700_Jx0.170_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_13/PreComp_GLRF_L12_T2.094_alpha10.000_J4.3700_Jx0.170_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_13/PreComp_GLRF_L12_T4.189_alpha10.000_J4.3700_Jx0.170_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_13/PreComp_GLRF_L12_T7.854_alpha10.000_J4.3700_Jx0.170_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
)


Nfiles=9
Nstates=1
NObs=12

asd=$(( ${Nfiles} * ${Nstates} ))

fileState=$(( ${SLURM_ARRAY_TASK_ID}%${asd} ))
Obs=$(( ${SLURM_ARRAY_TASK_ID}/${asd} ))

FileNum=$(( ${fileState}%${Nfiles} ))
StateNum=$(( ${fileState}/${Nfiles} ))



echo "File Number" $FileNum "  -  StateNum" $StateNum "   - Obs " ${Obs}

mkdir $output_Folder

params="logEvo 1 Nsteps 1e50 MAX_COUNTER 2 Obs_Val "$Obs
#params="logEvo 1 Nsteps 1e8 MAX_COUNTER 4 Obs_Val "$Obs
#params="logEvo 0 Nsteps 500 MAX_COUNTER 1 Obs_Val "$Obs
#params="logEvo 0 Nsteps 2000000 MAX_COUNTER 20000 Obs_Val 0"
#params="logEvo 0 Nsteps 200 MAX_COUNTER 2 Obs_Val 2"

StateDesc=(
#"DDDDDDDDDDDD"
#"DDDUUUDDDUUU"
#"DUDDUDDUDDUD"
#"DUDUDUDUDUDU"
"DUDUDUUDUDUD"
#"DUDUUDUDDUDU"
#"DUUDDUUDDUUD"
)



echo "Running system"
echo $PreComp
echo "State"
echo ${SLURM_ARRAY_TASK_ID}
echo "python Evolution_States.py " ${PreComp[${FileNum}]} " stateDesc " ${StateDesc[${StateNum}]} $params " -o " $output_Folder

time python Evolution_States.py ${PreComp[$FileNum]} stateDesc ${StateDesc[${StateNum}]} $params -o $output_Folder