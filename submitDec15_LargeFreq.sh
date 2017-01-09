#!/bin/bash 
#SBATCH -J StatesL13_Nov23 # A single job name for the array 
#SBATCH -n 1 # Number of cores 
#SBATCH -N 1 # All cores on one machine 
#SBATCH -p general # Partition 
#SBATCH --mem 8000 # Memory request 
#SBATCH -t 0-01:30 # Maximum execution time (D-HH:MM) 
#SBATCH -o StatesL13_Nov23_%A_%a.out # Standard output 
#SBATCH -e StatesL13_Nov23_%A_%a.err # Standard error

output_Folder="Dec_15/"
PreComp=(
"Dec_15/PreComp_GLRF_L12_T0.126_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_15/PreComp_GLRF_L12_T0.140_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_15/PreComp_GLRF_L12_T0.157_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_15/PreComp_GLRF_L12_T0.180_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_15/PreComp_GLRF_L12_T0.209_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_15/PreComp_GLRF_L12_T0.251_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_15/PreComp_GLRF_L12_T0.314_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_15/PreComp_GLRF_L12_T0.419_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_15/PreComp_GLRF_L12_T0.628_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_15/PreComp_GLRF_L12_T0.785_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_15/PreComp_GLRF_L12_T1.047_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_15/PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_15/PreComp_GLRF_L12_T1.571_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_15/PreComp_GLRF_L12_T3.142_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_15/PreComp_GLRF_L12_T6.283_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
)


Nfiles=15
Nstates=3
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
"DUDUDUDUDUDU"
"DUDUDUUDUDUD"
"DUDUUDUDDUDU"
#"DUUDDUUDDUUD"
)



echo "Running system"
echo $PreComp
echo "State"
echo ${SLURM_ARRAY_TASK_ID}
echo "python Evolution_States.py " ${PreComp[${FileNum}]} " stateDesc " ${StateDesc[${StateNum}]} $params " -o " $output_Folder

time python Evolution_States.py ${PreComp[$FileNum]} stateDesc ${StateDesc[${StateNum}]} $params -o $output_Folder