#!/bin/bash 
#SBATCH -J StatesL13_Nov23 # A single job name for the array 
#SBATCH -n 1 # Number of cores 
#SBATCH -N 1 # All cores on one machine 
#SBATCH -p general # Partition 
#SBATCH --mem 8000 # Memory request 
#SBATCH -t 0-06:00 # Maximum execution time (D-HH:MM) 
#SBATCH -o StatesL13_Nov23_%A_%a.out # Standard output 
#SBATCH -e StatesL13_Nov23_%A_%a.err # Standard error

output_Folder="Dec_8/"
PreComp=(
# "Dec_4/PreComp_GLRF_L12_T10.472_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T10.472_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T12.566_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T12.566_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T1.571_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T3.491_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T3.491_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T3.927_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T3.927_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T4.488_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T4.488_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T5.236_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T5.236_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T6.981_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T6.981_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T0.785_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T1.047_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T0.419_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T0.628_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T1.257_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T3.142_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T6.283_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T7.854_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T15.708_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T0.419_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T0.628_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T1.257_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T3.142_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T6.283_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T7.854_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Dec_4/PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_4/PreComp_GLRF_L12_T0.314_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_4/PreComp_GLRF_L12_T0.314_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_4/PreComp_GLRF_L12_T0.209_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_4/PreComp_GLRF_L12_T0.209_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_4/PreComp_GLRF_L12_T0.126_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_4/PreComp_GLRF_L12_T0.126_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_4/PreComp_GLRF_L12_T0.063_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
"Dec_4/PreComp_GLRF_L12_T0.063_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T8.976_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L12_T8.976_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
# "Dec_4/PreComp_GLRF_L14_T15.708_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_30/PreComp_GLRF_L12_T15.708_alpha10.000_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_30/PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0300_hx0.1300_hy0.0700Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_27/PreComp_GLRF_L12_T20.944_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_27/PreComp_GLRF_L12_T20.944_alpha1.130_J1.0000_Jx0.170_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_27/PreComp_GLRF_L12_T31.416_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_27/PreComp_GLRF_L12_T31.416_alpha1.130_J1.0000_Jx0.170_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_27/PreComp_GLRF_L12_T15.708_alpha0.730_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_27/PreComp_GLRF_L12_T15.708_alpha0.730_J1.0000_Jx0.170_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_27/PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_27/PreComp_GLRF_L12_T5.560_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_27/PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.170_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_27/PreComp_GLRF_L12_T15.708_alpha2.270_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_27/PreComp_GLRF_L12_T15.708_alpha2.270_J1.0000_Jx0.170_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010ExpEig.npy"
#"PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.3100_hy0.0100_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010ExpEig.npy"
#"PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.6700_hy0.0100_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010ExpEig.npy"
#"PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx1.1900_hy0.0100_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010ExpEig.npy"
#"PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx2.1900_hy0.0100_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010ExpEig.npy"
#"PreComp_GLRF_L12_T15.708_alpha4.200_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010ExpEig.npy"
#"PreComp_GLRF_L12_T2.732_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010ExpEig.npy"
#"PreComp_GLRF_L12_T5.712_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010ExpEig.npy"
#"Nov_19/PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.170_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_19/PreComp_GLRF_L13_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig.npy"
#"Nov_19/PreComp_GLRF_L13_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.3100_hy0.0100Cutoff_0.00000010ExpEig.npy"
#"Nov_19/PreComp_GLRF_L13_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.6700_hy0.0100Cutoff_0.00000010ExpEig.npy"
#"Nov_19/PreComp_GLRF_L13_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx1.1900_hy0.0100Cutoff_0.00000010ExpEig.npy"
#"Nov_19/PreComp_GLRF_L13_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx2.1900_hy0.0100Cutoff_0.00000010ExpEig.npy"
#"Nov_19/PreComp_GLRF_L13_T15.708_alpha4.200_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig.npy"
#"Nov_19/PreComp_GLRF_L13_T2.732_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig.npy"
#"Nov_19/PreComp_GLRF_L13_T5.712_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig.npy"
#"Nov_19/PreComp_GLRF_L8_T0.100_alpha1.130_J-1.0000_Jx-0.300_eps0.0000_hz0.1700_hx0.1000_hy0.2200Cutoff_0.00000010ExpEig.npy"
#"Nov_19/PreComp_GLRF_L14_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_19/PreComp_GLRF_L14_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.3100_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_19/PreComp_GLRF_L14_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.6700_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_19/PreComp_GLRF_L14_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx1.1900_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_19/PreComp_GLRF_L14_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx2.1900_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_19/PreComp_GLRF_L14_T15.708_alpha4.200_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_19/PreComp_GLRF_L14_T2.732_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
#"Nov_19/PreComp_GLRF_L14_T5.712_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100Cutoff_0.00000010ExpEig_LessObs.npy"
)


Nfiles=8
Nstates=1
NObs=12

asd=$(( ${Nfiles} * ${Nstates} ))

fileState=$(( ${SLURM_ARRAY_TASK_ID}%${asd} ))
Obs=$(( ${SLURM_ARRAY_TASK_ID}/${asd} ))
#Obs=$(( ${Obs} * 4 ))

FileNum=$(( ${fileState}%${Nfiles} ))
StateNum=$(( ${fileState}/${Nfiles} ))

echo "File Number" $FileNum "  -  StateNum" $StateNum "   - Obs " ${Obs}



mkdir $output_Folder

params="logEvo 1 Nsteps 1e200 MAX_COUNTER 1 Obs_Val "$Obs
#params="logEvo 1 Nsteps 1e6 MAX_COUNTER 2 Obs_Val "$Obs
#params="logEvo 0 Nsteps 20000 MAX_COUNTER 200 Obs_Val 5"
#params="logEvo 0 Nsteps 2000000 MAX_COUNTER 20000 Obs_Val 0"
#params="logEvo 0 Nsteps 200 MAX_COUNTER 2 Obs_Val 2"

StateDesc=(
#"UUDUDUDUDUDU"
#"UDDUDUDUDUDU"
#"UDUUDUDUDUDU"
#"UDUDDUDUDUDU"
#"UDUDUUDUDUDU"
"UDUDUDDUDUDU"
# "UUUUUUUUUUUU"
# "UUUUUUDDDDDD"
# "UUUDDDUUUDDD"
# "UUDDUUDDUUDD"
# "UDDUUDDUUDDU"
# "UDUUDUUDUUDU"
# "UDUDDUDUUDUD"
# "UDUDUDUDDUDU"
# "UDUDUDUUDUDU"
# "UDUDUDDUDUDU"
# "UDUDUDUDUDUD"
)


echo "Running system"
echo $PreComp
echo "State"
echo ${SLURM_ARRAY_TASK_ID}
echo "python Evolution_States.py " ${PreComp[${FileNum}]} " stateDesc " ${StateDesc[${StateNum}]} $params " -o " $output_Folder

time python Evolution_States.py ${PreComp[$FileNum]} stateDesc ${StateDesc[${StateNum}]} $params -o $output_Folder