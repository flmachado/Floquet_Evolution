#!/bin/bash 
#SBATCH -J States12 # A single job name for the array 
#SBATCH -n 1 # Number of cores 
#SBATCH -N 1 # All cores on one machine 
#SBATCH -p general # Partition 
#SBATCH --mem 2000 # Memory request 
#SBATCH -t 0-01:00 # Maximum execution time (D-HH:MM) 
#SBATCH -o States_%A_%a.out # Standard output 
#SBATCH -e States_%A_%a.err # Standard error

output_Folder="Nov_11/"
#PreComp="PreComp_GLRF_L12_T15.708_alpha1.500_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100_Log1_Nsteps10000000000_MAX_COUNTER2_12Cutoff_0.00000010.npy"
#PreComp="PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100_Log1_Nsteps10000000000_MAX_COUNTER2_12Cutoff_0.00000010.npy"

#PreComp="PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.3100_hy0.0100_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010ExtraTime.npy"
#PreComp="PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.6700_hy0.0100_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010ExtraTime.npy"
#PreComp="PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx1.1900_hy0.0100_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010ExtraTime.npy"
#PreComp="PreComp_GLRF_L12_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx2.1900_hy0.0100_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010ExtraTime.npy"
#PreComp="PreComp_GLRF_L12_T15.708_alpha1.500_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100_Log1_Nsteps10000000000_MAX_COUNTER2_12Cutoff_0.00000010ExtraTime.npy"
#PreComp="PreComp_GLRF_L12_T15.708_alpha4.200_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010ExtraTime.npy"
#PreComp="PreComp_GLRF_L12_T2.732_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010ExtraTime.npy"
#PreComp="PreComp_GLRF_L12_T5.712_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010ExtraTime.npy"
#PreComp="PreComp_GLRF_L4_T15.708_alpha1.130_J1.0000_Jx0.000_eps0.0000_hz0.0000_hx0.1300_hy0.0100_Log1_Nsteps10000000000_MAX_COUNTER2_12Cutoff_0.00000010ExtraTime.npy"
PreComp="PreComp_GLRF_L8_T0.100_alpha1.130_J-1.0000_Jx-0.300_eps0.0000_hz0.1700_hx0.1000_hy0.2200_Log1_Nsteps100000000_MAX_COUNTER2_12Cutoff_0.00000010.npy"
mkdir $output_Folder

params="logEvo 1 Nsteps 1e10 MAX_COUNTER 0"

echo "Running system"
echo $PreComp
echo "State"
echo ${SLURM_ARRAY_TASK_ID}
echo "python Evolution_States.py " $PreComp " state " ${SLURM_ARRAY_TASK_ID} " " $params " -o " $output_Folder

time python Evolution_States.py $PreComp state ${SLURM_ARRAY_TASK_ID} $params -o $output_Folder