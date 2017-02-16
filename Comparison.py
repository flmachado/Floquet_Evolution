import numpy as np
import matplotlib.pyplot as plt
import sys

from Simulation import *
from Compute_Evolution import *

if sys.argv[1] == '-h':
    print("Usage python Comparison.py MPI_file")
    print("The script automatically generates the matrix and the evolution of the ED based on the params of the MPI_file and computes the differences in the observables")

# Process MPI_file
# Get Params
MPI_filename = sys.argv[1]
MPI_file = open(MPI_filename, 'r')

params = {}

readingParams = True
for linecount, line in enumerate(MPI_file):
    line = line[:-1]
    if line == '--BEGIN DATA--':
        break

    line = line.split(',')

    if len(line) < 2:
        continue

    params[line[0]] = line[1]

# Import Data
dataset = np.genfromtxt( MPI_filename, skip_header=linecount+1, delimiter=',', names = True)

# Build the ED Matrix

ED_filename = 'ED_' + MPI_filename.split('/')[-1].split('.')[0] + '.npy'
print ED_filename

ED_params = {
    'L':  int(params['L']),
    'T': 2*np.pi/ float(params['w']),
    'alpha': float(params['alpha']),
    'J': float(params['J']),
    'Jx': float(params['Jx']),
    'epsilon': 0,
    'hz': float(params['hz']),
    'hy': float(params['hy']),
    'hx': float(params['hx']),
    'fil': ED_filename,
    'dir': './',
}

Generate_Long_Range_Floquet(ED_params)

# Run ED simulation
Nsteps = len(dataset['EE'].flatten()) -1

stateNum = 0
for i in range(int(params['L'])):
    if dataset['Sz_%d'%i][0] == 0:
        stateNum += 1<<i
        
state = np.zeros(2**int(params['L']), dtype='complex')
state[stateNum] = 1

ED_Data = []


for i in [0]:#range(int(params['L'])):
    res_filename = ED_filename.split('.')[0] + "_Spin%d.npy"%(i)
    Compute_Evolution(
        Nsteps,
        state,
        stateDesc = "X",
        result_fil = res_filename,
        preComputation_fil = ED_params['fil'],
        logEvo = 0,
        MAX_COUNTER = 1,
        SVD_Cutoff = 1e-7,
        output_folder = './',
        ObsVal = i)

    spin_data = np.load(res_filename)[()]

    ED_Data.append([spin_data['energies'], spin_data['valuesZ'], spin_data['entropy'] ])
    
print np.shape(dataset['DeffL'].flatten())
print np.shape((ED_Data[0][0]).flatten())
#print np.shape(dataset['DeffL'].flatten() - ED_Data[0][0].flatten())
    
# Compare Data:
fig, ax = plt.subplots()

for i in range(len(ED_Data)):
    dif = np.abs(dataset['DeffL'].flatten()- ED_Data[i][0].flatten())
    plt.plot(dif, label = 'Energy File=%d'%i)
plt.xlabel('Time')
plt.ylabel('Energy Difference')
plt.legend()

fig, ax = plt.subplots()
for i in range(len(ED_Data)):
    plt.plot(np.abs(dataset['Sz_%d'%i].flatten()- ED_Data[i][1].flatten()), label = 'Spin %d'%i)
plt.xlabel('Time')
plt.ylabel('Spin Expectation Difference')
plt.legend()


fig, ax = plt.subplots()
for i in range(len(ED_Data)):
    plt.plot(np.abs(dataset['EE'].flatten()- ED_Data[i][2].flatten()), label = 'Spin %d'%i)
plt.xlabel('Time')
plt.ylabel('Entanglement Entropy')
plt.legend()
plt.show()



    



    
    


