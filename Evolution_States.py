from Compute_Evolution import *
import sys

preComputation_Fil = sys.argv[1]
#StateNum = sys.argv[2]

States = np.load(preComputation_Fil)[()]['states']

args = {'Nsteps': 10**8,
        'state': -1,
        'logEvo': 1,
        'MAX_COUNTER': 2,
        'SVD_Cutoff': 1e-7,
        'EIG_COUNTER': 25,
        '-o': './'}

for i in range(2, len(sys.argv), 2):
    if sys.argv[i] in args.keys():
        if sys.argv[i] == '-o':
            args[ sys.argv[i]] = sys.argv[i+1]
        else:
            args[ sys.argv[i]] = float(sys.argv[i+1])

Nsteps = int(args['Nsteps'])
StateNum = int(args['state'])
logEvo = args['logEvo']
MAX_COUNTER = args['MAX_COUNTER']
EIG_COUNTER = args['EIG_COUNTER']
SVD_Cutoff = args['SVD_Cutoff']
output_folder = args['-o']

fil = preComputation_Fil + "_State%d"%( StateNum) + ('_Log%d_Nsteps%d_MAX_COUNTER%d_12'%( logEvo, Nsteps, args['MAX_COUNTER'] ) )

print "Computing Evolution for state: ", States[StateNum]
print "Saving it as:"
print fil


Compute_Evolution(Nsteps,
                  States[StateNum],
                  result_fil = fil,
                  preComputation_fil = preComputation_Fil,
                  logEvo = logEvo,
                  MAX_COUNTER = MAX_COUNTER,
                  EIG_COUNTER = EIG_COUNTER,
                  SVD_Cutoff = SVD_Cutoff,
                  output_folder = output_folder)
 
