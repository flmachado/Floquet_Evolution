from  Simulation import *
import sys

args = {'L': 8,
        'T': 0.1,
        'J': -1,
        'Jx': -0.3,
        'Omega': 1,
        'A': 1.3,
        'dA': 0.3,
        'U': 1,
        'hz': 0.17,
        'epsilon': 0.00,
        'alpha': 1.13,
        'eta': 0,
        'hy': 0.22,
        'hx': 0.1,
        'Nsteps': 10**8,
        'Rep': 1,
        'state': -1,
        'logEvo': 1,
        'MAX_COUNTER': 2,
        'SVD_Cutoff': 1e-7,
        'EIG_COUNTER': 25,
        '-o': './'}

changed_state = False
changed_Omega = False
for i in range(1, len(sys.argv), 2):
    if sys.argv[i] in args.keys():
        if sys.argv[i] == 'state':
            print "ASD"
            changed_state = True
        if sys.argv[i] == 'Omega':
            changed_Omega = True

        if sys.argv[i] == '-o':
            args[ sys.argv[i]] = sys.argv[i+1]
        else:
            args[ sys.argv[i]] = float(sys.argv[i+1])
if not changed_state:
    args['state'] = 2**args['L'] - 1
    
print args

L = int(args['L'])
Omega = float(args['Omega'])
if changed_Omega:
    T = 2*np.pi/Omega
else:
    T = float(args['T'])

J = float(args['J'])
Jx = float(args['Jx'])
hz = float(args['hz'])
epsilon = float(args['epsilon'])
alpha = float(args['alpha'])

eta=float(args['eta'])
hy = float(args['hy'])
hx = float(args['hx'])

A = float(args['A'])
dA = float(args['dA'])
U = float(args['U'])

state = int(args['state'])
Nsteps = float(args['Nsteps'])
Rep = int(args['Rep'])
logEvo = bool(args['logEvo'])
SVD_Cutoff = float(args['SVD_Cutoff'])
EIG_COUNTER = int(args['EIG_COUNTER'])

"""
Implemented Floquet evolution generators

GCP   -   Generate_Chetans_Prethermal(L,T2,J,epsilon,hz)
GFNN  -   Generate_Floquet_NearestNeighbour(L,T,W,epsilon,eta)
GFO   -   Generate_Floquet_Operator(L,T,W,epsilon,eta)
GLRF  -   Generate_Long_Range_Floquet( L, T, alpha, J, hz, epsilon, hx, hy)
GF    -   Generate_Francisco(L, T, J, U, A)

"""

Floquet= {'GCP':  [Generate_Chetans_Prethermal,
                   {'L': L,
                    'T': T,
                    'J': J,
                    'epsilon': epsilon,
                    'eta': eta,
                    'hy': hy,
                    'hx': hx},
                   'GCP_L%d_T%.3f_J%.4f_eps%.4f_eta%.4f_hy%.4f_hx%.4f'% (L, T, J, epsilon, eta, hy,hx)],
          
          'GFNN': [Generate_Floquet_NearestNeighbour,
                   {'L': L,
                    'T': T,
                    'J': J,
                    'epsilon': epsilon,
                    'hz': hz,
                    'hy': hy,
                    'hx': hx},
                   'GFNN_L%d_T%.3f_J%.4f_epsilon%.4f_hz%.4f_hy%.4f_hx%.4f'% (L, T, J, epsilon, hz,hy,hx)],
          
          'GFO':  [Generate_Floquet_Operator,
                   {'L': L,
                    'T': T,
                    'J': J,
                    'epsilon': epsilon,
                    'eta': eta},
                   'GFO_L%d_T%.3f_J%.4f_eps%.4f_eta%.4f'% (L, T, J, epsilon, eta)],

          'GLRF': [Generate_Long_Range_Floquet,
                   {'L': L,
                    'T': T,
                    'alpha': alpha,
                    'J': J,
                    'Jx': Jx,
                    'epsilon': epsilon,
                    'hz': hz,
                    'hy': hy,
                    'hx': hx},
                   'GLRF_L%d_T%.3f_alpha%.3f_J%.4f_Jx%.3f_eps%.4f_hz%.4f_hx%.4f_hy%.4f'% (L, T, alpha, J, Jx, epsilon, hz, hx, hy)],
          'GF': [Generate_Francisco,
                 {'L':L,
                  'Omega':Omega,
                  'J':J,
                  'U':U,
                  'A':A,
                  'dA':dA},
                 'GF_L%d_Omega%.3f_J%.3f_U%.3f_A%.3f_dA%.3f'% (L, Omega, J,U, A, dA)]
          }
Choice = 'GLRF'

fil = Floquet[Choice][2] + ('_Log%d_Nsteps%d_MAX_COUNTER%d_12'%( logEvo, Nsteps, args['MAX_COUNTER'] ) )

fil = fil + "Cutoff_%.8f"%SVD_Cutoff

print "Filename: ",fil

info = 0
for i in range(Rep):
    print "\nRepetition: %d" % (i)
    output = Compute_Evolution(L, Floquet[Choice][0], Floquet[Choice][1], Nsteps, state, fil = 'R%d_'%(i) + fil, evolution = 'single',  logEvo = logEvo, MAX_COUNTER = args['MAX_COUNTER'], EIG_COUNTER = EIG_COUNTER, SVD_Cutoff = SVD_Cutoff, output_folder = args['-o'])
 
#np.save("Ave_R%d_"%(Rep) + fil, s)
