import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as slinalg
import time

def Compute_Evolution(Nsteps, state, result_fil, preComputation_fil, evolution='single',  logEvo = False, MAX_COUNTER = 1, EIG_COUNTER = 10, SVD_Cutoff = 1e-4, output_folder = './', ObsVal = 2):

    preComputation = np.load(preComputation_fil)[()]
    #print preComputation
    HPrethermal = preComputation['HPrethermal']
    Floquet     = preComputation['Floquet']
    U           = preComputation['U']
    Udag        = preComputation['Udag']
    Diag        = preComputation['Diag']
    Diag = Diag**(MAX_COUNTER)
    Obs_0       = preComputation['Obs']
    
    Obs_0 = [Obs_0[ObsVal],Obs_0[-1]]

    Obs_t       = Obs_0
    print "Number of Obs_0: " , len(Obs_0)
    res         = preComputation['res']
    L           = preComputation['L']
    InfTemp     = preComputation['InfTemp']
    eigHPrethermal = preComputation['eigHPrethermal']
    
    w = Diag
    
    values = []
    energies = []
    alpha = []

    states = [state]
    
    #print "Looking at state:", state
                      
    if res:
        print "Using Matrix Diagonalization"
        stateleft = []
        stateright = []
        
        for state in states:
            stateleft.append( np.conj(Udag.dot(state)))
            temp = []
            for i in range(len(Obs_0)):
                temp.append( Udag.dot(Obs_0[i]).dot(state).T )
            stateright.append(temp)

            #print (np.conj(state).dot(U).dot(Udag).dot(Obs_0[i]).dot(state) )
            #print np.conj(state).dot(Obs_0[i]).dot(state)


        for i in range(len(Obs_0)):
            alpha.append( Udag.dot( Obs_0[i] ).dot(U) )

        for i in range(len(Obs_0)-1):
            obs0 = []
            for k in range(len(states)):
                #print np.shape(alpha[i])
                #print np.shape(stateleft[k])
                #print np.shape(stateright[k])
                obs0.append( (stateleft[k].dot(alpha[i]).dot( stateright[k][i] ) )[0,0] )
                
                #print obs0[-1]
                #print states[k].dot(Obs_0[i]).dot(Obs_0[i]).dot(states[k])
                #print np.diag(Udag.dot(U))
                #print np.diag(U.dot(Udag))
                #print np.diag(Obs_0[i].dot(Obs_0[i]) )
                #print states[k].dot(U).dot(Obs_0[i]).dot(Obs_0[i]).dot(Udag).dot(states[k])
                #print "\n"
                #print i, k , (np.conj(stateleft[k]).dot( stateright[k] ) )[0,0]

            values.append([obs0])
        
        en0 = []
        for k in range(len(states)):
            en0.append( (stateleft[k].dot(alpha[-1]).dot( np.conj(stateleft[k].T)) )[0,0] )
        energies.append(en0)

    else:
        print "Using power of matrix"
        for i in range(len(Obs_0)-1):
            obs0 = []
            for state in states:
                obs0.append( (state.dot(Obs_0[i]).dot(Obs_0[i]).dot(state))[0,0] )
            values.append([obs0])
    
        en0 = []
        for state in states:
            en0.append( (state.dot(HPrethermal).dot(state)))
        energies.append(en0)
        FloquetEvo = Floquet

    c = 0
    cc = 1

    times = [0]
    counter = 0
    eigCounter = 0
    dt = 1

    f = open ('t.out','w')
    f.write('Finish precomputation\n')
    f.write('%s\n'%(time.localtime(time.time() ) ) )
    f.close()

    def compute_Expectations(prod, i ):
        if res:
            if i < len(Obs_0) - 1:
                #prod = prod.dot(Obs_0[i]) 
                obst = []
                for k in range(len( states)):

                    # print np.multiply(stateleft[k] ,np.conj(Diag)).shape
                    # print stateleft[k].dot(np.diag(np.conj(Diag)) ).shape
                    # print max( np.abs(np.multiply(stateleft[k] ,np.conj(Diag)) - stateleft[k].dot(np.diag(np.conj(Diag)) ) ))
                    # print ""
                    # print np.shape(Diag)
                    # print stateright[k][i].T.shape, stateright[k][i].shape
                    # print (stateright[k][i]*Diag).shape
                    # print np.multiply(Diag , stateright[k][i]).shape
                    # print np.multiply( stateright[k][i].T, Diag).T.shape
                    # print np.diag(Diag).dot(stateright[k][i]).shape
                    # print max( np.abs(np.multiply(Diag , stateright[k][i])- np.diag(Diag).dot(stateright[k][i])) )
                    # print ""
                    # print ""
                    obst.append( (
                        np.multiply(stateleft[k] ,np.conj(Diag)).dot( prod ).dot(
                            np.multiply( stateright[k][i].T, Diag).T)
                    )[0,0] )

                    #print (stateleft[k].dot( prod ).dot(stateright[k][i])  )[0,0], states[k].dot( np.linalg.matrix_power(np.conj(Floquet).T, times[-1]+dt)).dot(Obs_0[i]).dot( np.linalg.matrix_power( Floquet, times[-1]+dt) ).dot(states[k])[0,0]
                values[i].append(obst)

            elif i == len(Obs_0) - 1:
                ent = []
                for k in range(len(states)):
                    ent.append( (
                        np.multiply(stateleft[k] , np.conj(Diag)).dot( prod ).dot(
                            np.multiply( np.conj(stateleft[k]), Diag).T)
                    )[0,0] )
                energies.append(ent)

        else:
            if i < len(Obs_0) - 1:
                obst = []
                prod = prod.dot(Obs_0[i]) 
                for state in states:
                    obst.append( (np.conj(state).dot( prod ).dot(state) )[0,0] )
                values[i].append(obst)
            elif i == len(Obs_0) - 1:
                ent = []
                for state in states:
                    ent.append( (np.conj(state).dot( prod ).dot(state))[0,0] )
                energies.append(ent)


    while times[-1] < Nsteps:
        #print times[-1]
        # Evolve the different observables:
        for i in range(len(Obs_0)):

            if logEvo:
                if res == True:
                    #prod = np.diag( np.conj(Diag) ).dot(alpha[i]).dot(np.diag( Diag) )
                    #U.dot(np.diag(np.conj(Diag))).dot(alpha[i]).dot( np.diag(Diag) ).dot(Udag) 
                    #compute_Expectations(prod, i)
                    compute_Expectations(alpha[i], i)
                    
                                
                else:
                    prod =  np.conj(FloquetEvo).T.dot(Obs_t[i]).dot( FloquetEvo)
                    Obs_t[i] = prod

                    if eigCounter == EIG_COUNTER:
                        print "Checking Numerics"
                        print times[-1]
                        Need_SVD = EigenValueError(FloquetEvo , SVD_Cutoff )

                        eigCounter = 0
                        #l = min(np.abs(np.linalg.eigvals( FloquetEvo) ) )
                        #print l
                        if Need_SVD: #l < 1 - SVD_Cutoff:
                            #print "Lowest absolute eigenvalue", l
                            print "Applying SVD procedure"
                            print "Time: ", times[-1]
                            (U, s, V) = np.linalg.svd( FloquetEvo )
                            FloquetEvo = U.dot(V)
                    else:
                        eigCounter += 1

                    compute_Expectations(prod, i)
                        
                    # if i < len(Obs_0) - 1:
                    #     for state in states:
                    #         obst.append( (state.dot( prod ).dot(Obs_0[i]).dot(state) )[0,0] )
                    #     values[i].append(obst)
                    # elif i == len(Obs_0) - 1:
                    #     for state in states:
                    #         ent.append( (state.dot( prod ).dot(state))[0,0] )
                    #     energies.append([ent])
                        
            else:
                if res:
                    #prod = np.diag(np.conj(Diag) ).dot(alpha[i]).dot( np.diag(Diag) )
                    #compute_Expectations(prod, i)
                    compute_Expectations(alpha[i], i)
           
                else:
                    prod = np.conj(Floquet).T.dot(Obs_t[i]).dot(Floquet)
                    Obs_t[i] = prod
                    compute_Expectations(prod, i)
                    #values[i].append(prod[state,state])

#                if i < len(Obs_0) - 1:
#                    fft[i].append( np.fft.fft(values_FM[i]))
        if logEvo:
            #print ""
            #print "dt: ",dt
            #print times[-1]
            #print counter

            times.append( times[-1] + dt )
            if counter < MAX_COUNTER:
                counter += 1
            else:
                counter = 0 
                dt *= 2
                if res:
                    w= w * w
                    w = w / np.abs(w)
                else:
                    FloquetEvo = FloquetEvo.dot(FloquetEvo)

            if res:
                Diag = Diag * w
                Diag = Diag/np.abs(Diag)
                    
        else:
            times.append(times[-1] + MAX_COUNTER)
            if res:
                Diag = Diag * w
                Diag = Diag/np.abs(Diag)

        
    times = np.array(times[:])

    for i in range(len(Obs_0)-1):
        #print values[i]
        values[i] = np.array(values[i])

    energies = np.array(energies)
    if  np.max(np.abs( np.imag( energies) ) )> 1e-14:
        print "TOO MUCH IMAGINARY PART"
        print energies
        print np.max(np.abs( np.imag( energies) ) )

    info = {'L': L,
            'state': state,
            'Nsteps': Nsteps,
            'result_fil': result_fil,
            'preComputation_fil': preComputation_fil,
            'evolution': evolution,
            'logEvo': logEvo,
            'InfTemp': InfTemp,
            'SVD_Cutoff': SVD_Cutoff,
            'EigHPrethermal': eigHPrethermal,
            'EigFloquet': w}
            
    output = {'values': values,
              'energies': energies,
              'times': times,
              'info': info,
              }
    print ""
    print result_fil + ".npy"
    print output_folder + result_fil
    np.save(output_folder + result_fil, output)
    
    return output
        
            
