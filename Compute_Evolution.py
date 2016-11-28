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

    print preComputation.keys()
    D_U         = preComputation['D_U']
    D_Udag      = preComputation['D_Udag']
    D_Diag      = preComputation['D_Diag']
    
    if not logEvo:
        Diag = Diag**(MAX_COUNTER)
    print "MAX_COUNTER: ", MAX_COUNTER

    Obs_0       = preComputation['Obs']    
    Obs_0 = [Obs_0[ObsVal],Obs_0[-1]]

    Obs_t       = Obs_0
    print "Number of Obs_0: " , len(Obs_0)
    res         = preComputation['res']
    L           = preComputation['L']
    InfTemp     = preComputation['InfTemp']
    eigHPrethermal = preComputation['eigHPrethermal']
    
    w = Diag

    D_w = D_Diag
    
    values = []
    D_values = []
    
    energies = []
    D_energies = []
    
    alpha = []
    D_alpha = []

    states = [state]
    
    #print "Looking at state:", state
                      
    if res:
        print "Using Matrix Diagonalization"
        stateleft = []
        stateright = []

        D_stateleft = []
        D_stateright = []
        
        for state in states:
            stateleft.append( np.conj(Udag.dot(state)))
            temp = []
            for i in range(len(Obs_0)):
                temp.append( Udag.dot(Obs_0[i]).dot(state).T )
            stateright.append(temp)

            D_stateleft.append( np.conj(D_Udag.dot(state)))
            temp = []
            for i in range(len(Obs_0)):
                temp.append( D_Udag.dot(Obs_0[i]).dot(state).T )
            D_stateright.append(temp)

            #print (np.conj(state).dot(U).dot(Udag).dot(Obs_0[i]).dot(state) )
            #print np.conj(state).dot(Obs_0[i]).dot(state)


        for i in range(len(Obs_0)):
            alpha.append( Udag.dot( Obs_0[i] ).dot(U) )

        for i in range(len(Obs_0)):
            D_alpha.append( D_Udag.dot( Obs_0[i] ).dot(D_U) )

            
        for i in range(len(Obs_0)-1):
            obs0 = []
            D_obs0 = []
            for k in range(len(states)):
                obs0.append( (stateleft[k].dot(alpha[i]).dot( stateright[k][i] ) ) [0,0] )
                D_obs0.append( (D_stateleft[k].dot(D_alpha[i]).dot( D_stateright[k][i] ) ) [0,0] )

            values.append([obs0])
            D_values.append([D_obs0])
        
        en0 = []
        D_en0 = []
        for k in range(len(states)):
            en0.append( (stateleft[k].dot(alpha[-1]).dot( np.conj(stateleft[k].T)) ) [0,0] )
            D_en0.append( (D_stateleft[k].dot(D_alpha[-1]).dot( np.conj(D_stateleft[k].T)) ) )
            
        energies.append(en0)
        D_energies.append(D_en0)

    else:
        print "Using power of matrix"
        for i in range(len(Obs_0)-1):
            obs0 = []
            for state in states:
                obs0.append( (state.dot(Obs_0[i]).dot(Obs_0[i]).dot(state))  [0,0] )
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

    def compute_Expectations(prod, i, D_prod = -1, compute_D = False ):
        if res:
            if i < len(Obs_0) - 1:
                obst = []
                for k in range(len( states)):
                    obst.append( (
                        np.multiply(stateleft[k] ,np.conj(Diag)).dot( prod ).dot(
                            np.multiply( stateright[k][i].T, Diag).T)
                    )[0,0] )

                    #print "ASDASD", (np.multiply(stateleft[k] ,np.conj(Diag)).dot( prod ).dot(
                    #    np.multiply( stateright[k][i].T, Diag).T))
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

            if compute_D:
                if i < len(Obs_0) - 1:
                    D_obst = []
                    for k in range(len( states)):
                        D_obst.append( (
                            np.multiply(D_stateleft[k] ,np.conj(D_Diag)).dot( D_prod ).dot(
                            np.multiply( D_stateright[k][i].T, D_Diag).T)
                        )[0,0] )
                    #print (stateleft[k].dot( prod ).dot(stateright[k][i])  )[0,0], states[k].dot( np.linalg.matrix_power(np.conj(Floquet).T, times[-1]+dt)).dot(Obs_0[i]).dot( np.linalg.matrix_power( Floquet, times[-1]+dt) ).dot(states[k])[0,0]
                    D_values[i].append(D_obst)

                elif i == len(Obs_0) - 1:
                    D_ent = []
                    for k in range(len(states)):
                        D_ent.append( (
                            np.multiply(D_stateleft[k] , np.conj(D_Diag)).dot( D_prod ).dot(
                                np.multiply( np.conj(D_stateleft[k]), D_Diag).T)
                        ))#[0,0] )
                    D_energies.append(D_ent)
                
        else:
            if i < len(Obs_0) - 1:
                obst = []
                prod = prod.dot(Obs_0[i]) 
                for state in states:
                    obst.append( (np.conj(state).dot( prod ).dot(state) )  [0,0] )
                values[i].append(obst)
            elif i == len(Obs_0) - 1:
                ent = []
                for state in states:
                    ent.append( (np.conj(state).dot( prod ).dot(state))  [0,0] )
                energies.append(ent)


    while times[-1] < Nsteps:
        #print times[-1]
        # Evolve the different observables:
        for i in range(len(Obs_0)):

            if logEvo:
                if res == True:
                    compute_Expectations(alpha[i], i, D_prod = D_alpha[i], compute_D = True )
                    
                                
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
                    compute_Expectations(alpha[i], i, D_prod = D_alpha[i], compute_D = True)
           
                else:
                    prod = np.conj(Floquet).T.dot(Obs_t[i]).dot(Floquet)
                    Obs_t[i] = prod
                    compute_Expectations(prod, i)
                    #values[i].append(prod[state,state])

        if logEvo:

            times.append( times[-1] + dt )
            if counter < MAX_COUNTER:
                counter += 1
            else:
                counter = 0 
                dt *= 2
                if res:
                    w = w * w
                    w = w / np.abs(w)
                    D_w = D_w * D_w
                    D_w = D_w / np.abs(D_w)
                else:
                    FloquetEvo = FloquetEvo.dot(FloquetEvo)

            if res:
                Diag = Diag * w
                Diag = Diag/np.abs(Diag)

                D_Diag = D_Diag * D_w
                D_Diag = D_Diag/np.abs(D_Diag)
                    
        else:
            times.append(times[-1] + MAX_COUNTER)
            if res:
                Diag = Diag * w
                Diag = Diag/np.abs(Diag)

        
    times = np.array(times[:])

    for i in range(len(Obs_0)-1):
        #print values[i]
        values[i] = np.array(values[i])
        D_values[i] = np.array( D_values[i] )

    energies = np.array( energies)
    D_energies = np.array(D_energies)
    print D_energies
    #print D_values
    if  np.max(np.abs( np.imag( energies) ) )> 1e-14:
        print "TOO MUCH IMAGINARY PART"
        #print energies
        print np.max(np.abs( np.imag( energies) ) )

    info = {'L': L,
            'args': preComputation['args'],
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
              'D_values': D_values,
              'D_energies': D_energies,              
              'times': times,
              'info': info,
              }

    print ""
    print result_fil + ".npy"
    print output_folder + result_fil

    #np.save("here", output)
    np.save(output_folder + result_fil, output)
    
    return output
        
            
