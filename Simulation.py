import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as slinalg

from random import *

#Constants of interest
sigmax = np.array([[0,1],
                   [1,0]])
sigmay = np.array([[0,-1j],
                   [1j,0]])
sigmaz = np.array([[1,0],
                   [0,-1]])
iden   = np.array([[1,0],
                   [0,1]])

def Ewald( L,x, alpha):
    return ( 1.0/ (np.sin( np.pi * x/L) / (np.pi / L))**alpha)

def EigenValueError(mat, err):

    CO = np.linalg.eigvalsh( (mat + np.conj( mat ).T)/2)
    SI = np.linalg.eigvalsh( (mat - np.conj( mat ).T)/2j ) 

    l = np.array( np.sort( CO**2 ) )
    k = np.array( np.sort( SI**2 ) )
    
    ab = l + k[::-1]
    
    if min(ab) < 1 - err:
        return True
    else:
        return False

def EigenVectorsUnitary(mat):

    (w_real, v) = np.linalg.eigh( (mat + np.conj(mat).T)/2)
    w = []
    print np.shape(v)
    print np.shape(mat)
    
    Eig = np.conj(v.T).dot( mat ).dot( v )   
    #print Eig
    #print Eig
    #print Eig.shape
    #print np.max(np.abs( Eig - np.diag(np.diag(Eig)) ) )
    #print w
    #print np.linalg.eigvals(mat)
    return (np.diag(Eig), v)

def Eigenvalues(Floquet):

    eig = np.linalg.eig(Floquet)[0]
    return np.sort(np.angle(-eig))

    
def outer(A,B):
    sa = np.shape(A)
    sb = np.shape(B)

    #print A, sa
    #print B, sb

    l = []
    for i in range(sa[0]):
        l.append([])
        for o in range(sa[1]):
            l[-1].append( A[i][o] * B)
            
    C = np.bmat(l)
    #print C
    #print np.shape(C), ":" , sa, sb
    return C
 

def SigmaTerms(sigma, L, indices):

    mat = np.array([[1]])
    for k in range(0,L):
        if k in indices:
            mat = outer(sigma, mat)
        else:
            mat = outer(iden, mat)
    return mat

def Generate_Floquet_Operator(L,T,W,epsilon,eta):
    print "\nUSING THE FLOQUET_OPERATOR FUNCTION\n"
    # Hamniltonia
    ########## First Half of evolution
    # Precession term:
    prec = np.zeros((2**L, 2**L))
    for i in range(2**L):
        n_ones = 0
        t = i
        while t > 0:
            n_ones += t%2
            t = int(t/2)
        prec[i][i] = L - 2*n_ones
    prec = eta * prec
    #Interaction term
    
    wei = [complex(0)]
    for i in range(1,L):
        wei.append( complex(Ewald(L,i,1)) )
    
    interations = np.zeros((2**L, 2**L), dtype = 'complex')
    #np.identity(2**L) * wei[0]

    
    for i in range(L):
        for j in range(i+1,L):
            mat = iden
            if i == 0:
                mat = sigmaz
                
            for k in range(1,L):
                if j ==k or i==k:
                    mat = outer(sigmaz, mat)
                else:
                    mat = outer( iden, mat)
            interations += mat * wei[j-i] * W
    
    Uf1 = slinalg.expm2(- 1j* T*( interations +  prec))
    
    ########## Second Half of evolutions
    
    SingleSpin = iden * np.cos(np.pi/2 * (1+epsilon)) - 1j*sigmax * np.sin( np.pi/2 * (1+epsilon))
    Uf2 = SingleSpin
    for i in range(1,L):
        Uf2 = outer(SingleSpin, Uf2)
    
    ##########
    Floquet = np.dot(Uf2, Uf1)
    (w_t, v_t) = EigenVectorsUnitary( Floquet )
    (w,v) = np.linalg.eig( Floquet )

    res = True
    test = np.allclose( np.conj(v).T.dot(v) ,  np.eye(2**L))
    res = res and test
    print "Unitary: ", test
    test = np.allclose(v.dot( np.diag(w).dot( np.conj(v).T)) , np.dot(Uf2, Uf1))
    res = res and test
    print "Decomp: ", test
    test = np.allclose(v.dot( np.diag(w**2).dot( np.conj(v).T)) , np.dot(Uf2, Uf1).dot(Uf2.dot(Uf1)) )
    res = res and test
    print "Decomp2: ", test
    test = np.allclose(v.dot( np.diag(w**3).dot( np.conj(v).T)) , np.dot(Uf2, Uf1).dot(Uf2.dot(Uf1)).dot(Uf2.dot(Uf1)) )
    res = res and test
    print "Decomp3: ", test
    
    return (v, w, np.conj(v).T, np.dot(Uf2, Uf1) , res)
    

def Generate_Floquet_NearestNeighbour(args):
    print "\nUSING THE NEAREST NEIGHBOUR FUNCTION\n"
    # Hamniltonia
    ########## First Half of evolution
    # Precession term:

    L = args['L']
    T = args['T']
    J = args['J']
    epsilon = args['epsilon']
    hx = args['hx']
    hy = args['hy']
    hz = args['hz']

    
    Zprecession = np.zeros( (2**L, 2**L), dtype = 'complex')
    for i in range(L):
        mat = iden
        if i==0:
            mat = sigmaz
        for k in range(1,L):
            if i == k:
                mat = outer(sigmaz, mat)
            else:
                mat = outer(iden, mat)
        Zprecession += mat

    Xprecession = np.zeros( (2**L, 2**L), dtype = 'complex')
    for i in range(L):
        mat = iden
        if i==0:
            mat = sigmax
        for k in range(1,L):
            if i == k:
                mat = outer(sigmax, mat)
            else:
                mat = outer(iden, mat)
        Xprecession += mat

    Yprecession = np.zeros( (2**L, 2**L), dtype = 'complex')
    for i in range(L):
        mat = iden
        if i==0:
            mat = sigmax
        for k in range(1,L):
            if i == k:
                mat = outer(sigmax, mat)
            else:
                mat = outer(iden, mat)
        Yprecession += mat


    #Interaction term
    
    interactions = np.zeros((2**L, 2**L), dtype = 'complex')
    #np.identity(2**L) * wei[0]
    
    for i in range(L):
        j = (i+1)%L

        mat = iden
        if i == 0 or j==0:
            mat = sigmaz
            
        for k in range(1,L):
            if j ==k or i==k:
                mat = outer(sigmaz, mat)
            else:
                mat = outer( iden, mat)
        interactions += mat * J
    
    Uf1 = slinalg.expm2(- 1j* T*( interactions + hz*Zprecession + hy*Yprecession + hx*Xprecession))

    ########## Second Half of evolutions
    
    SingleSpin = iden * np.cos(np.pi/2 * (1+epsilon)) - 1j*sigmax * np.sin( np.pi/2 * (1+epsilon))
    Uf2 = SingleSpin
    for i in range(1,L):
        Uf2 = outer(SingleSpin, Uf2)

    ##########
    HPrethermal = interactions + hx*Xprecession
    
    return (HPrethermal, Uf2.dot(Uf1))
    

def Generate_Chetans_Prethermal(L,T2,J,epsilon,hz, hy, hx):
    print "\nUSING CHETAN's FUNCTION\n"
    # Hamniltonia
    ########## First Half of evolution
    # Precession term:
    Zprecession = np.zeros( (2**L, 2**L), dtype = 'complex')
    for i in range(L):
        mat = iden
        if i==0:
            mat = sigmaz
        for k in range(1,L):
            if i == k:
                mat = outer(sigmaz, mat)
            else:
                mat = outer(iden, mat)
        Zprecession += mat * (-hz)
        
    Yprecession = np.zeros( (2**L, 2**L), dtype = 'complex')
    for i in range(L):
        mat = iden
        if i==0:
            mat = sigmay
        for k in range(1,L):
            if i == k:
                mat = outer(sigmay, mat)
            else:
                mat = outer(iden, mat)
        Zprecession += mat * (-hy)
        

    #Interaction term
    
    interations = np.zeros((2**L, 2**L), dtype = 'complex')
    #np.identity(2**L) * wei[0]
    
    for i in range(L):
        j = (i+1)%L

        mat = iden
        if i == 0 or j==0:
            mat = sigmaz
            
        for k in range(1,L):
            if j ==k or i==k:
                mat = outer(sigmaz, mat)
            else:
                mat = outer( iden, mat)
        interations += mat * (-J)
    

    ########## Second Half of evolutions

    Xprecession = np.zeros( (2**L, 2**L), dtype = 'complex')
    for i in range(L):
        mat = iden
        if i==0:
            mat = sigmax
        for k in range(1,L):
            if i == k:
                mat = outer(sigmax, mat)
            else:
                mat = outer(iden, mat)
        Xprecession += mat * np.pi/2 * ( 1 + epsilon)
        
    Uf1 = slinalg.expm2(- 1j*( Zprecession + interations + Yprecession + Xprecession*(1+hx)))
    Uf2 = slinalg.expm2(- 1j*T2*(interations + Zprecession + hx*Xprecession) )
    ##########

    #(U, s, V) = np.linalg.svd( np.dot(Uf2, Uf1))
    #print "SVD: ", np.allclose( U.dot(np.diag(s).dot( V) ) , np.dot(Uf2, Uf1) ) 

    #print s
    #print w

    #print "SVD: ", np.allclose( U.dot(np.diag(w).dot( V_prime) ) , np.dot(Uf2, Uf1) ) 
    
    return (v, w, np.conj(v).T, np.dot(Uf2, Uf1), res)


def Generate_Long_Range_Floquet( args):

    L = args['L']
    T = args['T']
    alpha = args['alpha']
    J = args['J']
    Jx = args['Jx']
    epsilon = args['epsilon']
    hx = args['hx']
    hy = args['hy']
    hz = args['hz']

    fil = args['fil']
    
    print "\nLong_Range Floquet Function\n"
    # Hamiltonian
    ########## First Half of evolution
    # Precession term:

    print "Generate ZPrecession"
    Zprecession = np.zeros( (2**L, 2**L), dtype = 'complex')
    for i in range(L):
        Zprecession += SigmaTerms(sigmaz, L, [i])

    #print "Zprecession"
    #print np.diag(Zprecession)

    print "Generate XPrecession"
    Xprecession = np.zeros( (2**L, 2**L), dtype = 'complex')
    for i in range(L):
        Xprecession += SigmaTerms(sigmax, L, [i])

    print "Generate YPrecession"
    Yprecession = np.zeros( (2**L, 2**L), dtype = 'complex')
    for i in range(L):
        Yprecession += SigmaTerms(sigmay, L, [i])

    print "Generate XXPrecession"
    XXprec = np.zeros((2**L, 2**L), dtype = 'complex')
    # NO CYCLIC CONDITION
    for i in range(L-1):
        j = (i+1)
        XXprec += SigmaTerms(sigmax, L, [i,j])


    # CONSIDER ONLY NEAREST NEIGHBOURS
    # interactions = np.zeros((2**L, 2**L), dtype = 'complex')
    # for i in range(L):
    #     j = (i+1)%L
    #     interactions += SigmaTerms(sigmaz, L, [i,j]) * J 

    # CONSIDER LONG RANGE INTERACTION - No Cyclic conditions
    print "Generate Interactions"
    interactions = np.zeros((2**L, 2**L), dtype = 'complex')
    for i in range(L-1):
        for j in range(i+1, L):
            interactions += SigmaTerms(sigmaz, L, [i,j]) * J * 1/(j-i)**alpha #Ewald(L, j-i, alpha)
            #print Ewald(L, j-i, alpha)
    ########## Second Half of evolutions

    H2 = np.pi/2 * (1+epsilon) * Xprecession # Reuse calculation of sum sigma_x
    SingleSpin = iden * np.cos(np.pi/2 * (1+epsilon)) - 1j*sigmax * np.sin( np.pi/2 * (1+epsilon))

    Uf2 = SingleSpin
    for i in range(1,L):
        Uf2 = outer(SingleSpin, Uf2)
    
        
    H1 = interactions + hz*Zprecession + hx*Xprecession + hy*Yprecession + Jx * XXprec
    #print H1
    print "Compute Uf1"
    Uf1 = slinalg.expm2(- 1j*T* H1)
    
    HPrethermal = (interactions + hx*Xprecession + Jx * XXprec)/L
  
    (w,v) = np.linalg.eigh(HPrethermal)
    groundstate = np.argmin(w)
    print "Range of HPrethermal Hamiltonian: ", np.min(w), np.max(w)
    states = []
    states.append(v[:,groundstate])

    rang = np.max(w) - np.min(w)
    delta = 0.05
    Nstates = 20
    
    for i in range(1,Nstates):
        #print i
        for o in range(2**L):
            if abs(w[o]- np.min(w) - rang * float(i)/Nstates) < delta:
                states.append(v[:, o] )
                break
            
    FM = states[0] * 0
    FM[2**L-1] = 1    
    states.append(FM)

    for dom in range(1,5):
        size = L/dom
        counter =1
        l = 1
        index = 0
        for i in range(0, L):
            index += l*2**i
            if counter < size:
                l = 1 - l
                counter += 1
            else:
                counter = 1            
        temp = states[0] * 0
        temp[index] = 1
        states.append(temp)

    eigHPrethermal = w
    InfTemp = np.sum(eigHPrethermal)
        
    #print "Computing HPrethermal Eigenvalues"
    #eigHprethermal = np.linalg.eigvalsh( HPrethermal )

    #(w_t,v_t) = np.linalg.eig(Floquet)
    print "Computing Floquet Eigenvalues"
    Floquet = Uf2.dot(Uf1)
    (w, v) = EigenVectorsUnitary( Floquet )
    eigFloquet = w

    U = v
    Udag = np.conj(v).T
    Diag = np.array(w)


    print "Testing Decomposition"
    res = True
    test1 = np.allclose( Udag.dot(U) ,  np.eye(2**L))
    test = np.allclose(U.dot(np.diag(Diag) ).dot(Udag) , Floquet)
    res = res and test and test1


    Obs_0 = []
    O = SigmaTerms(sigmaz, L, [L/2])
    Obs_0.append( O )
    #Obs_0.append( SigmaTerms(sigmaz, L, [1]) )

    Obs_0.append(HPrethermal)

    
    preComputation = {'HPrethermal': HPrethermal,
                      'Floquet':     Floquet,
                      'U':           U,
                      'Udag':        Udag,
                      'Diag':        Diag,
                      'Obs':         Obs_0,
                      'res':         res,
                      'L':           L,
                      'InfTemp':     InfTemp,
                      'args':        args,
                      'eigHPrethermal': eigHPrethermal
                      }

    np.save('PreComp_'+fil, preComputation)
    print "Finished Precomputation"
    print "Saved to:"
    print 'PreComp_'+fil+'.npy'
        
    return (states, 'PreComp_'+fil+'.npy')

def Generate_Francisco( args):
    
    L = args['L']
    Omega = args['Omega']

    T = 2*np.pi / Omega

    J = args['J']
    U = args['U']
    
    A = args['A'] * Omega
    dA = args['dA'] * Omega

    
    print "\n Francisco\n"
    # Hamiltonian
    ########## First Half of evolution
    # Precession term:

    StaggeredZprecession = np.zeros( (2**L, 2**L), dtype = 'complex')
    for i in range(L):
        StaggeredZprecession += SigmaTerms(sigmaz, L, [i]) * (-1)**(i+1)

    print StaggeredZprecession
    print ""    
        
    XXprec = np.zeros((2**L, 2**L), dtype = 'complex')
    for i in range(L-1):
        j = (i+1)
        XXprec += SigmaTerms(sigmax, L, [i,j])

    YYprec = np.zeros((2**L, 2**L), dtype = 'complex')
    for i in range(L-1):
        j = (i+1)
        YYprec += SigmaTerms(sigmay, L, [i,j])

    print XXprec + YYprec
    print ""
        
    ZZprec = np.zeros((2**L, 2**L), dtype = 'complex')
    for i in range(L-1):
        j = (i+1)
        ZZprec += SigmaTerms(sigmaz, L, [i,j])

    print ZZprec
    print ""    
    LinearTerm = np.zeros((2**L, 2**L), dtype = 'complex')
    for i in range(L):       
        LinearTerm += SigmaTerms(sigmaz, L, [i]) *(i+1)
    print LinearTerm
    print ""

    H1 = -(J/2)*(XXprec + YYprec) + (U/4) * ZZprec - (Omega-A)/4 * StaggeredZprecession - (dA/2) * LinearTerm
    H2 = -(J/2)*(XXprec + YYprec) + (U/4) * ZZprec - (Omega+A)/4 * StaggeredZprecession + (dA/2) * LinearTerm

    print "H1"
    #print H1
    #print ""

    #plt.imshow( np.abs(H1) > 1e-10)
    #plt.show()

    #plt.imshow( np.abs(H2) > 1e-10)
    #plt.show()
    st = []
    for i in range(2**L):
        counter = 0
        temp = i
        while i > 0:
            counter += i%2
            i = i/2
        if counter == L/2:
            st.append(temp)

    print st
        
    
    print "ST: ",  len(st)
    for i in st:
        for o in range(2**L):
            if np.abs(H1[i][o]) > 1e-10 and ( not o in st):
                print "PROBLEM", i,o
    H1 = H1[st,:]
    H1 = H1[:, st]

    H2 = H2[st,:]
    H2 = H2[:,st]
    ## Only the columns of which are the sum of n different powers of two matter in the half-filling system.
      
   
    Uf1 = slinalg.expm2(- 1j*T/2 * H1)
    Uf2 = slinalg.expm2(- 1j*T/2 * H2)

    def xi (x):
        return 2*x/ np.pi * np.cos( np.pi * x / 2 ) / (1-x**2)

    print "##########\n"
    print xi( (A-dA)/Omega)
    print xi( (A+dA)/Omega)
    print ""
    
    XXYYOdd = np.zeros((2**L, 2**L), dtype = 'complex')
    XXYYEven = np.zeros((2**L, 2**L), dtype = 'complex')
    
    for i in range(0,L-1):
        j = i+1
        if (i+1) % 2 == 0:
            XXYYEven += SigmaTerms(sigmax, L, [i,j]) + SigmaTerms(sigmay, L, [i,j])
        else:
            XXYYOdd += SigmaTerms(sigmax, L, [i,j]) + SigmaTerms(sigmay, L, [i,j])

    
    HPrethermal = -xi( (A+dA)/Omega ) * (XXYYEven/2) - xi( (A-dA)/Omega ) * (XXYYOdd/2) + (U/J)*(ZZprec/4)
    HPrethermal = HPrethermal / L
    HPrethermal = HPrethermal[:, st]
    HPrethermal = HPrethermal[st, :]
    
    (w,v) = np.linalg.eigh(HPrethermal)
    groundstate = np.argmin(w)
    print "Range of HPrethermal Hamiltonian: ", np.min(w), np.max(w)
    states = []
    states.append(v[:,groundstate])

    rang = np.max(w) - np.min(w)
    delta = 0.1

    for i in range(1,6):
        #print i
        for o in range(2**L):
            if abs(w[o]- np.min(w) - rang * float(i)/6.0) < delta:
                states.append(v[:, o] )
                break
            
    #print len(states)
    
    
    
    # for i in range(2**L-1):
    #     groundstate[i] == 0
    # groundstate[2**L-1] = 1
    
    return (HPrethermal, w, Uf2.dot(Uf1), states, st) #HPrethermal acts over time T, H2, acts over a Delta term
    #if L <= 4:
    #    s = np.abs(slinalg.expm2(-1j * T * HPrethermal * J * L) - Uf1.dot(Uf2) ) 
    #    print np.max(np.max(s))
    #    plt.imshow( s )
    #    plt.show()
        
    #(w,v) = np.linalg.eigh(HPrethermal)

    #groundstate = np.argmin(w)
    #print groundstate
    #groundstate = v[:,groundstate]

    groundstate = np.zeros((1, 2**L))
    groundstate[2**L-1] = 1
    #statesInt = []
    #for i in range(2**L):
    #    if np.abs(groundstate[i]) > 1e-10:
    #        statesInt.append(i)
    #print statesInt
  
    return (HPrethermal, Uf2.dot(Uf1), groundstate) #HPrethermal acts over time T, H2, acts over a Delta term

    
def Compute_r(L, T, W, epsilon, eta, neighbour = False):

    if neighbour:
        Floquet = Generate_Floquet_NearestNeighbour(L,T,W,epsilon, eta)
    else:
        Floquet = Generate_Floquet_Operator(L,T,W,epsilon, eta)
        
    quasi = Eigenvalues(Floquet)
    
    #print "sort_quasi:\n", quasi
    delta = quasi[1:] - quasi[:-1]
    #print "Delta:\n", delta
    
    r = []
    for i in range(len(delta)-1):
        #print delta[i]
        #print delta[i+1]
        #print min( delta[i+1], delta[i]), max( delta[i+1], delta[i])
        if delta[i+1] == 0.0 and delta[i] ==0:
            continue
        r.append( min( delta[i+1], delta[i]) / max( delta[i+1], delta[i]))
    
    r = np.array(r)
    
    
    print "<r>: ", np.mean(r)

    return np.mean(r)
    

def Compute_Evolution(Nsteps, state, result_fil, preComputation_fil, evolution='single',  logEvo = False, MAX_COUNTER = 4, EIG_COUNTER = 10, SVD_Cutoff = 1e-4, output_folder = './'):

    preComputation = np.load(preComputation_fil)[()]
    #print preComputation
    HPrethermal = preComputation['HPrethermal']
    Floquet     = preComputation['Floquet']
    U           = preComputation['U']
    Udag        = preComputation['Udag']
    Diag        = preComputation['Diag']
    Obs_0       = preComputation['Obs']
    Obs_t       = Obs_0
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
            times.append(times[-1] + 1)
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
    np.save(output_folder + result_fil, output)
    
    return output
        
            
