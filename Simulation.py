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

    #print "\n Error "
    
    CO = np.linalg.eigvalsh( (mat + np.conj( mat ).T)/2)
    SI = np.linalg.eigvalsh( (mat - np.conj( mat ).T)/2j ) 

    #print CO
    #print SI

    #print ""
    
    l = np.array( np.sort( CO**2 ) )
    k = np.array( np.sort( SI**2 ) )

    #print l
    #print k
    
    ab = l + k[::-1]

    #print ab

    #print ""
    
    if min(ab) < 1 - err:
        return True
    else:
        return False


def Eigenvalues(Floquet):

    eig = np.linalg.eig(Floquet)[0]
    return np.sort(np.angle(-eig))


    # #print np.dot(Floquet, np.conj(Floquet).T) #Checks out
    # print "FLOQUET"
    # print Floquet
    # print ""
        
    # (w,v) =  np.linalg.eigh(Floquet + np.conj(Floquet).T )
    # (t1,temp) =  np.linalg.eig(Floquet)
    # print "Eigenvalues: ", t1
    # ## Eigenvectors satisfy dot(a, v[:, i]) = w[i] * v[:, i] which means that the COLUMNS are the eigenvectors, hence us transposing
    # v = v.T

    # w.sort()
    # t2 = np.real(t1)
    # t12.sort()

    # print "w_eigh ", w
    # print "w_eig ", 2*t2
    # print "dw ", w - 2*t2
    
    # if np.max(np.abs(np.sort(w) - np.sort(2*t2))) > 1e-2:
    #     print "STOP NOT MATCHING"
    #     print "STOP NOT MATCHING"
    #     print "STOP NOT MATCHING"
    #     print "STOP NOT MATCHING"

    # print ""
    # spect = []
    # for i in range(len(v)):
    #     print "Norm of vector: ", np.dot(np.conj(v[i]).T, v[i])
    #     k = np.dot( np.conj(v[i]).T , np.dot(Floquet, v[i])  )
    #     print "Eigenvalue: ", k
    #     print "Eigenvalue Norm: ", np.abs(k)
    #     print ""
    #     spect.append(k)
        
            
    # spect = np.array(spect, dtype = 'complex')
    
    # print spect
    # print t1
    # print "\n\n"
    
    # tquasi = -np.angle(t1)
    # tquasi.sort()
    # tz = np.real( 1j* np.log(t1))
    # tz.sort()
    # print tz
    # quasi = -np.angle(spect)
    # quasi.sort()

    # print "Q1: ", tquasi
    # print "Q2: ",quasi
    # print "dQ: ",tquasi - quasi
    # if max(np.abs(tquasi - quasi)) > 1e-7:
    #     print "STOP", "Eigenvalues are not matching"
    #     print "STOP", "Eigenvalues are not matching"
    #     print "STOP", "Eigenvalues are not matching"
    #     print "STOP", "Eigenvalues are not matching"
    #     return 0
    
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
    (w,v) = np.linalg.eig(np.dot(Uf2, Uf1))

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
    
    print "\nLong_Range Floquet Function\n"
    # Hamiltonian
    ########## First Half of evolution
    # Precession term:

    Zprecession = np.zeros( (2**L, 2**L), dtype = 'complex')
    for i in range(L):
        Zprecession += SigmaTerms(sigmaz, L, [i])

    #print "Zprecession"
    #print np.diag(Zprecession)

    Xprecession = np.zeros( (2**L, 2**L), dtype = 'complex')
    for i in range(L):
        Xprecession += SigmaTerms(sigmax, L, [i])
        
    Yprecession = np.zeros( (2**L, 2**L), dtype = 'complex')
    for i in range(L):
        Yprecession += SigmaTerms(sigmay, L, [i])

    
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
    Uf1 = slinalg.expm2(- 1j*T* H1)
    
    HPrethermal = (interactions + hx*Xprecession + Jx * XXprec)/L

    (w,v) = np.linalg.eigh(HPrethermal)
    groundstate = np.argmin(w)
    groundstate = v[:,groundstate]

    # for i in range(2**L-1):
    #     groundstate[i] == 0
    # groundstate[2**L-1] = 1
    
    return (HPrethermal, Uf2.dot(Uf1), groundstate) #HPrethermal acts over time T, H2, acts over a Delta term


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
    

def Compute_Evolution(L, Floq, args, Nsteps, state, fil = 'test.out', evolution='single',  logEvo = False, MAX_COUNTER = 4, EIG_COUNTER = 10, SVD_Cutoff = 1e-4, output_folder = './'):

    (HPrethermal, Floquet, groundstate) = Floq( args )
    InfTemp = np.trace(HPrethermal)
    #print Floquet

    # (w,v) = np.linalg.eig(Floquet)

    # U = v
    # Udag = np.conj(v).T
    # Diag = w

    # angles = np.angle(w)
    # print angles
    # fil = 'J=%.2f    omega=%.2f   alpha=%.2f  hx=%.2f hy=%.2f hz=%.2f' % (args['J'], 1.0/args['T'], args['alpha'], args['hx'], args['hy'], args['hz']) 
    # plt.title(fil)
    # n, bins, patches = plt.hist(angles, 50, normed=1, range=[-3.5, 3.5])
    # plt.ylabel('Normalized Histrogram')
    # plt.xlabel('Complex Phase')
    # plt.savefig(fil + '.png')
    # plt.show()
    # return 0

    
    # res = True
    # test = np.allclose( Udag.dot(U) ,  np.eye(2**L))
    # res = res and test
    # print "Unitary: ", test
    # test = np.allclose(U.dot(np.diag(w)).dot(Udag) , Floquet)
    # res = res and test
    # print "Decomp: ", test
    # test = np.allclose(U.dot( np.diag(w**2)).dot(Udag) , Floquet.dot(Floquet))
    # res = res and test
    # print "Decomp2: ", test
    # test = np.allclose(U.dot( np.diag(w**3)).dot(Udag) , Floquet.dot(Floquet).dot(Floquet))
    # res = res and test
    # print "Decomp3: ", test

    res = False

    #print "Floquet"
    #print Floquet
    #print ""
    #print ""

    # If single        Compute a random spin in a random physical state
    # If all           Compute all the spins in a random physical state
    # If magnetization Compute only \sum_{i} \sigma_i^z

    Obs_0 = []
    Obs_t = []
    alpha = []
    energy_gs = []
    energy_FM = []
    energy_AM = []
    if res:
        print "Using Matrix Diagonalization"
    else:
        print "Using power of matrix"
    

    
    print "Looking at spins 3 and 6"
    
    Obs_0.append( SigmaTerms(sigmaz, L, [2]) )
    Obs_0.append( SigmaTerms(sigmaz, L, [5]) )

    for i in Obs_0:
        Obs_t.append( i )

    Obs_0.append(HPrethermal)
    Obs_t.append(HPrethermal)

    values_gs = []
    values_FM = []
    values_AM = []

    FM = groundstate*0
    FM[2**L-1] = 1

    AM = groundstate*0
    index = 0
    for i in range(0, L-1, 2):
        index += 2**i
    AM[index] = 1

   

    fft = []
    for i in range(len(Obs_0)-1):
        #print np.shape(Obs_0[i])
        #print Obs_0[i][state,state]
        
        values_gs.append( [ groundstate.dot(Obs_0[i]).dot(Obs_0[i]).dot(groundstate)[0,0] ] ) 
        values_FM.append( [ AM.dot(Obs_0[i]).dot(Obs_0[i]).dot(AM)[0,0] ] ) 
        values_AM.append( [ FM.dot(Obs_0[i]).dot(Obs_0[i]).dot(FM)[0,0] ] ) 
        fft.append([1])
        
    energy_gs.append(  groundstate.dot(HPrethermal).dot(groundstate) ) 
    energy_FM.append(  AM.dot(HPrethermal).dot(AM) ) 
    energy_AM.append(  FM.dot(HPrethermal).dot(FM) )
    
    c = 0
    cc = 1

    FloquetEvo = Floquet
    times = [0]
    counter = 0
    eigCounter = 0
    dt = 1
    while times[-1] < Nsteps:
        # Evolve the different observables:
        for i in range(len(Obs_0)):
            #print i
            if logEvo:
                if res == True:
                    prod = U.dot(np.diag(np.conj(Diag))).dot(alpha[i]).dot( np.diag(Diag) ).dot(Udag)  # state Spin (finite energy)
                    if i < len(Obs_0) - 1:
                        values[i].append( groundstate.dot( prod ).dot(groundstate) )
                    elif i == len(Obs_0) - 1:
                        energy.append(  groundstate.dot( prod ).dot(groundstate) )
                        #energy.append( prod[state,state])
                        
                
                
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

                        
                   
                    if i < len(Obs_0) - 1:
                        values_gs[i].append( (groundstate.dot( prod ).dot(Obs_0[i]).dot(groundstate) )[0,0] )
                        values_AM[i].append( (AM.dot( prod ).dot(Obs_0[i]).dot(AM) )[0,0] )
                        values_FM[i].append( (FM.dot( prod ).dot(Obs_0[i]).dot(FM) )[0,0] )

                    elif i == len(Obs_0) - 1:
                        energy_gs.append( (groundstate.dot( prod ).dot(groundstate))[0,0] )
                        energy_AM.append( (AM.dot( prod ).dot(AM))[0,0] )
                        energy_FM.append( (FM.dot( prod ).dot(FM))[0,0] )

                        
            else:
                if res:
                    prod = U.dot(np.diag(np.conj(Diag))).dot(alpha[i]).dot( np.diag(Diag) ).Udag  # state Spin (finite energy)
                    values[i].append(prod[state,state])

            
                else:
                    prod = np.conj(Floquet).T.dot(Obs_t[i]).dot(Floquet)
                    Obs_t[i] = prod
                    if i < len(Obs_0) - 1:
                        values_gs[i].append( (groundstate.dot( prod ).dot(Obs_0[i]).dot(groundstate) )[0,0] )
                        values_AM[i].append( (AM.dot( prod ).dot(Obs_0[i]).dot(AM) )[0,0] )
                        values_FM[i].append( (FM.dot( prod ).dot(Obs_0[i]).dot(FM) )[0,0] )
                    elif i == len(Obs_0) - 1:
                        energy_gs.append( (groundstate.dot( prod ).dot(groundstate))[0,0] )
                        energy_AM.append( (AM.dot( prod ).dot(AM))[0,0] )
                        energy_FM.append( (FM.dot( prod ).dot(FM))[0,0] )

                if i < len(Obs_0) - 1:
                    fft[i].append( np.fft.fft(values_FM[i]))
        if logEvo:
            #print ""
            #print "dt: ",dt
            #print times[-1]
            #print counter
            times.append( times[-1] + dt )
            if res:
                if counter < MAX_COUNTER:
                    counter += 1
                else:
                    Diag = Diag * Diag
                    Diag = Diag / np.abs(Diag)
                    dt *=2
                    counter = 0 
            else:
                if counter < MAX_COUNTER:
                    counter += 1
                else:
                    FloquetEvo = FloquetEvo.dot(FloquetEvo)
                    dt *=2
                    counter = 0
                    
        else:
            times.append(times[-1] + 1)
            if res:
                Diag = Diag * w
                Diag = Diag/np.abs(Diag)


                
    #for ob in Obs_t:
    #    print ""
    #    print ""
    #    print ob

        
    times = np.array(times[:])
    values_gs = np.array(values_gs)
    values_FM = np.array(values_FM)
    values_AM = np.array(values_AM)

    if  np.max(np.abs( np.imag( energy_AM) ) )> 1e-14 or np.max(np.abs( np.imag( energy_FM) ) )> 1e-14:
        print "TOO MUCH IMAGINARY PART"
        print np.max(np.abs( np.imag( energy_AM) ) )
        print np.max(np.abs( np.imag( energy_FM) ) )
    energy_gs = np.array(np.real(energy_gs) )
    energy_FM = np.array(np.real(energy_FM) )
    energy_AM = np.array(np.real(energy_AM) )

    info = {'L': L,
            'args': args,
            'Floquet': Floq.__name__,
            'state': state,
            'Nsteps': Nsteps,
            'fil': fil,
            'evolution': evolution,
            'logEvo': logEvo,
            'InfTemp': InfTemp,
            'SVD_Cutoff': SVD_Cutoff}
            
    output = {'values_gs': values_gs,
              'values_FM': values_FM,
              'values_AM': values_AM,
              'energy_gs': energy_gs,
              'energy_FM': energy_FM,
              'energy_AM': energy_AM,
              'times': times, 'info': info,
              'fft': fft}
    print ""
    print fil + ".npy"
    np.save(output_folder + fil, output)
    
    return output
        
            
