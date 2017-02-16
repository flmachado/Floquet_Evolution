import numpy as np
import numpy.linalg as linalg


sigmax = np.array([[0,1],
                   [1,0]])
sigmay = np.array([[0,-1j],
                   [1j,0]])
sigmaz = np.array([[1,0],
                   [0,-1]])
iden   = np.array([[1,0],
                   [0,1]])


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


def Entropy(state, reduced, L):

    rho_size = 2**(L-reduced)
    red_size = 2**reduced

    #print np.abs(state)
    
    rho = np.zeros( (rho_size, rho_size), dtype='complex' )
    for i in range(rho_size):
        for o in range(rho_size):
            temp = 0
            for k in range(red_size):
                #print i,o,k
                #print (o<<reduced) + k
                temp +=  state[ (i << reduced) + k]*np.conj( state[(o<<reduced) + k] )
            rho[i,o] = temp
            
    eigval =linalg.eigvalsh(rho)
    ent = 0
    for i in eigval:
        ent -= i * np.log(i)
        #print i, ent

    return ent
    
    
