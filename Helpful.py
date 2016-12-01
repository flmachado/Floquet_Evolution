import numpy as np


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
