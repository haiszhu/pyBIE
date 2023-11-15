import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.gauss import gauss
from utils.cheby import cheby

def interpmat(n,m,qntype='G'):
#
#
#
    if m==n:
        P = np.eye(n)
    else:
        if qntype=='G':
            x, w1, D1 = gauss(n)
            y, w2, D2 = gauss(m)
        else:
            x, w1, D1 = cheby(n)
            y, w2, D2 = cheby(m)
        x = np.squeeze(x); y = np.squeeze(y)
        V = np.ones((n,n))
        for j in range(1,n):
            V[:,j] = V[:,j-1]*x  # Vandermonde, original nodes
        R = np.ones((m,n))
        for j in range(1,n):
            R[:,j] = R[:,j-1]*y # % monomial eval matrix @ y
        P = np.linalg.solve(V.transpose(),R.transpose()).transpose()

    return P


def test_interpmat():

    P = interpmat(3,6,'C')
    print(P)

if __name__ == '__main__':
    test_interpmat()
