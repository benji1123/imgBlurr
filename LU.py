import numpy as np
import sys, time


'''
Stack Overflow: "Python Progress Bar"
by user: eusoubrasileiro
'''
def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


'''
LU Decomposition
Returns (L, U)
'''
def lu(A):
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    N = A.shape[0]

    for i in progressbar(range(N), "LU Decomp: ", 50):
        U[i, i] = (A[i, i] - np.dot(L[i, :i], U[:i, i]))
        for k in range(i+1, N):
            U[i,k] = A[i,k] - np.dot(L[i,:i], U[:i,k])
        # diagonals are '1'
        L[i, i] = 1
        for k in range(i+1, N):
            L[k, i] = (A[k, i]-np.dot(L[k,:i], U[:i,i])) / U[i,i]
    return L, U


'''
Solves Y
Y = L mask^{-1}
'''
def solve_Y(L, b):
    y = np.zeros_like(b)
    for i in range(b.shape[0]):
        s = np.dot(L[i, :i], y[:i])
        y[i] = b[i] - s
    return y


'''
Solves mask^{-1}
'''
def solve_X(U, y):
    x = np.zeros_like(y)
    for i in range(x.shape[0], 0, -1):
      x[i-1] = (y[i-1] - np.dot(U[i-1, i:], x[i:])) / U[i-1, i-1]
    return x


'''
Solves Ax = B
given A, B
'''
def solve(A, b):
    L, U = lu(A)
    y = solve_Y(L,b)
    x = solve_X(U,y)
    return x