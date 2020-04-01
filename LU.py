import numpy as np
from scipy.linalg import solve

def sum(A, L, U, i, k):
    sum = 0
    for j in range(i):
        sum += np.dot(L[i,j], U[j,k])
    return sum


def lu(A):
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    N = A.shape[0]

    for i in range(N):
        L[i, i] = 1
        for k in range(i, N):
            U[i,k] = A[i,k] - np.dot(L[i,:i], U[:i,k])
        for k in range(i+1, N):
            L[k, i] = (A[k, i]-np.dot(L[k,:i], U[:i,i])) / U[i,i]
    return L, U


def inv(mask):
    L, U = lu(mask)
    I = np.identity(mask.shape[0])
    Y = solve(L, I)
    mask_inv = solve(U, Y)
    return mask_inv


# Sources
# https://tutorialspoint.dev/computer-science/engineering-mathematics/doolittle-algorithm-lu-decomposition
