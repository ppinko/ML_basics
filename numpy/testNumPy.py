import numpy as np
import pandas as pd

A = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
B = np.array([[1, 2], [3, 4], [5, 6]])
print('Size of A = {0}'.format(A.shape))
print('Size of B = {0}'.format(B.shape))

# C = A * B
C = np.dot(A, B)
print('Size of C = {0}'.format(C.shape))
print(C)

D = np.array([[1, 2], [3, 4]])
D2 = np.array([[1, 2], [3, 4]])
print(D)
D = np.power(D, 2)
print(D)
D2 = np.subtract(D, D2)
print(D2)
E = D.sum()
print(E)
F1 = np.array([[1, 2], [3, 4]])
F2 = np.linalg.inv(F1)
print(F2)
F3 = np.dot(F1, F2)
print(F3)

G = np.array([[1, 2, 3]])
print(G)
print(G.shape)
