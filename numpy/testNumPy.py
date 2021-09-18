import numpy as np
import pandas as pd

A = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
print(type(A))
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
print('\n')

# column wise multiplication
M1 = np.array([[1, 2], [3, 4]])
M2 = np.column_stack((M1, np.multiply(M1[:, 0], M1[:, 1])))
print(M1)
print(M2)

# test rotation
print('/nRotation of the np.array:')
R1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('Before transformation: \n {0}'.format(R1))
R2 = np.rot90(R1, k=1)
print('After (left) transformation: \n {0}'.format(R2))
R3 = np.rot90(R1, k=-1)
print('After (right) transformation: \n {0}\n'.format(R3))

# assignment to a part of the matrix
print('\nAssignment to a part of the matrix:')
A1 = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [10, 11, 12, 13]])
print('Before assignment:\n{0}'.format(A1))
zeros = np.zeros((2, 2), dtype=int)
A1[1:3, 1:3] = zeros
print(A1)

# concatenate arrays
print('\nConcatenate arrays:')
Con1 = np.array(object=[[1, 2], [3, 4]])
print('Con1 = {}'.format(Con1))
Con2 = np.array(object=[[1], [3]])
print('Con2 = {}'.format(Con2))
# if axis = 0, then new rows will be added
Con3 = np.concatenate((Con1, Con2.transpose()), axis=0)
print('Con3 = {}'.format(Con3))
# if axis = 1, then new columns will be added
Con4 = np.concatenate((Con1, Con2), axis=1)
print('Con4 = {}'.format(Con4))
