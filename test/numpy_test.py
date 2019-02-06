import numpy as np

a1 = np.array([1.0, 2.0])
a2 = np.array([
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 2.0]
])
a3 = np.arange(24)
a4 = np.array([1, 2])
a4 = a4.reshape((1, 2))
a5 = np.array([[1, 2]])

a6 = np.array([[1], [2]])

print(a1)

print("\n")
print(a2)

print("\n")
print(a3)

print("\n")
print(a4)
print(a4.ndim)
print(a4.shape)

print("\n")
print(a5)
print(a5.ndim)
print(a5.shape)

print("\n")
print(a6)
print(a6.shape)

print('\n')
a = np.array([[1]])
print(a.ndim, '\n', a.shape)
print('\n')
print(a2[1:3])
