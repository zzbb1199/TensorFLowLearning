
import numpy as np

a = np.array([2,2,1])
softmax_a = (np.exp(a)/np.sum(np.exp(a)))
print('softmax-a',softmax_a)
print('sum=',np.sum(softmax_a))