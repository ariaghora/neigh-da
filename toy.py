import numpy as np

def logsm(x):
    ex = np.exp(x-np.max(x, 1))
    return np.log(ex / ex.sum(1, keepdims=True))

a = np.array([[0.5, 0.5]])

logsm(a)

e = -(a * logsm(a))

print(e)