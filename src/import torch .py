import numpy as np
import torch
A= np.arange(6).reshape(1,2,3)
print(A)
print(np.rollaxis(A,2).shape)
