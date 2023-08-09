import torch
import math
import numpy as np 
import matplotlib.pyplot as plt

'''print("Pytorch Version:", torch.__version__)
print(device)'''

#figuring out what the device is cuda or the cpu - 
# if cuda is present it will be the gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#set a grid for computing image
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

#load into pytorch
x = torch.Tensor(X)
y = torch.Tensor(Y)

#transfer the process to the gpu
x = x.to(device)
y = y.to(device)

#get a gaussian function
a = 1
pi = math.pi
f = 0.001
phase = 180

z = a * torch.sin((2.0 * pi * f * (x+y)) + phase) * torch.exp(-((x**2)+(y**2))/2.0)    

plt.imshow(z.cpu().numpy())
plt.tight_layout()
plt.show()

