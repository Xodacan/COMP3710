import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Y, X = np.mgrid[-1.5:1.5:0.005, -1.5:1.5:0.005]

x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y) #important! 
zs = z.clone() #Updated!
ns = torch.zeros_like(z)

# transfer to the GPU device
z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

c = torch.tensor(0.5 + 0.5j, dtype=torch.complex64).to(device)

#Mandelbrot Set
for i in range(500):
    #Compute the new values of z: z^2 + x 
    zs_ = zs*zs + c
    #Have we diverged with this new value? 
    not_diverged = torch.abs(zs_) < 4.0 
    #Update variables to compute
    ns += not_diverged
    zs = zs_

fig = plt.figure(figsize=(16,10))

def processFractal(a):
        """Display an array of iteration counts as a
             colorful picture of a fractal."""
        a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
        img = np.concatenate([10+20*np.cos(a_cyclic),
        30+50*np.sin(a_cyclic),
        155-80*np.cos(a_cyclic)], 2)
        img[a==a.max()] = 0
        a = img
        a = np.uint8(np.clip(a, 0, 255))
        return a

plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()