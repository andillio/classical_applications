import numpy as np 
import scipy.fftpack as sp 
import matplotlib.pyplot as plt

#def getKernel(x,s, hbar_):
#	Y, X = np.meshgrid(x,x)
#	g = np.exp(-(X/(2*s))**2)*np.exp(-(Y/(2*s))**2)
#	g /= (2*np.pi*hbar_)*(2*np.pi*s**2)**(1/2.)
#	return sp.fft2(sp.fftshift(g))

def getKernel(kx, s, hbar_):
    KY, KX = np.meshgrid(kx, kx)
    ax = 1./(2.*s**2)

    g = np.exp(-KX**2 / (4*ax))*np.exp(-KY**2 / (4*ax))
    #g /= (2*np.pi*hbar_)*(2*np.pi*s**2)**(1/2.)*(ax/np.pi/np.sqrt(2.))
    g /= hbar_*np.sqrt(np.pi)/s # really unclear why this is the normalization constant... but it is
    return g



def f_H(rho, K, x, u, hbar_, L, shift = False):
    Y, X = np.meshgrid(x,x)
    f = np.zeros((len(x),len(u)))

    for i in range(len(u)):
        u_i = u[i]
        dR = -(Y-X) # this may be negative
        dR[dR > L/2] -= L
        dR[dR < -L/2] += L
        K_u = np.exp(((dR)*u_i/hbar_) *1j)
        Ff = sp.fft2(sp.fftshift(K_u*rho))
        f[:,i] = np.diagonal((sp.ifft2(Ff*K)))
    if shift:
        return np.fft.fftshift(f.transpose(), 1)
    return f.transpose() # not sure if it supposed to be fftshifted


