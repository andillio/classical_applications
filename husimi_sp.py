import numpy as np
import scipy.integrate as sp
import numpy as np 


# x has to be defined from -L/2, L/2 or this won't work
def K_H(x, x_, u, hbar, mpart, sig_x, L, full = False):
	X, X_, U = np.meshgrid(x, x_, u) # x is on axis 1, x_ is on axis 0, u is on axis 2

	X_[X_ - X > L/2] -= L # shift x_ so it is centered at a given x value
	X_[X_ - X < -L/2] += L

	arg = (-(X - X_)**2)/(4.*sig_x**2) + 0j
	arg -= U*X_*mpart*(1.j/hbar)
	denom = np.sqrt(2.*np.pi*hbar/mpart)
	denom *= (2.*np.pi*sig_x**2)**(.25)
	if full:
		return np.exp(arg)/denom, X, X_, U
	return np.exp(arg)/denom

# x has to be defined from -L/2, L/2 or this won't work
def K_H_(x, x_, u, hbar_, sig_x, L, full = False):
	X, X_, U = np.meshgrid(x, x_, u) # x is on axis 1, x_ is on axis 0, u is on axis 2

	X_[X_ - X > L/2] -= L # shift x_ so it is centered at a given x value
	X_[X_ - X < -L/2] += L

	arg = (-(X - X_)**2)/(4.*sig_x**2) + 0j
	arg -= U*X_*(1.j/hbar_)
	denom = np.sqrt(2.*np.pi*hbar_)
	denom *= (2.*np.pi*sig_x**2)**(.25)
	if full:
		return np.exp(arg)/denom, X, X_, U
	return np.exp(arg)/denom


def f_H(psi, K, dx, shift = False):
	f = (np.abs(psi_H(psi, K, dx))**2).transpose()
	if shift:
		return np.fft.fftshift(f,1)
	return f


def psi_H(psi, K, dx):
	integrand = psi[:, None, None]*K
	return dx*integrand.sum(axis = 0)
#	return dx*sp.simps(integrand, axis = 0)

