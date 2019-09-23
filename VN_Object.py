import numpy as np 
import scipy.fftpack as sp
import utils as u
import husimi as hu
import os

# 1D Von Neumann solver
class VN1D(object):

	def __init__(self, rho = False, K = False, N = 256, C = 2e-6, f = 5, kernel = True,
		Mtot = 1., L = 1., hbar_ = 1e-6, mpart = 4e-6, hbar = 4e-12, dt = 1e-3, label = 'g'):

		self.label = label

		self.N = N # grid cells

		self.hbar_ = hbar_
		self.hbar = hbar
		self.mpart = mpart

		# density matrix, NxN complex numbers, diagonal sum is 1 NOT Mtot
		if rho:
			self.rho = rho
		else:
			self.rho = np.zeros((N,N)) + 0j 

		self.C = C # poisson constant

		self.Mtot = Mtot # total mass

		self.L = L # side length

		self.dx = L/N

		self.x = self.dx*(.5+np.arange(-self.N/2, self.N/2))

		self.sig_x = self.dx*f

		self.T_scale = np.pi*2./np.sqrt(np.abs(C*Mtot/L)) # time scaling factor

		self.kx = 2*np.pi*sp.fftfreq(N, d = L/N)
		Kj,Ki = np.meshgrid(-self.kx,self.kx)
		dK2 = Ki**2 - Kj**2
		self.HK = np.exp(self.hbar_*dK2*dt*self.T_scale/4. *1.0j) #kinetic term for Hamiltonian

		self.u = hbar_*sp.fftshift(self.kx)
		self.du = self.u[1] - self.u[0]

		self.omega0 = np.sqrt(np.abs(self.C*self.Mtot/self.L))
		self.u_c = self.L*self.omega0/(2.*np.pi)

		self.dt = dt

		self.K = None
		if kernel:
			if K is None:
				self.setKernel()
			else:
				self.refer2Kernel(K) # Husimi kernel


	def setKernel(self):
		self.K = hu.getKernel(self.kx, self.sig_x, self.hbar_)

	def refer2Kernel(self, K):
		self.K = K

	def getRhoX(self):
		return self.Mtot*(np.abs(np.diagonal(self.rho)))

	def getRhoU(self):
		rho_u = np.diagonal(sp.ifft(sp.fft(self.rho, axis = 1),axis=0))
		return self.Mtot*u.normalize(rho_u, self.du)

	def getPS(self):
		#rho = u.makeOverDensity(u.normalize((np.abs(np.diagonal(self.rho))), self.dx))
		#rho = u.makeOverDensity( u.normalize( self.getRhoX(), self.dx ) )
		rho = self.getRhoX()
		return self.kx[self.kx>0], np.abs(sp.fft(rho)[self.kx>0])**2

	def getPhaseSpace(self, shift = 1):
		return hu.f_H(self.rho, self.K, self.x, self.u, self.hbar_, self.L)

	def getNorm(self):
		return ((np.abs(np.diagonal(self.rho))).sum()*self.dx)

	def getPotential(self):
		rho_r = self.getRhoX()
		return u.compute_phi(rho_r, self.C, self.dx, self.kx)

	def getMaxE(self):
		return np.max(u.a_from_Phi(self.getPotential()))

	def setThermalDistr(self, Sa, v_th, lam0 = 1., dn = 0., v_bump = 0.):
		v = self.hbar_*4*np.pi*np.pi*np.array(Sa)/(lam0*self.L*np.sqrt(self.C))
		w_pre = np.exp(-(v/v_th)**2 + 0j) + dn*np.exp(-((v - v_bump)/v_th)**2 + 0j)
		w_ = u.normalize(w_pre,1)

		for i in range(len(Sa)):
			phi = self.x*self.hbar_*np.pi*2.*Sa[i]/self.L
			psi_ = np.exp(phi/self.hbar_*1.j)
			self.rho += w_[i]*np.outer(np.conj(psi_),psi_)

		self.rho = self.rho/((np.abs(np.diagonal(self.rho))).sum()*self.dx)

	def setColdDistr(self):
		psi_ = np.array(np.ones(self.N))
		self.rho = np.outer(np.conj(psi_),psi_)
		self.rho = self.rho/((np.abs(np.diagonal(self.rho))).sum()*self.dx)

	def addVelocityPerturb(self, lam, v0, sin = False):
		phi = np.zeros(self.N)
		for i in range(len(lam)):
			phi += v0*self.u_c*lam[i] / (2*np.pi) *np.cos(2*np.pi*self.x/lam[i]  + np.pi/2.*sin)/len(lam)
		psi_ = np.exp(phi/self.hbar_*1.j)
		self.rho = self.rho*np.outer(np.conj(psi_),psi_)
		self.rho = self.rho/((np.abs(np.diagonal(self.rho))).sum()*self.dx)

	def addSpatialPerturb(self, lam, delta0, sin = False):
		psi_ = 1.
		n_lam = len(lam)
		for i in range(n_lam):
			psi_ += delta0*np.sin(2.*np.pi*self.x/(lam[i]*self.L) + np.pi/2.*sin)/n_lam
		psi_ = np.sqrt(psi_)
		self.rho = self.rho*np.outer(np.conj(psi_),psi_)
		self.rho = self.rho/((np.abs(np.diagonal(self.rho))).sum()*self.dx)
	

	def update(self, dt_, iters):
		for j in range(iters):
			self.rho = sp.ifft(sp.fft(self.rho, axis = 1),axis=0)
			self.rho *= self.HK
			self.rho = sp.fft(sp.ifft(self.rho, axis = 1),axis=0)

			phi_r = self.getPotential()
			Vi, Vj = np.meshgrid(phi_r, phi_r)
			dV = Vi - Vj
			self.rho *= np.exp(-1j*dt_*self.T_scale*dV/(self.hbar_))

			self.rho = sp.ifft(sp.fft(self.rho, axis = 1),axis=0)
			self.rho *= self.HK
			self.rho = sp.fft(sp.ifft(self.rho, axis = 1),axis=0)

	def readyDir(self,ofile):
		u.readyDir(ofile, "rho")

	def dropData(self,i,ofile_):
		np.save("../" + ofile_ + "/rho/" + "drop" + str(i) + ".npy", self.rho)




# 2D Von Neumann solver
class VN2D(object):

	def __init__(self, rho = False, N = 256, C = 2e-6,
		Mtot = 1., L = 1., hbar_ = 1e-6, mpart = 4e-6, hbar = 4e-12, dt = 1e-3, label = 'g'):

		self.label = label

		self.N = N # grid cells

		self.hbar_ = hbar_
		self.hbar = hbar
		self.mpart = mpart

		# density matrix, NxN complex numbers, diagonal sum is 1 NOT Mtot
		if rho:
			self.rho = rho
		else:
			self.rho = np.zeros((N,N)) + 0j 

		self.C = C # poisson constant

		self.Mtot = Mtot # total mass

		self.L = L # side length

		self.dx = L/N

		self.x = self.dx*(.5+np.arange(-self.N/2, self.N/2))

		self.T_scale = np.pi*2./np.sqrt(np.abs(C*Mtot/L)) # time scaling factor

		kx = 2*np.pi*sp.fftfreq(N, d = L/N)
		self.kx, self.ky = np.meshgrid(kx,kx.copy())
		# wtf do i do here? one for x and one for y then outer product?
		Kj,Ki = np.meshgrid(-kx,kx)
		dK2 = Ki**2 - Kj**2
		self.HK = np.exp(self.hbar_*dK2*dt*self.T_scale/4. *1.0j) #kinetic term for Hamiltonian
		self.HK = np.einsum("ab,cd->abcd", self.HK, self.HK.copy())

		self.u = hbar_*sp.fftshift(self.kx)
		self.du = self.u[1] - self.u[0]

		self.omega0 = np.sqrt(np.abs(self.C*self.Mtot/self.L))
		self.u_c = self.L*self.omega0/(2.*np.pi)

		self.dt = dt

	def getRhoX(self):
		return self.Mtot*np.abs(np.einsum("aacc->ac", self.rho))

	def getRhoU(self):
		rho_u = sp.ifft(sp.fft(sp.ifft(sp.fft(
			self.rho, axis = 3), axis = 2), axis = 1),axis=0)
		return sp.fftshift(self.Mtot*np.abs(np.einsum("aacc->ac", rho_u)))

	def getNorm(self):
		return ((np.abs(np.einsum("aacc->ac", self.rho))).sum()*self.dx)

	def getPotential(self):
		rho_r = self.getRhoX()
		return u.compute_phi2D(rho_r, self.C, self.dx, self.kx, self.ky)

	def setThermalDistr(self, Sa, v_th, lam0 = 1., dn = 0., v_bump = 0.):
		v = self.hbar_*4*np.pi*np.pi*np.array(Sa)/(lam0*self.L*np.sqrt(self.C))
		w_pre = np.exp(-(v/v_th)**2 + 0j) + dn*np.exp(-((v - v_bump)/v_th)**2 + 0j)
		w_ = u.normalize(w_pre,1)
		N = self.N

		x_ = self.dx*(.5+np.arange(-self.N/2, self.N/2))
		rhoX = np.zeros((N,N)) + 0j
		for i in range(len(Sa)):
			phi = x_*self.hbar_*np.pi*2.*Sa[i]/self.L
			psi_ = np.exp(phi/self.hbar_*1.j)
			rhoX += w_[i]*np.outer(np.conj(psi_),psi_)
		rhoY = rhoX.copy()
		self.rho = np.einsum("ab,cd->abcd", rhoX, rhoY)

		self.rho = self.rho/self.getNorm()


	def setColdDistr(self):
		self.rho = np.ones((self.N, self.N, self.N, self.N))
		self.rho = self.rho/self.getNorm()


	def addVelocityPerturb(self, lamX, lamY, v0, sin = False):
		phiX = np.zeros(self.N) +0j
		phiY = np.zeros(self.N) +0j
		x_ = self.dx*(.5+np.arange(-self.N/2, self.N/2))
		for i in range(len(lamX)):
			phiX += v0*self.u_c*lamX[i] / (2*np.pi) *np.cos(2*np.pi*x_/lamX[i]  + np.pi/2.*sin)/len(lamX)
		for i in range(len(lamY)):
			phiY += v0*self.u_c*lamY[i] / (2*np.pi) *np.cos(2*np.pi*x_/lamY[i]  + np.pi/2.*sin)/len(lamY)
		psiX_ = np.exp(phiX/self.hbar_*1.j)
		rhoX = np.outer(np.conj(psiX_),psiX_)
		psiY_ = np.exp(phiY/self.hbar_*1.j)
		rhoY = np.outer(np.conj(psiY_),psiY_)

		self.rho = self.rho*np.einsum("ab,cd->abcd", rhoX, rhoY)
		self.rho = self.rho/self.getNorm()

	def addSpatialPerturb(self, lamx, lamy, delta0, sin = False):
		psiX_ = 1.
		psiY_ = 1.
		n_lamx, n_lamy = len(lamx), len(lamy)
		x_ = self.dx*(.5+np.arange(-self.N/2, self.N/2))
		for i in range(n_lamx):
			psiX_ += delta0*np.sin(2.*np.pi*x_/(lamx[i]*self.L) + np.pi/2.*sin)/n_lamx
		for i in range(n_lamy):
			psiY_ += delta0*np.sin(2.*np.pi*x_/(lamy[i]*self.L) + np.pi/2.*sin)/n_lamy
		psiX_ = np.sqrt(psiX_)
		rhoX = np.outer(np.conj(psiX_),psiX_)
		psiY_ = np.sqrt(psiY_)
		rhoY = np.outer(np.conj(psiY_),psiY_)

		self.rho = self.rho*np.einsum("ab,cd->abcd", rhoX, rhoY)
		self.rho = self.rho/self.getNorm()
	
	def update(self, dt_, iters):
		for j in range(iters):
			self.rho = sp.ifft(sp.fft(sp.ifft(sp.fft(
				self.rho, axis = 3), axis = 2), axis = 1),axis=0)
			self.rho *= self.HK
			self.rho = sp.fft(sp.ifft(sp.fft(sp.ifft(
				self.rho, axis = 3), axis = 2), axis = 1),axis=0)

			phi_r = self.getPotential()
			ones = np.ones((self.N,self.N))
			V1 = np.einsum("il,jm->ijlm", phi_r, ones)
			V2 = np.einsum("jm,il->ijlm", phi_r, ones)
			dV = V1 - V2
			self.rho *= np.exp(-1j*dt_*self.T_scale*dV/(self.hbar_))

			self.rho = sp.ifft(sp.fft(sp.ifft(sp.fft(
				self.rho, axis = 3), axis = 2), axis = 1),axis=0)
			self.rho *= self.HK
			self.rho = sp.fft(sp.ifft(sp.fft(sp.ifft(
				self.rho, axis = 3), axis = 2), axis = 1),axis=0)

	def readyDir(self,ofile):
		u.readyDir(ofile, "rho")

	def dropData(self,i,ofile_):
		np.save("../" + ofile_ + "/rho/" + "drop" + str(i) + ".npy", self.rho)
