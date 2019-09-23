import numpy as np 
import scipy.fftpack as sp
import utils as u
import husimi_sp as hu
import os

# 1D schroedinger poisson solver
class SP1D(object):

	def __init__(self, psi = False, K = False, N = 256, C = 2e-6, f = 5,
		Mtot = 1., L = 1., hbar_ = 1e-6, mpart = 4e-6, hbar = 4e-12, label = 'b'):

		self.N = N # grid cells

		self.label = label

		self.hbar_ = hbar_
		self.hbar = hbar
		self.mpart = mpart

		# wavefunction, N complex numbers, square norm sum is 1 NOT Mtot
		if psi:
			self.psi = psi
		else:
			self.psi = np.zeros(N) + 0j 

		self.C = C # poisson constant

		self.Mtot = Mtot # total mass

		self.L = L # side length

		self.dx = L/N

		self.x = self.dx*(.5+np.arange(-self.N/2, self.N/2))

		self.sig_x = self.dx*f

		self.T_scale = np.pi*2./np.sqrt(np.abs(C*Mtot/L)) # time scaling factor

		self.kx = 2*np.pi*sp.fftfreq(N, d = L/N)

		self.u = hbar_*sp.fftshift(self.kx)
		self.du = self.u[1] - self.u[0]

		self.omega0 = np.sqrt(np.abs(self.C*self.Mtot/self.L))
		self.u_c = self.L*self.omega0/(2.*np.pi)

		self.K = None
		if K is None:
			self.setKernel()
		else:
			self.refer2Kernel(K) # Husimi kernel


	def setKernel(self):

		x_ = self.x.copy()

		u = self.hbar_*sp.fftshift(self.kx)
		
		self.K = hu.K_H_(self.x, x_, u, self.hbar_, self.sig_x, self.L)


	# sets the Kernel as a reference
	def refer2Kernel(self, K):
		self.K = K

	def getKernel(self):
		return self.K

	def getRhoX(self):
		return self.Mtot*(np.abs(self.psi))**2

	def getRhoU(self):
		rho = np.abs(sp.fft(self.psi))**2
		return self.Mtot*u.normalize(rho, self.du)

	def getPS(self):
		#rho = u.makeOverDensity(u.normalize(((np.abs(self.psi))**2), self.dx))
		#rho = u.makeOverDensity( u.normalize( self.getRhoX(), self.dx ) )
		rho = self.getRhoX()
		return self.kx[self.kx>0], np.abs(sp.fft(rho)[self.kx>0])**2

	def getPhaseSpace(self, shift = 1):
		return hu.f_H(self.psi, self.K, self.dx, shift)

	def getNorm(self):
		return np.sqrt((np.abs(self.psi)**2).sum()*self.dx)

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
			self.psi += np.sqrt(w_[i])*psi_

		self.psi = self.psi/self.getNorm()

	def setColdDistr(self):
		self.psi = np.array(np.ones(self.N ))
		self.psi = self.psi/self.getNorm()

	def addVelocityPerturb(self, lam, v0, sin = False):
		phi = np.zeros(self.N)
		n_lam = len(lam)
		for i in range(len(lam)):
			phi += v0*self.u_c*lam[i] / (2*np.pi) *np.cos(2*np.pi*self.x/lam[i] + np.pi/2.*sin)/n_lam
		self.psi = self.psi*np.exp(phi/self.hbar_*1.j)
		self.psi = self.psi/self.getNorm()

	def addSpatialPerturb(self, lam, delta0,sin = False):
		psi_ = 1.
		n_lam = len(lam)
		for i in range(n_lam):
			psi_ += delta0*np.sin(2.*np.pi*self.x/(lam[i]*self.L)+np.pi/2.*sin)/n_lam
		self.psi = self.psi*np.sqrt(psi_)
		self.psi = self.psi/self.getNorm()


	def update(self, dt_, iters):
		for j in range(iters):
			# update position half-step
			psi_k = sp.fft(self.psi)
			psi_k *= np.exp(-1j*dt_*self.T_scale*self.hbar*(self.kx**2)/(4*self.mpart))
			self.psi = sp.ifft(psi_k)

			phi_r = self.getPotential()
			Vr = phi_r*self.mpart # the PE field for an individual particle

			# update momentum full-step
			self.psi *= np.exp(-1j*dt_*self.T_scale*Vr/(self.hbar))

			# update position half-step
			psi_k = sp.fft(self.psi)
			psi_k *= np.exp(-1j*dt_*self.T_scale*self.hbar*(self.kx**2)/(4*self.mpart))
			self.psi = sp.ifft(psi_k)

	def readyDir(self,ofile):
		u.readyDir(ofile, "psi")

	def dropData(self,i,ofile_):
		np.save("../" + ofile_ + "/psi/" + "drop" + str(i) + ".npy", self.psi)



# 2D schroedinger poisson solver
class SP2D(object):

	def __init__(self, psi = False, N = 256, C = 2e-6,
		Mtot = 1., L = 1., hbar_ = 1e-6, mpart = 4e-6, hbar = 4e-12, label = 'b'):

		self.N = N # grid cells

		self.hbar_ = hbar_
		self.hbar = hbar
		self.mpart = mpart

		# wavefunction, N complex numbers, square norm sum is 1 NOT Mtot
		if psi:
			self.psi = psi
		else:
			self.psi = np.zeros((N,N)) + 0j 

		self.C = C # poisson constant

		self.Mtot = Mtot # total mass

		self.L = L # side length

		self.dx = L/N

		self.x = self.dx*(.5+np.arange(-self.N/2, self.N/2))

		self.T_scale = np.pi*2./np.sqrt(np.abs(C*Mtot/L)) # time scaling factor

		kx = 2*np.pi*sp.fftfreq(N, d = L/N)
		self.kx, self.ky = np.meshgrid(kx,kx.copy())

		self.u = hbar_*sp.fftshift(kx)
		self.du = self.u[1] - self.u[0]

		self.omega0 = np.sqrt(np.abs(self.C*self.Mtot/self.L))
		self.u_c = self.L*self.omega0/(2.*np.pi)

		self.label = label


	def getRhoX(self):
		return self.Mtot*(np.abs(self.psi))**2

	def getRhoU(self):
		rho = np.abs(sp.fft2(self.psi))**2
		rho = sp.fftshift(rho)
		return self.Mtot*u.normalize(rho, self.du)

	def getNorm(self):
		return np.sqrt((np.abs(self.psi)**2).sum()*self.dx)

	def getPotential(self):
		rho_r = self.getRhoX()
		return u.compute_phi2D(rho_r, self.C, self.dx, self.kx, self.ky)

	def setThermalDistr(self, Sa, v_th, lam0 = 1., dn = 0., v_bump = 0.):
		v = self.hbar_*4*np.pi*np.pi*np.array(Sa)/(lam0*self.L*np.sqrt(self.C))
		w_pre = np.exp(-(v/v_th)**2 + 0j) + dn*np.exp(-((v - v_bump)/v_th)**2 + 0j)
		w_ = u.normalize(w_pre,1)

		x_ = self.dx*(.5+np.arange(-self.N/2, self.N/2))
		psiX = np.zeros(self.N) + 0j
		for i in range(len(Sa)):
			phi = x_*self.hbar_*np.pi*2.*Sa[i]/self.L
			psiX_ = np.exp(phi/self.hbar_*1.j)
			psiX += np.sqrt(w_[i])*psiX_

		self.psi = np.outer(psiX, psiX.copy())

		self.psi = self.psi/self.getNorm()

	def setColdDistr(self):
		self.psi = np.array(np.ones((self.N, self.N)))
		self.psi = self.psi/self.getNorm()

	def addVelocityPerturb(self, lamx, lamy, v0, sin = False):
		phiX = np.zeros(self.N)
		phiY = phiX.copy()
		x_ = self.dx*(.5+np.arange(-self.N/2, self.N/2))
		n_lamx, n_lamy = len(lamx), len(lamy)
		for i in range(len(lamx)):
			phiX += v0*self.u_c*lamx[i] / (2*np.pi) *np.cos(2*np.pi*x_/lamx[i]  + np.pi/2.*sin)/n_lamx
		for i in range(len(lamy)):
			phiY += v0*self.u_c*lamy[i] / (2*np.pi) *np.cos(2*np.pi*x_/lamy[i]  + np.pi/2.*sin)/n_lamy
		psiX = np.exp(phiX/self.hbar_*1.j)
		psiY = np.exp(phiY/self.hbar_*1.j)
		psi = np.outer(psiX,psiY)
		self.psi = self.psi*psi
		self.psi = self.psi/self.getNorm()


	def addSpatialPerturb(self, lamx, lamy, delta0, sin = False):
		psiX_ = 1.
		psiY_ = 1.
		n_lamx, n_lamy = len(lamx), len(lamy)
		x_ = self.dx*(.5+np.arange(-self.N/2, self.N/2))
		for i in range(n_lamx):
			psiX_ += delta0*np.sin(2.*np.pi*x_/(lamx[i]*self.L) + np.pi/2.*sin)/n_lamx
		for i in range(n_lamy):
			psiY_ += delta0*np.sin(2.*np.pi*x_/(lamy[i]*self.L) + np.pi/2.*sin)/n_lamy
		psi_ = np.outer(psiX_,psiY_)
		self.psi = self.psi*np.sqrt(psi_)
		self.psi = self.psi/self.getNorm()

	def update(self, dt_, iters):
		for j in range(iters):
			# update position half-step
			psi_k = sp.fft2(self.psi)
			psi_k *= np.exp(-1j*dt_*self.T_scale*self.hbar*(self.kx**2 + self.ky**2)/(4*self.mpart))
			self.psi = sp.ifft2(psi_k)

			phi_r = self.getPotential()
			Vr = phi_r*self.mpart # the PE field for an individual particle

			# update momentum full-step
			self.psi *= np.exp(-1j*dt_*self.T_scale*Vr/(self.hbar))

			# update position half-step
			psi_k = sp.fft2(self.psi)
			psi_k *= np.exp(-1j*dt_*self.T_scale*self.hbar*(self.kx**2 + self.ky**2)/(4*self.mpart))
			self.psi = sp.ifft2(psi_k)

	def readyDir(self,ofile):
		u.readyDir(ofile, "psi")

	def dropData(self,i,ofile_):
		np.save("../" + ofile_ + "/psi/" + "drop" + str(i) + ".npy", self.psi)