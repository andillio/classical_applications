import numpy as np 
import scipy.fftpack as sp
import utils as u
import os
import husimi_sp as hu


# 1D mutli-stream schreodinger poisson solver
class MS1D(object):

	def __init__(self, Psi = False, K = None, N = 256, C = 2e-6, f = 5,
		Mtot = 1., L = 1., hbar_ = 1e-6, mpart = 4e-6, hbar = 4e-12, label = 'kx'):

		self.label = label

		self.N = N # grid cells

		self.hbar_ = hbar_
		self.hbar = hbar
		self.mpart = mpart

		# wavefunction array, n_streams x N complex numbers, square norm sum is 1 NOT Mtot
		if Psi:
			self.Psi = Psi
		else:
			self.Psi = None 

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

	def refer2Kernel(self, K):
		self.K = K

	def getRhoX(self):
		return self.Mtot*((np.abs(self.Psi))**2).sum(axis = 0)

	def getRhoU(self):
		rho_u = (np.abs(sp.fft(self.Psi))**2).sum(axis = 0)
		rho = sp.fftshift(rho)
		return self.Mtot*u.normalize(rho_u, self.du)

	def getPS(self):
		#rho = u.makeOverDensity(u.normalize(((np.abs(self.Psi))**2).sum(axis = 0), self.dx))
		#rho = u.makeOverDensity( u.normalize( self.getRhoX(), self.dx ) )
		rho = self.getRhoX()
		return self.kx[self.kx>0], np.abs(sp.fft(rho)[self.kx>0])**2

	def getPhaseSpace(self, shift = 1):
		HM = np.zeros((self.N,self.N)) # husimi distribution for multi phase space
		for i in range(len(self.Psi)):
			HM += hu.f_H(self.Psi[i], self.K, self.dx,shift)
		return HM

	def getNorm(self):
		return np.sqrt((np.abs(self.Psi)**2).sum()*self.dx)

	def getPotential(self):
		rho_r = self.getRhoX()
		return u.compute_phi(rho_r, self.C, self.dx, self.kx)

	def getMaxE(self):
		return np.max(u.a_from_Phi(self.getPotential()))


	def setThermalDistr(self, Sa, v_th, lam0 = 1., dn = 0., v_bump = 0.):
		v = self.hbar_*4*np.pi*np.pi*np.array(Sa)/(lam0*self.L*np.sqrt(self.C))
		w_pre = np.exp(-(v/v_th)**2 + 0j) + dn*np.exp(-((v - v_bump)/v_th)**2 + 0j)
		w_ = u.normalize(w_pre,1)

		Psi = []

		for i in range(len(Sa)):
			phi = self.x*self.hbar_*np.pi*2.*Sa[i]/self.L
			psi_ = np.exp(phi/self.hbar_*1.j)
			Psi.append(np.sqrt(w_[i])*psi_)

		self.Psi = np.array(Psi)
		self.Psi = self.Psi/self.getNorm()

	def setColdDistr(self):
		self.Psi = np.array([np.ones(self.N)])
		self.Psi = self.Psi/self.getNorm()

	def addVelocityPerturb(self, lam, v0, sin = False):
		phi = np.zeros(self.N)
		for i in range(len(lam)):
			phi += v0*self.u_c*lam[i] / (2*np.pi) *np.cos(2*np.pi*self.x/lam[i] + np.pi/2.*sin)/len(lam)
		self.Psi = self.Psi*np.exp(phi/self.hbar_*1.j)
		self.Psi = self.Psi/self.getNorm()


	def addSpatialPerturb(self, lam, delta0, sin = False):
		psi_ = 1.
		n_lam = len(lam)
		for i in range(n_lam):
			psi_ += delta0*np.sin(2.*np.pi*self.x/(lam[i]*self.L + np.pi/2. * sin))/n_lam
		self.Psi = self.Psi*np.sqrt(psi_)
		self.Psi = self.Psi/self.getNorm()

	def update(self, dt_, iters):
		for j in range(iters):
			# update position half-step
			psi_k = sp.fft(self.Psi)
			psi_k *= np.exp(-1j*dt_*self.T_scale*self.hbar*(self.kx**2)/(4*self.mpart))
			self.Psi = sp.ifft(psi_k)

			phi_r = self.getPotential()
			Vr = phi_r*self.mpart # the PE field for an individual particle

			# update momentum full-step
			self.Psi *= np.exp(-1j*dt_*self.T_scale*Vr/(self.hbar))

			# update position half-step
			psi_k = sp.fft(self.Psi)
			psi_k *= np.exp(-1j*dt_*self.T_scale*self.hbar*(self.kx**2)/(4*self.mpart))
			self.Psi = sp.ifft(psi_k)

	def readyDir(self,ofile):
		u.readyDir(ofile, "Psi")

	def dropData(self,i,ofile_):
		np.save("../" + ofile_ + "/Psi/" + "drop" + str(i) + ".npy", self.Psi)




class MS2D(object):

	def __init__(self, Psi = False, N = 256, C = 2e-6,
		Mtot = 1., L = 1., hbar_ = 1e-6, mpart = 4e-6, hbar = 4e-12, label = 'kx'):

		self.label = label

		self.N = N # grid cells

		self.hbar_ = hbar_
		self.hbar = hbar
		self.mpart = mpart

		# wavefunction array, n_streams x N complex numbers, square norm sum is 1 NOT Mtot
		if Psi:
			self.Psi = Psi
		else:
			self.Psi = None 

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


	def getRhoX(self):
		return self.Mtot*((np.abs(self.Psi))**2).sum(axis = 0)

	def getRhoU(self):
		rho = (np.abs(sp.fft2(self.Psi))**2).sum(axis = 0)
		rho = sp.fftshift(rho)
		return self.Mtot*u.normalize(rho, self.du)

	def getNorm(self):
		return np.sqrt((np.abs(self.Psi)**2).sum()*self.dx)

	def getPotential(self):
		rho_r = self.getRhoX()
		return u.compute_phi2D(rho_r, self.C, self.dx, self.kx, self.ky)

	def setThermalDistr(self, Sa, v_th, lam0 = 1., dn = 0., v_bump = 0.):
		v = self.hbar_*4*np.pi*np.pi*np.array(Sa)/(lam0*self.L*np.sqrt(self.C))
		w_pre = np.exp(-(v/v_th)**2 + 0j) + dn*np.exp(-((v - v_bump)/v_th)**2 + 0j)
		w_ = u.normalize(w_pre,1)

		Psi = []

		for i in range(len(Sa)):
			phiX = self.x*self.hbar_*np.pi*2.*Sa[i]/self.L
			psiX_ = np.exp(phiX/self.hbar_*1.j)
			for j in range(len(Sa)):
				phiY = self.x*self.hbar_*np.pi*2.*Sa[j]/self.L
				psiY_ = np.exp(phiY/self.hbar_*1.j)
				psi = np.outer(psiX_,psiY_)
				Psi.append(np.sqrt(w_[i]*w_[j])*psi)

		self.Psi = np.array(Psi)
		self.Psi = self.Psi/self.getNorm()

	def setColdDistr(self):
		self.Psi = np.array([np.ones((self.N, self.N))])
		self.Psi = self.Psi/((np.abs(self.Psi)**2).sum()*self.dx)

	def addVelocityPerturb(self, lamx, lamy, v0, sin = False):
		phiX, phiY = np.zeros(self.N) +0j, np.zeros(self.N)+0j
		for i in range(len(lamx)):
			phiX += v0*self.u_c*lamx[i] / (2*np.pi) *np.cos(2*np.pi*self.x/lamx[i] + np.pi/2.*sin)/len(lamx)
		for j in range(len(lamy)):
			phiY += v0*self.u_c*lamy[i] / (2*np.pi) *np.cos(2*np.pi*self.x/lamy[i] + np.pi/2.*sin)/len(lamy)
		psiX = np.exp(phiX/self.hbar_*1.j)
		psiY = np.exp(phiY/self.hbar_*1.j)
		psi = np.outer(psiX, psiY)
		self.Psi = self.Psi*psi
		self.Psi = self.Psi/self.getNorm()

	def addSpatialPerturb(self, lamx, lamy, delta0, sin = False):
		psiX_ = 1.
		psiY_ = 1.
		n_lamx, n_lamy = len(lamx), len(lamy)
		for i in range(n_lamx):
			psiX_ += delta0*np.sin(2.*np.pi*self.x/(lamx[i]*self.L + np.pi/2. * sin))/n_lamx
		for i in range(n_lamy):
			psiY_ += delta0*np.sin(2.*np.pi*self.x/(lamy[i]*self.L + np.pi/2. * sin))/n_lamy
		self.Psi = self.Psi*np.sqrt(np.outer(psiX_, psiY_))
		self.Psi = self.Psi/self.getNorm()

	def update(self, dt_, iters):
		for j in range(iters):
			# update position half-step
			psi_k = sp.fft2(self.Psi)
			psi_k *= np.exp(-1j*dt_*self.T_scale*self.hbar*(self.kx**2 + self.ky**2)/(4*self.mpart))
			self.Psi = sp.ifft2(psi_k)

			phi_r = self.getPotential()
			Vr = phi_r*self.mpart # the PE field for an individual particle

			# update momentum full-step
			self.Psi *= np.exp(-1j*dt_*self.T_scale*Vr/(self.hbar))

			# update position half-step
			psi_k = sp.fft2(self.Psi)
			psi_k *= np.exp(-1j*dt_*self.T_scale*self.hbar*(self.kx**2 + self.ky**2)/(4*self.mpart))
			self.Psi = sp.ifft2(psi_k)

	def readyDir(self,ofile):
		u.readyDir(ofile)

	def dropData(self,i,ofile_):
		np.save("../" + ofile_ + "/Psi/" + "drop" + str(i) + ".npy", self.Psi)
