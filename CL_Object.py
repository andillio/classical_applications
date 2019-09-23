import numpy as np 
import scipy.fftpack as sp
import utils as u
import os
from scipy.optimize import brentq

class CL1D(object):

	def __init__(self, r = None, v = None, n = None, N = 256, C = 1, CS = 2e-6,
		Mtot = 1., L = 1., hbar_ = 1e-6, mpart = 4e-6, label = 'r--'):

		self.label = label

		self.N = N

		self.n = n

		if r:
			self.r = r
		else:
			self.r = np.zeros(n)

		if v:
			self.v = v
		else:
			self.v = np.zeros(n)

		self.w = np.zeros(n)
		self.n_i = None

		self.C = C
		self.CS = CS

		self.Mtot = Mtot

		self.L = L

		self.mpart = mpart

		self.hbar_ = hbar_

		self.dx = L/N

		self.x = self.dx*(.5+np.arange(-self.N/2, self.N/2))

		self.T_scale = np.pi*2./np.sqrt(np.abs(C*Mtot/L)) # time scaling factor

		self.kx = 2*np.pi*sp.fftfreq(N, d = L/N)

		self.omega0 = np.sqrt(np.abs(self.C*self.Mtot/self.L))
		self.u_c = self.L*self.omega0/(2.*np.pi)
		self.u = hbar_*sp.fftshift(self.kx)


	def getRhoX(self, shift = 1):
		if shift:
			return sp.fftshift(u.fast_CIC_deposit(self.r, self.mpart, self.N))
		else:
			return u.fast_CIC_deposit(self.r, self.mpart, self.N)

	def getRhoU(self, u_min = None, u_max = None):
		if u_min is None:
			u_min = np.min(self.u)
		if u_max is None:
			u_max = np.max(self.u)
		n, edges = np.histogram(self.v/self.u_c, range = (u_min,u_max), bins = self.N, normed = True)
		du_ = (edges[1] - edges[0])
		return u.normalize(n, du_)

	def getPS(self):
		#rho = u.makeOverDensity( u.normalize( self.getRhoX(), self.dx ) )
		rho = self.getRhoX()
		return self.kx[self.kx>0], np.abs(sp.fft(rho)[self.kx>0])**2

	def getPotential(self):
		rho_r = self.getRhoX()
		return u.compute_phi(rho_r, self.C, self.dx, self.kx)

	def getMaxE(self):
		return np.max(u.a_from_Phi(self.getPotential()))


	def getNorm(self):
		return self.mpart*self.n

	def setThermalDistr(self, Sa, v_th, lam0 = 1., dn = 0., v_bump = 0.):
		n_streams = len(Sa)
		y = np.array([])
		v = np.array([])
		weight = np.zeros(n_streams)
		self.n_i = np.zeros(n_streams)

		v__ = self.hbar_*4*np.pi*np.pi*np.array(Sa)/(lam0*self.L*np.sqrt(self.CS))
		w_pre = np.exp(-(v__/v_th)**2 + 0j) + dn*np.exp(-((v__ - v_bump)/v_th)**2 + 0j)
		weight = u.normalize(w_pre,1)

		self.n_i[Sa != 0] = (weight[Sa != 0]*self.n/weight.sum()).astype(int)
		#self.n_i[Sa == 0] = self.n - self.n_i.sum()
		self.n_i = (self.n_i).astype(int)
		self.n_i[Sa == 0] = 0
		self.n_i[Sa == 0] = self.n - self.n_i.sum()

		for i in range(len(Sa)):
		    n_i_ = self.n_i[i]
		    y_ = (i/float(n_streams) + np.arange(n_i_))*self.L/float(n_i_)
		    S_ = Sa[i]
		    v_ = np.ones(n_i_)*self.hbar_*4*np.pi*np.pi*S_/(lam0*self.L*np.sqrt(np.abs(self.CS)))*self.u_c
		    v = np.concatenate((v, v_),0)
		    y = np.concatenate((y,y_),0)

		self.r = y
		self.v = v

	def setColdDistr(self):
		self.r = np.arange(self.n)*self.L/float(self.n)
		self.v = np.zeros(self.n)
		self.n_i = np.array([self.n])

	def addVelocityPerturb(self, lam, v0, sin = False):
		n_lam = len(lam)
		for i in range(len(lam)):
			self.v += -v0*self.u_c*np.sin(2.*np.pi*self.r/lam[i]  + np.pi/2.*sin)/n_lam

	def addSpatialPerturb(self, lam, delta0, sin = False):
		y = np.array([])
		for i in range(len(self.n_i)):
			n_i_ = self.n_i[i]
			if n_i_ > 0:
				y_ = self.getPositions(n_i_, lam, delta0, sin)
				y = np.concatenate((y,y_),0)
		self.r = y
	def getPositions(self, n, lam, delta0, sin):
		r = np.zeros(n)
		cp = (0. + np.arange(n + 2))/(n + 2)
		mass = lambda z, cp_:  z - self.perturb(z, lam, delta0, sin) - cp_ + self.perturb(0,lam,delta0, sin)
		for i in range(1, len(cp)-1):
			i_ = i - 1
			r[i_] = brentq(mass, 0, 1., args=cp[i])
		return r
	def perturb(self, z, lam, delta0, sin):
		rval = 0
		n_lam = len(lam)
		for i in range(len(lam)):
			k_ = 2.*np.pi/lam[i]
			rval += delta0*np.cos(k_*z + np.pi/2. * sin)/(k_*n_lam)
		return rval

	def update(self, dt_, iters):
		for j in range(iters):
			self.r += self.v*dt_*self.T_scale/2.
			u.array_make_periodic(self.r, self.w)

			a = u.CIC_acc(self.r,self.mpart,self.N,self.C,self.dx, self.kx)

			self.v += a*dt_*self.T_scale

			self.r += self.v*dt_*self.T_scale/2.
			u.array_make_periodic(self.r, self.w)

	def readyDir(self,ofile):
		try:
			os.mkdir("../" + ofile + "/r")
		except OSError:
			pass
		try:
			os.mkdir("../" + ofile + "/v")
		except OSError:
			pass

	def dropData(self,i,ofile_):
		np.save("../" + ofile_ + "/r/" + "drop" + str(i) + ".npy", self.r)
		np.save("../" + ofile_ + "/v/" + "drop" + str(i) + ".npy", self.v)



class CL2D(object):

	def __init__(self, rx = None, ry = None, vx = None, vy = None, n = None, N = 256, 
		C = 1, CS = 2e-6,Mtot = 1., L = 1., hbar_ = 1e-6, mpart = 4e-6, label = 'r--'):

		self.label = label

		self.N = N

		self.n = n

		if rx or ry:
			self.rx = rx
			self.ry = ry
		else:
			self.rx = np.zeros(n)
			self.ry = np.zeros(n)

		if vx or vy:
			self.vx = vx
			self.vy = vy
		else:
			self.vx = np.zeros(n)
			self.vy = np.zeros(n)

		self.w = np.zeros(n)
		self.n_i = None

		self.C = C
		self.CS = CS

		self.Mtot = Mtot

		self.L = L

		self.mpart = mpart

		self.hbar_ = hbar_

		self.dx = L/N

		self.x = self.dx*(.5+np.arange(-self.N/2, self.N/2))

		self.T_scale = np.pi*2./np.sqrt(np.abs(C*Mtot/L)) # time scaling factor

		kx = 2*np.pi*sp.fftfreq(N, d = L/N)
		self.kx, self.ky = np.meshgrid(kx,kx.copy())

		self.omega0 = np.sqrt(np.abs(self.C*self.Mtot/self.L))
		self.omega0S = np.sqrt(np.abs(self.CS*self.Mtot/self.L))
		self.u_c = self.L*self.omega0/(2.*np.pi)
		self.u_cS = self.L*self.omega0S/(2.*np.pi)
		self.u = hbar_*sp.fftshift(self.kx)

		self.u_max = 2*np.pi*self.N*self.hbar_/(2.*self.L)/self.u_cS


	def getRhoX(self, shift = 1):
		if shift:
			return sp.fftshift(u.fast_CIC_deposit2D(self.rx, self.ry, self.mpart, self.N))
		else:
			return u.fast_CIC_deposit2D(self.rx, self.ry, self.mpart, self.N)

	def getRhoU(self, u_min = None, u_max = None):
		if u_min is None:
			u_min = -self.u_max
		if u_max is None:
			u_max = self.u_max
		n, edgesx, edgesy = np.histogram2d(self.vx/self.u_c, self.vy/self.u_c, 
			range = [[u_min,u_max],[u_min,u_max]], bins = self.N, normed = True)
		du_ = (edgesx[1] - edgesx[0])
		return u.normalize(n, du_)

	def getPotential(self):
		rho_r = self.getRhoX()
		return u.compute_phi2D(rho_r, self.C, self.dx, self.kx, self.ky)

	def getNorm(self):
		return self.mpart*self.n

	def setThermalDistr(self, Sa, v_th, lam0 = 1., dn = 0., v_bump = 0.):
		n_streams = len(Sa)*len(Sa)
		rx = np.array([])
		ry = np.array([])
		vx = np.array([])
		vy = np.array([])
		self.n_i = np.zeros((len(Sa),len(Sa)))

		v__ = self.hbar_*4*np.pi*np.pi*np.array(Sa)/(lam0*self.L*np.sqrt(self.CS))
		w_pre = np.exp(-(v__/v_th)**2 + 0j) + dn*np.exp(-((v__ - v_bump)/v_th)**2 + 0j)
		w_pre = np.outer(w_pre, w_pre)
		weight = u.normalize(w_pre,1)

		for i in range(len(Sa)):
			S_x = Sa[i]
			for j in range(len(Sa)):
				n_i_ = int(np.sqrt(weight[i,j]*self.n))
				S_y = Sa[j]
				v_y = np.ones(n_i_*n_i_)*self.hbar_*4*np.pi*np.pi*S_y/(lam0*self.L*np.sqrt(np.abs(self.CS)))*self.u_c
				v_x = np.ones(n_i_*n_i_)*self.hbar_*4*np.pi*np.pi*S_x/(lam0*self.L*np.sqrt(np.abs(self.CS)))*self.u_c
				y_ = (j/np.sqrt(n_streams) + np.arange(n_i_))*self.L/float(n_i_)
				x_ = (i/np.sqrt(n_streams) + np.arange(n_i_))*self.L/float(n_i_)

				y_ = np.outer(np.ones(n_i_), y_).flatten()
				x_ = np.outer(x_, np.ones(n_i_)).flatten()

				vx = np.concatenate((vx, v_x),0)
				vy = np.concatenate((vy, v_y),0)
				rx = np.concatenate((rx,x_),0)
				ry = np.concatenate((ry,y_),0)

				self.n_i[i,j] = len(y_)

		self.ry = ry
		self.rx = rx
		self.vx = vx
		self.vy = vy
		self.n_i = self.n_i.flatten()

		print str((self.n - self.n_i.sum())/float(self.n)), " fraction particles not used"
		self.n = self.n_i.sum()
		self.mpart = self.Mtot/self.n



	# TODO
	def setColdDistr(self):
		self.r = np.arange(self.n)*self.L/float(self.n)
		self.v = np.zeros(self.n)
		self.n_i = np.array([self.n])

	# TODO
	def addVelocityPerturb(self, lam, v0, sin = False):
		n_lam = len(lam)
		for i in range(len(lam)):
			self.v += -v0*self.u_c*np.sin(2.*np.pi*self.r/lam[i]  + np.pi/2.*sin)/n_lam

	# TODO
	def addSpatialPerturb(self, lam, delta0, sin = False):
		y = np.array([])
		for i in range(len(self.n_i)):
			n_i_ = self.n_i[i]
			if n_i_ > 0:
				y_ = self.getPositions(n_i_, lam, delta0, sin)
				y = np.concatenate((y,y_),0)
		self.r = y
	def getPositions(self, n, lam, delta0, sin):
		r = np.zeros(n)
		cp = (0. + np.arange(n + 2))/(n + 2)
		mass = lambda z, cp_:  z - self.perturb(z, lam, delta0, sin) - cp_ + self.perturb(0,lam,delta0, sin)
		for i in range(1, len(cp)-1):
			i_ = i - 1
			r[i_] = brentq(mass, 0, 1., args=cp[i])
		return r
	def perturb(self, z, lam, delta0, sin):
		rval = 0
		n_lam = len(lam)
		for i in range(len(lam)):
			k_ = 2.*np.pi/lam[i]
			rval += delta0*np.cos(k_*z + np.pi/2. * sin)/(k_*n_lam)
		return rval

	# TODO
	def update(self, dt_, iters):
		for j in range(iters):
			self.r += self.v*dt_*self.T_scale/2.
			u.array_make_periodic(self.r, self.w)

			a = u.CIC_acc(self.r,self.mpart,self.N,self.C,self.dx, self.kx)

			self.v += a*dt_*self.T_scale

			self.r += self.v*dt_*self.T_scale/2.
			u.array_make_periodic(self.r, self.w)

	def readyDir(self,ofile):
		u.readyDir(ofile, "rx")
		u.readyDir(ofile, "ry")
		u.readyDir(ofile, "vx")
		u.readyDir(ofile, "vy")

	def dropData(self,i,ofile_):
		np.save("../" + ofile_ + "/rx/" + "drop" + str(i) + ".npy", self.rx)
		np.save("../" + ofile_ + "/ry/" + "drop" + str(i) + ".npy", self.ry)
		np.save("../" + ofile_ + "/vx/" + "drop" + str(i) + ".npy", self.vx)
		np.save("../" + ofile_ + "/vy/" + "drop" + str(i) + ".npy", self.vy)
