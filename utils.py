import numpy as np 
import pylab as pyl 
import scipy.fftpack as sp
import time
import sys
import matplotlib.pyplot as plt
import os
import pyfftw
import shutil

# computes the potential from the wavefunction
def compute_phi(rho_r, C_, dx, kx = None, fac = 500):
	rho_k = sp.fft(rho_r)
	phi_k = None
	#if kx is None:
	kx = 2*np.pi*sp.fftfreq(len(rho_r), d = dx)
	with np.errstate(divide ='ignore', invalid='ignore'):
		# np.errstate(...) suppresses the divide-by-zero warning for k = 0
		phi_k = -(C_*rho_k)/(kx**2)
	phi_k[0] = 0.0 # set k=0 component to 0
	phi_r = np.real(sp.ifft(phi_k))
	return phi_r

def compute_phi2D(rho_r, C_, dx, kx, ky):
    # TODO: we know rho_r is real, so using rfft would be better 
    rho_k = sp.fft2(rho_r)
    with np.errstate(divide='ignore', invalid='ignore'):
        # np.errstate(...) suppresses the divide-by-zero warning for k=0
        phi_k = -(C_*rho_k)/(kx**2+ky**2) 
    phi_k[0,0] = 0.0 # zero the k=0 component 
    phi_r = np.real(sp.ifft2(phi_k))
    return phi_r

def compute_phi3D(rho_r):
	fftObj = pyfftw.FFTW(rho_r, s.fft_out_phi, axes = (0,1,2))
	rho_k = fftObj()
	phi_k = None
	with np.errstate(divide ='ignore', invalid='ignore'):
	    # np.errstate(...) suppresses the divide-by-zero warning for k = 0
	    phi_k = -(C*rho_k)/((s.spec_grid))
	phi_k[0,0,0] = 0.0 # set k=0 component to 0
	ifftObj = pyfftw.FFTW(phi_k, s.fft_out_phi, axes = (0,1,2), direction = "FFTW_BACKWARD")
	phi_r = ifftObj()
	return phi_r

def normalize(y,dx):
	return y/(np.sum(y)*dx)

def makeOverDensity(y):
	return y - np.mean(y)

def array_make_periodic(x,w):
    w[x>=1] += 1
    x[x>=1.] -= 1.

    w[x<0] -= 1 
    x[x<0.]  +=1.

def fast_CIC_deposit(x,mi,Ngrid,periodic=1):
    """cloud in cell density estimator
    """
    if ((np.size(mi)) < (np.size(x))):
        m=x.copy()
        m[:]=mi
    else:
        m=mi

    dx = 1./Ngrid
    rho = np.zeros(Ngrid)
 
    left = x-0.5*dx
    right = left+dx
    xi = np.int32(left/dx)
    frac = (1.+xi-left/dx)
    ind = pyl.where(left<0.)
    frac[ind] = (-(left[ind]/dx))
    xi[ind] = Ngrid-1
    xir = xi.copy()+1
    xir[xir==Ngrid] = 0
    rho  = pyl.bincount(xi,  weights=frac*m, minlength=Ngrid)
    rho2 = pyl.bincount(xir, weights=(1.-frac)*m, minlength=Ngrid)

    rho += rho2
    
    return rho*Ngrid

# TODO: wtf is going on here
def fast_CIC_deposit2D(x,y,mpart,Ngrid,periodic=1):
    """cloud in cell density estimator
    """

    dx = 1./Ngrid
    rho = np.zeros((Ngrid,Ngrid))
 
    left_x = x-0.5*dx
    left_y = y-0.5*dx

    xi = np.int32(left_x/dx)
    yi = np.int32(left_y/dx)
    
    frac_x = (1.+xi-left_x/dx) # fraction in left bin
    frac_y = (1.+yi-left_y/dx)

    ind_x = pyl.where(left_x<0.) # handles edge cases
    ind_y = pyl.where(left_y<0.) # handles edge cases

    frac_x[ind_x] = (-(left_x[ind_x]/dx))
    frac_y[ind_y] = (-(left_y[ind_y]/dx))
    
    xi[ind_x] = Ngrid-1
    yi[ind_y] = Ngrid-1
    
    xir = xi.copy()+1
    yir = yi.copy()+1

    xir[xir==Ngrid] = 0
    yir[yir==Ngrid] = 0

    rho[xi,yi] += frac_x*frac_y*mpart
    rho[xir,yi] += (1.-frac_x)*frac_y*mpart
    rho[xi,yir] += (1.-frac_y)*frac_x*mpart
    rho[xir,yir] += (1.-frac_x)*(1.-frac_y)*mpart
    
    return rho

# TODO: broken af
def CIC_deposit2D(rx, ry, mpart, N, periodic = 1):
	density = np.zeros((N,N))
	dx = 1./N

	i_left =  (rx - dx/2.)  # index of cell containing left edge
	i_left[i_left < 0] += 1.
	i_left = (i_left/dx).astype(np.int32)
	
	j_left =  (ry - dx/2.)  # index of cell containing left edge
	j_left[i_left < 0] += 1.
	j_left = (j_left/dx).astype(np.int32)

	x_frac = 1.5*dx - (rx - (i_left%(N-1)) * dx) # fraction of deposit contianed in left cell 
	y_frac = 1.5*dx - (ry - (j_left%(N-1)) * dx)

	x_frac[x_frac>1.] -= 1
	y_frac[y_frac>1.] -= 1

	density[i_left, j_left] += mpart*x_frac*y_frac
	density[(i_left + 1)%N, j_left] += mpart*(1. - x_frac)*y_frac
	density[i_left, (j_left + 1)%N] += mpart*(1. - y_frac)*x_frac
	density[(i_left + 1)%N, (j_left + 1)%N] += mpart*(1. - x_frac)*(1. - y_frac)

	return density



def CIC_acc(x,m,Ngrid,C,dx,kx):
    dx = 1./Ngrid
    xg = (0.5+np.arange(Ngrid))/Ngrid
    rho = fast_CIC_deposit(x,m,Ngrid)
    Phi = compute_phi(rho, C, dx, kx)
    a = a_from_Phi(Phi)
    left = x-0.50000*dx
    xi = np.int64(left/dx)
    frac = (1.+xi-left/dx)
    ap = (frac)*(np.roll(a,0))[xi] + (1.-frac) * (np.roll(a,-1))[xi]
    return ap

def a_from_Phi(Phi):
    """Calculate  - grad Phi  from Phi assuming a periodic domain
    domain the is 0..1 and dx=1./len(Phi)
    """
    N = len(Phi)
    dx = 1./N
    a = - central_difference(Phi)/dx
    return a

def central_difference(y):
    """ Central difference:  (y[i+1]-y[i-1])/2 
    """
    return (np.roll(y,-1)-np.roll(y,1))/2


def remaining(done, total, start):
	Dt = time.time() - start
	return hms((Dt*total)/float(done) - Dt)

# given a time T in s
# returns (hours, mins, secs) remaining
def hms(T):
	r = T
	hrs = int(r)/(60*60)
	mins = int(r%(60*60))/(60)
	s = int(r%60)
	return (hrs, mins, s)

def repeat_print(string):
    sys.stdout.write('\r' +string)
    sys.stdout.flush()

def getMetaKno(name, **kwargs):
	f = open("../" + name + "/" + name + "Meta.txt")

	print "reading meta info..."
	
	metaParams = {}

	for key_ in kwargs.keys():
		metaParams[key_] = None

	for line in f.readlines():
		for key_ in kwargs.keys():
			if key_ + ":" in line:
				print line
				number = line.split(":")[1]
				#metaParams[key_] = re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]
				if key_ == "N" or key_ == "frames" or key_ == "n":
					metaParams[key_] = int(number)
				else:
					metaParams[key_] = float(number)

	return metaParams

# gets the names of files in the given directory organized by time stamp (ascending)
def getNames(name):
	files = ["../" + name + "/" + file for file in os.listdir("../" + name) if (file.lower().endswith('.npy'))]
	files.sort(key=os.path.getmtime)
	return files


def orientPlot():
	plt.rc("font", size=22)

	plt.figure(figsize=(6,6))

	fig,ax = plt.subplots(figsize=(6,6))

	plt.rc("text", usetex=True)

	plt.rcParams["axes.linewidth"]  = 1.5

	plt.rcParams["xtick.major.size"]  = 8

	plt.rcParams["xtick.minor.size"]  = 3

	plt.rcParams["ytick.major.size"]  = 8

	plt.rcParams["ytick.minor.size"]  = 3

	plt.rcParams["xtick.direction"]  = "in"

	plt.rcParams["ytick.direction"]  = "in"

	plt.rcParams["legend.frameon"] = 'False'


def GetIndexes(T,N,times):
	inds = []
	for i in range(len(times)):
		t = times[i]
		ind_ = int(t*N/T)
		ind_ = np.min([ind_, N-1])
		ind_ = np.max([0,ind_])
		inds.append(ind_)
	return inds


def readyDir(ofile, tag):
	try: # attempt to make the directory
		os.mkdir("../" + ofile + "/" + tag)
	except OSError:
		try: # assuming directory already exists, delete it and try again
			print "removing and recreating an existing directory"
			shutil.rmtree("../" + ofile + "/" + tag)
			readyDir(ofile, tag)
		except OSError:
			pass


def ding():
	dur1 = .15
	dur2 = .15
	freq1 = 600
	freq2 = 700
	os.system('play  --no-show-progress --null --channels 1 synth %s sine %f' % (dur1, freq1))
	time.sleep(.04)
	os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (dur2, freq2))


class figObj(object):

	def __init__(self):
		self.ax = None
		self.im = None

		self.axSP = None
		self.imSP = None

		self.axMS = None
		self.imMS = None

		self.axVN = None
		self.imVN = None

		self.axCL = None
		self.imCL = None

		self.frames = None

		self.Tfinal = None

		self.decimate = None
		self.times = None
		self.SP = None
		self.shift = None
		self.colorscale = None
		self.CL_res_shift = None
		self.tags = None
		self.inds = None
		self.axis = None

		self.N_images = None
		self.fileNames_r = None
		self.fileNames_v = None
		self.fileNames_rx = None
		self.fileNames_vx = None
		self.fileNames_ry = None
		self.fileNames_vy = None
		self.fileNames_psi = None
		self.fileNames_Psi = None
		self.fileNames_rho = None

		self.x = None
		self.dx = None
		self.L = None
		self.N = None

		self.K = None

		self.meta = None


def setFileNames(fo, name, tags = ["SP", "VN", "MS", "CL"]):
	if "CL" in tags:
		fo.fileNames_r = getNames(name + "/" + "r")
		fo.fileNames_v = getNames(name + "/" + "v")
		fo.N_images = len(fo.fileNames_r)
	if "MS" in tags:
		fo.fileNames_Psi = getNames(name + "/" + "Psi")
		fo.N_images = len(fo.fileNames_Psi)
	if "SP" in tags:
		fo.fileNames_psi = getNames(name + "/" + "psi")
		fo.N_images = len(fo.fileNames_psi)
	if "VN" in tags:
		fo.fileNames_rho = getNames(name + "/" + "rho")
		fo.N_images = len(fo.fileNames_rho)
	if "CL2D" in tags:
		fo.fileNames_rx = getNames(name + "/" + "rx")
		fo.fileNames_vx = getNames(name + "/" + "vx")
		fo.fileNames_ry = getNames(name + "/" + "ry")
		fo.fileNames_vy = getNames(name + "/" + "vy")
		fo.N_images = len(fo.fileNames_rx)