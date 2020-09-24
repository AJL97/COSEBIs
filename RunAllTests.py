'''
This script contains the additive Gaussianity test as described in Sellentin et al. 2017 (DOI: 10.1093/mnras/stx2491)
'''
import numpy as np
import matplotlib.pyplot as plt
import sys
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
import create_cosebis as cc
from matplotlib import cm

'''
Class UsefulFuncs contains the test functions and runs in the background
Class RunTests runs the test functions and should be called in another script
'''

class UsefulFuncs(object):

	def __init__(self,bins=55,N=7,sample_size=819,type='En',savedir=None,dfile=None,noise=False,Gaussian=False,mini=True,vmin_zero=True):
		self._histobins = 40
		self._bins = bins
		self._N = N
		self._Samples = sample_size
		self._type = type
		self._savedir = savedir
		if dfile is not None:
			self._data = np.load(dfile)
		self._noise = noise
		self._Gaussian = Gaussian
		if self._Gaussian:
			print ('='*20)
			print ("Using Gaussian data")
		if ('xip' in np.char.lower(type) or 'xim' in np.char.lower(type)) and dfile is not None:
			self._data = np.swapaxes(self._data,1,2)
		self._covmat = None
		self._cormat = None
		self._white = None
		self.mini = mini
		self.vmin_zero = vmin_zero
		
	'''Function to reshape the data (by combining first two dimensions)'''
	def reshape(self):
		samples = self._Samples
		nmax = self._N
		bins = self._bins
		x = self._data
		
		max_val = bins*nmax
		x_reshaped = np.zeros((max_val,samples))
		c = 0

		for bin in range(bins):
			for mode in range(nmax):
				x_reshaped[c,:] = x[bin,mode,:]
				c += 1
		return x_reshaped
	
	'''Function to create the covariance matrix of the given data'''
	def Covmat(self,plot=True):
		#print ('='*20)
		#print ('Creating cov-mat for {0}....'.format(self._type))
		
		nmax = self._N
		samples = self._Samples
		self._covmat = np.zeros((nmax,nmax))
		data = self._data
		
		for i in range(nmax):
			vec1 = data[i,:]
			meanvec1 = np.mean(vec1)
			for j in range(nmax):
				vec2 = data[j,:]
				meanvec2 = np.mean(vec2)
				
				self._covmat[i,j] = np.sum((vec1-meanvec1)*(vec2-meanvec2))/(samples-1)
		
		if plot:
			fig,ax = plt.subplots()
			im = ax.imshow(self._covmat)#,origin='lower')
			fig.colorbar(im)
			ax.set_title('covmat {0}'.format(self._type))
			plt.savefig('{0}/covmat_{1}.png'.format(self._savedir,self._type),dpi=385)
			plt.show()
			
	
	'''Function to create the correlation matrix of the given data'''
	def Cormat(self,plot=True):
		#print ('='*20)
		#print ('Creating cor-mat for {0}....'.format(self._type))
		
		if self._covmat is None:
			self.Covmat(plot=False)
		
		if self._covmat is not None:
			
			self._cormat = np.zeros_like(self._covmat)
			#Looping over the covariance matrix 
			for i in range(self._covmat.shape[0]):
				for j in range(self._covmat.shape[1]):
					self._cormat[i,j] = self._covmat[i,j]/(np.sqrt(self._covmat[i,i]*self._covmat[j,j]))
	
		if plot:
			fig,ax = plt.subplots()
			im = ax.imshow(self._cormat)
			fig.colorbar(im)
			ax.set_title('cormat {0}'.format(self._type))
			plt.savefig('{0}/cormat_{1}.png'.format(self._savedir,self._type),dpi=385)
			plt.show()
		
	'''Function used if bins*N>>samples'''
	def ReduceData(self,dir='data_xi_pm/log250_slop1.0_max400/data_xipm/',tdir='data_xi_pm/log250_slop1.0_max400/thetas/theta.txt'):
		print ('='*20)
		print ('Reducing {0} data size....'.format(self._type))
		
		bin_indx = []
		
		C = cc.Create_Cosebis(self._noise,dir,tdir)
		C.read_files()
		
		files_xip = C._files_xip
		files_xim = C._files_xim
	
		reds_bins = np.array([11,22,33,44,55,66,77,88,99,1010])
	
		for bin in range(len(files_xip)):
			num_xip = int(re.search(r'\d+', files_xip[bin,0][47:53]).group())
			num_xim = int(re.search(r'\d+', files_xim[bin,0][47:53]).group())
			if num_xip in reds_bins and num_xip == num_xim:
				bin_indx.append(bin)
		
		thetas_log = np.loadtxt('data_xi_pm/log250_slop1.0_max400/thetas/theta.txt')
		thetas_lin = np.linspace(1.0,400,80)
	
		indxs = list()
	
		for i in range(len(thetas_lin)):
			theta_lin = thetas_lin[i]
			new_indx = np.argmin(abs(thetas_log-theta_lin))
			if i >= 1 and thetas_log[new_indx] == thetas_log[indxs[-1]]:
				break
			indxs.append(new_indx)
		
		length = len(thetas_log[indxs[-1]:])
		for j in range(length-1):
			indxs.append((len(thetas_log)-length+1)+j)
			
		self._data = self._data[bin_indx]
		self._data = self._data[:,indxs,:]
		self._N = len(indxs)

	'''Function to whiten submatrix'''
	def Whiten(self,cov,nonwhites):
		
		dim = nonwhites.shape[0]
		
		hilf = np.copy(cov)
		
		transform = np.linalg.cholesky(hilf).T
		invtrans = np.linalg.inv(transform)
		
		means = np.mean(nonwhites,axis=1)
		w = np.zeros_like(nonwhites)
		for s in range(self._Samples):
			temp = np.zeros(dim)
			tempwhitened = np.zeros(dim)
			for p in range(dim):
				temp[p] = nonwhites[p,s] - means[p]
			tempwhitened = np.dot(invtrans.T,temp)
			
			w[:,s] = tempwhitened

		return w
		
		
	'''Function to whiten the data'''
	def WhitenData(self,plot=True):
		#print ('='*20)
		#print ('Whitening {0} data....'.format(self._type))
		
		data = self._data
		cov = self._covmat
		samples = self._Samples
		dim = self._covmat.shape[0]

		self._white = self.Whiten(cov,data)
		
		if plot:	
			self._data = self._white
			old_type = self._type
			self._type = self._type +'_whitened'
			cov = self.Covmat()
			self._type = old_type
			
	def miniwhitening(self,i,j):
		
		mininonwhites = np.zeros((2,self._Samples))
		for s in range(self._Samples):
			mininonwhites[0,s] = self._data[i,s]
			mininonwhites[1,s] = self._data[j,s]

		minicov = np.zeros((2,2))
		minicov[0,0] = self._covmat[i,i]
		minicov[1,1] = self._covmat[j,j]
		minicov[0,1] = self._covmat[i,j]
		minicov[1,0] = self._covmat[j,i]
		
		minimean = np.zeros(2)
		minimean[0] = np.mean(self._data[i,:])
		minimean[1] = np.mean(self._data[j,:])
		
		w = self.Whiten(minicov,mininonwhites)
		self._white = w
		
	'''Function to create the trans-covariance matrix of the data'''
	def TransCovmat(self,TestType='Add',plot=True):

		data = self._data
		samples = self._Samples
		dim = self._covmat.shape[0]
		
		if self._Gaussian:
			Gdata = np.zeros_like(self._data)
			for j in range(self._N):
				mean = np.mean(self._data[j,:])
				std = np.std(self._data[j,:])
				Gdata[j,:] = np.random.normal(loc = mean, scale = std, size = samples)
			
			
			self._data = Gdata
			self.Covmat(plot=False)
			self.WhitenData(plot=False)
			self._type = self._type +'_Gaussian'
		
		data = self._data
		
		self.RunAddTest()
		
		if plot:
			fig,ax = plt.subplots(1,1,figsize=(8,8))
	
			vmin = np.min(self.TestMasked)
			vmax = np.max(self.TestMasked)
		
			cmap = plt.cm.gnuplot
			cmap.set_bad(color='black')
			
			im = ax.imshow(self.TestMasked,vmin=vmin,vmax=vmax)
			div = make_axes_locatable(ax)
			cax = div.append_axes("right",size="5%", pad=0.05)
			fig.colorbar(im,cax=cax)
		
			plt.savefig('{0}/transcovmat_{1}.png'.format(self._savedir,self._type),dpi=385)
			plt.show()
			plt.close()
			
		return self.TestMasked

	'''Function that loops over all combinations (to evaluate MISE error)'''
	def RunAddTest(self):
		
		dim = self._covmat.shape[0]

		Add_MISE = np.zeros((dim,dim))
		for n in range(self._N):
			for m in range(n,self._N):
				if n == m:
					continue
				val = self.AddTest(n,m)
				Add_MISE[n,m] = val
				Add_MISE[m,n] = val
				
		self.TestMasked = np.ma.masked_where(Add_MISE == 0, Add_MISE)
			
	def joint_dist(self,sample_set,save_file=None,i=0,j=0):

		joint_hist, edges = np.histogramdd(np.swapaxes(sample_set,0,1),bins=20)
		
		if save_file==None:
			return joint_hist, edges
		plt.imshow(joint_hist)
		plt.savefig(self._savedir+save_file+'/'+'joint_dist{0}{1}'.format(i,j),dpi=385,bbox_inches='tight')
		plt.close()
		
		return joint_dist, edges
		
	'''Function that applies the Add test'''
	def AddTest(self,i,j):
		
		if self.mini:
			self.miniwhitening(i,j)
		data = self._white #Dim = (redshift bins, Theta/Cosebi values, Samples)
		samples = self._Samples
		hist = np.zeros(samples)
		L = -5.5
		U = 5.5
		width = abs(U-L)/self._histobins
		
		for s in range(samples):
			if not self.mini:
				hist[s] = 0.5*(data[i,s] + data[j,s])
			elif self.mini:
				hist[s] = 0.5*(data[0,s]+data[1,s])
				
		Addgram, bin_edges = np.histogram(hist,self._histobins,(L,U))
		scale = 1./(samples*width)
		Addgram = Addgram*scale

		mis = 0
		diff_scales = np.ones(self._histobins)
		preds = []
		for b in range(self._histobins):
		
			xl = bin_edges[b]
			xu = bin_edges[b+1]
			xaxis = 0.5*(xl+xu)
			add_pred = (1./np.sqrt(2.*np.pi/2.))*np.exp(-0.5*xaxis*xaxis*2)
			preds.append(add_pred)
			hits = Addgram[b]
			diff = add_pred - hits
			mis += abs(diff)
			
		return mis/(self._histobins*1.0)
	
	def get_histogram(self,samples,bins,mean,std):
		
		L = mean - (5*std)
		U = mean + (5*std) 
		
		hist, bin_edges = np.histogram(samples,bins=bins,range=(L,U))
		
		return hist, bin_edges
	
class RunTests(object):
	
	def __init__(self,En,bins):
		self._En = En
		self._bins = bins
	
	'''Function to create the transcovriance, covariance, and correlation matrices'''
	def create_TransCorMats(self,gauss=False):
		'''
		@Params:
		En = COSEBIs, mult. dim. arr
		bins = Number of redshift bins, int
		gauss = Using actual data or Gauss version, bool
		'''	
		
		En = self._En
		bins = self._bins
		
		TEST = UsefulFuncs(type='En',N=250,savedir='Created_En/log250_max400_slop1.3',Gaussian=gauss,mini=True,vmin_zero=False)
	
		c = 0
		all_transcovs = np.zeros((bins*En.shape[1],bins*En.shape[1]))
		all_cormats = np.zeros_like(all_transcovs)
		all_covmats = np.zeros_like(all_transcovs)
		N = En.shape[1] #modes
	
		#Looping over all redshift bin combinations
		for n in range(bins):
			for m in range(bins-n):
				TEST._data = En[c,:,:]
				TEST._bins = 1
				TEST._N = N
				TEST._Samples = En.shape[2]
				
				TEST.Covmat(plot=False)
				TEST.Cormat(plot=False)
				TEST.WhitenData(plot=False)
				transcov = TEST.TransCovmat(plot=False)
			
				#Transcovariance matrix (2x because matrix is symmetric)
				all_transcovs[(n*N)+(m*N):(m*N)+N+(n*N),(n*N):(n*N)+N] = transcov
				all_transcovs[(n*N):(n*N)+N,(n*N)+(m*N):(m*N)+N+(n*N)] = transcov
				
				#Correlation matrix (2x because matrix is symmetric)
				all_cormats[(n*N)+(m*N):(m*N)+N+(n*N),(n*N):(n*N)+N] = TEST._cormat
				all_cormats[(n*N):(n*N)+N,(n*N)+(m*N):(m*N)+N+(n*N)] = TEST._cormat

				#Covariance matrix (2x because matrix is symmetric)
				all_covmats[(n*N)+(m*N):(m*N)+N+(n*N),(n*N):(n*N)+N] = TEST._covmat
				all_covmats[(n*N):(n*N)+N,(n*N)+(m*N):(m*N)+N+(n*N)] = TEST._covmat
			
				c += 1
	
		return all_transcovs, all_cormats, all_covmats

	'''Function to create 300 transcovriance matrices given COSEBI data transformed to Gaussian data'''
	def create_Gauss(self):
		'''
		@Params:
		En = COSEBIs, mult. dim. arr
		bins = Number of redshift bins, int
		'''
		
		En = self._En
		bins = self._bins
		
		TEST = UsefulFuncs(type='En',N=250,savedir='Created_En/log250_max400_slop1.3',Gaussian=True,mini=True,vmin_zero=False)
	
		TEST._Gaussian = True
		gauss = []
		
		print ('Running Add Test 300x on Gaussian data...')
		
		for g in range(300):
			c = 0
			trans_gauss = np.zeros((bins*En.shape[1],bins*En.shape[1]))
		
			#Looping over all redshift bin combinations
			for n in range(bins):
				for m in range(bins-n):
				
					TEST._data = En[c,:,:]
					TEST._Samples = En.shape[2]
					TEST._bins = 1
					TEST._N = En.shape[1]
				
					TEST.Covmat(plot=False)
					TEST.Cormat(plot=False)
					TEST.WhitenData(plot=False)
					transcov = TEST.TransCovmat(plot=False)
				
					#(Gaussian) Transcovariance matrix (2x because matrix is symmetric)
					trans_gauss[(n*7)+(m*7):(m*7)+7+(n*7),(n*7):(n*7)+7] = transcov
					trans_gauss[(n*7):(n*7)+7,(n*7)+(m*7):(m*7)+7+(n*7)] = transcov
				
					c += 1
			
			if g%10 == 0:
				print ('{0}/300 tests done.'.format(g))	
				
			gauss.append(trans_gauss)
		
		return gauss