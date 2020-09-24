import numpy as np
import matplotlib.pyplot as plt
import sys 
import os
import re
import scipy.integrate as si
import T_functions_log as Tlog
from scipy import integrate	
		
class Create_Cosebis(object):

	def __init__(self,noise,dir,tdir):
		self._dir = dir
		self._files_xip_noisy = []
		self._files_xim_noisy = []
		self._files_xip = []
		self._files_xim = []
		self._file_theta = tdir
		self._data_xip = None
		self._data_xim = None
		self._thetas = None
		self._noise = noise
		self._datContent = [i.strip().split() for i in open("roots_0.5_300.dat").readlines()]
		self._p = 10
		self._dtheta = None

	''' Function to read specific files, not necessary when given the xips and xims beforehand'''
	def read_files(self):
		
		dir = self._dir
		print ('='*20)
		print ('Searching folder {0} for xi data-files'.format(dir))
		pindxs_noisy = []
		mindxs_noisy = []
		pindxs = []
		mindxs = []

		# r=root, d=directories, f = files
		for r, d, f in os.walk(dir):
			if len(d) != 0:
				dirs = d
			
		folders = []
		for folder in dirs:
			folder_files_xip = []
			folder_files_xim = []
			folder_files_xip_noisy = []
			folder_files_xim_noisy = []
			pindxs = []
			mindxs = []
			for r,d,f in os.walk(dir+folder):
				for file in f:
					if 'xip' in file and 'noise' in file:
						pindxs_noisy.append(int(re.search(r'\d+', file).group()))
						folder_files_xip_noisy.append(os.path.join(r,file))
						continue
					elif 'xim' in file and 'noise' in file:
						mindxs_noisy.append(int(re.search(r'\d+', file).group()))
						folder_files_xim_noisy.append(os.path.join(r,file))
						continue
					elif 'xip' in file:
						pindxs.append(int(re.search(r'\d+', file).group()))
						folder_files_xip.append(os.path.join(r,file))
						continue
					elif 'xim' in file:
						mindxs.append(int(re.search(r'\d+', file).group()))
						folder_files_xim.append(os.path.join(r,file))
						continue
					elif 'theta' in file:
						self._file_theta = os.path.join(r,file)
						continue

			if len(folder_files_xip) != 0 and len(folder_files_xim) != 0:
				self._files_xip.append(np.array(folder_files_xip)[np.argsort(pindxs)])
				self._files_xim.append(np.array(folder_files_xim)[np.argsort(mindxs)])
				folders.append(int(re.search(r'\d+', folder).group()))

		folders = self.sort(folders)

		self._files_xip_noisy = np.array(self._files_xip_noisy)
		self._files_xim_noisy = np.array(self._files_xim_noisy)
		self._files_xip = np.array(self._files_xip)[folders]
		self._files_xim = np.array(self._files_xim)[folders]
	
		#Data is in order (1,1),...(1,10),(2,2),...(9,10),(10,10)
	
	'''Function to sort the above loaded files'''
	def sort(self,arr):
		arr_c = np.copy(arr)
		dictionary = dict({'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[],'0':[]})
		indx_dictionary = dict({'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[],'0':[]})
		indx = 0
		for i in range(len(arr_c)):
			digit = str(arr_c[i])[-1]
			dictionary[digit].append(arr_c[i])
			indx_dictionary[digit].append(indx)
			indx += 1
		indices_arr = np.array([])
		for indx, key in enumerate(dictionary):
			arr_key = dictionary[key]
			arr_indx = np.argsort(arr_key)
			indices = indx_dictionary[key]
			indices = np.array(indices)[arr_indx]
			indices_arr = np.concatenate((indices_arr,np.array(indices)))
		
		indices_arr = np.array(indices_arr)
		indices_arr = indices_arr.astype(int)
		
		return indices_arr
	
	'''Extracting the xi-data from the files'''
	def xi_data(self):
		
		print ('='*20)
		print ('Loading the files ...')
		
		if self._noise:
			files_xim = self._files_xim_noisy
			files_xip = self._files_xip_noisy
		else:
			files_xim = self._files_xim
			files_xip = self._files_xip
			
		
		binned_data_xip = np.zeros((files_xip.shape[0],files_xip.shape[1],250))
		binned_data_xim = np.zeros((files_xim.shape[0],files_xim.shape[1],250))
		
		for d in range(len(files_xim)):
			for f in range(len(files_xim[d])):
			
				xim = np.loadtxt(files_xim[d,f])
				xip = np.loadtxt(files_xip[d,f])
				
				binned_data_xip[d,f,:] = xip
				binned_data_xim[d,f,:] = xim
		
			
		self._data_xip = binned_data_xip
		self._data_xim = binned_data_xim
		
		print ('xi data loaded with shapes {0}'.format(self._data_xip.shape))
		self._thetas = np.loadtxt(self._file_theta)
		print ('thetas loaded with shape {0}'.format(self._thetas.shape))
		self._bins = self._data_xip.shape[0]
		
		'''
		binned_data arrays shape: (55,32,931),
		where 55 is number of bin pairs,
		32 is number of thetas
		931 is number of simulations
		'''
		
	def cosebis(self,n=7,min=False,get_data=True,ranges='1.0_400'):
	
		if get_data:
			self.read_files()
			self.xi_data()
		
		all_xi_p = self._data_xip
		all_xi_m = self._data_xim
		thetas = self._thetas
		
		theta_min = np.min(thetas)
		theta_max = np.max(thetas)
		
		samples_p = all_xi_p.shape[1]
		samples_m = all_xi_m.shape[1]
	
		samples = samples_p
	
		if samples_p < samples_m:
			xi_m = all_xi_m[:,0:samples_p,:]
		elif samples_m < samples_p:
			xi_p = all_xi_p[:,0:samples_m,:]
			samples = samples_m
			
		bins = self._data_xip.shape[0]
		En = np.zeros((bins,n,samples))	
		
		print ('='*20)
		print ('Creating Cosebis for {0} modes ...'.format(n))
		print ('... Might take a while ...')
		
		datContent = [i.strip().split() for i in open("roots_{0}.dat".format(ranges)).readlines()]
		cvals = [i.strip().split() for i in open("c_values_{0}.dat".format(ranges)).readlines()]
		Nn = np.loadtxt('norms_{0}.dat'.format(ranges))
		

		nmax=n
		r = np.zeros((nmax,nmax+1))
		c = np.zeros((nmax,nmax+2))
		
		#ROOTS MATRIX
		for i in range(r.shape[0]):
			dat = datContent[i]
			for j in range(r.shape[1]):
				float_true = self.isfloat(dat[j])
				if float_true:
					r[i,j] = np.float128(dat[j])
				elif "I" in dat[j]:
					r[i,j] = np.float128(str(np.abs(to_complex(dat[j]))))
				else:
					break
		
		#C MATRIX
		for i in range(nmax):
			cval = cvals[i]
			for j in range(c.shape[1]):
				float_true_c = self.isfloat(cval[j])
				if float_true_c:
					c[i,j] = np.float128(cval[j])
				
		#N-values	
		for i in range(nmax):
			Nn[i] = np.float128(str(Nn[i]))
				
		colors = ['grey','red','green','blue','orange','pink','brown']*10
		
		plot_bins = [0,10,19,27,34,40,45,49,52,54]
		act_bins = [11,22,33,44,55,66,77,88,99,1010]
		W = 0
		for bin in range(bins):
			print (bin)
			Tps = []
			Tms = []
			for mode in range(1,n+1):
				
				T = Tlog.T_functions_log(thetas,mode,Nn[mode-1])
				T.T_m_log(c,mode)
				T.T_p_log(r,mode)
				T_p = T._t_p
				T_m = T._t_m
				
				Tps.append(T_p)
				Tms.append(T_m)
				
				integrands = []
				for sample in range(1,samples+1):
					
					sample_xi_p = all_xi_p[bin,sample-1,:]
					sample_xi_m = all_xi_m[bin,sample-1,:]
					
					factor = []
					
					
					if min:
						factor = (T_p*sample_xi_p) - (T_m*sample_xi_m)
					else:
						factor = (T_p*sample_xi_p) + (T_m*sample_xi_m)
					
					integrand = thetas*factor
					integrands.append(integrand)
					dthetas = thetas[1:]-thetas[:-1]
					
					if self._dtheta == None:
						En[bin,mode-1,sample-1] = (np.pi**2)*0.5*integrate.trapz(thetas*factor,thetas)/((180*60)**2)
					else:
						En[bin,mode-1,sample-1] = (np.pi**2)*0.5*integrate.trapz(integrand,dx=self._dtheta)/((180*60)**2)
	
		if min:
			print ("Cosebi modes created and stored in 'Bn_noise_0.5_100.npy'.")
		else:
			print ("Cosebi modes created and stored in 'En_noise_0.5_100.npy'.")
			
		return En
	
	def isfloat(self,value):
		try:
			float(value)
			return True
		except ValueError:
			return False
