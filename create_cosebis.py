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
		self._data_xip = None
		self._data_xim = None
		self._thetas = None
		self._noise = noise
		self._datContent = [i.strip().split() for i in open("roots_0.5_300.dat").readlines()]
		self._p = 10
		self._dtheta = None
		
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
	
		print ("Cosebi modes created")
		
		return En
	
	def isfloat(self,value):
		try:
			float(value)
			return True
		except ValueError:
			return False
