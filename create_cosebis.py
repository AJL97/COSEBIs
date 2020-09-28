'''
TO DO: 
- SPLIT cosebis FUNCTION INTO MULTIPLE SMALLER ONES SUCH THAT IT IS MORE CLEAR
- MAKE SURE THAT THE INIT DOES NOT HOLD TOO MANY STUFF
'''

import numpy as np
import matplotlib.pyplot as plt
import sys 
import os
import re
import math
import scipy.integrate as si
from scipy import integrate	

class T_functions_log(object):
	
	def __init__(self,thetas,nmax,Nn):
		
		self._thetas = thetas
		self._z = np.log(thetas/min(thetas))
		self._zmax = np.log(max(thetas)/min(thetas))
		self._nmax = nmax
		self._c = np.zeros((nmax,nmax+2))
		self._r = np.zeros((nmax,nmax+1))
		self._t_p = None
		self._t_m = None
		self._Nn = Nn
	
	def a_n_2(self,c):
		
		tot_sum = 0
		for i in range(self._nmax+2):
			tot_sum += (c[self._nmax-1,i]*math.factorial(i))/((-2)**(i+1))
		
		return 4*self._Nn*tot_sum
	
	def a_n_4(self,c):
		
		tot_sum = 0
		
		for i in range(self._nmax+2):
			tot_sum += (c[self._nmax-1,i]*math.factorial(i))/((-4)**(i+1))
		
		return 12*self._Nn*tot_sum
		
	def d_n_m(self,m,c):
		c_nm = self._Nn*c[self._nmax-1,m]
		tot_sum = 0
		for i in range(m,self._nmax+2):
			tot_sum += self._Nn*c[self._nmax-1,i]*math.factorial(i)*((-2)**(m-i-1))*((3*(2**(m-i-1)))-1)
		return c_nm + ((4/math.factorial(m))*tot_sum)
		
	def T_m_log(self,c,nmax):
		
		a_n2 = self.a_n_2(c)
		a_n4 = self.a_n_4(c)
			
		t_m = (a_n2*np.exp(-2*self._z)) - (a_n4*np.exp(-4*self._z))

		for m in range(self._nmax+1):
			d_nm = self.d_n_m(m,c)
			t_m += d_nm*(self._z**m)
		
		self._t_m = t_m
	
	def T_p_log(self,r,nmax):
		
		t_p = np.ones_like(self._z)
		
		for j in range(1,nmax+2):
			t_p *= (self._z - r[nmax-1,j-1])
		t_p *= self._Nn
		
		self._t_p = t_p
		
class Create_Cosebis(object):

	def __init__(self):
		self._data_xip = None
		self._data_xim = None
		self._thetas = None
		self._dtheta = None
			    
	def cosebis(self,n=7,min=False,ranges='1.0_400'):
		
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
		
		#ROOTS MATRIX
		r = self.roots(datContent,nmax)
		
		#C MATRIX
		c = self.c_matrix(cvals,nmax)
				
		#N-values	
		for i in range(nmax):
			Nn[i] = np.float128(str(Nn[i]))
				
		colors = ['grey','red','green','blue','orange','pink','brown']*10
		
		plot_bins = [0,10,19,27,34,40,45,49,52,54]
		act_bins = [11,22,33,44,55,66,77,88,99,1010]
		W = 0
		
		for bin in range(bins):
		
			Tps = []
			Tms = []
			for mode in range(1,n+1):
				
				T = T_functions_log(thetas,mode,Nn[mode-1])
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
	
	def c_matrix(self,cvals,nmax):
		
		c = np.zeros((nmax,nmax+2))
		
		for i in range(nmax):
			cval = cvals[i]
			for j in range(c.shape[1]):
				float_true_c = self.isfloat(cval[j])
				if float_true_c:
					c[i,j] = np.float128(cval[j])
		
		return c
	
	def roots(self,datContent,nmax):
		
		r = np.zeros((nmax,nmax+1))
		
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
		return r
	
	def isfloat(self,value):
		try:
			float(value)
			return True
		except ValueError:
			return False
