'''
Creating the T-functions using eqs. 36 and 38 from https://arxiv.org/pdf/1002.2136.pdf
'''

import numpy as np
import math
from scipy.special import gamma, gammainc, exp1
import scipy.integrate as si
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class T_functions_log(object):
	
	def __init__(self,thetas,nmax,Nn):
		
		self._thetas = thetas
		self._z = np.log(thetas/min(thetas))
		self._zmax = np.log(max(thetas)/min(thetas))
		#print (self._zmax)
		self._nmax = nmax
		self._c = np.zeros((nmax,nmax+2))
		self._r = np.zeros((nmax,nmax+1))
		self._t_p = None
		self._t_m = None
		self._Nn = Nn
	
	def inc_gamma(self,k, j):
		gamma_2 = gammainc(j+1,k*self._zmax)
		gamma_tot =  gamma_2
		return gamma_tot/((-k)**(j+1))

	def J(self,k,j):
	
		self.k = k
		self.j = j
		y = si.romberg(self.int_funct,0,self._zmax)
		return y
		
	def N_1(self,c10,c11):
		J = self.J
		denom_A = c10*((c10*J(1,0))+(c11*J(1,1))+(J(1,2)))
		denom_B = c11*((c10*J(1,1))+(c11*J(1,2))+(J(1,3)))
		denom_C = (c10*J(1,2)) + (c11*J(1,3)) + J(1,4)
		denom = denom_A + denom_B + denom_C
		nom = np.exp(self._zmax)-1
		return np.sqrt(nom/denom)
		
	def N(self):
		total_sum = 0
		J = self.J
		for i in range(self._nmax+2):
			c_ni = self._c[self._nmax-1,i]
			sum_j = 0
			
			for j in range(self._nmax+2):
				sum_j += self._c[self._nmax-1,j]*J(1,i+j)
			total_sum += sum_j*c_ni
		Nn = np.sqrt((np.exp(self._zmax) - 1)/total_sum)
		self._Nn = Nn
		
	def int_funct(self,z):
		return np.exp(self.k*z)*(z**self.j)
		
	def function_to_integrate(self,z):
		product = 1
		for i in range(self._nmax+1):
			product *= (z - self._r[self._nmax-1,i])
		return np.exp(z)*(product**2)
		
	def normgral(self):
		norm = si.romberg(self.function_to_integrate,0,self._zmax)
		return np.sqrt((np.exp(self._zmax)-1)/norm)
	
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