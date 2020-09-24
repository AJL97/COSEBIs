''' Example program of how to create COSEBIs and perform the gaussian Addtest on'''

import numpy as np
import create_cosebis as cc
import RunAllTests as tests
import matplotlib.pyplot as plt
import sys

'''Function to create the COSEBIs given specific xipms files'''
def create_Cosebis(xip,xim,thetas,dir='',tdir='',dtheta=None,ranges='0.5_300'):
	
	CC = cc.Create_Cosebis(noise=False,dir=dir,tdir=tdir)
	
	#Setting the xipm's and thetas	
	CC._data_xip = xip
	CC._data_xim = xim
	CC._thetas = thetas
	
	if dtheta != None:
		CC._dtheta = dtheta
	
	#Storing the E-modes in En	
	En = CC.cosebis(n=7,min=False,get_data=False,ranges=ranges)
	
	return En		
	
'''Function to test the COSEBIs on Gaussianity'''
def Test_COSEBIs(En,bins):
	'''
	@Params:
	En = COSEBIs, mult. dim. arr
	bins = Number of redshift bins, int
	'''
	
	DoTests = tests.RunTests(En,bins)
	
	#Executing the Addtest, resulting in the transcovariance, correlation, and covariance matrix	
	trans, cor, cov = DoTests.create_TransCorMats(gauss=False)
	
	#Executing the Addtest 300x, but now with a gaussian set, to determine significance 
	gauss = DoTests.create_Gauss()
		
	return trans, cor, cov, gauss
	
def main():
	
	#Loading the xip/xim data
	xips = np.load('Created_xipm/lin10000_slop0.9_max400/xip_88.npy')
	xims = np.load('Created_xipm/lin10000_slop0.9_max400/xim_88.npy')
	thetas = np.loadtxt('data_xi_pm/lin10000_slop0.9_max400/thetas/theta.txt')
	
	#Setting the angular range
	indxs = np.where(thetas <= 300)
	thetas = thetas[indxs]
	xips = xips[:,:,indxs]
	xims = xims[:,:,indxs]
	
	#Creating the E-modes
	En = create_Cosebis(xips,xims,thetas)
	
	#Running the Addtest
	trans,cor,cov,gauss = Test_COSEBIs(En,1)
	
	#Calculating the significance of the results
	trans = np.ma.masked_where(trans==0,trans)
	trans = (trans-np.nanmean(gauss,axis=0))/np.nanstd(gauss,axis=0)
	
	plt.imshow(abs(trans),origin='lower')
	plt.colorbar()
	plt.show()
	
if __name__ == "__main__":
	sys.exit(main())