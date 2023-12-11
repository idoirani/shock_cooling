import numpy as np
from astropy import table
from astropy.io import ascii,fits
import matplotlib.pyplot as plt
from astropy import constants 
from scipy.optimize import curve_fit
import math
import os 
import glob 
from astropy.time import Time
from astropy.io import fits
import astropy.wcs as fitswcs
import astropy.units as u
from scipy import interpolate
plt.rcParams.update({
	"text.usetex": True,
	"font.family": "sans-serif",
	"font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
	"text.usetex": True,
	"font.family": "serif",
	"font.serif": ["Palatino"],
})

c=constants.c.value*1e10
cAA = constants.c.value*1e10
pc=constants.pc.cgs.value
Mpc=1e6*pc
jy=1e-23 #erg/s/cm^2/Hz



def zp_band_vega(Filter):
	# in angstrom
	Lambda = vega_spec[:,0]
	Filter["col1"] = Filter["col1"].astype(float)
	T_filt_interp=np.interp(Lambda,Filter["col1"],Filter["col2"],left=0,right=0)
	trans_flux=np.trapz( vega_spec[:,1]*Lambda*T_filt_interp,Lambda)
	mag=-2.5*np.log10(trans_flux)+0.03
	return  mag


def zp_piv_wl_ab(piv_wl):
	# in angstrom
	c=constants.c.value*1e10
	zp_band_ab=-2.5*np.log10((piv_wl**2)/c)-48.6
	return  zp_band_ab

def zp_band_ab(T):
	pivot_wl2=np.trapz(T["col2"]*T["col1"],T["col1"])/ np.trapz(T["col2"]/T["col1"],T["col1"])
	pivot_wl=np.sqrt(pivot_wl2)
	zp_band_ab=zp_piv_wl_ab(pivot_wl)
	
	return  zp_band_ab, pivot_wl


#unchecked
def SynPhot(Lambda,Flux,Filter, sys='AB',ret_flux=False):
	if sys.lower()=='vega':
		ZPflux_vec=np.interp(Lambda,vega_spec[:,0] ,vega_spec[:,1] ,left=0,right=0)
	if sys.lower()=='ab':
		ZPflux=3631
		ZPflux=ZPflux*1e-23
		ZPflux_vec=ZPflux*cAA*(Lambda**(-2))

	Filter["col1"] = Filter["col1"].astype(float)
	T_filt_interp=np.interp(Lambda,Filter["col1"],Filter["col2"],left=0,right=0)

	trans_flux=np.trapz(Flux*Lambda*T_filt_interp,Lambda)
	norm_flux=np.trapz(ZPflux_vec*T_filt_interp*Lambda,Lambda)

	mag=-2.5*np.log10(trans_flux/norm_flux)
	if sys.lower()=='vega':
		mag=-2.5*np.log10(trans_flux/norm_flux)+0.03
	if ret_flux:
		return trans_flux#/norm_flux
	return mag  

def delta_Vega(Filter):
	ZPflux=3631
	ZPflux=ZPflux*1e-23
	ZPflux_vec=ZPflux*cAA*(vega_spec[:,0]**(-2))

	mag =  SynPhot(vega_spec[:,0] ,ZPflux_vec,Filter, sys='AB')
	#mag = SynPhot(vega_spec[:,0] ,vega_spec[:,1] ,Filter, sys='AB')
	mvega  =  SynPhot(vega_spec[:,0] ,vega_spec[:,1],Filter, sys='AB')#+0.03

	return  mag - mvega
def delta_Vega2(Filter):

	mvega = SynPhot(vega_spec[:,0] ,vega_spec[:,1] ,Filter, sys='Vega') 
	mag = SynPhot(vega_spec[:,0] ,vega_spec[:,1] ,Filter, sys='AB')

	return  mag - mvega


#


def calibrate_spectra_to_phot(lam,f_lam,t_spec,t_data,M_data,M_err,filter_transmission,n=7,plot_lc=False):

	poly,_=polynomial_smooth(t_data,M_data,M_err,n=n)
	realmag=poly.eval(t_spec)
	synmag=SynPhot(lam,f_lam,filter_transmission)

	corrective_factor=10**(-0.4*(realmag-synmag))
	f_lam_corr=corrective_factor*f_lam
	if plot_lc:
		t0=np.min(t_data)
		tmax=np.max(t_data)
		time_vec=np.linspace(t0,tmax,1000) 

		plt.figure()
		plt.plot(time_vec,poly.eval(time_vec),'r--',label='{0}th order polynomial fit'.format(n))
		plt.errorbar(t_data,M_data,M_err,ls='',marker='s',capsize=0,color='r',label='data')
		plt.errorbar(t_spec,realmag,ls='',marker='*',markersize=10,capsize=0,color='g',label='interpolated magnitude')

		plt.legend()
		plt.xlabel('time (d)')
		plt.ylabel('magnitude')
		plt.gca().invert_yaxis()
		plt.show()
	return f_lam_corr


def generate_syn_phot_sequence(spectra,t_spec,t_data,M_data,M_err,filter_transmission,n=7,plot_lc=False,show=True,sys='AB'):

	poly,_=polynomial_smooth(t_data,M_data,M_err,n=n)
	synmag=[]
	for spec in spectra:
		lam=spec[spec.colnames[0]]
		f_lam=spec[spec.colnames[1]]
		synmag.append(SynPhot(lam,f_lam,filter_transmission,sys=sys))

	if plot_lc:
		t0=np.min(t_data)
		tmax=np.max(t_data)
		time_vec=np.linspace(t0,tmax,1000) 
		plt.figure()
		plt.plot(time_vec,poly.eval(time_vec),'r--',label='{0}th order polynomial fit'.format(n))
		plt.errorbar(t_data,M_data,M_err,ls='',marker='s',capsize=0,color='r',label='data')
		plt.errorbar(t_spec,synmag,ls='',marker='*',markersize=10,capsize=0,color='g',label='synthetic magnitude')

		plt.legend()
		plt.xlabel('Time (d)')
		plt.ylabel('Magnitude {0}'.format(sys))
		plt.gca().invert_yaxis()
		if show:
			plt.show()

	return synmag







def bb_F(lam,T,r,EBV=0,EBV_mw=0, R_v=3.1,z=0,LAW='MW'):
	A_v=R_v*EBV
	A_v_mw=3.1*EBV_mw
	
	flux=1.191/(lam**(5))/(np.exp((1.9864/1.38064852)/(lam*T))-1)*(np.pi*(r)**2) 
	#flux=apply(ccm89(lam*1e4, A_v, R_v), flux)
	#flux=apply(ccm89(lam*1e4*(1+z), A_v_mw, 3.1), flux)
	flux=apply_extinction(1e4*lam,flux,EBV=EBV,EBV_mw=EBV_mw,R_v=R_v,z=z,LAW=LAW)
	return flux  

from extinction import ccm89,calzetti00, apply

def apply_extinction(lam,flam,EBV=0,EBV_mw=0, R_v=3.1,z=0,LAW='MW'):
	if EBV>0:
		if LAW == 'Cal00':
			A_v=R_v*EBV
			flux=apply(calzetti00(lam, A_v, R_v,unit='aa'), flam)
		#elif LAW == 'SMC':
		#	ex = SMC_Gordon03(lam*(1+z))
		#	ex.set_EBmV(EBV)
		#	A_v = ex.Av
		#	Alam  = A_v*ex.AlamAv 
		#	#import ipdb; ipdb.set_trace()
		#	flux = flam*10**(-0.4*Alam/0.87)
		#elif LAW == 'LMC':
		#	ex = LMC_Gordon03(lam*(1+z))
		#	ex.set_EBmV(EBV)
		#	A_v = ex.Av
		#	Alam  = A_v*ex.AlamAv 
		#	flux = flam*10**(-0.4*Alam/1.08)
		else:
			A_v=R_v*EBV
			flux=apply(ccm89(lam, A_v, R_v,unit='aa'), flam)
	else:
		flux = flam
	if EBV_mw>0:
		A_v_mw=3.1*EBV_mw
		flux=apply(ccm89(lam*(1+z), A_v_mw, 3.1,unit='aa'), flux)
	return flux  


def convert_band(data, band, band_new, input1 = 'ZTF_r', input2 = 'ZTF_g'):
	data_new = data.copy()
	cond_filter = data['filter'] == band
	times = np.array(data['t'][cond_filter])
	mag1 = np.array([get_inter_mag(t, data, input1)[0] for t in times])
	mag2 = np.array([get_inter_mag(t, data, input2)[0] for t in times])
	mag1_err = np.array([get_inter_mag(t, data, input1)[1] for t in times])
	mag2_err = np.array([get_inter_mag(t, data, input2)[1] for t in times])
	color = mag2-mag1
	color[np.isnan(color)] =0 
	color_err = np.sqrt(mag1_err**2+mag2_err**2)
	color_err[np.isnan(color_err)] =0 

	f_corr = filter_correction(band, band_new, input1, input2)
	corr = f_corr(color)
	corr_err = f_corr(color-color_err) + corr
	data_new['AB_MAG'][cond_filter] = data['AB_MAG'][cond_filter] + corr
	data_new['AB_MAG_ERR'][cond_filter] = np.sqrt(data['AB_MAG_ERR'][cond_filter]**2 + corr_err**2)
	data_new['filter'][cond_filter] = band_new
	return data_new, corr


def get_inter_mag_array(t_array, phot, filt, t_max = 3, verbose = False,t_col = 't_rest'):
	lis = list(map(lambda t: get_inter_mag(t, phot, filt, t_max = t_max, verbose = verbose,t_col = t_col), t_array))
	arr = np.array(lis)
	return arr


def get_inter_mag(t, phot, filt,mag_col = 'AB_MAG', mag_col_err = 'AB_MAG_ERR',filt_col = 'filter',t_col = 't_rest',t_max=2):
	mag = phot[mag_col][phot[filt_col]==filt]
	magerr = phot[mag_col_err][phot[filt_col]==filt]
	T  = phot[t_col][phot[filt_col]==filt]  
	if len(mag) == 0:
		print('no points with the filter {0}'.format(filt))
		return np.nan,np.nan ,np.nan 
	elif len(mag) == 1:
		if (T[0] - t)<0.1:
			return mag[0],magerr[0],T[0]
		else: 
			print('not enough points with the filter {0}'.format(filt))
			return np.nan,np.nan ,np.nan             
	if (t>np.max(T))|(t<np.min(T)):
		print('outside of valid region for interpolation')
		return np.nan,np.nan ,np.nan  
	else:     
		i1 = np.argwhere(t>T)[-1]
		i2 = np.argwhere(t>T)[-1]+1
		t1 = T[i1]
		t2 = T[i2]
		m1 = mag[i1]
		m2 = mag[i2]
		m1_err = magerr[i1]
		m2_err = magerr[i2]
		a = (t2  - t)/(t2-t1)
		b = (t-t1)/(t2-t1)
		m = a*m1 + b*m2
		m_err = np.sqrt(a**2*m1_err**2 + b**2*m2_err**2)

		#if t2 - t1 > 5:
		#    print('warning: dt > {0} days'.format(t_max))
		if min((t2  - t),(t-t1)) > t_max:
			print('maximum time diff > {0} days. Returning nan'.format(t_max))
			return np.nan,np.nan ,np.nan  
		return m[0],m_err[0],(t2-t1)[0]



def hierarchical_offset(dat,band0,band_list,t_max = 1,t_col ='t'):
	dat_filt = {}
	off = {}
	for filt in band_list:
		dat_filt[filt] = dat[dat['filter']==filt]
	dat_filt[band0] = dat[dat['filter']==band0]
	lens = [len(np.unique(dat_filt[filt][t_col])) for filt in band_list]
	args = np.flip(np.argsort(lens))

	for a in args: 
		filt = band_list[a]
		try:
			if (len(dat_filt[filt])<len(dat_filt[band0]))&(len(dat_filt[filt])!=0):
				if np.max(dat_filt[filt][t_col])<np.min(dat_filt[band0][t_col]):
					off[filt] = 0 
				elif np.min(dat_filt[filt][t_col])>np.max(dat_filt[band0][t_col]):
					off[filt] = 0
				else:
					mag_int = get_inter_mag_array(dat_filt[filt][t_col], dat, band0, t_max = t_max,t_col = t_col)[:,0]
					mag_re = dat_filt[filt]['AB_MAG']
					diff = mag_int - mag_re
					iqr = np.nanpercentile(diff,75)-np.nanpercentile(diff,25)
					if iqr>0:
						diff = diff[np.abs(diff-np.nanmedian(diff))<2*iqr]
					#mag_int = np.interp(dat_filt[filt][t_col],dat_filt[band0][t_col],dat_filt[band0]['AB_MAG'])
					
					off[filt] = np.nanmedian(diff)
			elif (len(dat_filt[filt])==0)|(len(dat_filt[band0])==0):
				off[filt] = 0 
			else:
				if np.min(dat_filt[band0][t_col])>np.max(dat_filt[filt][t_col]):
					off[filt] = 0 
				elif np.max(dat_filt[band0][t_col])<np.min(dat_filt[filt][t_col]):
					off[filt] = 0 
				else:
					mag_int = get_inter_mag_array(dat_filt[band0][t_col], dat, filt, t_max = t_max,t_col = t_col)[:,0]
					#mag_int = np.interp(dat_filt[band0][t_col],dat_filt[filt][t_col],dat_filt[filt]['AB_MAG'])
					mag_re = dat_filt[band0]['AB_MAG']
					diff = mag_re-mag_int
					iqr = np.percentile(diff,75)-np.percentile(diff,25)
					diff = diff[np.abs(diff-np.nanmedian(diff))<2*iqr]
					#mag_int = np.interp(dat_filt[filt][t_col],dat_filt[band0][t_col],dat_filt[band0]['AB_MAG'])
					
					off[filt] = np.nanmedian(diff)
		except: 
			try:
					mag_int = get_inter_mag_array(dat_filt[band0][t_col], dat, filt, t_max = t_max,t_col = t_col)[:,0]
					#mag_int = np.interp(dat_filt[band0][t_col],dat_filt[filt][t_col],dat_filt[filt]['AB_MAG'])
					mag_re = dat_filt[band0]['AB_MAG']
					diff = mag_re-mag_int
					iqr = np.percentile(diff,75)-np.percentile(diff,25)
					diff = diff[np.abs(diff-np.nanmedian(diff))<2*iqr]
					#mag_int = np.interp(dat_filt[filt][t_col],dat_filt[band0][t_col],dat_filt[band0]['AB_MAG'])
					off[filt] = np.nanmedian(diff)
			except:         
				import ipdb; ipdb.set_trace()


		dat_filt[filt]['AB_MAG'] = dat_filt[filt]['AB_MAG'] + off[filt]
		dat_filt[filt]['absmag'] = dat_filt[filt]['absmag'] + off[filt]
		dat[dat['filter']==filt] = dat_filt[filt]
		dat['filter'][dat['filter']==filt] = band0
		dat_filt[band0] = dat[dat['filter']==band0]
	return dat, off

def plot_offset_corr(dat,off,band0,band_list,t_col = 't' ):
	plt.rcParams.update({
	"text.usetex": False,
	})

	dat_filt = {}
	for filt in band_list:
		dat_filt[filt] = dat[dat['filter']==filt]
	dat_filt[band0] = dat[dat['filter']==band0]

	dat_filt_orig = {}
	for filt in band_list:
		dat_filt_orig[filt] = dat[dat['filter_orig']==filt]
	dat_filt_orig[band0] = dat[dat['filter_orig']==band0]

	try:                
		plt.figure()
		plt.plot(dat_filt[band0]['t'],dat_filt[band0]['AB_MAG'],marker = '*',ls = '', label = band0 +' new', color= 'y', alpha = 0.5)
		for filt in band_list:
			plt.plot(dat_filt_orig[filt]['t'],dat_filt_orig[filt]['AB_MAG'],marker = 'o',ls = '', label = filt +' new')
			plt.plot(dat_filt_orig[filt]['t'],dat_filt_orig[filt]['AB_MAG']-off[filt],marker = 'x',ls = '', label = filt +' original')
		plt.xlabel(t_col+' [days]')
		plt.ylabel('AB Mag')
		plt.gca().invert_yaxis()
		plt.legend()
	except: 
		import ipdb; ipdb.set_trace()
	plt.rcParams.update({
	"text.usetex": True,
	})
	pass

def filter_correction(filt1,filt2, input1 ='ZTF_r' , input2 ='ZTF_g' ):
	T_array = np.logspace(2,5,100)
	mag1 = generate_bb_mag(T_array,filt1)
	mag2 = generate_bb_mag(T_array,filt2)
	mag_r = generate_bb_mag(T_array,input1)
	mag_g = generate_bb_mag(T_array,input2)
	gr_color_bb = mag_g - mag_r
	corr_factor = mag2 - mag1
	corr = interpolate.interp1d(gr_color_bb,corr_factor)

	return corr






def generate_bb_mag(T_array,filt,r = 1e14, d = 3.08e26, Rv=3.1, EBV = 0,EBV_MW = 0, LAW = 'MW', sys = 'AB'):
	v = 67.4*d/3.08e24
	z = v/300000
	Trans  = filter_transmission[filt]
	lam  = np.linspace(1000,20000,1901)
	
	m_array = []
	for i,T in enumerate(T_array):
		flux = bb_F(lam*1e-4,T*1e-4,(r/d)*1e10,EBV=EBV,EBV_mw = EBV_MW,z = z, R_v = Rv, LAW = LAW)*1e-13                                                                  
		m = SynPhot(lam,flux,Trans, sys=sys)
		m_array.append(m) 
	m_array = np.array(m_array) 
	return m_array 












def cosmo_dist(z,    H0 = 67.4, omega_M = 0.315, omega_L = 0.684,omega_R=0,omega_K=0,c= 2.998e5, n=int(1e4)):
	'''
	Compute cosmolgical distance for given z

	Parameters
	----------
	z : float 
		cosmological redshift 
	H0: float
		Hubble's constant. default is Planck 2014 value of 67.32    
	
	omega_M: float
		Matter energy density/critical density. default is Planck 2014 value of 0.32
	omega_L: float
		Vacuum energy density/critical density. default is Planck 2014 value of 0.68
	omega_R: float
		Radiation energy density/critical density. default is 0
	omega_K: float
		Curvature energy density/critical density. default is 0
	c: float
		speed of light in km s^-1
	n: int
		parameter to indicate the resolution of the trapz integration involved in the computation. 
	
	Returns
	-------
	Comv_dist\Lum_dist\Ang_dist\Ang_scale: float
		Comoving\luminosity\angular distance in units of Mpc. 
		Ang_scale in units of Kpc/"

	'''
	x=np.linspace(0,z,n)
	H=H0*np.sqrt(omega_M*(1+x)**3+omega_R*(1+x)**4+omega_K*(1+x)**2+omega_L)
	r=np.trapz(c/H,x)   #proper distance
	if omega_K>0:
		S=np.sin(r)
	elif omega_K<0:
		S=np.sinh(r)
	elif omega_K==0:
		S=r
	Comv_dist=S
	Lum_dist=S*(1+z)
	Ang_dist=S/(1+z)
	arcsec=(180/np.pi)*3600
	Ang_scale=Ang_dist*1e3/arcsec 

	return Lum_dist,Comv_dist,Ang_dist,Ang_scale

def cosmo_vol(z,solid_ang=4*np.pi,H0=67.32,omega_M=0.32,omega_L=0.68,omega_R=0,omega_K=0,c= 2.998e5, n=int(1e3)):
	x=np.linspace(0,z,1000)
	V=0
	Comv_dist_last=0
	for i in range(len(x)):
		_,Comv_dist,Ang_dist,_ = cosmo_dist(z=x[i],H0=H0,omega_M=omega_M,omega_L=omega_L,omega_R=omega_R,omega_K=omega_K,c=c,n=int(1e3))
		dV = solid_ang*(Comv_dist-Comv_dist_last)*Ang_dist**2
		Comv_dist_last=Comv_dist
		V=V+dV
	return V

#def apply_extinction(lam,flux,EBV=0,EBV_mw=0, R_v=3.1,z=0):
#    A_v=R_v*EBV
#    A_v_mw=3.1*Ebv_MW
#    
#    flux=1.191/(lam**(5))/(np.exp((1.9864/1.38064852)/(lam*T))-1)*(np.pi*(r)**2) 
#    flux=apply(ccm89(lam*1e4, A_v, R_v), flux)
#    flux=apply(ccm89(lam*1e4*(1+z), A_v_mw, 3.1), flux)
#
#    return flux  
	
def dist_mod(z,n=1e3,H0=73,omega_M=0.27,omega_L=0.73,omega_R=0,omega_K=0):
	d_Mpc=cosmo_dist(z,H0=H0,omega_M=omega_M,omega_L=omega_L,omega_R=omega_R,omega_K=omega_K ,n=int(n))[0]
	d_pc=1e6*d_Mpc
	dm=5*np.log10(d_pc)-5
	return dm

def mag2flux(mag,piv_wl):
	zp=zp_piv_wl_ab(piv_wl)
	flux=10**(-0.4*(mag-zp)) 
	return flux

def flux2mag(flux,piv_wl):
	zp=zp_piv_wl_ab(piv_wl)
	mag=-2.5*np.log10(flux)+zp 
	return mag

def magerr2fluxerr(magerr,flux):
	fluxerr=np.abs(-2.303/2.5*magerr*flux)
	return fluxerr

def fluxerr2magerr(fluxerr,flux):
	magerr=np.abs(-2.5/2.303*fluxerr/flux)
	return magerr
def maggie2cgs(f):
	return f*3631*1e-23

def f_lam2f_nu(f_lam_AA,lam_AA):
	nu=c/lam_AA
	f_nu=lam_AA*f_lam_AA/nu
	return f_nu

def f_nu2f_lam(f_nu,lam_AA):
	f_lam_AA=(c/lam_AA**2)*f_nu
	return f_lam_AA
def estimate_galaxy_mass_W2(W2mag,dist_Mpc,type = 'all',W2err = 0):
	# according to Wen et al 2013
	lam_AA =46179
	dist_cm = 1e6*constants.pc.cgs.value*dist_Mpc
	nu=c/lam_AA
	f_lam = mag2flux(W2mag,lam_AA)
	f_nu = f_lam2f_nu(f_lam,lam_AA)
	f_nu_err = magerr2fluxerr(W2err,f_nu)
	r_err = f_nu_err/f_nu
	L_nu = f_nu * 4*np.pi*dist_cm**2

	nuLnu=nu*L_nu
	Ls = constants.L_sun.cgs.value
	if type =='all':
		A = -0.04
		B = 1.12
		sigma_A = 0.001
		sigma_B = 0.001
	elif type =='composites':
		A = 1.081
		B = 1.041
		sigma_A = 0.003
		sigma_B = 0.001
	elif type =='H II':
		A = 0.779
		B = 1.019
		sigma_A = 0.002
		sigma_B = 0.001
	elif type =='AGNs':
		A = 1.132
		B = 1
		sigma_A = 0.008
		sigma_B = 0.002        
	elif type =='ETG':
		A = 0.761
		B = 1.044
		sigma_A = 0.001
		sigma_B = 0.001  
	elif type =='LTG':
		A = 0.679
		B = 1.033
		sigma_A = 0.002
		sigma_B = 0.001  

	logM = A+B*np.log10(nuLnu/Ls)
	unc_flux = B*r_err/2.303
	unc = np.sqrt(sigma_A**2+(sigma_B*np.log10(nuLnu/Ls))**2+unc_flux**2)
	
	
	return logM, unc 


def estimate_SFR_UV(f_lam,lam_AA,d_mpc):

	f_nu=f_lam2f_nu(f_lam,lam_AA)
	#d_Mpc,_,_,_=cosmo_dist(z)
	D=d_Mpc*Mpc
	lum_nu=f_nu*4*np.pi*D**2
	SFR=1.46e-28*lum_nu #Salim 2007
	#SFR=1.40e-28*lum_nu #Kennicut 1998

	return SFR


def estimate_SFR_FUV_W4(FUVmag,W4mag,d_Mpc):
	lam_W4 = 23e4
	lam_FUV = 1542

	c = constants.c.value*1e10 
	f_W4 = mag2flux(W4mag,lam_W4)
	f_nuW4=f_lam2f_nu(f_W4,lam_W4)
	D=d_Mpc*Mpc
	lum_nu23=f_nuW4*4*np.pi*D**2
	nu23 = c/lam_W4
	L23 = nu23 * lum_nu23

	f_UV = mag2flux(FUVmag,lam_FUV)
	f_nuUV=f_lam2f_nu(f_UV,lam_FUV)
	D=d_Mpc*Mpc
	lum_nuUV=f_nuUV*4*np.pi*D**2
	nuUV = c/lam_FUV
	L_UV = nuUV * lum_nuUV

	L_UV_corr = L_UV + 3.89*L23 #Cortese et al 2012/Hao et al 2011
	L_nu_UV_corr = L_UV_corr/nuUV
	SFR=1.46e-28*L_nu_UV_corr #Salim et al 2007

	return SFR

def estimate_SFR_NUV_W4(NUVmag,W4mag,d_Mpc,NUVmag_err = None,W4mag_err = None, err = False,z=0):
	lam_W4 = 23e4
	lam_NUV = 2274

	c = constants.c.value*1e10 
	f_W4 = mag2flux(W4mag,lam_W4)
	f_nuW4=f_lam2f_nu(f_W4,lam_W4)
	D=d_Mpc*Mpc
	lum_nu23=f_nuW4*4*np.pi*D**2
	nu23 = c/lam_W4
	L23 = nu23 * lum_nu23

	f_UV = mag2flux(NUVmag,lam_NUV)
	f_nuUV=f_lam2f_nu(f_UV,lam_NUV)
	D=d_Mpc*Mpc
	lum_nuUV=f_nuUV*4*np.pi*D**2
	nuUV = c/lam_NUV
	L_UV = (1+z)*(nuUV * lum_nuUV)
	alpha = 2.26
	alpha_err = 0.09

	L_UV_corr = L_UV + alpha*L23 #Cortese et al 2012/Hao et al 2011
	
	L_nu_UV_corr = L_UV_corr/nuUV
	SFR=1.46e-28*L_nu_UV_corr #Salim et al 2007


	if err:
		L_UV_err = L_UV*10**(-0.4*NUVmag_err)
		L23_err = L23*10**(-0.4*W4mag_err)
		L_UV_corr_err = np.sqrt(np.nansum([L_UV_err**2,(alpha*L23_err)**2,(alpha_err*L23)**2]))
		L_nu_UV_corr_err = L_UV_corr_err/nuUV
		SFR_err = 1.46e-28*L_nu_UV_corr_err
		return SFR,SFR_err
	else:
		return SFR


def estimate_SFR_W4(W4mag,d_Mpc, err = False):
	lam_W4 = 23e4
	c = constants.c.value*1e10 
	f_W4 = mag2flux(W4mag,lam_W4)
	f_nuW4=f_lam2f_nu(f_W4,lam_W4)
	D=d_Mpc*Mpc
	lum_nu23=f_nuW4*4*np.pi*D**2
	nu23 = c/lam_W4
	L23 = nu23 * lum_nu23
	Lsun = constants.L_sun.cgs.value
	logSFR=0.915*np.log10(L23/Lsun)-8.01 #Cluver et al
	SFR = 10**logSFR
	alpha = 0.023
	beta  = 0.2
	logSFR_Err  = np.sqrt((alpha*np.log10(L23))**2+beta**2)
	SFR_err = logSFR_Err*SFR

	if err:
		return SFR,SFR_err
	else:
		return SFR


def estimate_SFR_W3(W3mag,d_Mpc, err = False):
	lam_W3 = 12e4
	c = constants.c.value*1e10 
	f_W3 = mag2flux(W3mag,lam_W3)
	f_nuW3=f_lam2f_nu(f_W3,lam_W3)
	D=d_Mpc*Mpc
	lum_nu12=f_nuW3*4*np.pi*D**2
	nu12 = c/lam_W3
	L12 = nu12 * lum_nu12
	Lsun = constants.L_sun.cgs.value
	logSFR=1.43*np.log10(L12/Lsun)-13.17 #Cluver et al
	SFR = 10**logSFR
	alpha = 0.161
	beta  = 1.66
	logSFR_Err  = np.sqrt((alpha*np.log10(L12))**2+beta**2)
	SFR_err = logSFR_Err*SFR

	if err:
		return SFR,SFR_err
	else:
		return SFR


def estimate_SFR_UV_MAG(mag,dist_mpc,piv_wl=2274, err = False, magerr = None):
	zp=zp_piv_wl_ab(piv_wl)
	f_lam=10**(-0.4*(mag-zp))
	f_nu=f_lam2f_nu(f_lam,piv_wl)
	D=dist_mpc*Mpc
	lum_nu=f_nu*4*np.pi*D**2
	#SFR=1.4e-28*lum_nu #Kennicut et al 1998
	SFR=1.46e-28*lum_nu #Salim et al 2007
	if err:
		f_lam_err=np.abs(-2.303/2.5*magerr*f_lam)
		SFR_err = (f_lam_err/f_lam)*SFR
		return SFR, SFR_err
	else:    
		return SFR


def estimate_SFR_UV_absMAG(abs_mag,piv_wl):
	zp=zp_piv_wl_ab(piv_wl)
	f_lam=10**(-0.4*(abs_mag-zp))
	f_nu=f_lam2f_nu(f_lam,piv_wl)
	D=10*pc
	lum_nu=f_nu*4*np.pi*D**2
	SFR=1.46e-28*lum_nu #Salim et al 2007
	return SFR

def Integrate_SED(time,data_full,plot_sed=0,z=0,EBV=0,EBV_mw=0, R_v_host=3.1,dm='z'):
	if dm == 'z':
		dm = dist_mod(z)
	L_bol = np.zeros((np.shape(data_full)[1],1)).flatten()
	L_bol_err = np.zeros((np.shape(data_full)[1],1)).flatten()
	for i in range(len(time)):
		lam=piv_wl[data_full[:,i]!=99]
		lam=lam/(1+z)
		dof+=len(lam)
		f_lam=(1+z)*data_full[data_full[:,i]!=99,i]
		f_lam=apply_extinction(lam,f_lam,EBV=EBV,EBV_mw=Ebv_MW, R_v_host=R_v_host,z=z)
		f_err=data_full_e[data_full_e[:,i]!=99,i]
		L_bol[i] = np.trapz(lam,f_lam)
		L_bol_err[i] = np.sqrt(np.trapz(lam,f_err**2))
		
	
	
	return L_bol,L_bol_err

class Polynomial:
	
	def __init__(self, *coefficients):
		""" input: coefficients are in the form a_n, ...a_1, a_0 
		"""
		self.coefficients = list(coefficients) # tuple is turned into a list
	 
	def __repr__(self):
		"""
		method to return the canonical string representation 
		of a polynomial.
		"""
		return "Polynomial" + str(tuple(self.coefficients))
	def eval(self, x):    
		res = 0
		for index, coeff in enumerate(self.coefficients[::-1]):
			res += coeff * x** index
		return res 
	def poly_func(self, t,*coeffs):
		self.coefficients=coeffs
		M=self.eval(t)
		return M

def polynomial_smooth(t_data,M_data,M_err=None,n=5):
	coeffs_init=np.ones_like(np.arange(0,n))
	poly=Polynomial(*coeffs_init)

	popt, _ = curve_fit(poly.poly_func, t_data, M_data, sigma=M_err,p0=coeffs_init)
	return poly, popt     
	

def bin_lc_to_interval(t,f,delta,err='None',ctype='med'):
	'''
	bin flux sequence (f) along time/wavelength (t) and bin to intervals of delta. 
	This function will divide the data range into intervals of size delta, and combine all data points falling into this range. 
	'''
	
	if (len(t)==1):
		if err!='None':
			
			return t,f, err
		else:
			return t,f
	else:    
		
		if ctype == 'med':
			func=np.nanmedian
		elif ctype == 'mean':
			func=np.nanmean
		else: 
			raise Exception('Error: Unknown combine method')
		
		t_range=np.max(t)-np.min(t)
		N=int(math.ceil(t_range/delta))+1
		t_range=np.min(t)+delta*np.arange(N)
		f_bin=np.zeros_like(t_range)
		t_bin=np.zeros_like(t_range)
		if err!='None':
			f_err_bin=np.zeros_like(t_range)
		for n in np.arange(1,N):
			cond = (t>=t_range[n-1])&(t<t_range[n])
			if np.sum(cond)>0:
				f_bin[n-1]=func(f[cond])
				t_bin[n-1]=func(t[cond])
				if err!='None':
					f_err_bin[n-1]=np.sqrt(np.nanmean(err[cond]**2))/len(err[cond])
			else:
				f_bin[n-1]=np.nan
				t_bin[n-1]=np.nan
				if err!='None':
					f_err_bin[n-1]=np.nan
		
		cond2 =  ~np.isnan(f_bin)
		t_bin=t_bin[cond2][0:-1]
		f_bin=f_bin[cond2][0:-1]

		if err!='None':            
			f_err_bin=f_err_bin[cond2][0:-1]
			
			return t_bin,f_bin,f_err_bin   

		else:            
			return t_bin,f_bin



def bin_lc_to_interval2(t,m,delta,m_err='None',ctype='med'):
	'''
	bin mag sequence (m) along time/wavelength (t) and bin to intervals of delta. 
	This function will divide the data range into intervals of size delta, and combine all data points falling into this range. 
	'''
	f = 10**(-0.4*m)
	
	if m_err=='None':
		import ipdb; ipdb.set_trace()
		t_bin,f_bin = bin_lc_to_interval(t,f,delta,ctype=ctype)
		m_bin  = -2.5*np.log10(f_bin)

		return t_bin,m_bin
	else:
		f_err = magerr2fluxerr(m_err,f)
		t_bin,f_bin,f_err_bin = bin_lc_to_interval(t,f,delta,err=f_err,ctype=ctype)
		m_err_bin  = fluxerr2magerr(f_err_bin,f_bin)
		m_bin  = -2.5*np.log10(f_bin)
		return t_bin,m_bin,m_err_bin


def cumulative(variable):
	idx = np.argsort(variable)
	variable = variable[idx]
	cumul = list(np.arange(start = 1,stop = len(variable)+1)/len(variable))
	variable = np.insert(variable, 0, np.min(variable))
	cumul = np.insert(cumul, 0, 0)

	return variable, cumul

def Vega_nMgy_2_ABmag(flux_nMGY, band = 'W2'):
	Vega_mag = 22.5 - 2.5*np.log10(flux_nMGY)
	if band == 'W1':
		AB_mag = Vega_mag +2.699
	elif band == 'W2':
		AB_mag = Vega_mag +3.339
	elif band == 'W3':
		AB_mag = Vega_mag +5.174
	elif band == 'W4':
		AB_mag = Vega_mag +6.620
		

		
	return AB_mag 




def size_mass(Ms,Mp=10**10.2,rp=8.6,a=0.17,b=0.5,d=6):
	r80 = rp*(Ms/Mp)**a*(0.5*(1+(Ms/Mp)**d))**((b-a)/d)
	return r80 





def color(mag1,mag2,magerr1,magerr2):
	col = mag1 - mag2
	colerr = np.sqrt(magerr1**2 + magerr2**2)
	return col,colerr 



def get_inter_mag(t, phot, filt,t_max=2):
	mag = phot['AB_MAG'][phot['filter']==filt]
	magerr = phot['AB_MAG_ERR'][phot['filter']==filt]
	T  = phot['t_rest'][phot['filter']==filt]  
	if len(mag) == 0:
		print('no points with the filter {0}'.format(filt))
		return np.nan,np.nan ,np.nan 
	elif len(mag) == 1:
		if (T[0] - t)<0.1:
			return mag[0],magerr[0],T[0]
		else: 
			print('not enough points with the filter {0}'.format(filt))
			return np.nan,np.nan ,np.nan             
	if (t>np.max(T))|(t<np.min(T)):
		print('outside of valid region for interpolation')
		return np.nan,np.nan ,np.nan  
	else:     
		i1 = np.argwhere(t>T)[-1]
		i2 = np.argwhere(t>T)[-1]+1
		t1 = T[i1]
		t2 = T[i2]
		m1 = mag[i1]
		m2 = mag[i2]
		m1_err = magerr[i1]
		m2_err = magerr[i2]
		a = (t2  - t)/(t2-t1)
		b = (t-t1)/(t2-t1)
		m = a*m1 + b*m2
		m_err = np.sqrt(a**2*m1_err**2 + b**2*m2_err**2)
		if t2-t1> 0.0135+(1e9)*t/3e10:
			print('warning: dt>R/C')
		#if t2 - t1 > 5:
		#    print('warning: dt > {0} days'.format(t_max))
		if min((t2  - t),(t-t1)) > t_max:
			print('maximum time diff > {0} days. Returning nan'.format(t_max))
			return np.nan,np.nan ,np.nan  
		return m[0],m_err[0],(t2-t1)[0]

def color_curve(dat, filt, filt2,tt = '', tmax = 10, tmax_inter = 3):
	
	dat.sort('t_rest')
	time = np.unique(np.round(2*dat['t_rest'])/2)
	#tt = 2
	if tt=='':
		tt = dat['t_rest'][dat['filter']==filt]+0.01
		tt = tt[tt<tmax]

	if len(tt)==0:
		pass
	f1     =  np.array(list(map(lambda t: get_inter_mag(t, dat, filt,t_max = tmax_inter)[0],tt)))
	f1err  =  np.array(list(map(lambda t: get_inter_mag(t, dat, filt,t_max = tmax_inter)[1],tt)))
	f2    =  np.array(list(map(lambda t: get_inter_mag(t, dat, filt2,t_max = tmax_inter)[0],tt)))
	f2err =  np.array(list(map(lambda t: get_inter_mag(t, dat, filt2,t_max = tmax_inter)[1],tt)))
	x = f1 - f2
	xerr = np.sqrt(f1err**2+ f2err**2)
	tt = np.array(tt)
	#plt.errorbar(x,y,xerr = xerr, yerr = yerr, **kwargs)
	return tt, x, xerr








#sp1.
#u.photon / u.cm**2 / u.s / u.micron
#
#    if (len(t)==1):
#        if err!='None':
#            return t,m, err
#        else:
#            return t,m
#    else:    
#        
#        if ctype == 'med':
#            func=np.nanmedian
#        elif ctype == 'mean':
#            func=np.nanmean
#        else: 
#            raise Exception('Error: Unknown combine method')
#        
#        t_range=np.max(t)-np.min(t)
#        N=int(math.ceil(t_range/delta))+1
#        t_range=np.min(t)+delta*np.arange(N)
#        f_bin=np.zeros_like(t_range)
#        t_bin=np.zeros_like(t_range)
#        for n in np.arange(1,N):
#            cond = (t>=t_range[n-1])&(t<t_range[n])
#            if np.sum(cond)>0:
#                f_bin[n-1]=func(f[cond])
#                t_bin[n-1]=func(t[cond])
#            else:
#                f_bin[n-1]=np.nan
#                t_bin[n-1]=np.nan
#        if err!='None':
#            f_err_bin=np.zeros_like(t_range)
#            for n in np.arange(1,N):
#                cond = (t>=t_range[n-1])&(t<t_range[np])
#                f_err_bin[n]=np.sqrt(np.sum(err[cond]**2))/len(err[cond])
#            cond2 =  ~np.isnan(f_bin)
#            t_bin=t_bin[cond2][0:-1]
#            f_bin=f_bin[cond2][0:-1]
#            f_err_bin=f_err_bin[cond2][0:-1]
#            m_bin  = -2.5*np.log10(f_bin)
#            m_err_bin  = fluxerr2magerr(f_err_bin,f_bin)
#            return t_bin,m_bin,m_err_bin   
#
#        else:
#            cond2 =  ~np.isnan(f_bin)
#            t_bin=t_bin[cond2][0:-1]
#            f_bin=f_bin[cond2][0:-1]
#            m_bin  = -2.5*np.log10(f_bin)
#            return t_bin,m_bin
#
#
#
#
#
#






#def correct_extinction(lam,F_Ha,F_Hb,R=3.1):
#    #lam in micron
#    k=lambda l: -2.156+1.509/l-0.198/l**2+0.011/l**3
#    l_Ha=0.6564
#    l_Hb=0.4861 
#    Av=R*(np.log(2.86) - np.log(F_Ha/F_Hb))/(0.4*(k(l_Ha)-k(l_Hb)))
#    A_lam=10**(-0.4*Av*k(lam)/R)
#    return A_lam
#