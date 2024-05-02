import numpy as np
from astropy import table
from astropy.io import ascii
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy import constants 
import os
import sys
from SNeSCOPE.PhotoUtils import *
from extinction import ccm89,calzetti00, apply
from scipy.optimize  import minimize,curve_fit
import tqdm
from numba import jit, njit 
import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
import pandas as pd
#from params import *
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
## priors 
rad_high=3e15
rad_low=4e12
temp_high=5e5
temp_low=1000
sep = os.sep
# constants 
c_cgs = constants.c.cgs.value
k_B_eVK = constants.k_B.to('eV/K').value
k_B = constants.k_B.cgs.value
sigma_sb=constants.sigma_sb.cgs.value
cAA  = c_cgs*1e8
h = constants.h.cgs.value
eVs = constants.h.to('eV s').value
eV2K =1/constants.k_B.to('eV / K ')
eV2K = eV2K.value


def interp_data(time,data,datcol='flux'):
		data_interp=np.interp(time,data['real_time'],data[datcol])
		return data_interp


def plot_error_photo(filter,color,marker):
		filter_photometry=data[data['filter']==filter]
		plt.errorbar(filter_photometry['real_time'],filter_photometry['flux'],filter_photometry['fluxerr'],marker=marker,color=color)


def bb_F(lam,T,r,EBV=0,EBV_mw=0, R_v=3.1,z=0,LAW='MW'):
	A_v=R_v*EBV
	A_v_mw=3.1*EBV_mw
	
	flux=1.191/(lam**(5))/(np.exp((1.9864/1.38064852)/(lam*T))-1)*(np.pi*(r)**2) 
	#flux=apply(ccm89(lam*1e4, A_v, R_v), flux)
	#flux=apply(ccm89(lam*1e4*(1+z), A_v_mw, 3.1), flux)
	flux=apply_extinction(1e4*lam,flux,EBV=EBV,EBV_mw=EBV_mw,R_v=R_v,z=z,LAW=LAW)
	return flux  

def apply_extinction(lam,flam,EBV=0,EBV_mw=0, R_v=3.1,z=0,LAW='MW'):
	if EBV>0:
		if LAW == 'Cal00':
			A_v=R_v*EBV
			flux=apply(calzetti00(lam, A_v, R_v,unit='aa'), flam)
		else:
			A_v=R_v*EBV
			flux=apply(ccm89(lam, A_v, R_v,unit='aa'), flam)

	else:
		flux = flam
	if EBV_mw>0:
		A_v_mw=3.1*EBV_mw
		flux=apply(ccm89(lam*(1+z), A_v_mw, 3.1,unit='aa'), flux)
	return flux  




def bb_fit(lam,f_lam,lims,f_lam_err=None,include_errors=True,EBV=0,Ebv_MW=0,R_v=3.1,z=0,ret_cov = False):
	func= lambda lam,T,r: bb_F(lam,T,r,EBV=EBV,EBV_mw=Ebv_MW, R_v=R_v, z=z)

	if include_errors:  
		popt, pcov = curve_fit(func, lam, f_lam, sigma=f_lam_err,bounds=lims,method='trf')

	else:
		popt, pcov = curve_fit(func, lam, f_lam,bounds=lims,method='trf')    

	perr = np.sqrt(np.diag(pcov))
	if ret_cov:
		return popt, perr,pcov
	else: 
		return popt, perr




def fit_bb_sequence(time,data_full,data_full_e,EBV=0,Ebv_MW=0,R_v=3.1,plot_sed=0,z=0):
	T_bb=np.zeros_like(time)        
	r_bb=np.zeros_like(time)
	T_bb_e=np.zeros_like(time)
	r_bb_e=np.zeros_like(time)
	chi2_vec=np.zeros_like(time)
	dof_vec=np.zeros_like(time)
	dof=0
	for i in range(len(time)):
		lam=piv_wl[data_full[:,i]!=99]*1e-4
		lam=lam/(1+z)
		dof+=len(lam)
		f_lam=(1+z)*data_full[data_full[:,i]!=99,i]*1e13
		f_err=(1+z)*data_full_e[data_full_e[:,i]!=99,i]*1e13
		
		try:
			popt, perr=bb_fit(lam,f_lam,lims,f_lam_err=f_err,include_errors=include_errors,EBV=EBV,Ebv_MW=Ebv_MW,R_v=R_v,z=z)
		except:
			import ipdb; ipdb.set_trace()
		T_bb[i]=popt[0]*1e4
		r_bb[i]=popt[1]*d*1e-10
		T_bb_e[i]=perr[0]*1e4
		r_bb_e[i]=perr[1]*d*1e-10

		c2=chi2(lam,f_lam,f_err,T_bb[i],r_bb[i],EBV=EBV,Ebv_MW=Ebv_MW,R_v=R_v,z=z)
		chi2_vec[i]=np.sum(c2)
		dof_vec[i] = len(c2) - 2
		if np.inf in (r_bb_e[0],T_bb_e[0]):
				print("ATTENTION: there are infinite values for the error bars. The covariance matrix could not be estimated, and the fit is unreliable for these points.")
		if plot_sed==1:
			plt.figure()
			plt.title("sed fit at t={0}".format(time[i]))
			plt.errorbar(lam*1e4,f_lam*1e-13,f_err*1e-13,marker='o',color='r',linestyle="None")
			l=np.linspace(0.75*min(lam),1.25*max(lam),1000)*1e4

			plt.plot(l,bb_F(l*1e-4,popt[0],popt[1],EBV=EBV,EBV_mw=Ebv_MW,R_v=R_v,z=z)*1e-13)
			plt.xlabel('Wavelength (angstrom)')
			plt.ylabel('flux(cgs/\AA)')
			plt.xscale('log')
			plt.yscale('log')
	Chi2=np.sum(chi2_vec)
	return T_bb,T_bb_e,r_bb, r_bb_e, Chi2, dof, chi2_vec,dof_vec



def generate_bb_mag(T_array,filt,dic_transmission,r = 1e14, d = 3.08e26, Rv=3.1, EBV = 0,EBV_MW = 0, LAW = 'MW'):
	v = 67.4*d/3.08e24
	z = v/300000
	Trans  = dic_transmission[filt]
	lam  = np.linspace(1000,20000,1901)
	
	m_array = []
	for i,T in enumerate(T_array):
		flux = bb_F(lam*1e-4,T*1e-4,(r/d)*1e10,EBV=EBV,EBV_mw = EBV_MW,z = z, R_v = Rv, LAW = LAW)*1e-13                                                                  
		m = SynPhot(lam,flux,Trans)
		m_array.append(m) 
	m_array = np.array(m_array) 
	return m_array 



@njit
def fast_trapz(y,x):
	trap = np.trapz(y,x)
	return trap

@njit
def fast_interp(x,xp,yp):
	interp = np.interp(x,xp,yp)
	return interp

def SynPhot_fast_AB(Lambda,Flux,Filter_lam,Filter_T, sys='AB',ret_flux=False):
	cAA = 2.99792e+18
	ZPflux=3631
	ZPflux=ZPflux*1e-23
	ZPflux_vec=ZPflux*cAA*(Lambda**(-2))
	T_filt_interp=np.interp(Lambda,Filter_lam,Filter_T,left=0,right=0)
	trans_flux=fast_trapz(Flux*Lambda*T_filt_interp,Lambda)
	norm_flux=fast_trapz(ZPflux_vec*T_filt_interp*Lambda,Lambda)
	mag=-2.5*np.log10(trans_flux/norm_flux)
	return mag  


def SynPhot_fast(Lambda,Flux,Filter, sys='AB',ret_flux=False):
	if sys.lower()=='vega':
		ZPflux_vec=fast_interp(Lambda,vega_spec[:,0] ,vega_spec[:,1] ,left=0,right=0)
	if sys.lower()=='ab':
		ZPflux=3631
		ZPflux=ZPflux*1e-23
		ZPflux_vec=ZPflux*cAA*(Lambda**(-2))

	Filter["col1"] = Filter["col1"].astype(float)
	T_filt_interp=np.interp(Lambda,Filter["col1"],Filter["col2"],left=0,right=0)

	trans_flux=fast_trapz(Flux*Lambda*T_filt_interp,Lambda)
	norm_flux=fast_trapz(ZPflux_vec*T_filt_interp*Lambda,Lambda)
	if norm_flux == 0:
		import ipdb; ipdb.set_trace()
	mag=-2.5*np.log10(trans_flux/norm_flux)
	if sys.lower()=='vega':
		mag=-2.5*np.log10(trans_flux/norm_flux)+0.03
	if ret_flux:
		return trans_flux/norm_flux 
	return mag  


@njit
def B_nu(nu_eV,T_eV):
	nu_hz = nu_eV/eVs
	B_nu = (2*h*(nu_hz)**3/c_cgs**2)*(np.exp(nu_eV/T_eV)-1)**(-1)
	return B_nu



@njit
def L_BB(tday, L_break,t_break_days,t_tr_day):
	t_tilda = tday/t_break_days #same units
	L = (t_tilda**(-4/3)+t_tilda**(-0.172)*f_corr_SW(tday,t_tr_day))
	L = L_break*L
	return L
@njit
def L_SW(tday, L_break,t_break_days,t_tr_day):
	t_tilda = tday/t_break_days #same units
	L = (t_tilda**(-0.172))*f_corr_SW(tday,t_tr_day)
	L = L_break*L
	return L

@njit
def L_SW2(tday, L_break,t_break_days,t_tr_day):
	t_tilda = tday/t_break_days #same units
	L = (t_tilda**(-0.172))*f_corr_SW(tday,t_tr_day)
	L = L_break*L
	return L
@njit
def L_SW(tday,v85,fM,k34,R13 ,t_tr_day):
	t_tilda = tday #same units
	A = 0.94
	alpha = 0.8
	a = 1.67
	f_corr = A*np.exp(-(a*tday/t_tr_day)**alpha) 
	L = (t_tilda**(-0.172))*f_corr
	L_break = 2e42*((v85/fM/k34)**(-0.086))*(v85**2*R13/k34)
	L = L_break*L
	return L
@njit
def T_color_bb(tday, T_break,t_break_days):
	t_tilda = tday/t_break_days #same units
	T = T_break*np.minimum(0.97*t_tilda**(-1/3), t_tilda**(-0.45))
	return  T
@njit
def T_color_SW(tday, T_break,t_break_days):
	t_tilda = tday/t_break_days #same units
	T = T_break*t_tilda**(-0.45)
	return  T

@njit
def f_corr_SW(tday,t_tr_day):
	A = 0.9
	alpha = 0.5
	a = 2
	f_corr = A*np.exp(-(a*tday/t_tr_day)**alpha) 
	return f_corr

@njit
def break_params_bo(R13, beta_bo,rho_bo9,k34,Menv):
	v_bo_9 = beta_bo*29979245800.0/1e9
	t_trans = 0.85*R13/v_bo_9 #hrs
	L_trans = 3.98e42*R13**(2/3)*v_bo_9**(5/3)*rho_bo9**(-1/3)*k34**(-4/3) #erg/s
	T_trans = 8.35*R13**(-1/3)*v_bo_9**(0.463)*rho_bo9**(-0.083)*k34**(-1/3) #eV
	t_trasnp = 18.9*np.sqrt(Menv)*(R13*rho_bo9)**(0.14)*k34**(0.64)*(v_bo_9)**(-0.36) #days
	return t_trans , T_trans, L_trans,t_trasnp

@njit
def break_params_new(R13, v85,fM,Menv,k34):
	t_trans = 0.86*R13**1.26*v85**(-1.13)*(fM*k34)**(-0.13)        #hrs
	L_trans = 3.69e42*R13**(0.78)*v85**(2.11)*fM**0.11*k34**(-0.89) #erg/s
	T_trans = 8.19*R13**(-0.32)*v85**(0.58)*fM**(0.03)*k34**(-0.22) #eV
	t_trasnp =19.5*np.sqrt(Menv*k34/v85) #days
	return t_trans , T_trans, L_trans,t_trasnp

def bo_params_from_phys(fM,vstar85,R13,k034=1):
	beta_bo = 0.033039*fM**0.129356*vstar85**1.12936*k034**0.129356/R13**0.258713
	rho_bo9 = 1.1997*fM**0.322386/R13**1.64477/vstar85**0.677614/k034**0.677614
	return beta_bo,rho_bo9
def Lbo_energy_from_phys(fM,vstar85,R13,k034=1):
	beta_bo,rho_bo9 = bo_params_from_phys(fM,vstar85,R13,k034=k034)
	vbo9 = beta_bo*29979245800/1e9

	Ebo = 2.2e47*R13**2*vbo9*k034**(-1) #erg
	tbo = 90*k034**(-1)*rho_bo9**(-1)*R13**(-1)*vbo9**(-2) #s
	t_light = R13*1e13/29979245800
	tbo = max(tbo,t_light)
	L_bo = Ebo/tbo #erg/s
	L_inf = 0.33*4*np.pi*(R13*1e13)**2*(rho_bo9*1e-9)*( beta_bo*29979245800)**3 #erg/s
	log_hnu_peak = 1.4+vbo9**0.5+(0.25-0.05*vbo9**0.5)*np.log10(rho_bo9)
	Tbo = (10**log_hnu_peak)/3
	return Ebo,tbo,L_bo,L_inf,Tbo


def phys_params_from_bo(rho_bo9,beta_bo,R13,k034=1,n=3/2,beta1 = 0.1909):
	rho_bo = rho_bo9*1e-9
	kappa = 0.34*k034
	Rstar = R13*1e13
	vbo = beta_bo*29979245800
	vsstar = vbo*(rho_bo*kappa*beta_bo*Rstar/(n+1))**(-beta1*n)
	fM = (4*np.pi/3)*rho_bo*(rho_bo*kappa*beta_bo/(n+1))**n*(Rstar)**(n+3)
	v85 = vsstar/10**8.5
	fM = fM/constants.M_sun.cgs.value
	return v85,fM


@njit
def validity(R13, v85,rho_bo9,k34,Menv):
	t_trans_hrs , T_trans_eV, L_trans_ergs,t_trasnp = break_params_new(R13, v85,rho_bo9,k34,Menv)
	t07eV = 2.67*t_trans_hrs*(T_trans_eV/5)**2.2 #days
	t_up = min(t07eV,t_trasnp) #days
	t_down = 0.2*t_trans_hrs**(-0.1)*(L_trans_ergs/10**41.5)**0.55*(T_trans_eV/5)**(-2.21) #hrs
	t_down = t_down/24 #days
	return t_down,t_up
	
@njit
def validity2(R13, v85,fM,k34,Menv):
	t07eV = 6.86*R13**(0.56)*v85**(0.16)*k34**(-0.61)*fM**(-0.06)  #days
	t_trasnp  = 19.5*np.sqrt(Menv*k34/v85)
	t_up = min(t07eV,t_trasnp/2) #days
	t_down = 17*R13/60 #hrs
	t_down = t_down/24 #days
	return t_down,t_up


@njit
def L_nu_MSW(tday, nu_eV,R13,v85,fM,Menv,k34):

	t_trans , T_trans, L_trans,t_tr_day =break_params_new(R13, v85,fM,Menv,k34)
	L_break_41_5erg = L_trans/10**41.5
	T_break_5eV = T_trans/5
	t_break_days = t_trans/24
	L_break_42_5erg = L_trans/10**42.5
	L_bb = L_BB(tday, L_break_41_5erg,t_break_days,t_tr_day)*10**(41.5)
	T_col =  T_color_bb(tday, T_break_5eV,t_break_days)*5
	t_tilda = tday/t_break_days

	R = R13*1e13
	

	####BO parameter formula####
	r_col_nu = R + 2.179e13*(L_break_42_5erg**0.483)*T_break_5eV**(-1.9737)*t_tilda**(0.805)*nu_eV**(-0.0775)*k34**(-0.07) #cm
	T_col_nu = 5.468*(L_break_42_5erg**0.0544)*(T_break_5eV**0.916)*(t_tilda**-0.421)*nu_eV**0.2468*k34**(0.22) # in eV
	k_col_nu = 0.028*L_break_42_5erg**(-0.3665)*(T_break_5eV**0.5644)*k34**(-0.47)*(t_tilda**-0.1909)*(nu_eV**-1.664 ) #cm2g-1
	k_es = k34*0.345
	epc_col_nu = k_col_nu/(k_col_nu+k_es)

	L_nu1 = L_bb*(np.pi)*B_nu(nu_eV,0.85*T_col)/(sigma_sb*(0.85*T_col*eV2K)**4)/1e25
	L_nu2 = L_bb*(np.pi)*B_nu(nu_eV,0.74*T_col)/(sigma_sb*(0.74*T_col*eV2K)**4)/1e25
	L_nu3 = B_nu(nu_eV,T_col_nu)*((4*np.pi)**2/np.sqrt(3))*(r_col_nu**2)*(np.sqrt(epc_col_nu)/(1+np.sqrt(epc_col_nu)))/1e25

	L_nu  = 1e25*(L_nu1**(-5)+L_nu2**(-5)+L_nu3**(-5))**(-0.2)  #in formula min but using smoothed version from yoni 
	return L_nu




@njit
def L_nu_eps(tday, nu_eV,R13,k34,t_break_days , T_break_5eV, L_break_42_5erg,t_tr_day):
	L_bb = L_BB(tday, L_break_42_5erg,t_break_days,t_tr_day)*10**(42.5)
	T_col =  T_color_bb(tday, T_break_5eV,t_break_days)*5
	t_tilda = tday/t_break_days
	R = R13*1e13
	r_col_nu = R + 2.179e13*(L_break_42_5erg**0.483)*T_break_5eV**(-1.9737)*t_tilda**(0.805)*nu_eV**(-0.0775)*k34**(-0.07) #cm
	T_col_nu = 5.468*(L_break_42_5erg**0.0544)*(T_break_5eV**0.916)*(t_tilda**-0.421)*nu_eV**0.2468*k34**(0.22) # in eV
	k_col_nu = 0.028*L_break_42_5erg**(-0.3665)*(T_break_5eV**0.5644)*k34**(-0.47)*(t_tilda**-0.1909)*(nu_eV**-1.664 ) #cm2g-1
	k_es = k34*0.345
	epc_col_nu = k_col_nu/(k_col_nu+k_es)
	L_nu1 = L_bb*(np.pi)*B_nu(nu_eV,0.85*T_col)/(sigma_sb*(0.85*T_col*eV2K)**4)/1e25
	L_nu3 = B_nu(nu_eV,T_col_nu)*((4*np.pi)**2/np.sqrt(3))*(r_col_nu**2)*(np.sqrt(epc_col_nu)/(1+np.sqrt(epc_col_nu)))/1e25
	L_nu_eps  = 1e25*(L_nu1**(-5)+L_nu3**(-5))**(-0.2)  
	return L_nu_eps


@njit
def L_nu_new_formula(tday, nu_eV,R13,v85,fM,Menv,k34):
	t_trans , T_trans, L_trans,t_tr_day =break_params_new(R13, v85,fM,Menv,k34)
	T_break_5eV = T_trans/5
	t_break_days = t_trans/24
	L_break_42_5erg = L_trans/10**42.5
	T_col =  T_color_bb(tday, T_break_5eV,t_break_days)*5
	L_bb = L_BB(tday, L_break_42_5erg,t_break_days,t_tr_day)*10**(42.5)
	if (nu_eV < 3.5*T_col): #&(nu_eV < 6)
		L_eps = L_nu_eps(tday, nu_eV,R13,k34,t_break_days , T_break_5eV, L_break_42_5erg,t_tr_day)
		T_col_mod = 0.85*T_col
		L_bb_nu = L_bb*(np.pi)*B_nu(nu_eV,T_col_mod)/(sigma_sb*(T_col_mod*eV2K)**4)
		L_nu = (L_eps**(-5)+L_bb_nu**(-5))**(-0.2)
	elif (nu_eV >= 3.5*T_col):
		T_col_mod = 0.85*(R13/tday)**(0.13)*T_col
		L_nu = 1.2*L_bb*(np.pi)*B_nu(nu_eV,T_col_mod)/(sigma_sb*(T_col_mod*eV2K)**4)
	return L_nu
#@njit
#def L_nu_new_formula(tday, nu_eV,R13,v85,fM,Menv,k34):
#	t_trans , T_trans, L_trans,t_tr_day =break_params_new(R13, v85,fM,Menv,k34)
#	T_break_5eV = T_trans/5
#	t_break_days = t_trans/24
#	L_break_42_5erg = L_trans/10**42.5
#	T_col =  T_color_bb(tday, T_break_5eV,t_break_days)*5
#	L_bb = L_BB(tday, L_break_42_5erg,t_break_days,t_tr_day)*10**(42.5)
#	if (nu_eV < 3.5*T_col): #&(nu_eV < 6)
#		L_eps = L_nu_eps(tday, nu_eV,R13,k34,t_break_days , T_break_5eV, L_break_42_5erg,t_tr_day)
#		L_nu = L_eps
#	elif (nu_eV >= 3.5*T_col):
#		T_col_mod = 0.85*(R13/tday)**(0.13)*T_col
#		L_nu = 1.2*L_bb*(np.pi)*B_nu(nu_eV,T_col_mod)/(sigma_sb*(T_col_mod*eV2K)**4)
#	return L_nu



@njit                           
def L_nu_MSW_reduced(tday, nu_eV,R13,v85,fM,Menv,k34):
	t_trans , T_trans, L_trans,t_tr_day =break_params_new(R13, v85,fM,Menv,k34)
	L_break_41_5erg = L_trans/10**41.5
	T_break_5eV = T_trans/5
	t_break_days = t_trans/24
	L_break_42_5erg = L_trans/10**42.5
	L_bb = L_BB(tday, L_break_41_5erg,t_break_days,t_tr_day)*10**(41.5)
	T_col =  T_color_bb(tday, T_break_5eV,t_break_days)*5
	L_nu = Lnu_reduced(nu_eV,T_col,L_bb)
	return L_nu

@njit
def Lnu_reduced(nu_eV,T_col,L_bb):
	x = nu_eV/T_col
	T_col5 = T_col/5
	eps_a = 0.0055*x**(-1.664)*(T_col/5)**(-1.0996)
	T_col_nu = 1.63*(x**0.247)*(T_col)
	eps_term = (np.sqrt(eps_a)/(1+np.sqrt(eps_a)))
	rr_nu = ((x**(-0.155))*(T_col5)**(-0.1))
	L_nu1 = B_nu(nu_eV,0.85*T_col)/((0.85)**4)
	L_nu2 = B_nu(nu_eV,0.74*T_col)/((0.74)**4)
	L_nu3 = (8/np.sqrt(3)) * rr_nu * eps_term * B_nu(nu_eV,T_col_nu) 
	L_nu  = ((np.pi/sigma_sb)*L_bb/((T_col*eV2K)**4))*(L_nu1**(-5)+L_nu2**(-5)+L_nu3**(-5))**(-0.2)  #in formula min but using smoothed version from yoni 
	return L_nu

@njit
def Lnu_eps_reduced(nu_eV,T_col,L_bb):
	x = nu_eV/T_col
	T_col5 = T_col/5
	eps_a = 0.0055*x**(-1.664)*(T_col/5)**(-1.0996)
	T_col_nu = 1.63*(x**0.247)*(T_col)
	eps_term = (np.sqrt(eps_a)/(1+np.sqrt(eps_a)))
	rr_nu = ((x**(-0.155))*(T_col5)**(-0.1))
	L_nu1 = B_nu(nu_eV,0.85*T_col)/((0.85)**4)
	#L_nu2 = B_nu(nu_eV,0.74*T_col)/((0.74)**4)
	L_nu3 = (8/np.sqrt(3)) * rr_nu * eps_term * B_nu(nu_eV,T_col_nu) 
	L_nu  = ((np.pi/sigma_sb)*L_bb/((T_col*eV2K)**4))*(L_nu1**(-5)+L_nu3**(-5))**(-0.2)  #in formula min but using smoothed version from yoni 
	return L_nu


@njit
def Lnu_LT(nu_eV,T_col,L_bb):
	if (nu_eV < 3.5*T_col):#&(nu_eV < 6):
		L_eps = Lnu_eps_reduced(nu_eV,T_col,L_bb)
		L_nu = L_eps	
	elif (nu_eV > 3.5*T_col):
		#T_col_mod = 0.69*R13**(0.06)*T_col
		L_bb425 = L_bb/10**42.5
		T_col5 = T_col/5
		T_col_mod = 1.11*L_bb425**0.03*(T_col5)**0.18*T_col
		L_nu = 1.2*L_bb*(np.pi)*B_nu(nu_eV,T_col_mod)/(sigma_sb*(T_col_mod*eV2K)**4)
	return L_nu

#@njit
#def Lnu_LT(nu_eV,T_col,L_bb):
#	if (nu_eV < 3*T_col):#&(nu_eV < 6):
#		L_eps = Lnu_eps_reduced(nu_eV,T_col,L_bb)
#		L_nu = L_eps	
#	elif (nu_eV > 3*T_col):
#		#T_col_mod = 0.69*R13**(0.06)*T_col
#		L_bb425 = L_bb/10**42.5
#		T_col5 = T_col/5
#		T_col_mod = 1.11*L_bb425**0.03*(T_col5)**0.18
#		L_nu = L_bb*(np.pi)*B_nu(nu_eV,T_col_mod)/(sigma_sb*(T_col_mod*eV2K)**4)
#	return L_nu
@njit                           
def L_nu_LT(tday, nu_eV,R13,v85,fM,Menv,k34):
	t_trans , T_trans, L_trans,t_tr_day =break_params_new(R13, v85,fM,Menv,k34)
	T_break_5eV = T_trans/5
	t_break_days = t_trans/24
	L_break_42_5erg = L_trans/10**42.5
	L_bb = L_BB(tday, L_break_42_5erg,t_break_days,t_tr_day)*10**(42.5)
	T_col =  T_color_bb(tday, T_break_5eV,t_break_days)*5
	L_nu = Lnu_LT(nu_eV,T_col,L_bb)
	return L_nu

@njit
def L_nu_g_MSW(tday, nu_eV,R13,v85,fM,Menv,k34):
	t_trans , T_trans, L_trans,t_tr_day =break_params_new(R13, v85,fM,Menv,k34)
	L_break_41_5erg = L_trans/10**41.5
	T_break_5eV = T_trans/5
	t_break_days = t_trans/24
	
	L_bb = L_BB(tday, L_break_41_5erg,t_break_days,t_tr_day)*10**(41.5)
	T_col =  T_color_bb(tday, T_break_5eV,t_break_days)*5
	L_nubb = L_bb*(np.pi)*B_nu(nu_eV,T_col)/(sigma_sb*(T_col*eV2K)**4)
	L_nu2 = L_bb*(np.pi)*B_nu(nu_eV,0.74*T_col)/(sigma_sb*(0.74*T_col*eV2K)**4)
	L_nu  = (L_nubb**(-5)+L_nu2**(-5))**(-0.2)  #in formula min but using smoothed version from yoni 
	#L_nu  = np.minimum(L_nubb,L_nu2)
	return L_nu


@njit
def L_nu_gray(tday, nu_eV,R13,v85,fM,k34,Menv):
	t_trans , T_trans, L_trans,t_tr_day =break_params_new(R13, v85,fM,Menv,k34)
	L_break_41_5erg = L_trans/10**41.5
	T_break_5eV = T_trans/5
	t_break_days = t_trans/24
	L_bb = L_BB(tday, L_break_41_5erg,t_break_days,t_tr_day)*10**(41.5)
	T_col =  T_color_bb(tday, T_break_5eV,t_break_days)*5
	L_nubb = L_bb*(np.pi)*B_nu(nu_eV,T_col)/(sigma_sb*(T_col*eV2K)**4)
	return  L_nubb


@njit
def f_nu_freq_dep(tday,lam_AA,R13,v85,fM,k34,Menv,d = 3.08e26):
	lam_cm = (lam_AA/1e8)
	nu_hz = c_cgs/lam_cm
	nu_eV = nu_hz*eVs
	L = L_nu_MSW(tday, nu_eV,R13,v85,fM,Menv,k34)
	f_nu = L/(4*np.pi*d**2)
	f_lam = f_nu2f_lam(f_nu,lam_AA)
	return f_lam,f_nu

@njit
def f_nu_reduced(tday,lam_AA,R13,v85,fM,k34,Menv,d = 3.08e26):
	lam_cm = (lam_AA/1e8)
	nu_hz = c_cgs/lam_cm
	nu_eV = nu_hz*eVs
	L = L_nu_MSW_reduced(tday, nu_eV,R13,v85,fM,Menv,k34)
	f_nu = L/(4*np.pi*d**2)
	f_lam = f_nu2f_lam(f_nu,lam_AA)
	return f_lam,f_nu

@njit
def f_nu_LT(tday,lam_AA,R13,v85,fM,k34,Menv,d = 3.08e26):
	lam_cm = (lam_AA/1e8)
	nu_hz = c_cgs/lam_cm
	nu_eV = nu_hz*eVs
	L = L_nu_LT(tday, nu_eV,R13,v85,fM,Menv,k34)
	f_nu = L/(4*np.pi*d**2)
	f_lam = f_nu2f_lam(f_nu,lam_AA)
	return f_lam,f_nu


@njit
def f_nu_general(*args,lam_AA,func,d = 3.08e26):
	lam_cm = (lam_AA/1e8)
	nu_hz = c_cgs/lam_cm
	nu_eV = nu_hz*eVs
	L = func(nu_eV,*args)
	f_nu = L/(4*np.pi*d**2)
	f_lam = f_nu2f_lam(f_nu,lam_AA)
	return f_lam,f_nu


def f_nu_new_formula(tday,lam_AA,R13,v85,fM,k34,Menv,d = 3.08e26):
	lam_cm = (lam_AA/1e8)
	nu_hz = c_cgs/lam_cm
	nu_eV = nu_hz*eVs
	L = L_nu_new_formula(tday, nu_eV,R13,v85,fM,Menv,k34)
	f_nu = L/(4*np.pi*d**2)
	f_lam = f_nu2f_lam(f_nu,lam_AA)
	return f_lam,f_nu

@njit
def f_nu_UV_sup(tday,lam_AA,R13,v85,fM,k34,Menv,d = 3.08e26):
	nu_hz = c_cgs/(lam_AA/1e8)
	nu_eV = nu_hz*eVs
	L = L_nu_g_MSW(tday, nu_eV,R13,v85,fM,Menv,k34)
	f_nu = L/(4*np.pi*d**2)
	f_lam = f_nu2f_lam(f_nu,lam_AA)
	return f_lam,f_nu

@njit
def f_nu_gray(tday, lam_AA,R13,v85,fM,k34,Menv,d = 3.08e26):
	nu_hz = c_cgs/(lam_AA/1e8)
	nu_eV = nu_hz*eVs
	L = L_nu_gray(tday, nu_eV,R13,v85,fM,k34,Menv)
	f_nu = L/(4*np.pi*d**2)
	f_lam = f_nu2f_lam(f_nu,lam_AA)
	return f_lam



def MW_F(tday,lam,R13,v85,fM,k34,Menv,d = 3.08e26, EBV=0,EBV_mw=0, R_v=3.1,z=0,LAW='MW'):

	#f_lam = np.array(list(map(lambda l: f_nu_freq_dep(tday,l,R13,v85,fM,k34,Menv,d = d)[0], lam )))
	f_lam = np.array(list(map(lambda l: f_nu_new_formula(tday,l,R13,v85,fM,k34,Menv,d = d)[0], lam )))

	flux_corr = apply_extinction(lam,f_lam,EBV=EBV,EBV_mw=EBV_mw,R_v=R_v,z=z,LAW=LAW)
	return flux_corr


def MW_F_red(tday,lam,R13,v85,fM,k34,Menv,d = 3.08e26, EBV=0,EBV_mw=0, R_v=3.1,z=0,LAW='MW'):
	#f_lam = np.array(list(map(lambda l: f_nu_reduced(tday,l,R13,v85,fM,k34,Menv,d = d)[0], lam )))
	f_lam = f_nu_reduced(tday,lam,R13,v85,fM,k34,Menv,d = d)[0]
	flux_corr = apply_extinction(lam,f_lam,EBV=EBV,EBV_mw=EBV_mw,R_v=R_v,z=z,LAW=LAW)
	return flux_corr


def MW_F_LT(tday,lam,R13,v85,fM,k34,Menv,d = 3.08e26, EBV=0,EBV_mw=0, R_v=3.1,z=0,LAW='MW'):
	#f_lam = np.array(list(map(lambda l: f_nu_freq_dep(tday,l,R13,v85,fM,k34,Menv,d = d)[0], lam )))
	f_lam = np.array(list(map(lambda l: f_nu_LT(tday,l,R13,v85,fM,k34,Menv,d = d)[0], lam )))

	flux_corr = apply_extinction(lam,f_lam,EBV=EBV,EBV_mw=EBV_mw,R_v=R_v,z=z,LAW=LAW)
	return flux_corr
def MW_F_UV_sup(tday,lam,R13,v85,fM,k34,Menv,d = 3.08e26, EBV=0,EBV_mw=0, R_v=3.1,z=0,LAW='MW'):
	f_lam = np.array(list(map(lambda l: f_nu_UV_sup(tday,l,R13,v85,fM,k34,Menv,d = d)[0], lam )))
	flux_corr = apply_extinction(lam,f_lam,EBV=EBV,EBV_mw=EBV_mw,R_v=R_v,z=z,LAW=LAW)
	return flux_corr

def MW_F_gray(tday,lam,R13,v85,fM,k34,Menv,d = 3.08e26, EBV=0,EBV_mw=0, R_v=3.1,z=0,LAW='MW'):
	f_lam = np.array(list(map(lambda l: f_nu_gray(tday,l,R13,v85,fM,k34,Menv,d = d), lam )))
	flux_corr = apply_extinction(lam,f_lam,EBV=EBV,EBV_mw=EBV_mw,R_v=R_v,z=z,LAW=LAW)
	return flux_corr
	



def generate_cooling_mag_single(T_array, R_array ,filter_transmission,z=0, d = 3.08e26, Rv=3.1, EBV = 0,EBV_MW = 0, LAW = 'MW',H0 = 67.4):
	if len(T_array)!=len(R_array):
		raise Exception('R and T arrays have to be the same length') 
	v = H0*d/3.08e24
	z = v/300000
	Trans  = filter_transmission

	lam  = np.linspace(np.min(Trans[:,0]),np.max(Trans[:,0]),90)
	mm = []
	for i in range(len(T_array)):
		flux = f_nu_bb(lam,T_array[i],R_array[i],d = d)[0]
		if EBV>0:
			flux = apply_extinction(lam,flux,EBV=EBV,EBV_mw=EBV_MW, R_v=Rv,z=z,LAW=LAW)
		#m = SynPhot_fast(lam,flux,Trans)
		m = SynPhot_fast_AB(lam,flux,Trans[:,0],Trans[:,1])
		mm.append(m)	
	m_array = np.array(mm)
	return m_array 
			

					
 
def generate_MW_mag(time,R13,v85,fM,k34,Menv,filt_list,filter_transmission_dic,z=0, d = 3.08e26, Rv=3.1, EBV = 0,EBV_MW = 0, LAW = 'MW',H0 = 67.4,UV_sup = False,reduced = False):
	if UV_sup:
		func =  MW_F_UV_sup
	else: 
		if reduced: 
			func = MW_F_LT
		else: 
			func = MW_F
	v = H0*d/3.08e24
	z = v/300000
	#lam  = np.linspace(1000,10000,90)

	m_array = {}
	for filt in filt_list:
		mm = []
		Trans  = filter_transmission_dic[filt]
		lam  = np.linspace(np.min(Trans[:,0]),np.max(Trans[:,0]),90)

		for t in time:			
			flux = func(t,lam,R13,v85,fM,k34,Menv,d = d, EBV=EBV,EBV_mw=EBV_MW, R_v=Rv,z=z,LAW=LAW)
			m = SynPhot_fast_AB(lam,flux,Trans[:,0],Trans[:,1])
			mm.append(m)	
		m_array[filt] = np.array(mm)
	return m_array 



def generate_MW_mag_single(time,R13,v85,fM,k34,Menv,filt,filter_trans,z=0, d = 3.08e26, Rv=3.1, EBV = 0,EBV_MW = 0, LAW = 'MW',H0 = 67.4,UV_sup = False,reduced=True):
	if UV_sup:
		func =  MW_F_UV_sup
	else: 
		if reduced: 
			func = MW_F_LT
		else:
			func = MW_F
	
	v = H0*d/3.08e24
	z = v/300000
	#Trans  = filter_transmission_fast[filt]
	Trans  = filter_trans

	Trans_l = Trans[:,0]
	Trans_T = Trans[:,1]
	lam  = np.linspace(np.min(Trans_l),np.max(Trans_l),20)
	m_array = {}
	mm = []
	for t in time:
		flux = func(t,lam,R13,v85,fM,k34,Menv,d = d, EBV=EBV,EBV_mw=EBV_MW, R_v=Rv,z=z,LAW=LAW)
		m = SynPhot_fast_AB(lam,flux,Trans_l,Trans_T)
		mm.append(m)
	m_array[filt] = np.array(mm)    	
	return m_array
	
def generate_MW_flux(time,lam,R13,v85,fM,k34,Menv, d = 3.08e26, Rv=3.1, EBV = 0,EBV_MW = 0, LAW = 'MW',H0 = 67.4,UV_sup = False,old_form = False,reduced = False,gray = False):
	if UV_sup:
		func =  MW_F_UV_sup
	elif old_form:
		func = MW_F_red
	elif gray:
		func = MW_F_gray
	else: 
		if reduced: 
			func = MW_F_LT
		else:
			func = MW_F
	
	
	v = H0*d/3.08e24
	z = v/300000
	f_array = {}
	mm = []
	if isinstance(time,(float, int)):
		time = np.array([time])
	
	f_array = np.zeros((len(time),len(lam)))
	for i in range(len(time)):
		flux = func(time[i],lam,R13,v85,fM,k34,Menv,d = d, EBV=EBV,EBV_mw=EBV_MW, R_v=Rv,z=z,LAW=LAW)
		f_array[i,:] = flux
	return f_array 



@njit
def f_nu2f_lam(f_nu,lam_AA):
	c_AA = 2.99792458e+18
	f_lam_AA=(c_AA/lam_AA**2)*f_nu
	return f_lam_AA

@njit
def f_nu_bb(lam_AA,T,R,d = 3.08e26):
	lam_cm = (lam_AA/1e8)
	nu_hz = c_cgs/lam_cm
	nu_eV = nu_hz*eVs
	T_eV = T/eV2K
	L_bb = 4*np.pi*R**2
	L_nubb = L_bb*(np.pi)*B_nu(nu_eV,T_eV)
	f_nu = L_nubb/(4*np.pi*d**2)
	f_lam = f_nu2f_lam(f_nu,lam_AA)
	return f_lam,f_nu

@njit
def f_nu_red(lam_AA,T,L_bb,d = 3.08e26):
	lam_cm = (lam_AA/1e8)
	nu_hz = c_cgs/lam_cm
	nu_eV = nu_hz*eVs
	T_eV = T/eV2K
	L_nu_red = Lnu_reduced(nu_eV,T_eV,L_bb)
	f_nu = L_nu_red/(4*np.pi*d**2)
	f_lam = f_nu2f_lam(f_nu,lam_AA)
	return f_lam,f_nu


@njit
def Lnu_reduced(nu_eV,T_col,L_bb):
	x = nu_eV/T_col
	T_col5 = T_col/5
	eps_a = 0.0055*x**(-1.664)*(T_col/5)**(-1.0996)
	T_col_nu = 1.63*(x**0.247)*(T_col)
	eps_term = (np.sqrt(eps_a)/(1+np.sqrt(eps_a)))
	rr_nu = ((x**(-0.155))*(T_col5)**(-0.1))
	L_nu1 = B_nu(nu_eV,0.85*T_col)/((0.85)**4)
	L_nu2 = B_nu(nu_eV,0.74*T_col)/((0.74)**4)
	L_nu3 = (8/np.sqrt(3)) * rr_nu * eps_term * B_nu(nu_eV,T_col_nu) 
	L_nu  = ((np.pi/sigma_sb)*L_bb/((T_col*eV2K)**4))*(L_nu1**(-5)+L_nu2**(-5)+L_nu3**(-5))**(-0.2)  #in formula min but using smoothed version from yoni 
	return L_nu



class model_freq_dep_SC(object):
	def __init__(self,R13,v85,fM,Menv,t0,k34,filter_transmission = None,**model_kwargs):
		self.R13  = R13
		self.v85  = v85
		self.fM   = fM
		self.Menv = Menv
		self.k34  = k34
		self.t0 = t0
		t_break_hrs,T_break_eV,L_break,t_tr_day =  break_params_new(R13, v85,fM,Menv,k34)
		self.L_break       = L_break
		self.t_break_days  = t_break_hrs/24
		self.t_tr_day      = t_tr_day
		self.T_break       = T_break_eV/k_B_eVK
		t_down,t_up = validity2(R13, v85,fM,k34,Menv)
		self.t_down = t_down
		self.t_up   = t_up
		self.filter_transmission = filter_transmission
		inputs={'ebv':0
				,'Rv':3.1
				,'LAW':'MW'
				,'UV_sup':False
				,'reduced':False
				,'distance':3.08e19
				,'validity':'all'}                            
		inputs.update(model_kwargs)
		self.Rv = inputs.get('Rv')  
		self.LAW = inputs.get('LAW')  
		self.UV_sup = inputs.get('UV_sup')
		self.reduced = inputs.get('reduced')
		self.ebv = inputs.get('ebv')  
		self.Rv = inputs.get('Rv')  
		self.distance =inputs.get('distance')  
		self.validity = inputs.get('validity')
		if self.validity == 'force lower':
			self.t_down = 0

	def T_evolution(self,time): 
		time = time.flatten()
		T_break = self.T_break
		t_break_days = self.t_break_days
		T = T_color_bb(time, T_break,t_break_days)
		return T

	def L_evolution(self, time):
		time = time.flatten()
		t_break_days = self.t_break_days
		L_break = self.L_break
		t_tr_day = self.t_tr_day
		L_evo = L_BB(time, L_break,t_break_days,t_tr_day)
		return L_evo  

	def R_evolution(self,time): 
		time = time.flatten()
		Tevo = self.T_evolution(time)
		Levo = self.L_evolution(time)
		R = np.sqrt(Levo/(4*np.pi*sigma_sb*(Tevo)**4))
		return R
	def mags(self,time,filt_list): 
		m_array = generate_MW_mag(time,self.R13,self.v85,self.fM,self.k34,self.Menv,filt_list, self.filter_transmission
									, d =  self.distance, Rv=self.Rv, EBV = self.ebv
									,EBV_MW = 0, LAW = self.LAW,UV_sup = self.UV_sup,reduced = self.reduced)
		return m_array
	def mags_single(self,time,filt): 
		m_array = generate_MW_mag_single(time,self.R13,self.v85,self.fM,self.k34,self.Menv,filt, self.filter_transmission[filt]
									, d =  self.distance, Rv=self.Rv, EBV = self.ebv
									,EBV_MW = 0, LAW = self.LAW,UV_sup = self.UV_sup,reduced = self.reduced)
		return m_array   
	def flux(self,time,lam): 
		f_array = generate_MW_flux(time,lam,self.R13,self.v85,self.fM,self.k34,self.Menv
									, d =  self.distance, Rv=self.Rv, EBV = self.ebv
									,EBV_MW = 0, LAW = self.LAW,UV_sup = self.UV_sup,reduced = self.reduced, old_form = False, gray = False)

		return f_array  
	def likelihood(self,dat,sys_err = 0.05,nparams = 7):
		t0 = self.t0 
		ebv = self.ebv
		Rv = self.Rv
		LAW = self.LAW
		d = self.distance
		t_down  = self.t_down 
		t_up    = self.t_up   
		dat = dat[(dat['t_rest']>t_down)&(dat['t_rest']<t_up)]
		filt_list = np.unique(dat['filter'])
		chi2_dic = {}
		dof = 0
		chi_tot = 0
		N_band = []
		for filt in filt_list:
			dat_filt = dat[dat['filter']==filt]
			time = dat_filt['t_rest']-t0
			m_array = self.mags(time,filt_list = [filt])
			mag = m_array[filt]
			mag_obs = dat_filt['absmag']
			mag_err = dat_filt['AB_MAG_ERR'] + sys_err
			c2 = chi_sq(mag_obs,mag_err, mag)
			N = len(c2)
			chi2_dic[filt] = c2
			dof = dof+N
			chi_tot = chi_tot +np.sum(chi2_dic[filt])
		dof = dof - nparams
		
		rchi2 = chi_tot/dof
			
		if chi_tot == 0:
			logprob = -np.inf
		elif dof <= 0:
			logprob = -np.inf
		else:
			logprob = log_chi_sq_pdf(chi_tot, dof)
		if np.isnan(logprob):
			import ipdb; ipdb.set_trace()
		return logprob, chi_tot, dof
	def likelihood_cov(self,dat,inv_cov,sys_err = 0,nparams = 7):
		t0 = self.t0 
		ebv = self.ebv
		Rv = self.Rv
		LAW = self.LAW
		d = self.distance
		t_down  = self.t_down 
		t_up    = self.t_up   
		cond_valid = (dat['t_rest']-t0>t_down)&(dat['t_rest']-t0<t_up)
		args_valid = np.argwhere(cond_valid).flatten()
		if np.sum(cond_valid) == 0:
			chi_tot = 0
			logprob = -np.inf
			dof = -1
			return logprob, chi_tot, dof
		inds = [np.min(args_valid),np.max(args_valid)+1]

		dat = dat[args_valid]
		#_,dat_res = compute_resid(dat.copy(),self)
		dof = len(dat)
		filt_list = np.unique(dat['filter'])
		delta = np.zeros(dof,)
		for filt in filt_list:
			cond_filt = dat['filter']==filt
			args_filt = np.argwhere(cond_filt).flatten()
			dat_filt = dat[args_filt]
			time = dat_filt['t_rest']-t0
			mag = dat['absmag'][args_filt]
			mags =   self.mags_single(time,filt=filt)
			res = np.array(mag - mags[filt])
			delta[cond_filt] = res
		#cov = cov[inds[0]:inds[1],inds[0]:inds[1]]
		#inv_cov = np.linalg.inv(cov)
		inv_cov = inv_cov[inds[0]:inds[1],inds[0]:inds[1]]
		#prod = np.dot(delta,inv_cov)
		#chi_tot = np.dot(prod,delta)
		chi_tot = delta @ inv_cov @ delta
		dof = dof - nparams
		rchi2 = chi_tot/dof
		if chi_tot == 0:
			logprob = -np.inf
		elif dof <= 0:
			logprob = -np.inf
		else:
			#prob = scipy.stats.chi2.pdf(chi_tot, dof)
			#prob = float(prob)
			logprob = log_chi_sq_pdf(chi_tot, dof)
		if np.isnan(logprob):
			import ipdb; ipdb.set_trace()
		#logprob = np.log(prob)
		return logprob, chi_tot, dof
	def likelihood_bb(self,dat_bb,sys_err = 0.05,nparams = 10):
		t0 = self.t0 
		dof = 0
		chi_tot = 0
		
		time = dat_bb['t_rest']-t0
		L_evo = self.L_evolution(time)
		T_evo = self.T_evolution(time)
		err_L = np.sqrt((dat_bb['L_up']-dat_bb['L'])**2 + (sys_err*dat_bb['L'])**2)
		err_T = np.sqrt((dat_bb['T_up']-dat_bb['T'])**2 + (sys_err*dat_bb['T'])**2)

		c2_L = chi_sq(dat_bb['L'],err_L, L_evo)
		c2_T = chi_sq(dat_bb['T'],err_T, T_evo)

		N = len(c2_L) + len(c2_T)
		chi_tot = np.sum(c2_L) +np.sum(c2_T) 
		dof =N
		dof = dof - nparams
		
		rchi2 = chi_tot/dof

		logprob = -chi_tot
		return logprob, chi_tot, dof






class model_SW(object):
	def __init__(self,R13,v85,fM,Menv,t0,k34,filter_transmission = None,distance = 3.08e19,ebv = 0,Rv = 3.1, LAW = 'MW'):
		self.ebv = ebv
		self.Rv = Rv
		self.distance = distance
		self.LAW  = LAW
		self.R13  = R13
		self.v85  = v85
		self.fM   = fM
		self.Menv = Menv
		self.k34  = k34
		self.t0 = t0
		t_break_hrs,T_break_eV,L_break,t_tr_day =  break_params_new(R13, v85,fM,Menv,k34)
		self.L_break       = L_break
		self.t_break_days  = t_break_hrs/24
		self.t_tr_day      = t_tr_day
		self.T_break       = T_break_eV/k_B_eVK
		t_down,t_up = validity2(R13, v85,fM,k34,Menv)
		self.t_down = t_down
		self.t_up   = t_up
		self.filter_transmission = filter_transmission
	def T_evolution(self,time): 
		time = time.flatten()
		T_break = self.T_break
		t_break_days = self.t_break_days
		T = T_color_SW(time, T_break,t_break_days)
		return T
	def L_evolution(self, time):
		time = time.flatten()
		t_break_days = self.t_break_days
		L_break = self.L_break
		t_tr_day = self.t_tr_day
		L_evo = L_BB(time, L_break,t_break_days,t_tr_day)
		return L_evo  
	def R_evolution(self,time): 
		time = time.flatten()
		Tevo = self.T_evolution(time)
		Levo = self.L_evolution(time)
		R = np.sqrt(Levo/(4*np.pi*sigma_sb*(Tevo)**4))
		return R
	def mags(self,time,filt_list): 
		m_array={}
		for filt in filt_list:
			R_evo = self.R_evolution(time)
			T_evo = self.T_evolution(time)
			mag = generate_cooling_mag_single(T_evo, R_evo, self.filter_transmission[filt],d = self.distance, Rv=self.Rv, EBV = self.ebv,EBV_MW = 0, LAW = self.LAW)
			m_array[filt] = mag
		return m_array
	def mags_single(self,time,filt): 
		#m_array = generate_SC_mag_single(time,self.R13,self.v85,self.fM,self.k34,self.Menv,filt, self.filter_transmission[filt]
		#							, d =  self.distance, Rv=self.Rv, EBV = self.ebv
		#  							,EBV_MW = 0, LAW = self.LAW)
		R_evo = self.R_evolution(time)
		T_evo = self.T_evolution(time)
		m_array = generate_cooling_mag_single(T_evo, R_evo,self.filter_transmission[filt],d = self.distance, Rv=self.Rv, EBV = self.ebv,EBV_MW = 0, LAW = self.LAW)

		return m_array    	
	def flux(self,time,lam): 
		f_array = generate_MW_flux(time,lam,self.R13,self.v85,self.fM,self.k34,self.Menv
									, d =  self.distance, Rv=self.Rv, EBV = self.ebv
									,EBV_MW = 0, LAW = self.LAW,UV_sup = False,reduced = False, old_form = False, gray = True)

		return f_array 
	def likelihood(self,dat,sys_err = 0.05,nparams = 7):
		t0 = self.t0 
		ebv = self.ebv
		Rv = self.Rv
		LAW = self.LAW
		d = self.distance
		t_down  = self.t_down 
		t_up    = self.t_up   
		dat = dat[(dat['t_rest']>t_down)&(dat['t_rest']<t_up)]
		filt_list = np.unique(dat['filter'])
		chi2_dic = {}
		dof = 0
		chi_tot = 0
		N_band = []
		for filt in filt_list:
			dat_filt = dat[dat['filter']==filt]
			time = dat_filt['t_rest']-t0
			m_array = self.mags(time,filt_list = [filt])
			mag = m_array[filt]
			mag_obs = dat_filt['absmag']
			mag_err = dat_filt['AB_MAG_ERR'] + sys_err
			c2 = chi_sq(mag_obs,mag_err, mag)
			N = len(c2)
			chi2_dic[filt] = c2
			dof = dof+N
			chi_tot = chi_tot +np.sum(chi2_dic[filt])
		dof = dof - nparams
		
		rchi2 = chi_tot/dof
			
		if chi_tot == 0:
			logprob = -np.inf
		elif dof <= 0:
			logprob = -np.inf
		else:
			logprob = log_chi_sq_pdf(chi_tot, dof)
		if np.isnan(logprob):
			import ipdb; ipdb.set_trace()
		return logprob, chi_tot, dof
	def likelihood_cov(self,dat,inv_cov,sys_err = 0,nparams = 7):
		t0 = self.t0 
		ebv = self.ebv
		Rv = self.Rv
		LAW = self.LAW
		d = self.distance
		t_down  = self.t_down 
		t_up    = self.t_up   
		cond_valid = (dat['t_rest']-t0>t_down)&(dat['t_rest']-t0<t_up)
		args_valid = np.argwhere(cond_valid).flatten()
		if np.sum(cond_valid) == 0:
			chi_tot = 0
			logprob = -np.inf
			dof = -1
			return logprob, chi_tot, dof
		inds = [np.min(args_valid),np.max(args_valid)+1]

		dat = dat[args_valid]
		#_,dat_res = compute_resid(dat.copy(),self)
		dof = len(dat)
		filt_list = np.unique(dat['filter'])
		delta = np.zeros(dof,)
		for filt in filt_list:
			cond_filt = dat['filter']==filt
			args_filt = np.argwhere(cond_filt).flatten()
			dat_filt = dat[args_filt]
			time = dat_filt['t_rest']-t0
			mag = dat['absmag'][args_filt]
			mags =   self.mags_single(time,filt=filt)
			res = np.array(mag - mags)
			delta[cond_filt] = res
		#cov = cov[inds[0]:inds[1],inds[0]:inds[1]]
		#inv_cov = np.linalg.inv(cov)
		inv_cov = inv_cov[inds[0]:inds[1],inds[0]:inds[1]]
		#prod = np.dot(delta,inv_cov)
		#chi_tot = np.dot(prod,delta)
		chi_tot = delta @ inv_cov @ delta
		dof = dof - nparams
		rchi2 = chi_tot/dof
		if chi_tot == 0:
			logprob = -np.inf
		elif dof <= 0:
			logprob = -np.inf
		else:
			#prob = scipy.stats.chi2.pdf(chi_tot, dof)
			#prob = float(prob)
			logprob = log_chi_sq_pdf(chi_tot, dof)
		if np.isnan(logprob):
			import ipdb; ipdb.set_trace()
		#logprob = np.log(prob)
		return logprob, chi_tot, dof
	def likelihood_bb(self,dat_bb,sys_err = 0.05,nparams = 10):
		t0 = self.t0 
		dof = 0
		chi_tot = 0
		
		time = dat_bb['t_rest']-t0
		L_evo = self.L_evolution(time)
		T_evo = self.T_evolution(time)
		err_L = np.sqrt((dat_bb['L_up']-dat_bb['L'])**2 + (sys_err*dat_bb['L'])**2)
		err_T = np.sqrt((dat_bb['T_up']-dat_bb['T'])**2 + (sys_err*dat_bb['T'])**2)

		c2_L = chi_sq(dat_bb['L'],err_L, L_evo)
		c2_T = chi_sq(dat_bb['T'],err_T, T_evo)

		N = len(c2_L) + len(c2_T)
		chi_tot = np.sum(c2_L) +np.sum(c2_T) 
		dof =N
		dof = dof - nparams
		
		rchi2 = chi_tot/dof

		logprob = -chi_tot
		return logprob, chi_tot, dof





class model_SC(object):
	def __init__(self,R13,v85,fM,Menv,t0,k34,filter_transmission = None,distance = 3.08e19,ebv = 0,Rv = 3.1, LAW = 'MW'):
		self.ebv = ebv
		self.Rv = Rv
		self.distance = distance
		self.LAW  = LAW
		self.R13  = R13
		self.v85  = v85
		self.fM   = fM
		self.Menv = Menv
		self.k34  = k34
		self.t0 = t0
		t_break_hrs,T_break_eV,L_break,t_tr_day =  break_params_new(R13, v85,fM,Menv,k34)
		self.L_break       = L_break
		self.t_break_days  = t_break_hrs/24
		self.t_tr_day      = t_tr_day
		self.T_break       = T_break_eV/k_B_eVK
		t_down,t_up = validity2(R13, v85,fM,k34,Menv)
		self.t_down = t_down
		self.t_up   = t_up
		self.filter_transmission = filter_transmission
	def T_evolution(self,time): 
		time = time.flatten()
		T_break = self.T_break
		t_break_days = self.t_break_days
		T = T_color_bb(time, T_break,t_break_days)
		return T
	def L_evolution(self, time):
		time = time.flatten()
		t_break_days = self.t_break_days
		L_break = self.L_break
		t_tr_day = self.t_tr_day
		L_evo = L_BB(time, L_break,t_break_days,t_tr_day)
		return L_evo  
	def R_evolution(self,time): 
		time = time.flatten()
		Tevo = self.T_evolution(time)
		Levo = self.L_evolution(time)
		R = np.sqrt(Levo/(4*np.pi*sigma_sb*(Tevo)**4))
		return R
	def mags(self,time,filt_list): 
		m_array={}
		for filt in filt_list:
			R_evo = self.R_evolution(time)
			T_evo = self.T_evolution(time)
			mag = generate_cooling_mag_single(T_evo, R_evo, self.filter_transmission[filt],d = self.distance, Rv=self.Rv, EBV = self.ebv,EBV_MW = 0, LAW = self.LAW)
			m_array[filt] = mag
		return m_array
	def mags_single(self,time,filt): 
		#m_array = generate_SC_mag_single(time,self.R13,self.v85,self.fM,self.k34,self.Menv,filt, self.filter_transmission[filt]
		#							, d =  self.distance, Rv=self.Rv, EBV = self.ebv
		#  							,EBV_MW = 0, LAW = self.LAW)
		R_evo = self.R_evolution(time)
		T_evo = self.T_evolution(time)
		m_array = generate_cooling_mag_single(T_evo, R_evo,self.filter_transmission[filt],d = self.distance, Rv=self.Rv, EBV = self.ebv,EBV_MW = 0, LAW = self.LAW)

		return m_array    	
	def flux(self,time,lam): 
		f_array = generate_MW_flux(time,lam,self.R13,self.v85,self.fM,self.k34,self.Menv
									, d =  self.distance, Rv=self.Rv, EBV = self.ebv
									,EBV_MW = 0, LAW = self.LAW,UV_sup = False,reduced = False, old_form = False, gray = True)

		return f_array 
	def likelihood(self,dat,sys_err = 0.05,nparams = 7):
		t0 = self.t0 
		ebv = self.ebv
		Rv = self.Rv
		LAW = self.LAW
		d = self.distance
		t_down  = self.t_down 
		t_up    = self.t_up   
		dat = dat[(dat['t_rest']>t_down)&(dat['t_rest']<t_up)]
		filt_list = np.unique(dat['filter'])
		chi2_dic = {}
		dof = 0
		chi_tot = 0
		N_band = []
		for filt in filt_list:
			dat_filt = dat[dat['filter']==filt]
			time = dat_filt['t_rest']-t0
			m_array = self.mags(time,filt_list = [filt])
			mag = m_array[filt]
			mag_obs = dat_filt['absmag']
			mag_err = dat_filt['AB_MAG_ERR'] + sys_err
			c2 = chi_sq(mag_obs,mag_err, mag)
			N = len(c2)
			chi2_dic[filt] = c2
			dof = dof+N
			chi_tot = chi_tot +np.sum(chi2_dic[filt])
		dof = dof - nparams
		
		rchi2 = chi_tot/dof
			
		if chi_tot == 0:
			logprob = -np.inf
		elif dof <= 0:
			logprob = -np.inf
		else:
			logprob = log_chi_sq_pdf(chi_tot, dof)
		if np.isnan(logprob):
			import ipdb; ipdb.set_trace()
		return logprob, chi_tot, dof
	def likelihood_cov(self,dat,inv_cov,sys_err = 0,nparams = 7):
		t0 = self.t0 
		ebv = self.ebv
		Rv = self.Rv
		LAW = self.LAW
		d = self.distance
		t_down  = self.t_down 
		t_up    = self.t_up   
		cond_valid = (dat['t_rest']-t0>t_down)&(dat['t_rest']-t0<t_up)
		args_valid = np.argwhere(cond_valid).flatten()
		if np.sum(cond_valid) == 0:
			chi_tot = 0
			logprob = -np.inf
			dof = -1
			return logprob, chi_tot, dof
		inds = [np.min(args_valid),np.max(args_valid)+1]

		dat = dat[args_valid]
		#_,dat_res = compute_resid(dat.copy(),self)
		dof = len(dat)
		filt_list = np.unique(dat['filter'])
		delta = np.zeros(dof,)
		for filt in filt_list:
			cond_filt = dat['filter']==filt
			args_filt = np.argwhere(cond_filt).flatten()
			dat_filt = dat[args_filt]
			time = dat_filt['t_rest']-t0
			mag = dat['absmag'][args_filt]
			mags =   self.mags_single(time,filt=filt)
			res = np.array(mag - mags)
			delta[cond_filt] = res
		#cov = cov[inds[0]:inds[1],inds[0]:inds[1]]
		#inv_cov = np.linalg.inv(cov)
		inv_cov = inv_cov[inds[0]:inds[1],inds[0]:inds[1]]
		#prod = np.dot(delta,inv_cov)
		#chi_tot = np.dot(prod,delta)
		chi_tot = delta @ inv_cov @ delta
		dof = dof - nparams
		rchi2 = chi_tot/dof
		if chi_tot == 0:
			logprob = -np.inf
		elif dof <= 0:
			logprob = -np.inf
		else:
			#prob = scipy.stats.chi2.pdf(chi_tot, dof)
			#prob = float(prob)
			logprob = log_chi_sq_pdf(chi_tot, dof)
		if np.isnan(logprob):
			import ipdb; ipdb.set_trace()
		#logprob = np.log(prob)
		return logprob, chi_tot, dof
	def likelihood_bb(self,dat_bb,sys_err = 0.05,nparams = 10):
		t0 = self.t0 
		dof = 0
		chi_tot = 0
		
		time = dat_bb['t_rest']-t0
		L_evo = self.L_evolution(time)
		T_evo = self.T_evolution(time)
		err_L = np.sqrt((dat_bb['L_up']-dat_bb['L'])**2 + (sys_err*dat_bb['L'])**2)
		err_T = np.sqrt((dat_bb['T_up']-dat_bb['T'])**2 + (sys_err*dat_bb['T'])**2)

		c2_L = chi_sq(dat_bb['L'],err_L, L_evo)
		c2_T = chi_sq(dat_bb['T'],err_T, T_evo)

		N = len(c2_L) + len(c2_T)
		chi_tot = np.sum(c2_L) +np.sum(c2_T) 
		dof =N
		dof = dof - nparams
		
		rchi2 = chi_tot/dof

		logprob = -chi_tot
		return logprob, chi_tot, dof







def generate_cooling_mag(T_array, R_array ,filt,z=0, d = 3.08e26, Rv=3.1, EBV = 0,EBV_MW = 0, LAW = 'MW',H0 = 67.4):
	if len(T_array)!=len(R_array):
		raise Exception('R and T arrays have to be the same length') 
	v = H0*d/3.08e24
	z = v/300000
	lam  = np.linspace(1000,10000,90)
	mm = []
	for i in range(len(T_array)):
		flux = f_nu_bb(lam,T_array[i],R_array[i],d = d)[0]
		if EBV>0:
			flux = apply_extinction(lam,flux,EBV=EBV,EBV_mw=EBV_MW, R_v=Rv,z=z,LAW=LAW)
		Trans  = filter_transmission[filt]
		#m = SynPhot_fast(lam,flux,Trans)

		m = SynPhot_fast_AB(lam,flux,Trans['col1'],Trans['col2'])
		mm.append(m)	
	m_array = np.array(mm)
	return m_array 


@njit
def chi_sq(obs,err, mod):
	c2 = (obs-mod)**2/err**2
	return c2
from math import gamma


@njit
def log_chi_sq_pdf(x, k):

	logp = (k/2-1)*np.log(x)-0.5*x-0.5*k*np.log(2)-np.log(gamma(k/2))
	return logp


def compute_resid(Data,obj):
	filt_list = np.unique(Data['filter'])
	cond_dic = {}
	resid = {}
	obj.distance = 3.0856e+19
	Data['resid'] = np.zeros_like(Data['absmag'])
	Data['mod_mag'] = np.zeros_like(Data['absmag'])

	for band in filt_list:
		cond_dic[band] = (Data['filter']==band)&(Data['t']>0)
	# change to model_cooling_broken
	for i,band in enumerate(filt_list):
		t = Data['t_rest'][cond_dic[band]]-obj.t0
		mag = Data['absmag'][cond_dic[band]]
		mags =   obj.mags(t,filt_list=[band])
		res = np.array(mag - mags[band])
		resid[band] = res
		Data['resid'][cond_dic[band]] = res
		Data['mod_mag'][cond_dic[band]] = mags[band]

	return resid,Data



def construct_covariance_MG_v2(data,path_mat,path_key,dic_transmission,model_func = model_freq_dep_SC,from_bo =True, valid_inds = [],t0_vec = [0],**model_kwargs):
	print('Loading simulations')
	mat, key_mat, key_dic = get_sims(path_mat,path_key)
	global filter_transmission 
	filter_transmission = dic_transmission
	print('Constructing SN synthetic data from MG simulations')
	Data_filters2 = construct_sim(data,mat,t0_vec = t0_vec)
	if valid_inds ==[]:
		valid_inds = list(range(len(Data_filters2)))
	right_keys0 = [str(x) for x in valid_inds]
	#replicate every one of the keys in right_keys0 once for every t0 in t0_vec, so that the new keys are key+'_t0'+str(t0)
	right_keys = []
	for key in right_keys0:
		for t0 in t0_vec:
			right_keys.append(key+'_t0'+str(t0))
	Data_filters = {}
	for key in right_keys:
		Data_filters[key] = Data_filters2[key]	#Data_filters = Data_filters2
	N = len(data)
	print('Calculating residuals from analytic model')
	imax = 0
	resids = np.zeros((N,len(Data_filters)))
	for i,key in enumerate(list(Data_filters.keys())):
		sn_ind = key.split('_')[0]
		t0 = float(key.split('_t0')[1])
		sn_ind = int(sn_ind)
		data_sim = Data_filters[key]


		if not from_bo:
			R13,v85,fM,Menv =   [key_dic['Rstar'][sn_ind]/1e13
								,key_dic['v_shstar'][sn_ind]/10**8.5
								,key_dic['frhoM'][sn_ind]
								,key_dic['Menv'][sn_ind]/constants.M_sun.cgs.value]
		else: 
			R13,beta_bo,rho_bo9,Menv =   [key_dic['Rstar'][sn_ind]/1e13
								,key_dic['beta_bo'][sn_ind]
								,key_dic['rho_bo'][sn_ind]
								,key_dic['Menv'][sn_ind]/constants.M_sun.cgs.value]  
			v85,fM = phys_params_from_bo(rho_bo9,beta_bo,R13)
		obj =   model_func(R13,v85,fM,Menv,0,1, ebv = 0, distance = 3.0856e19,filter_transmission = filter_transmission,**model_kwargs)
		_,data_resid = compute_resid(data_sim,obj)
		t_down,t_up = validity2(R13, v85,fM,1,Menv)
		data_resid['is_valid'] = (data_resid['t_rest']<t_up)
		
		data_resid['resid'][~data_resid['is_valid']] = np.nan
		resids[:,i] = np.array(data_resid['resid'])
		imax = max(np.argwhere(data_resid['is_valid'])[-1][0]+1,imax)
		#data_resid['resid'][data_resid['resid']>4] = np.nan
	 
	if imax >= N:
		imax = N
	resids = resids[0:imax,:]
	print('Generating covariance matrix')
	resids_cov = np.zeros((imax,imax))

	tmax = data['t_rest'][imax-1]
	for i in range(imax):
		for j in range(imax):
			resids_cov[i,j] = np.nanmean(resids[i,:]*resids[j,:]) - np.nanmean(resids[i,:])*np.nanmean(resids[j,:])
	resids_mean = np.nanmean(resids[0:imax],axis = 1)
   
	return resids_cov,resids,tmax

def generate_full_covariance(data,path_mat,path_key,dic_transmission,sys_err = 0.1,covar = True,valid_inds = [], t0_vec = [0],**model_kwargs):
	if covar:
		resids_cov,_,t_max = construct_covariance_MG_v2(data,path_mat,path_key,dic_transmission = dic_transmission,model_func = model_freq_dep_SC,valid_inds=valid_inds,t0_vec = t0_vec,**model_kwargs)

		#resids_cov[np.isnan(resids_cov)] = 0   
		cov_obs = np.diagflat(np.array(data[(data['t_rest']<=t_max)]['AB_MAG_ERR'])**2+ sys_err**2)   
		#cov = resids_cov + cov_obs  
		#data_resid = data_resid['resid'][data_resid['t_rest']<tmax]
		#cov[np.isnan(cov)] = 0
		u,d,v = np.linalg.svd(resids_cov)
		A = u[:,:3] @ np.diag(d[:3]) @ v[:3,:]  
		if (np.linalg.eig(A+cov_obs)[0]<0).any():
			import ipdb; ipdb.set_trace
		cov_est = A+cov_obs
		inv_cov = np.linalg.inv(cov_est)
	else: 
		cov_obs = np.diagflat(np.array(data['AB_MAG_ERR'])**2+ sys_err**2)   
		cov_est = cov_obs  
		inv_cov = np.linalg.inv(cov_obs)
	return inv_cov,cov_est


def get_sim_params(sn_inds,**kwargs):
	inputs = {    'path_mat'    :  '/home/idoi/Dropbox/Objects/ZTF infant sample/Multigroup simulations/Multigroup simulations/RSG_batch_R03_20_removed_lines_Z1.mat'
				 ,'path_key'    :  '/home/idoi/Dropbox/Objects/ZTF infant sample/Multigroup simulations/Multigroup simulations/RSG_batch_R03_20_removed_lines_Z1_key.mat'
				 ,'path_w_key'  :  '/home/idoi/Dropbox/Objects/ZTF infant sample/Multigroup simulations/RSG_SED_R1_HiRes_key.mat'
				 ,'path_mat_hi' : '/home/idoi/Dropbox/Objects/ZTF infant sample/Multigroup simulations/RSG_SED_R1_HiRes.mat'
				 ,'path_key_hi' :  '/home/idoi/Dropbox/Objects/ZTF infant sample/Multigroup simulations/RSG_SED_R1_HiRes_key.mat'}

	inputs.update(kwargs)
	path_mat=inputs.get('path_mat')  
	path_key=inputs.get('path_key')  
	path_w_key=inputs.get('path_w_key')  
	path_key_hi=inputs.get('path_key_hi') 
	path_mat_hi=inputs.get('path_mat_hi') 
	mat, key_mat, key_dic = get_sims(path_mat,path_key)

	import scipy.io
	if mat == '':
		mat = scipy.io.loadmat(path_mat)
		mat = mat['SEDs_from_MG'][0]
	
	
	if key_mat == '':
		key_mat = scipy.io.loadmat(path_key)
		key_mat = key_mat['key'][0]
	
	if 'RSG_SED_batch1_key.mat' in path_key:
		key_dic = {}
		key_dic['names']   =    key_mat[0][0][0]
		key_dic['v_shstar'] =   key_mat[0][1][0]/10**8.5
		key_dic['E_env']   =    key_mat[0][3][0]
		key_dic['Rstar']   =    key_mat[0][4][0]/10**13
		key_dic['Mcore']   =    key_mat[0][5][0]/constants.M_sun.cgs.value
		key_dic['Menv']    =    key_mat[0][6][0]/constants.M_sun.cgs.value
		key_dic['E_inj']   =    key_mat[0][7][0]
		key_dic['frhoM']    =   key_mat[0][2][0]*(key_dic['Mcore']+key_dic['Menv'] )
	elif 'Full_batch_12_2022_Z_1_01_key.mat' in path_key:
		key_dic = {}
		key_dic['names']   =    key_mat[0][0][0]
		key_dic['v_shstar']  =   key_mat[0][18][0]/10**8.5
		key_dic['frhoM']    =   key_mat[0][19][0]/constants.M_sun.cgs.value
		key_dic['E_env']   =    key_mat[0][3][0]
		key_dic['Rstar']   =    key_mat[0][4][0]/10**13
		key_dic['Mcore']   =    key_mat[0][5][0]/constants.M_sun.cgs.value
		key_dic['Menv']    =     key_mat[0][6][0]/constants.M_sun.cgs.value
		key_dic['E_inj']   =   key_mat[0][7][0]
	params={}
	for sn in sn_inds:
		sn_ind = sn.split('_')[0]
		sn_ind = int(sn_ind)
		if len(sn.split('_'))>1:
			ebv = float(sn.split('_')[1])
			params[sn]  = [key_dic['v_shstar'][sn_ind]
						  ,key_dic['E_env'][sn_ind]
						  ,key_dic['Rstar'][sn_ind]
						  ,key_dic['Mcore'][sn_ind]
						  ,key_dic['Menv'][sn_ind]
						  ,key_dic['E_inj'][sn_ind]
						  ,key_dic['frhoM'][sn_ind]
						  #,key_dic['beta_bo'][sn_ind]
						  #,key_dic['rho_bo'][sn_ind]
						  ,ebv
						  ,key_dic['names'][sn_ind]]
		else:
			params[sn]  = [key_dic['v_shstar'][sn_ind]
						  ,key_dic['E_env'   ][sn_ind]
						  ,key_dic['Rstar'   ][sn_ind]
						  ,key_dic['Mcore'   ][sn_ind]
						  ,key_dic['Menv'    ][sn_ind]
						  ,key_dic['E_inj'   ][sn_ind]
						  ,key_dic['frhoM'   ][sn_ind]
						  ,key_dic['names'   ][sn_ind]]      
	return params

def construct_sim(data,mat, d = 3.0856e+19, t0_vec =[0] ):
	sn_skeleton = data.copy()
	filters2include = np.unique(sn_skeleton['filter'])
	runs = {str(i): mat[i] for i in range(len(mat))}
	Data_filters = {}
	times = {}
	instrum = {}
	piv  = {}
	Null_tab  = {}
	for key in  tqdm.tqdm(runs.keys()):
		for t0 in t0_vec:
			dat = runs[key]
			t = dat[0][0]/3600/24
			t = t-t[0]+t0 # t is measure relative to CC and not breakout. t-t[0] might remove first hour or so from explosion as simulations start at most from 2R/C and R<1e14
			f_lam = dat[1]/4/np.pi/d**2
			lam = dat[2]
			Tab = sn_skeleton[0:0]['t_rest','filter','piv_wl','AB_MAG','instrument']
			Tab = Tab.to_pandas()
			for filt in filters2include: 
				cond = (sn_skeleton['filter'] == filt)&(sn_skeleton['t_rest']<np.max(t))
				cond2 = (sn_skeleton['filter'] == filt)&(sn_skeleton['t_rest']>np.max(t))
				times[filt] = np.array(sn_skeleton['t_rest'][cond])
				piv[filt] =     sn_skeleton['piv_wl'][0]
				instrum[filt] = sn_skeleton['instrument'][0]
				Null_tab[filt] = sn_skeleton[cond2]['t_rest','AB_MAG']
				Null_tab[filt]['AB_MAG'] = Null_tab[filt]['AB_MAG']*np.nan

			for filt in filters2include:
			
				inds = list(map(lambda T: np.argwhere(t>T)[0][0],times[filt]))
				mag = []
				for i in inds:
					flux = f_lam[i,:]
					m = SynPhot_fast_AB(lam,flux,filter_transmission[filt][:,0],filter_transmission[filt][:,1])[0]
					mag.append(m)
				tab = table.Table([times[filt],mag], names = ['t_rest','AB_MAG'])
				if len(times[filt]>0):
					tab['filter'] = filt
					tab['piv_wl'] = piv[filt]
					tab['instrument'] = instrum[filt]

				else: 
					tab['filter']     = sn_skeleton[0:0]['filter']    
					tab['piv_wl']     = sn_skeleton[0:0]['piv_wl']     
					tab['instrument'] = sn_skeleton[0:0]['instrument']              
				tab = tab['t_rest','filter','piv_wl','AB_MAG','instrument']
				tab = tab.to_pandas()
				if len(Null_tab[filt])>0:
					Null_tab[filt]['filter'] = filt
					Null_tab[filt]['piv_wl'] = piv[filt]
					Null_tab[filt]['instrument'] = instrum[filt]
					Null_tab[filt] = Null_tab[filt]['t_rest','filter','piv_wl','AB_MAG','instrument']
					Null_tab[filt] = Null_tab[filt].to_pandas()
					tab = pd.concat([tab,Null_tab[filt]])
				Tab = pd.concat([Tab,tab])
			Tab =  table.Table.from_pandas(Tab)
			Tab.sort('t_rest')
			Tab['absmag'] = Tab['AB_MAG']
			Tab['jd'] = Tab['t_rest']
			Tab['t'] = Tab['t_rest']
			Data_filters[key+'_t0'+str(t0)] = Tab.copy()
	return Data_filters

def get_sims(path_mat,path_key):
	
	import scipy.io
	mat = scipy.io.loadmat(path_mat)
	mat = mat['SEDs_from_MG'][0]
	key_mat = scipy.io.loadmat(path_key)
	key_mat = key_mat['key'][0]
	
	if path_key.split(sep)[-1]=='RSG_SED_batch1_key.mat':
		key_dic = {}
		key_dic['names'] = key_mat[0][0][0]
		key_dic['v_shstar'] = key_mat[0][1][0]
		key_dic['frho'] = key_mat[0][2][0]
		key_dic['E_env'] = key_mat[0][3][0]
		key_dic['Rstar'] = key_mat[0][4][0]
		key_dic['Mcore'] = key_mat[0][5][0]
		key_dic['Menv'] = key_mat[0][6][0]
		key_dic['E_inj'] = key_mat[0][7][0]
		key_dic['beta_bo'] = key_mat[0][8][0]
		key_dic['rho_bo'] = key_mat[0][9][0]/1e-9
		key_dic['frhoM']    =     key_mat[0][2][0]*(key_dic['Mcore']+key_dic['Menv'] )
	elif path_key.split(sep)[-1]=='RSG_batch_R03_20_removed_lines_Z1_key.mat':
		key_dic = {}
		key_dic['names']   =    key_mat[0][0][0]
		key_dic['v_shstar']= key_mat[0][1][0]
		key_dic['frho']   =    key_mat[0][2][0]
		key_dic['E_env']   =    key_mat[0][3][0]
		key_dic['Rstar']   =    key_mat[0][4][0]
		key_dic['Mcore']   =    key_mat[0][5][0] 
		key_dic['Menv']    =     key_mat[0][6][0]
		key_dic['frhoM']   =    key_dic['frho']*(key_dic['Mcore']+key_dic['Menv'] )/constants.M_sun.cgs.value
		key_dic['E_inj']   =   key_mat[0][7][0]
		key_dic['tbo_s']   =   key_mat[0][8][0]
		key_dic['beta_bo'] =  key_mat[0][9][0]
		key_dic['rho_bo']  =   key_mat[0][10][0]/1e-9
		key_dic['frhoM']    =     key_mat[0][2][0]*(key_dic['Mcore']+key_dic['Menv'] )/constants.M_sun.cgs.value
	
	elif path_key.split(sep)[-1] == 'Full_batch_12_2022_Z_1_01_key.mat':
		key_dic = {}
		key_dic['names']   =    key_mat[0][0][0]
		key_dic['v_shstar']= key_mat[0][1][0]
		key_dic['frho']   =    key_mat[0][2][0]
		key_dic['E_env']   =    key_mat[0][3][0]
		key_dic['Rstar']   =    key_mat[0][4][0]
		key_dic['Mcore']   =    key_mat[0][5][0]
		key_dic['Menv']    =     key_mat[0][6][0]
		key_dic['E_inj']   =   key_mat[0][7][0]
		key_dic['beta_bo'] =  key_mat[0][9][0]
		key_dic['rho_bo']  =   key_mat[0][10][0]/1e-9
		key_dic['frhoM']    =     key_mat[0][2][0]*(key_dic['Mcore']+key_dic['Menv'] )/constants.M_sun.cgs.value
	return mat, key_mat, key_dic



def likelihood_freq_dep_SC(data,R13,v85,fM,Menv,t0,filter_transmission,k34=1,sys_err = 0.05,nparams = 7,**model_kwargs):
	obj = model_freq_dep_SC(R13,v85,fM,Menv,t0,k34,filter_transmission = filter_transmission,**model_kwargs)
	logprob, chi_tot, dof_tot = obj.likelihood(data, sys_err = sys_err ,nparams = nparams)
	return logprob, chi_tot, dof_tot 

def likelihood_freq_dep_SC_cov(data,inv_cov,R13,v85,fM,Menv,t0,filter_transmission,k34=1,sys_err = 0.05,nparams = 7,**model_kwargs): 
	obj = model_freq_dep_SC(R13,v85,fM,Menv,t0,k34,filter_transmission = filter_transmission,**model_kwargs)
	logprob, chi_tot, dof_tot = obj.likelihood_cov(data,inv_cov, sys_err = sys_err ,nparams = nparams)
	return logprob, chi_tot, dof_tot 


def likelihood_SC_cov(data,inv_cov,R13,v85,fM,Menv,t0,filter_transmission,k34=1,distance = 3.0856e19,ebv = 0,Rv = 3.1 , LAW = 'MW',sys_err = 0.05,nparams = 7,UV_sup = False,reduced=True):
	
	obj = model_SC(R13,v85,fM,Menv,t0,k34, distance =distance, ebv = ebv, Rv = Rv, LAW = LAW,filter_transmission = filter_transmission)
	logprob, chi_tot, dof_tot = obj.likelihood_cov(data,inv_cov, sys_err = sys_err ,nparams = nparams)
	return logprob, chi_tot, dof_tot 



def uniform_prior_transform(u,**kwargs):
	inputs={'priors':[np.array([15000,50000]),
					  np.array([10**14,10**15]),
					  np.array([-1.3,0]),
					  np.array([0.5,1]),
					  np.array([-1,0]),
					  np.array([1,5]),
					  np.array([0,-1.3]),
					  np.array([0.5,1])]}                            
	inputs.update(kwargs)
	priors=inputs.get('priors')
	def strech_prior(u,pmin,pmax):
		x = (pmax-pmin) * u + pmin
		return x
	prior_list = []
	for i,prior in enumerate(priors):
		pmin=prior[0]
		pmax=prior[1]
		prior_dist=strech_prior(u[i],pmin,pmax)
		prior_list.append(prior_dist)
	return prior_list

def loguniform_prior_transform(u,**kwargs):
	inputs={'priors':[np.array([15000,50000]),
					  np.array([10**14,10**15]),
					  np.array([-1.3,0]),
					  np.array([0.5,1]),
					  np.array([-1,0]),
					  np.array([1,5]),
					  np.array([0,-1.3]),
					  np.array([0.5,1])]}                            
	inputs.update(kwargs)
	priors=inputs.get('priors')
	def strech_prior(u,pmin,pmax):
		x = 10**((np.log10(pmax)-np.log10(pmin)) * u + np.log10(pmin))
		return x
	prior_list = []
	for i,prior in enumerate(priors):
		pmin=prior[0]
		pmax=prior[1]
		prior_dist=strech_prior(u[i],pmin,pmax)
		prior_list.append(prior_dist)
	return prior_list

def fit_freq_dep_SC(data, dic_transmission,k34 = 1, plot_corner = True,sys_err = 0.05,**kwargs):  
	inputs={'priors':[np.array([0.05,5]),
					  np.array([0.3,5]),
					  np.array([0.05,1000]),
					  np.array([0.1,10]),
					  np.array([-0.5,0.5]),
					  np.array([0,0.3]),
					  np.array([2,5])]
			,'prior_type':['log-uniform','log-uniform','log-uniform','log-uniform','uniform','uniform','uniform']
			,'maxiter':200000
			,'maxcall':500000
			,'nlive':250
			,'ebv':'fit'
			,'Rv':3.1
			,'LAW':'MW',
			'UV_sup':False,
			'reduced':True,
			'covariance':True
			,'inv_cov':''
			,'rec_time_lims':[-np.inf,np.inf]
			,'validity':'all'
			,'t_tr_min':-np.inf}                            
	inputs.update(kwargs)
 
	maxiter=inputs.get('maxiter')  
	maxcall=inputs.get('maxcall')  
	nlive=inputs.get('nlive')  
	ebv = inputs.get('ebv')  
	Rv = inputs.get('Rv')  
	LAW = inputs.get('LAW')  
	priors=inputs.get('priors') 
	UV_sup = inputs.get('UV_sup')
	reduced = inputs.get('reduced')
	covariance = inputs.get('covariance')
	inv_cov = inputs.get('inv_cov')
	rec_time_lims = inputs.get('rec_time_lims')
	t_tr_min = inputs.get('t_tr_min')
	validity = inputs.get('validity')
	prior_type = inputs.get('prior_type')
	if ebv == 'fit':
		if Rv != 'fit':
			priors = priors[0:6]
	else:
		priors = priors[0:5]	
	if inv_cov =='':
		 inv_cov=  np.diagflat(1/(np.array(data['AB_MAG_ERR'])**2+ sys_err**2))

	global filter_transmission_fast 
	filter_transmission_fast = dic_transmission
	data = data['t_rest','filter','absmag','AB_MAG_ERR']


	def prior_transform(u,priors = priors,prior_type = prior_type):
		x = np.zeros_like(u)
		for i in range(len(priors)):
			if prior_type[i] == 'uniform':
				x[i:i+1]=uniform_prior_transform(u[i:i+1],priors =priors[i:i+1] )
			elif prior_type[i] == 'log-uniform':
				x[i:i+1]=loguniform_prior_transform(u[i:i+1],priors =priors[i:i+1] )
		return x
	
	def myloglike(x):
		R13,v85,fM,Menv,t0,ebv = x   
		t07 = 6.86*R13**(0.56)*v85**(0.16)*k34**(-0.61)*fM**(-0.06) + t0
		t_tr =  19.5*np.sqrt(Menv*k34/v85)
		if (t07<rec_time_lims[1])&(t07>rec_time_lims[0])&(t_tr>t_tr_min):
			if covariance:
				loglike = likelihood_freq_dep_SC_cov(data,inv_cov,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 6,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast,validity = validity)[0]
			else:
				loglike = likelihood_freq_dep_SC(data,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 6,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast,validity = validity)[0]
		else: 
			loglike = -np.inf
		return loglike  
	def myloglike2(x):
		R13,v85,fM,Menv,t0,ebv,Rv = x   
		t07 = 6.86*R13**(0.56)*v85**(0.16)*k34**(-0.61)*fM**(-0.06) + t0
		t_tr =  19.5*np.sqrt(Menv*k34/v85) + t0
		t_down,t_up = validity2(R13, v85,fM,k34,Menv)
		if (t07<rec_time_lims[1])&(t07>rec_time_lims[0])&(t_tr>t_tr_min):   
			if covariance:
				loglike = likelihood_freq_dep_SC_cov(data,inv_cov,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 7,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast,validity = validity)[0]
			else: 
				loglike = likelihood_freq_dep_SC(data,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 7,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast,validity = validity)[0]
		else: 
			loglike = -np.inf
		return loglike
	def myloglike3(x):
		R13,v85,fM,Menv,t0 = x 
		t07 = 6.86*R13**(0.56)*v85**(0.16)*k34**(-0.61)*fM**(-0.06) + t0
		t_tr =  19.5*np.sqrt(Menv*k34/v85) + t0
		if (t07<rec_time_lims[1])&(t07>rec_time_lims[0])&(t_tr>t_tr_min):
			if covariance:
				loglike = likelihood_freq_dep_SC_cov(data,inv_cov,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 5,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast,validity = validity)[0]
			else: 
				loglike = likelihood_freq_dep_SC(data,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 5,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast,validity = validity)[0]
		else: 
			loglike = -np.inf
		return loglike

	if ebv == 'fit':
		if Rv == 'fit': 
			myloglike_choice = myloglike2
			ndim = 7
			labels = [r'$R_{13}$',r'$v_{s,8.5}$',r'$f_{\rhp}M$',r'$M_{env}$',r'$t_{0}$',r'$E(B-V)$',r'$R_V$']
			labels = labels        
		else:
			ndim = 6
			labels = [r'$R_{13}$',r'$v_{s,8.5}$',r'$f_{\rhp}M$',r'$M_{env}$',r'$t_{0}$',r'$E(B-V)$'] 
			myloglike_choice = myloglike
	else:
		ndim = 5
		labels = [r'$R_{13}$',r'$v_{s,8.5}$',r'$f_{\rhp}M$',r'$M_{env}$',r'$t_{0}$']
		myloglike_choice = myloglike3

	dsampler = dynesty.DynamicNestedSampler(myloglike_choice, prior_transform,  ndim = ndim,nlive=nlive,update_interval=600)
	dsampler.run_nested(maxiter=maxiter, maxcall=maxcall)
	#from multiprocessing import Pool
 	#use Pool to multiproccess
	#import ipdb; ipdb.set_trace()
#	with Pool(4,myloglike_choice,prior_transform) as pool:
#		dsampler = dynesty.DynamicNestedSampler(pool.myloglike_choice, pool.prior_transform,  ndim = ndim,nlive=nlive,update_interval=600,pool=pool)
#		dsampler.run_nested(maxiter=maxiter, maxcall=maxcall)
#
#	with Pool(4) as pool:
#		dsampler = dynesty.DynamicNestedSampler(myloglike_choice, prior_transform,  ndim = ndim,nlive=nlive,update_interval=600,pool=pool)
#		dsampler.run_nested(maxiter=maxiter, maxcall=maxcall)
#
	dresults = dsampler.results
	if plot_corner:
		# Plot a summary of the run.
		#rfig, raxes = dyplot.runplot(dresults)
		# Plot traces and 1-D marginalized posteriors.
		#tfig, taxes = dyplot.traceplot(dresults)    
		# Plot the 2-D marginalized posteriors.
		cfig, caxes = dyplot.cornerplot(dresults,labels=labels
								,label_kwargs={'fontsize':14}, color = '#0042CF',show_titles = True)
	## add labels!
	# Extract sampling results.
	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])  # normalized weights
	# Compute 10%-90% quantiles.
	quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
				 for samps in samples.T]
	# Compute weighted mean and covariance.
	mean, _ = dyfunc.mean_and_cov(samples, weights)
	# Resample weighted samples.
	#samples_equal = dyfunc.resample_equal(samples, weights)
	# Generate a new set of results with statistical+sampling uncertainties.
	#results_sim = dyfunc.simulate_run(dresults)

	return mean, quantiles,dresults

	
def fit_SC(data, dic_transmission,k34 = 1, plot_corner = True,sys_err = 0.05,**kwargs):  
	inputs={'priors':[np.array([0.05,5]),
					  np.array([0.3,5]),
					  np.array([0.05,1000]),
					  np.array([0.1,10]),
					  np.array([-0.5,0.5]),
					  np.array([0,0.3]),
					  np.array([2,5])]
			,'maxiter':100000
			,'maxcall':500000
			,'nlive':250
			,'ebv':'fit'
			,'Rv':3.1
			,'LAW':'MW',
			'UV_sup':False,
			'reduced':True,
			'covariance':True
			,'inv_cov':''
			,'rec_time_lims':[-np.inf,np.inf]}                            
	inputs.update(kwargs)
 
	maxiter=inputs.get('maxiter')  
	maxcall=inputs.get('maxcall')  
	nlive=inputs.get('nlive')  
	ebv = inputs.get('ebv')  
	Rv = inputs.get('Rv')  
	LAW = inputs.get('LAW')  
	priors=inputs.get('priors') 
	UV_sup = inputs.get('UV_sup')
	reduced = inputs.get('reduced')

	covariance = inputs.get('covariance')
	inv_cov = inputs.get('inv_cov')
	rec_time_lims = inputs.get('rec_time_lims')

	if ebv == 'fit':
		if Rv != 'fit':
			priors = priors[0:6]
	else:
		priors = priors[0:5]
	if inv_cov =='':
		 inv_cov=  np.diagflat(1/(np.array(data['AB_MAG_ERR'])**2+ sys_err**2))

	global filter_transmission_fast 
	filter_transmission_fast = dic_transmission
	data = data['t_rest','filter','absmag','AB_MAG_ERR']


	def prior_transform(u,priors = priors):
		x=uniform_prior_transform(u,priors =priors )
		return x

	def myloglike(x):
		R13,v85,fM,Menv,t0,ebv = x   
		t07 = 6.86*R13**(0.56)*v85**(0.16)*k34**(-0.61)*fM**(-0.06) + t0
		if (t07<rec_time_lims[1])&(t07>rec_time_lims[0]):
			if covariance:
				loglike = likelihood_SC_cov(data,inv_cov,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 6,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast)[0]
			else:
				loglike = likelihood_SC(data,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 6,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast)[0]
		else: 
			loglike = -np.inf
		return loglike  
	def myloglike2(x):
		R13,v85,fM,Menv,t0,ebv,Rv = x   
		t07 = 6.86*R13**(0.56)*v85**(0.16)*k34**(-0.61)*fM**(-0.06) + t0
		if (t07<rec_time_lims[1])&(t07>rec_time_lims[0]):        
			if covariance:
				loglike = likelihood_SC_cov(data,inv_cov,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 7,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast)[0]
			else: 
				loglike = likelihood_SC(data,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 7,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast)[0]
		else: 
			loglike = -np.inf
		return loglike
	def myloglike3(x):
		R13,v85,fM,Menv,t0 = x 
		t07 = 6.86*R13**(0.56)*v85**(0.16)*k34**(-0.61)*fM**(-0.06) + t0
		if (t07<rec_time_lims[1])&(t07>rec_time_lims[0]):   
			if covariance:
				loglike = likelihood_SC_cov(data,inv_cov,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 5,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast)[0]
			else: 
				loglike = likelihood_SC(data,R13,v85,fM,Menv,t0=t0,k34 = k34,ebv = ebv, Rv = Rv, sys_err = sys_err, LAW = LAW,nparams = 5,UV_sup=UV_sup,reduced=reduced,filter_transmission = filter_transmission_fast)[0]
		else: 
			loglike = -np.inf
		return loglike

	if ebv == 'fit':
		if Rv == 'fit': 
			myloglike_choice = myloglike2
			ndim = 7
			labels = [r'$R_{13}$',r'$v_{s,8.5}$',r'$f_{\rhp}M$',r'$M_{env}$',r'$t_{0}$',r'$E(B-V)$',r'$R_V$']
			labels = labels        
		else:
			ndim = 6
			labels = [r'$R_{13}$',r'$v_{s,8.5}$',r'$f_{\rhp}M$',r'$M_{env}$',r'$t_{0}$',r'$E(B-V)$'] 
			myloglike_choice = myloglike
	else:
		ndim = 5
		labels = [r'$R_{13}$',r'$v_{s,8.5}$',r'$f_{\rhp}M$',r'$M_{env}$',r'$t_{0}$']
		myloglike_choice = myloglike3

	dsampler = dynesty.DynamicNestedSampler(myloglike_choice, prior_transform,  ndim = ndim,nlive=nlive,update_interval=600)

	dsampler.run_nested(maxiter=maxiter, maxcall=maxcall)
	dresults = dsampler.results
	if plot_corner:
		# Plot a summary of the run.
		#rfig, raxes = dyplot.runplot(dresults)
		# Plot traces and 1-D marginalized posteriors.
		#tfig, taxes = dyplot.traceplot(dresults)    
		# Plot the 2-D marginalized posteriors.
		cfig, caxes = dyplot.cornerplot(dresults,labels=labels
								,label_kwargs={'fontsize':14}, color = '#0042CF',show_titles = True)
	## add labels!
	# Extract sampling results.
	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])  # normalized weights
	# Compute 10%-90% quantiles.
	quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
				 for samps in samples.T]
	# Compute weighted mean and covariance.
	mean, _ = dyfunc.mean_and_cov(samples, weights)
	# Resample weighted samples.
	#samples_equal = dyfunc.resample_equal(samples, weights)
	# Generate a new set of results with statistical+sampling uncertainties.
	#results_sim = dyfunc.simulate_run(dresults)

	return mean, quantiles,dresults


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

def plot_SED_data(t,data,filter_transmission,c_band = {},obj_mod = '', ax = plt.axes(), fig = plt.figure()):
	filt_list = np.unique(data['filter'])
	for filt in filt_list:
		if filt not in c_band.keys():
			c_band[filt] = 'k'
		f,ferr,_ = get_inter_mag(t, data, filt,mag_col = 'flux', mag_col_err = 'fluxerr')
		piv_wl = data['piv_wl'][data['filter']==filt][0]
		Tran = filter_transmission[filt]    
		w_eff = np.trapz(Tran[:,1],x = Tran[:,0])/np.max(Tran[:,1])
		#import ipdb; ipdb.set_trace()
		ax.errorbar(piv_wl,f,xerr = w_eff, yerr = ferr, marker = 'o', color = c_band[filt])
		if obj_mod!='':
			mags_mod = obj.mags_single([t],filt)
			flux_mod = maggie2cgs(10**(-0.4*mags_mod[filt]))
			flux_mod = f_nu2f_lam(flux_mod,piv_wl)
			ax.plot(piv_wl,flux_mod, marker = 'P',markersize = 8, color = c_band[filt])
	return fig,ax


def plot_SED_model(obj,t, ax = plt.axes(), fig = plt.figure(),label = '',plt_kwargs= {},**model_kwargs):
	wl = np.logspace(3.17,4.1,100)
	#test if t is an array or a single value
	if ~isinstance(t, (list, np.ndarray)):
		t = np.array([t])
	f_array = generate_MW_flux(t,wl,obj.R13,obj.v85,obj.fM,obj.k34,obj.Menv, d = obj.distance, Rv=obj.Rv, EBV = obj.ebv,EBV_MW = 0, LAW = obj.LAW,**model_kwargs)

	ax.plot(wl,f_array[0,:],label = label,**plt_kwargs)

	return fig,ax


def get_RW_params_from_bb(T,L,t,t_tr=20):
	eV2K = 11604.518121550082
	R13 = ((T*t**0.45)/eV2K/1.66)**4
	L208e42 = L/2.08e42/(0.9*np.exp(-(2*t/t_tr)**0.5))
	v85 = ((L208e42/R13)*t**-0.17)**(1/1.91)
	return R13,v85

def plot_SED_sequence(data,samples,weights,d_mpc,filter_transmission,c_band = {}, fig_sed = plt.figure(figsize=(20,15)),plt_kwargs= {},save_path = '',sn = '',**model_kwargs):
	piv_wl_dic = {}
	for filt in np.unique(data['filter','piv_wl']):
		piv_wl_dic[filt[0]] = filt[1]
	data = data[data['t_rest']>0]
	best =  samples[weights == np.max(weights)][0]
	if len(best)==6:
		R13,v85,fM,Menv,t0,ebv = best
		Rv = 3.1
	elif len(best)==7:
		R13,v85,fM,Menv,t0,ebv,Rv = best
	randind = np.random.choice(len(samples), size=50, replace=True, p=weights/weights.sum())
	random = samples[randind]
	#unpack best to R13,v85,fM,Menv,t0,ebv,Rv
	objs_rand = []
	obj_best= model_freq_dep_SC(R13,v85,fM,Menv,t0,1, ebv = ebv, Rv = Rv,**model_kwargs)
	obj_best.distance = d_mpc*3.0856e24
	for i in range(len(random)):
		if len( random[i]) ==7:
			R13_rand,v85_rand,fM_rand,Menv_rand,t0_rand,ebv_rand,Rv_rand = random[i]
		else: 
			R13_rand,v85_rand,fM_rand,Menv_rand,t0_rand,ebv_rand = random[i]
			Rv_rand = 3.1
		obj_rand = model_freq_dep_SC(R13_rand,v85_rand,fM_rand,Menv_rand,t0_rand,1, ebv = ebv_rand, Rv = Rv_rand,**model_kwargs)
		obj_rand.distance = d_mpc*3.0856e24
		objs_rand.append(obj_rand)
	#obj3 = model_freq_dep_SC(R13,v85,fM,Menv,t0,1, ebv = ebv, Rv = Rv, LAW = 'MW', distance = 3.0856e19, reduced = True)
	#obj_bb = model_SC       (R13,v85,fM,Menv,t0,1, ebv = ebv, Rv = Rv, LAW = 'MW', distance = 3.0856e19)
	t_log = np.logspace(-1,1.5,100)
	T_eV = obj_best.T_evolution(t_log)/eV2K
	T_hz = T_eV/eVs
	T_AA = cAA/T_hz
	#t7eV = t_log[T_AA/7 >1500][0]
	t2eV = t_log[T_eV<2][0]
	t1eV = t_log[T_eV<1][0]
	t15eV = t_log[T_eV<1.5][0]
	t13eV = t_log[T_eV<1.3][0]
	data.sort('t_rest')
	t_first = data['t_rest'][0]
	t_first =t_first +0.001
	if t_first - obj_best.t0 > obj_best.t_down:
		t_list = [t_first,t2eV,t15eV,t13eV,t1eV]
	else: 
		t_list = [t2eV,t15eV,t13eV,t1eV]
	#t_UV_first = data[data['instrument'] == 'Swift+UVOT']['t_rest'][0]+0.1
	#T_first = obj_best.T_evolution(t_first)[0]/eV2K	
	#T_first_UV = obj_best.T_evolution(t_UV_first)[0]/eV2K	
	for i,t in enumerate(t_list):
		ax_sed = plt.subplot(2,3,i+1)
		T =  obj_best.T_evolution(t)[0]/eV2K	
		plot_SED_data(t ,data,filter_transmission,c_band=c_band,fig = fig_sed, ax = ax_sed)
		#plot_SED_model(obj_best,t15eV ,fig = fig_sed, ax = ax_sed,old_form = True , reduced = False, UV_sup = False, gray = False,label = 'LT (old)',plt_kwargs= {'ls':'-','color':'r'})
		#plot_SED_model(obj_best,t15eV ,fig = fig_sed, ax = ax_sed,old_form = False , reduced = False, UV_sup = False, gray = False,label =  'New formula',plt_kwargs= {'ls':'-','color':'k'})
		#plot_SED_model(obj_best,t15eV ,fig = fig_sed, ax = ax_sed,old_form = False , reduced = True, UV_sup = False, gray = False,label =  'LT (new)',plt_kwargs= {'ls':'--','color':'k'})
		plot_SED_model(obj_best,t ,fig = fig_sed, ax = ax_sed ,old_form = False, reduced = False, UV_sup = False, gray = False,plt_kwargs= {'ls':'-.','color':(0.5,0.5,0.5,0.5)})
		plot_SED_model(obj_best,t ,fig = fig_sed, ax = ax_sed ,old_form = False, reduced = False, UV_sup = False, gray = True,plt_kwargs= {'ls':'-.','color':'orange','alpha':0.5})

		for obj in objs_rand:
			plot_SED_model(obj,t-(obj.t0-obj_best.t0) ,fig = fig_sed, ax = ax_sed,old_form = False , reduced = False, UV_sup = False, gray = False,plt_kwargs= {'ls':'-','color':(0.5,0.5,0.5, 0.8/len(random))})
		plt.text(0.1,0.25,'{0:.1f} eV'.format(T) ,transform = plt.gca().transAxes,fontsize = 18)
		plt.xscale('log')
		plt.yscale('log')
	plt.text(0.4,0.9,sn,transform = plt.gcf().transFigure,fontsize = 18)

	plt.subplots_adjust(hspace=0.15,wspace=0.2)
	plt.text(0.05,0.2, 'Flux [$erg\ s^{-1}\ cm^{-2}\ \AA^{-1}$]',transform = plt.gcf().transFigure,rotation = 'vertical',fontsize = 24)
	plt.text(0.45,0.05, 'Wavelength [$\AA$]',transform = plt.gcf().transFigure,rotation = 'horizontal',fontsize = 24)
	#plt.legend(loc = 'lower left',bbox_to_anchor = (1.13,-0.1,1.3,2),mode = 'expand',bbox_transform = plt.gca().transAxes,fontsize = 24)
	if save_path!='':
		plt.savefig(save_path, dpi = 300)
	return fig_sed



def plot_resid(dat,obj, c_band, lab_band, sigma = False,fig = 'create',ax = None):
	filt_list = np.unique(dat['filter'])
	cond_dic = {}
	resid = {}
	for band in filt_list:
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)

	# change to model_cooling_broken

	for i,band in enumerate(filt_list):
		t = dat['t_rest'][cond_dic[band]]-obj.t0
		mag = dat['absmag'][cond_dic[band]]
		mags =   obj.mags(t,filt_list=[band])
		res = mag - mags[band]
		if sigma: 
			magerr = dat['AB_MAG_ERR'][cond_dic[band]]
			res = res/magerr
		resid[band] = res
	
	if fig == 'create': 
		plt.figure(figsize=(15,6))
		ax = plt.axes()
	
	for i,band in enumerate(filt_list):
		if ~sigma:     
			ax.errorbar(dat['t_rest'][cond_dic[band]]-obj.t0, resid[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10, label = lab_band[band])
		else: 
			ax.plot(dat['t_rest'][cond_dic[band]]-tobj.t0, resid[band],marker = '*', ls = '',color = c_band[band], markersize = 10, label = lab_band[band])
	plt.plot([0.1,np.max(dat['t_rest'])],[0,0],'k--')
	#plt.xlim((-2,1.1*np.max(dat['t_rest'])))
	plt.xscale('log')
	plt.gca().invert_yaxis()
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	plt.ylabel('M - model (AB mag)')
	if sigma: 
		plt.ylabel('M - model ($\sigma$)')
	plt.legend()
	return fig, ax


def plot_resid_covariance(dat,obj,cov, fig = 'create', ax = None,figsize = (6,4), imax = -1):
	#use compute resid to get the resid vector
	resid,Data = compute_resid(dat.copy(),obj)
	#compute the validity time of the model given obj
	t_down = obj.t_down
	t_up = obj.t_up
	if np.shape(cov)[0]>len(Data):
		Data = Data[0:np.shape(cov)[0]]
	else:
		cov = cov[0:len(Data),0:len(Data)]
	cond_valid = (dat['t_rest']>t_down)&(dat['t_rest']<t_up)
	args_valid = np.argwhere(cond_valid).flatten()
	inds = [np.min(args_valid),np.max(args_valid)+1]
	cov = cov[inds[0]:inds[1],inds[0]:inds[1]]
	Data = Data[cond_valid]


	 #np.sqrt(np.trace(AA @ AA.T)) 
	# compute the SVD of the covariance matrix
	u, s, vh = np.linalg.svd(cov)

	N = np.shape(cov)[0]
	fractional_std = np.zeros((N,))
	for i in range(N):
		AA = u[:,:i] @ np.diag(s[:i]) @ vh[:i,:]  
		fractional_std[i] = np.std(AA)/np.std(cov)


	#use the matriced  to rotate the resid vector to the SVD basis
	rot_resid = np.dot(Data['resid'],vh)
	A = u[:,:3] @ np.diag(s[:3]) @ vh[:3,:] 
	#compute the error in the rotated basis
	rot_resid_err = np.sqrt(np.dot(Data['AB_MAG_ERR']**2,np.abs(vh)))
	if fig == 'create':
		fig = plt.figure(figsize=figsize)
		ax = plt.axes()
	ax.errorbar(np.arange(len(s[:imax])),rot_resid[:imax]/np.sqrt(s[:imax]),rot_resid_err[:imax]/np.sqrt(s[:imax]),label = 'Residuals in diagonal basis',ls = '', marker = 'o', color = 'b')
	ax.legend(fontsize = 14)
	ax.set_xlabel('SVD component',fontsize = 14)
	ax.set_ylabel('Residuals (sigma)',fontsize = 14)
	xlim= ax.get_xlim()

	ax.fill_between(np.arange(len(s[:imax])),-1,1, color = 'b', alpha = 0.2, label = 'Expected noise from the Covariance matrix')
	ax.plot(xlim,[0,0],'k--')
	ax.twiny()
	plt.plot(Data['t_rest'],Data['resid']/Data['AB_MAG_ERR'],label = 'Residuals in original basis',ls = '', marker = 'o', color = 'r')
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	ax.twinx()
	plt.plot(np.arange(len(s[:imax])),fractional_std[:imax], color = 'g')
	plt.ylabel('Fractional standard deviation',fontsize = 14)
	plt.legend(fontsize = 14)

	return rot_resid, s, fig, ax


class model_piro_cooling(object):
	def __init__(self, Me,Re, E51,t0, kappa=0.07, n=10, delta=1.1  ,distance = 3.08e26,ebv = 0,Rv = 3.1, LAW = 'MW'):
		self.ebv = ebv
		self.Rv = Rv
		self.distance = distance
		self.LAW  = LAW
		self.t0 = t0
		self.Me = Me
		self.Re = Re
		self.E51 = E51
		self.kappa = kappa
		self.n = n
		self.delta = delta


	def R_evolution(self,time): 
		time = time.flatten()
		R = Rph_shock_cooling_Piro2021(time, self.Me, self.Re, self.E51, kappa=self.kappa, n=self.n, delta = self.delta)[1]
		return R
	def L_evolution(self, time):
		time = time.flatten()
		L_evo = L_shock_cooling_Piro2021(time, self.Me, self.Re, self.E51, kappa=self.kappa, n=self.n, delta = self.delta)[1]
		return L_evo     
	def T_evolution(self,time): 
		time = time.flatten()

		#R_evo = self.R_evolution(time)
		#L_evo = self.L_evolution(time) 
		#T = (L_evo/(4*np.pi*sigma_sb*R_evo**2))**0.25       
		T = Tph_shock_cooling_Piro2021(time, self.Me, self.Re, self.E51, kappa=self.kappa, n=self.n, delta = self.delta)
		return T
	def mags(self,time,filt_list): 
		m_array={}
		for filt in filt_list:
			R_evo = self.R_evolution(time)
			T_evo = self.T_evolution(time)
			mag = generate_cooling_mag(T_evo, R_evo, filt,d = self.distance, Rv=self.Rv, EBV = self.ebv,EBV_MW = 0, LAW = self.LAW)
			m_array[filt] = mag
		return m_array    
	def likelihood(self,dat,sys_err = 0.05,nparams = 5):
		t0 = self.t0 
		ebv = self.ebv
		Rv = self.Rv
		LAW = self.LAW
		d = self.distance
		filt_list = np.unique(dat['filter'])
		chi2_dic = {}
		dof = 0
		chi_tot = 0
		N_band = []
		for filt in filt_list:
			dat_filt = dat[dat['filter']==filt]
			time = dat_filt['t_rest']-t0
			R_evo = self.R_evolution(time)
			T_evo = self.T_evolution(time)
			mag = generate_cooling_mag(T_evo, R_evo, filt,d = d, Rv=Rv, EBV = ebv,EBV_MW = 0, LAW = LAW)
			
			mag_obs = dat_filt['absmag']
			mag_err = dat_filt['AB_MAG_ERR'] + sys_err
			
			c2 = chi_sq(mag_obs,mag_err, mag)
			N = len(c2)
			#N_band.append(N)

			chi2_dic[filt] = c2#*N**2


			dof = dof+N

			chi_tot = chi_tot +np.sum(chi2_dic[filt])
		#N_band = np.array(N_band) 
		chi_tot = chi_tot#/np.sum(N_band**2)
		dof = dof - nparams
		
		rchi2 = chi_tot/dof
			
		
		if chi_tot == 0:
			logprob = -np.inf
		elif dof <= 0:
			logprob = -np.inf
		else:
			#prob = scipy.stats.chi2.pdf(chi_tot, dof)
			#prob = float(prob)
			
			logprob = log_chi_sq_pdf(chi_tot, dof)


		#logprob = np.log(prob)
		return logprob, chi_tot, dof


@njit
def  L_shock_cooling_Piro2021(ts, Me, Re, E51, kappa=0.07, n=10, delta = 1.1):
	'''
	Luminosity of shock cooling emission from extended material  
	Piro et al. (2021) https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P/abstract
	
	INPUTS:
		Me: mass of the extended material in unit of solar mass
		Re: radius of the extended material in unit of solar mass
		E51: energy gained from the shock passing through the extended material in unit of 1e51 erg
		ts: time in days
		kappa: assume a constant opacity
		n and delta: the radial dependence of the outer and inner density; the default values are adopted in Piro et al. 2021 with reference given to Chevalier & Soker 1989
		
	The inputs are converted into cgs units before the calculation 
	'''
	c = 29979245800
	ts = ts * 24*3600
	E = E51 * 1e51    
	M = Me * 1.988409870698051e+33
	R = Re * 69570000000
	
  
	K = (n-3) * (3-delta) / (4 * np.pi * (n-delta)) # K = 0.119 for default n and delta
	vt = ( (n-5)*(5-delta) / ((n-3)*(3-delta)) )**0.5 * (2*E/M)**0.5 #the transition velocity between the outer and inner regions
	td = ( 3*kappa*K*M / ((n-1)*vt*c) )**0.5 # the time at which the diffusion reaches the depth where the velocity is vt   
	
	prefactor = np.pi*(n-1)/(3*(n-5)) * c*R*vt**2 / kappa 
	L1 = prefactor * (td/ts)**(4/(n-2))
	L2 = prefactor * np.exp(-0.5 * ((ts/td)**2 - 1)) 
	Ls = np.zeros(len(ts))
	ix1 = ts < td
	Ls[ix1] = L1[ix1]
	Ls[~ix1] = L2[~ix1]
	return td/3600/24, Ls


@njit
def Rph_shock_cooling_Piro2021(ts, Me,Re, E51, kappa=0.07, n=10, delta=1.1):
	'''
	phot
	'''
	c = 29979245800
	ts = ts * 24*3600
	E = E51 * 1e51    
	M = Me * 1.988409870698051e+33
	R = Re * 69570000000
	
	
	K = (n-3) * (3-delta) / (4 * np.pi * (n-delta)) # K = 0.119 for default n and delta
	
	vt = ( (n-5)*(5-delta) / ((n-3)*(3-delta)) )**0.5 * (2*E/M)**0.5 #the transition velocity between the outer and inner regions    
	tph = (3*kappa*K*M/(2*(n-1)*vt**2))**0.5
	
	rph1 = (tph/ts)**(2/(n-1))*vt*ts
	rph2 = ((delta-1)/(n-1)*(ts**2/tph*2-1)+1)**(-1/(delta-1))*vt*ts
	
	Rphs = np.zeros(len(ts))
	ix1 = ts < tph
	Rphs[ix1] = rph1[ix1]
	Rphs[~ix1] = rph2[~ix1]
	return tph/3600/24, Rphs


@njit
def Tph_shock_cooling_Piro2021(ts, Me,Re, E51, kappa=0.07, n=10, delta=1.1):
	'''
	phot
	'''
	_,Rph = Rph_shock_cooling_Piro2021(ts, Me,Re, E51, kappa=kappa, n=n, delta=delta)
	
	_,Lph = L_shock_cooling_Piro2021(ts, Me,Re, E51, kappa=kappa, n=n, delta=delta)
	Tph = (Lph/(4*np.pi*sigma_sb*Rph**2))**0.25
	return Tph


def likelihood_piro_cooling(data, Me,Re, E51,t0, kappa=0.07, n=10, delta=1.1,distance = 3.0856e19,ebv = 0,Rv = 3.1 , LAW = 'MW',sys_err = 0.05,nparams = 4):
	obj = model_piro_cooling(Me,Re, E51,t0, kappa=kappa, n=n, delta=delta, distance =distance, ebv = ebv, Rv = Rv, LAW = LAW)
	t1 = Rph_shock_cooling_Piro2021(np.array([1]), Me,Re, E51, kappa=kappa, n=n, delta=delta)[0]
	data = data[data['t_rest']<0.95*t1+t0]


	logprob, chi_tot, dof = obj.likelihood(data, sys_err = sys_err ,nparams = nparams)


	return logprob, chi_tot, dof 

   

def fit_piro_cooling(data, plot_corner = True,sys_err = 0.05, LAW = 'MW',**kwargs): 
	inputs={'priors':[np.array([0.1,10]),
					  np.array([100,2000]),
					  np.array([0.1,10]),
					  np.array([-1,0])]
			,'maxiter':100000
			,'maxcall':500000
			,'nlive':250}                            
	inputs.update(kwargs)
	priors=inputs.get('priors')  
	maxiter=inputs.get('maxiter')  
	maxcall=inputs.get('maxcall')  
	nlive=inputs.get('nlive')  
   
	def prior_transform(u,priors = priors):
		x=uniform_prior_transform(u,priors =priors )
		return x
	def myloglike(x):
		Me,Re, E51,t0 = x   
		loglike = likelihood_piro_cooling(data,Me,Re, E51,t0,n=10,ebv = 0, Rv = 3.1, sys_err = sys_err, LAW = LAW)[0]
		return loglike  
	dsampler = dynesty.DynamicNestedSampler(myloglike, prior_transform, ndim = 4,nlive=nlive,update_interval=600)
	dsampler.run_nested(maxiter=maxiter, maxcall=maxcall)
	dresults = dsampler.results
	if plot_corner:
		# Plot a summary of the run.
		#rfig, raxes = dyplot.runplot(dresults)
		# Plot traces and 1-D marginalized posteriors.
		#tfig, taxes = dyplot.traceplot(dresults)    
		cfig, caxes = dyplot.cornerplot(dresults,labels=['M_e','R_e',r'$E_{51}$',r'$t_{exp}$']
								,label_kwargs={'fontsize':14}, color = '#0042CF',show_titles = True)    
	## add labels!
	# Extract sampling results.
	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])  # normalized weights
	# Compute 10%-90% quantiles.
	quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
				 for samps in samples.T]
	# Compute weighted mean and covariance.
	mean, _ = dyfunc.mean_and_cov(samples, weights)
	
	return mean, quantiles,dresults

def plot_lc_piro(dat,params, c_band, lab_band, offset, kappa = 0.07 ,ebv = 0, Rv = 3.1, LAW = 'Mw' ):
	if len(params)==4:
		Me,Re, E51,t0 = params
		n=10
		delta=1.1 
	elif len(params)==5:
		Me,Re, E51,t0,n = params 
		delta=1.1 
	elif len(params)==6:
		Me,Re, E51,t0,n,delta = params 

	obj = model_piro_cooling(Me,Re, E51,t0, kappa=kappa, n=n, delta=delta, distance =3.0856e19, ebv = ebv, Rv = Rv, LAW = LAW)
	t1 = Rph_shock_cooling_Piro2021(np.array([1]), Me,Re, E51, kappa=kappa, n=n, delta=delta)[0]

	filt_list = np.unique(dat['filter'])

	time_2 = np.logspace(-2,np.log10(0.95*t1),30)
	mags =   obj.mags(time_2,filt_list=filt_list)
	

	cond_dic = {}
	for band in filt_list:
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)

	plt.figure(figsize=(6,15))
	for i,band in enumerate(filt_list):
		plt.plot(time_2,mags[band]-offset[band],color =c_band[band] ,alpha = 0.5,ls = '-.',linewidth = 3)

		plt.errorbar(dat['t_rest'][cond_dic[band]]-t0, dat['absmag'][cond_dic[band]]-offset[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10)
		if np.sign(-offset[band]) == 1:
			string = lab_band[band]+' +{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == -1:
			string = lab_band[band]+' -{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == 0:
			string = lab_band[band]
		if lab_band[band]!='':
			plt.text(-1.5,np.mean(dat['absmag'][cond_dic[band]]-offset[band]),string, color =c_band[band] ) 
	plt.xlim((-2,1.1*np.max(dat['t_rest'])))

	plt.gca().invert_yaxis()
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	plt.ylabel('M (AB mag) + offset',fontsize = 14)
	pass




def plot_resid_piro(dat,params, c_band, lab_band, offset, kappa = 0.07 ,ebv = 0, Rv = 3.1, LAW = 'Mw', sigma = False):
	if len(params)==4:
		Me,Re, E51,t0 = params
		n=10
		delta=1.1 
	elif len(params)==5:
		Me,Re, E51,t0,n = params 
		delta=1.1 
	elif len(params)==6:
		Me,Re, E51,t0,n,delta = params 

	obj = model_piro_cooling(Me,Re, E51,t0, kappa=kappa, n=n, delta=delta, distance =3.0856e19, ebv = ebv, Rv = Rv, LAW = LAW)
	t1 = Rph_shock_cooling_Piro2021(np.array([1]), Me,Re, E51, kappa=kappa, n=n, delta=delta)[0]

	filt_list = np.unique(dat['filter'])


	cond_dic = {}
	resid = {}

	# change to model_cooling_broken

	for i,band in enumerate(filt_list):

		obj = model_piro_cooling(Me,Re, E51,t0, kappa=kappa, n=n, delta=delta, distance =3.0856e19, ebv = ebv, Rv = Rv, LAW = LAW)
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)&(dat['t']<0.95*t1)

		t = dat['t_rest'][cond_dic[band]]-t0
		mag = dat['absmag'][cond_dic[band]]
		mags =   obj.mags(t,filt_list=[band])
		res = mag - mags[band]
		if sigma: 
			magerr = dat['AB_MAG_ERR'][cond_dic[band]]
			res = res/magerr
		resid[band] = res
	

	plt.figure(figsize=(15,6))
	for i,band in enumerate(filt_list):
		if ~sigma:     
			plt.errorbar(dat['t_rest'][cond_dic[band]]-t0, resid[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10, label = lab_band[band])
		else: 
			plt.plot(dat['t_rest'][cond_dic[band]]-t0, resid[band],marker = '*', ls = '',color = c_band[band], markersize = 10, label = lab_band[band])
	plt.plot([0.1,np.max(dat['t_rest'])],[0,0],'k--')
	#plt.xlim((-2,1.1*np.max(dat['t_rest'])))
	plt.xscale('log')
	plt.gca().invert_yaxis()
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	plt.ylabel('M - model (AB mag)')
	if sigma: 
		plt.ylabel('M - model ($\sigma$)')
	plt.legend()
	pass




def plot_bb_with_model(obj,Temps,t0, fig = 'create'):
	logmax = np.max(np.log10(1.2*Temps['t_rest']))
	logmin = np.min(np.log10(0.8*Temps['t_rest']))

	time = np.logspace(logmin,logmax,100)
	if hasattr(obj,'t_up'):
		time = np.logspace(np.log10(obj.t_down),np.log10(obj.t_up),100)
	#T_pre = obj.T_evolution(time)
	#R_pre = obj.R_evolution(time)
	#T = np.zeros_like(T_pre)
	#R = np.zeros_like(R_pre)
	#for i in range(len(T_pre)):
	#    t,r = extinct_bb_bias(T_pre[i],R_pre[i],obj.ebv,Rv = obj.Rv,rng = np.array([2000,0.8e4]),z=0,LAW=obj.LAW)
	#    T[i] = t
	#    R[i] = r   
	#  
	if fig == 'create':
		fig = plt.figure(figsize=(14,6))

	T =obj.T_evolution(time)
	R =obj.R_evolution(time)
	ax1 = plt.subplot(1,3,1)
	plt.plot(time,T)

	plt.errorbar(Temps['t_rest']-t0,Temps['T'],np.vstack([Temps['T'] -Temps['T_down'],Temps['T_up']-Temps['T']]),marker = 'o',ls = '', elinewidth=2.5, ecolor= (0.5,0.5,0.5,0.5) , color = 'r', label = r'Fit')
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('rest-frame days from estimated explosion')
	plt.ylabel('T [$^{\circ}K$]')
	plt.xlim(0.4-t0,1.1*np.max(Temps['t_rest']))
	plt.ylim((5000,40000))
	ax2 =  plt.subplot(1,3,2)
	#plt.plot(t,power2(t,1e14,1,3.2e14,0.7,0))
	plt.plot(time,R)

	plt.errorbar(Temps['t_rest']-t0,Temps['R'],np.vstack([Temps['R'] -Temps['R_down'],Temps['R_up']-Temps['R']]),marker = 'o',ls = '', elinewidth=2.5, ecolor= (0.5,0.5,0.5,0.5) , color = 'r', label = r'Fit')
	plt.xlabel('rest-frame days from estimated explosion')
	plt.ylabel('R [cm]')
	plt.yscale('log')
	plt.xscale('log')
	plt.xlim(0.4-t0,1.1*np.max(Temps['t_rest']))
	plt.ylim((1e14,3e15))

	ax3 = plt.subplot(1,3,3)

	#plt.plot(t,power2(t,1e14,1,3.2e14,0.7,0))
	t_mid = (Temps['t_rest'][0:-1] + Temps['t_rest'][1:])/2
	v = np.diff(Temps['R'])/np.diff(Temps['t_rest'])/1e5/3600/24
	rerr = np.mean(np.vstack([Temps['R'] -Temps['R_down'],Temps['R_up']-Temps['R']]),axis = 0)

	L = Temps['L_bol_w_extrap']
	Lerr =Temps['L_bol_w_extrap_err']

	verr = (rerr[0:-1] + rerr[1:])*2/np.diff(Temps['t_rest'])/1e5/3600/24
	plt.plot(time,obj.L_evolution(time))
	#plt.plot(time_2,+Lni)

	#plt.plot(t_mid,v,marker = 'o',ls = '' , color = 'y', label = r'SN2022oqm')
	plt.errorbar(Temps['t_rest']-t0,L,Lerr,marker = 'o',ls = '', elinewidth=2.5, ecolor= (0.5,0.5,0.5,0.5) , color = 'b', label = r'Fit')

	plt.xlabel('rest-frame days from estimated explosion')
	plt.ylabel('L [erg/s]')
	plt.yscale('log')
	plt.xscale('log')
	plt.xlim(0.4-t0,1.1*np.max(Temps['t_rest']))
	return fig, [ax1, ax2, ax3]    

def plot_lc_with_model(dat,obj,t0, c_band, lab_band, offset, fig = 'create', ax = None,xlab_pos = -1.5):

	time_2 = np.logspace(-2,np.log10(np.max(dat['t_rest'])),30)
	if hasattr(obj,'t_up'):
		t_down = max(obj.t_down,0.001)
		time_2 = np.logspace(np.log10(t_down),np.log10(obj.t_up),30)
	if fig == 'create':
		fig = plt.figure(figsize=(6,15))
		ax = plt.axes()
	filt_list = np.unique(dat['filter'])
	cond_dic = {}
	for band in filt_list:
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)
	mags =   obj.mags(time_2,filt_list=filt_list)

	for i,band in enumerate(filt_list):
		try:
			ax.plot(time_2,mags[band]-offset[band],color =c_band[band] ,alpha = 0.5,ls = '-.',linewidth = 3, label = '')
		except:
			import ipdb;ipdb.set_trace()

		ax.errorbar(dat['t_rest'][cond_dic[band]]-t0, dat['absmag'][cond_dic[band]]-offset[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10, label = '')
		if np.sign(-offset[band]) == 1:
			string = lab_band[band]+' +{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == -1:
			string = lab_band[band]+' -{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == 0:
			string = lab_band[band]
		if lab_band[band]!='':
			ax.text(xlab_pos,np.mean(mags[band]-offset[band]),string, color =c_band[band] ) 
			
	ax.set_xlim((-2,1.1*np.max(time_2)))

	ax.invert_yaxis()
	ax.set_xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	ax.set_ylabel('M (AB mag) + offset',fontsize = 14)
	return fig,ax


def plot_lc_2model(dat, obj1, obj2, c_band, lab_band, offset, fig = 'create', ax = None):

	time_2 = np.logspace(-2,np.log10(np.max(dat['t_rest'])),30)
	time_2p=  time_2

	if hasattr(obj1,'t_up'):
		time_2 = np.logspace(np.log10(obj1.t_down),np.log10(obj1.t_up),30)
	if hasattr(obj2,'t_up'):
		time_2p = np.logspace(np.log10(obj2.t_down),np.log10(obj2.t_up),30)

	if fig == 'create':
		fig = plt.figure(figsize=(6,15))
		ax = plt.axes()
	filt_list = np.unique(dat['filter'])
	cond_dic = {}
	for band in filt_list:
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)
	mags =   obj1.mags(time_2,filt_list=filt_list)
	mags2 =   obj2.mags(time_2p,filt_list=filt_list)

	for i,band in enumerate(filt_list):
		ax.plot(time_2+obj1.t0,mags[band]-offset[band],color =c_band[band] ,alpha = 0.5,ls = '-.',linewidth = 3, label = '')
		ax.plot(time_2p+obj2.t0,mags2[band]-offset[band],color =c_band[band] ,alpha = 0.5,ls = '-',linewidth = 3, label = '')
		import ipdb; ipdb.set_trace()
		ax.errorbar(dat['t_rest'][cond_dic[band]], dat['absmag'][cond_dic[band]]-offset[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10, label = '')
		if np.sign(-offset[band]) == 1:
			string = lab_band[band]+' +{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == -1:
			string = lab_band[band]+' -{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == 0:
			string = lab_band[band]
		if lab_band[band]!='':
			ax.text(-1.5,np.mean(dat['absmag'][cond_dic[band]]-offset[band]),string, color =c_band[band] ) 
			
	ax.set_xlim((-2,1.1*np.max(time_2)))

	ax.invert_yaxis()
	ax.set_xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	ax.set_ylabel('M (AB mag) + offset',fontsize = 14)
	return fig,ax


def plot_lc_model_only(obj,filt_list,tmin,tmax,t0, c_band, lab_band,offset,validity = False, fig = 'create', ax = None,**kwargs):
	if fig == 'create':
		fig = plt.figure(figsize=(6,15))
		ax = plt.axes()
		print('create figure')

	time_2 = np.logspace(np.log10(tmin+t0),np.log10(tmax+t0),30)

	if validity:
		time_2 = np.logspace(np.log10(obj.t_down),np.log10(obj.t_up),30)
	mags =   obj.mags(time_2,filt_list=filt_list)
	for i,band in enumerate(filt_list):
		ax.plot(time_2+t0,mags[band]-offset[band],color =c_band[band],label = lab_band[band],**kwargs)
		
	#ax.set_xlim((0.01,1.1*np.max(time_2)))
	if fig == 'create':
		plt.gca().invert_yaxis()
	plt.xlabel('Rest-frame days since estimated explosion',fontsize = 14)
	plt.ylabel('M (AB mag) + offset',fontsize = 14)
	#plt.xscale('log')
	plt.legend()
	return fig,ax






def convert_bb(T,L):
	#using Faran et al. 
	Te3 = T/1000
	T30  = 3.1*Te3**3 - 58*Te3**2 +1630*Te3 -1730
	T15 = 0.69*Te3**3 -0.0914*Te3**2 +1150*Te3 -424
	L30 = L*(T30/T)**3
	L15 = L*(T15/T)**3
	R30 = np.sqrt(L30/(4*np.pi*sigma_sb*(T30)**4))
	R15 = np.sqrt(L15/(4*np.pi*sigma_sb*(T15)**4))
	return T30,L30,R30,T15,L15,R15




def plot_lc(dat,t0, c_band, lab_band, offset, fig = 'create', ax = None,figsize = (6,15), lab_x_loc = -1.5,fontsize=16):
	if fig == 'create':
		fig = plt.figure(figsize=figsize)
		ax = plt.axes()
	filt_list = np.unique(dat['filter'])
	cond_dic = {}
	for band in filt_list:
		cond_dic[band] = (dat['filter']==band)&(dat['t']>0)
	for i,band in enumerate(filt_list):
		ax.errorbar(dat['t_rest'][cond_dic[band]]-t0, dat['absmag'][cond_dic[band]]-offset[band],dat['AB_MAG_ERR'][cond_dic[band]],marker = '*', ls = '',color = c_band[band], markersize = 10, label = '')
		if np.sign(-offset[band]) == 1:
			string = lab_band[band]+' +{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == -1:
			string = lab_band[band]+' -{0}'.format(-offset[band])
		elif np.sign(-offset[band]) == 0:
			string = lab_band[band]
		if lab_band[band]!='':
			ax.text(lab_x_loc,np.nanmean(dat['absmag'][cond_dic[band]&(dat['t_rest']<30)]-offset[band]),string, color =c_band[band], fontsize = fontsize ) 
	ax.invert_yaxis()
	ax.set_xlabel('Rest-frame days since estimated explosion',fontsize = fontsize)
	ax.set_ylabel('M (AB mag) + offset',fontsize = fontsize)
	return fig,ax

