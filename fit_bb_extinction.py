#packages
import numpy as np
from astropy import table
from astropy.io import ascii
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, broyden1,broyden2,newton_krylov
from astropy import constants 
import os
import sys
from extinction import ccm89,calzetti00, apply, remove 
from params import path_package
sys.path.insert(1,path_package+'/barak')
from extinction_barak import SMC_Gordon03,LMC_Gordon03
sys.path.insert(1,path_package)
from PhotoUtils import cosmo_dist
from numba import jit,njit
##parameters 
import argparse
# parameters 
Filter_path='/home/idoi/Dropbox/Utils/Filters'
parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', type=str, help='Path to Data')
parser.add_argument('--dates', type=str,default='', help='Path to Dates fofr BB fit')
parser.add_argument('--name', type=str, help='SN name')
parser.add_argument('--write', type=str, help='write? answer 0 or 1')
parser.add_argument('--out_path', type=str, help='out path')
parser.add_argument('--plots', type=str, help='plot lc and sed? answer 0 or 1')
parser.add_argument('--z',default=-1, type=float, help='redshift')
parser.add_argument('--d_mpc',default=0, type=float, help='distance in MPC')
parser.add_argument('--t0',default=-1, type=float, help='reference time')
parser.add_argument('--Ebv_MW',default=0, type=float, help='MW extinction E(B-V)')
parser.add_argument('--EBV_host',default=0, type=float, help='Host extinction E(B-V)')
parser.add_argument('--LAW',default='MW', type=str, help='Host extinction LAW: MW/SMC/LMC/Cal00')
parser.add_argument('--fit_ex',default=False, type=bool, help='Fit extinction? True/False')
parser.add_argument('--modify_BB_MSW',default=False, type=str, help='Use the modified blackbody formula described in Morag Sapir and Waxman 2023')

from matplotlib import gridspec
plt.rcParams.update({
  "text.usetex": True,
})

args = parser.parse_args()
#plots 

 

Data_path=args.data
Dates_path=args.dates
sn=args.name
write= args.write
out_path = args.out_path
plots = args.plots
z=args.z
t0=args.t0
Ebv_MW   =  args.Ebv_MW  
EBV_host =  args.EBV_host
d_mpc = args.d_mpc
LAW = args.LAW #SMC|LMC|MW|Cal00  ####EDIT ME#####py36
fit_extinction=args.fit_ex == True
modify_BB_MSW = args.modify_BB_MSW
if modify_BB_MSW not in ['True','False']:
	raise ValueError('the argument modify_BB_MSW must be a either True or False')
modify_BB_MSW = modify_BB_MSW=='True'
plots = plots=='True'
write = write=='True'


plot_lc=plots
plot_sed=plots
#Data_path ='/home/idoi/Dropbox/Objects/ZTF20ablygyy/ZTF20ablygyy_lc_format.ascii'
#Dates_path='/home/idoi/Dropbox/Objects/ZTF20ablygyy/ZTF20ablygyy_lc_dates.ascii'
meta_path = '/home/idoi/Dropbox/Objects/ZTF infant sample/metadata_310722.ascii'
sys.path.append('/home/idoi/Dropbox/Objects/ZTF infant sample/analysis') 


meta = ascii.read(meta_path)
meta['dm'] = 5*np.log10(meta['distance'])+25  
if z == -1:
	z = meta['redshift'][meta['name']==sn][0]


if t0 ==-1:
	first_time= meta['t0'][meta['name']==sn]
else: 
	first_time = t0

if d_mpc == 0:
	d_mpc=meta['distance'][meta['name']==sn][0]

cosmo_distance_cm=d_mpc*1e6*constants.pc.to('cm').value
d=cosmo_distance_cm

filter_out_interp = ['MMIRS_J','MMIRS_H','MMIRS_Ks']



#z=0.0641 #19hgp
#Ebv_MW=0.019#19hgp
#EBV_host = 0#19hgp
#first_time= 2458641.6 #19hgp
#
#z=0.084 #21csp
#Ebv_MW=0 #21csp
#EBV_host = 0 #21csp
#first_time= 2459255.0 #21csp


sys_err = 0.05 # 0.1198


# write output? 
#out_path='/home/idoi/Dropbox/Objects/2021csp/simple_blackbody_fits.ascii'

#include errors?
include_errors=True

## priors 
rad_high=1e16
rad_low=4e12
temp_high=1e5
temp_low=1000

# constants 
c=constants.c.value*1e10
c_AA=constants.c.value*1e10

h=constants.h.value*1e23
k_B=constants.k_B.value*1e23
sigma_sb=constants.sigma_sb.cgs.value

# load data
data = ascii.read(Data_path)
if 't_rest' not in data.colnames:
	data['t'] = data['jd'] -  first_time
	data['rest_time'] = data['t']/(1+z)
else: 
	data['t_rest'].name = 'rest_time'
data['fluxerr'] = np.sqrt(data['fluxerr']**2+(data['flux']*sys_err)**2)

#data['fluxerr']=np.sqrt(data['fluxerr']**2+0.01*data['flux']**2)  
if Dates_path!='':
	dates= np.loadtxt(Dates_path)
else: 
	uvot = data[data['instrument'] == 'Swift+UVOT']
	tt = np.unique(np.round(uvot['t']*2)/2)
	ts = np.zeros_like(tt)
	ts[0] = np.max(uvot['t'][np.round(uvot['t']*2)/2 == tt[0]])+0.01
	for i in range(len(tt)):
		ts[i] = np.mean(uvot['t'][np.round(uvot['t']*2)/2 == tt[i]])
	ts[-1] = np.min(uvot['t'][np.round(uvot['t']*2)/2 == tt[-1]])-0.01
	dates = np.array(ts) +first_time
#FIT EXTINCTION?

Rv = np.linspace(2,5,10)
if LAW == 'LMC':
	Rv = np.linspace(3.41-0.001,3.41+0.001,3)
elif LAW == 'SMC':
	Rv = np.linspace(2.74-0.001,2.74+0.001,3)
EBVec = np.logspace(-2.5,-0.25,50)



sys_shift = {
	 'r_sdss'  :0.0
	,'g_sdss'  :0.0
	,'i_sdss'  :0.0
	,'z_sdss'  :0.0
	,'u_sdss'  :0.0
	,'ZTF_r'   :0.0
	,'ZTF_g'   :0.0
	,'ZTF_i'   :0.0
	,'u_swift' :0.0
	,'v_swift' :0.0
	,'b_swift' :0.0
	,'UVM2'    :0.0
	,'UVW2'    :0.0
	,'UVW1'    :0.0
	,'u_P60'   :0.0
	,'g_P60'   :0.0
	,'r_P60'   :0.0
	,'i_P60'   :0.0
	,'LT_u'    :0.0
	,'LT_g'    :0.0
	,'LT_r'    :0.0
	,'LT_i'    :0.0
	,'LT_z'    :0.0   
	,'LT_u'    :0.0
	,'LCO_u'   :0.0
	,'LCO_g'   :0.0
	,'LCO_r'   :0.0
	,'LCO_i'   :0.0 
	,'KAIT_B'   :0.0
	,'KAIT_V'   :0.0
	,'KAIT_R'   :0.0
	,'KAIT_I'   :0.0
	,'MMIRS_J':0
	,'MMIRS_H':0
	,'MMIRS_Ks':0}

filters=['r_sdss' 
		,'g_sdss' 
		,'i_sdss' 
		,'z_sdss' 
		,'u_sdss' 
		,'ZTF_r'  
		,'ZTF_g'  
		,'ZTF_i'  
		,'u_swift'
		,'v_swift'
		,'b_swift'
		,'UVM2'   
		,'UVW2'   
		,'UVW1'
		,'u_P60'
		,'g_P60'
		,'r_P60'
		,'i_P60'
		,'LT_u'
		,'LT_g'
		,'LT_r'
		,'LT_i'
		,'LT_z'
		,'NOT_u'
		,'NOT_g'
		,'NOT_r'
		,'NOT_i'
		,'NOT_z'
		,'LCO_u'
		,'LCO_g'
		,'LCO_r'
		,'LCO_i'
		,'LCO_V'
		,'LCO_B'
		,'KAIT_B'
		,'KAIT_V'
		,'KAIT_R'
		,'KAIT_I'
		,'KAIT_CLEAR'
		,'Ni_B'
		,'Ni_V'
		,'Ni_R'
		,'Ni_I'
		,'Ni_CLEAR'
		,'MMIRS_J'
		,'MMIRS_H'
		,'MMIRS_Ks']

markers={
	'r_sdss'   :'o'
	,'g_sdss'  :'o'
	,'i_sdss'  :'o'
	,'z_sdss'  :'o'
	,'u_sdss'  :'o'
	,'ZTF_r'   :'s' 
	,'ZTF_g'   :'s' 
	,'ZTF_i'   :'s' 
	,'u_swift' :'p' 
	,'v_swift' :'p' 
	,'b_swift' :'p' 
	,'UVM2'    :'p'
	,'UVW2'    :'p'
	,'UVW1'    :'p' 
	,'u_P60'   :'.'
	,'g_P60'   :'.'
	,'r_P60'   :'.'
	,'i_P60'   :'.'
	,'LT_u'    :'^'
	,'LT_g'    :'^'
	,'LT_r'    :'^'
	,'LT_i'    :'^'
	,'LT_z'    :'^'
	,'NOT_u'    :'P'
	,'NOT_g'    :'P'
	,'NOT_r'    :'P'
	,'NOT_i'    :'P'
	,'NOT_z'    :'P'
	,'LCO_u'   :'v'
	,'LCO_g'   :'v'
	,'LCO_r'   :'v'
	,'LCO_i'   :'v'
	,'LCO_V'   :'v'
	,'LCO_B'   :'v'
	,'KAIT_B':'P'
	,'KAIT_V':'P'
	,'KAIT_R':'P'
	,'KAIT_I':'P'
	,'KAIT_CLEAR':'P'
	,'Ni_B':'P'
	,'Ni_V':'P'
	,'Ni_R':'P'
	,'Ni_I':'P'
	,'Ni_CLEAR':'P'
	,'MMIRS_J':'*'
	,'MMIRS_H':'*'
	,'MMIRS_Ks':'*'}


colors={
	'r_sdss'   :'#EA0000'
	,'g_sdss'  :'#00BA41'
	,'i_sdss'  :'#D3DE00'
	,'z_sdss'  :'#7E0600'
	,'u_sdss'  :'m#6D00C2'
	,'ZTF_r'   :'#EA0000' 
	,'ZTF_g'   :'#00BA41' 
	,'ZTF_i'   :'#D3DE00' 
	,'u_swift' :'#6D00C2'
	,'v_swift' :'#00DCA7' 
	,'b_swift' :'#1300FF' 
	,'UVW2'    :'#060606'
	,'UVM2'    :'#FF37DE'   
	,'UVW1'    :'#AE0DBB'
	,'u_P60'   :'#6D00C2'
	,'g_P60'   :'#00BA41'
	,'r_P60'   :'#EA0000'
	,'i_P60'   :'#D3DE00'
	,'LT_u'    :'#6D00C2'
	,'LT_g'    :'#00BA41'
	,'LT_r'    :'#EA0000'
	,'LT_i'    :'#D3DE00'
	,'LT_z'    :'#7E0600'
	,'NOT_u'    :'#6D00C2'
	,'NOT_g'    :'#00BA41'
	,'NOT_r'    :'#EA0000'
	,'NOT_i'    :'#D3DE00'
	,'NOT_z'    :'#7E0600'
	,'LCO_u'   :'#6D00C2'
	,'LCO_g'   :'#00BA41'
	,'LCO_r'   :'#EA0000'
	,'LCO_i'   :'#D3DE00'
	,'LCO_V'   :'#00DCA7'
	,'LCO_B'   :'#1300FF'
	,'KAIT_B':'#1300FF'
	,'KAIT_V':'#00DCA7'
	,'KAIT_R':'#EA0000'
	,'KAIT_I':'#D3DE00'
	,'KAIT_CLEAR':'#B2607E'
	,'Ni_B':'#1300FF'
	,'Ni_V':'#00DCA7'
	,'Ni_R':'#EA0000'
	,'Ni_I':'#D3DE00'
	,'Ni_CLEAR':'#B2607E'
	,'MMIRS_J':'#755300'
	,'MMIRS_H':'#6C7500'
	,'MMIRS_Ks':'#6D3D39'}        

	
excluded_bands = []#['r_sdss' 
				 #,'g_sdss'
				 #,'i_sdss'
				 #,'z_sdss'
				 #,'u_sdss'
				 #,'ztf_r' 
				 #,'ztf_g' 
				 #,'ztf_i' ]



for filt in excluded_bands:
	idx = np.argwhere(data['filter']==filt).flatten()
	data.remove_rows(idx)


def interp_data(time,data,datcol='flux', out = False):
		if out == False:
			data_interp=np.interp(time,data['rest_time'],data[datcol], left = 99, right = 99)
		else: 
			data_interp=np.interp(time,data['rest_time'],data[datcol], left = 99, right = 99)
			if (np.min((data['rest_time'] - time))<2)&(len(data['rest_time'])==1):
				cond_min = (data['rest_time'] - time)==np.min((data['rest_time'] - time)) 
				data_interp[cond_min] = np.interp(time[cond_min],data['rest_time'],data[datcol])
		return data_interp


def plot_error_photo(filter,color,marker):
		filter_photometry=data[data['filter']==filter]
		plt.errorbar(filter_photometry['rest_time'],filter_photometry['flux'],filter_photometry['fluxerr'],marker=marker,color=color,linestyle = '')


def bb_F(lam,T,r,EBV=0,EBV_mw=Ebv_MW, R_v=3.1,z=0,LAW=LAW):
	A_v=R_v*EBV
	A_v_mw=3.1*Ebv_MW
	L_bb = 4*np.pi*r**2*sigma_sb*(T)**4
	#flux=1.191/(lam**(5))/(np.exp((1.9864/1.38064852)/(lam*T))-1)*(np.pi*(r)**2) 
	B = 1.191/(lam**(5))/(np.exp((1.9864/1.38064852)/(lam*T))-1)
	L = 4*np.pi**2*r**2*B
	flux = L/4/np.pi
	#flux=apply(ccm89(lam*1e4, A_v, R_v), flux)
	#flux=apply(ccm89(lam*1e4*(1+z), A_v_mw, 3.1), flux)
	flux=apply_extinction(1e4*lam,flux,EBV=EBV,EBV_mw=EBV_mw,R_v=R_v,z=z,LAW=LAW)
	return flux  



eV2K =1/constants.k_B.to('eV / K ')
eV2K = eV2K.value

eVs = constants.h.to('eV s').value
c_cgs = constants.c.cgs.value
h_cgs = constants.h.cgs.value

@njit
def B_nu(nu_eV,T_eV):
	nu_hz = nu_eV/eVs
	B_nu = (2*h_cgs*(nu_hz)**3/c_cgs**2)*(np.exp(nu_eV/T_eV)-1)**(-1)
	return B_nu
@njit
def L_nu_bb(nu_eV,T_eV,L_bb):
	B_n = B_nu(nu_eV,T_eV)
	L_nu = B_n*(np.pi)*L_bb/(sigma_sb*(T_eV*eV2K)**4)
	return L_nu


#@njit
def Lnu_reduced(nu_eV,T_eV,L_bb):
	x = nu_eV/T_eV
	T_eV5 = T_eV/5
	k_col_nu = 0.0055*(x)**-1.664*(T_eV5)**(-1.0996)

	#k_es = k34*0.34
	eps_a = 0.0055*x**(-1.664)*(T_eV5)**(-1.0996)
	#eps_a = k_col_nu/(k_es)
	T_col_nu = 1.63*(x**0.247)*(T_eV)
	#rr_nu = ((x**(-0.0775))*(T_eV5)**(-0.05))
	eps_term = (np.sqrt(eps_a)/(1+np.sqrt(eps_a)))
	rr_nu = ((x**(-0.155))*(T_eV5)**(-0.1))
	#L_nu3= (np.pi)*B_nu(nu_eV,T_col_nu  )/(sigma_b*(T_eV*eV2K)**4     )*(8/np.sqrt(3))*((T_col5)**(-0.1))*(x**(-0.155))*eps_term
	L_nu1 = B_nu(nu_eV,0.85*T_eV)/((0.85)**4)
	L_nu2 = B_nu(nu_eV,0.74*T_eV)/((0.74)**4)
	#L_nu3 = (8/np.sqrt(3))*rr_nu*L_bb*eps_term*(np.pi)*B_nu(nu_eV,T_col_nu)/(sigma_b*(T_eV*eV2K)**4)
	L_nu3 = (8/np.sqrt(3)) * rr_nu * eps_term * B_nu(nu_eV,T_col_nu) 
	
	argmin = (L_nu1**(-5)+L_nu2**(-5)+L_nu3**(-5))**(-0.2)
	L_nu  = ((np.pi/sigma_sb)*L_bb/((T_eV*eV2K)**4))*argmin  #in formula min but using smoothed version from yoni 
	
	return L_nu

def f_nu2f_lam(f_nu,lam_AA):
    f_lam_AA=(c_AA/lam_AA**2)*f_nu
    return f_lam_AA

def Llam_reduced(lam_um,T_k,r_bb,EBV = 0, EBV_mw = 0,R_v = 3.1, LAW = 'MW',z=0, d =cosmo_distance_cm):
	lam_AA = 1e4*lam_um
	A_v=R_v*EBV
	A_v_mw=3.1*Ebv_MW
	
	L_bb = 4*np.pi*r_bb**2*sigma_sb*T_k**4
	T_eV = T_k/eV2K
	lam_cm = (lam_AA/1e8)
	nu_hz = c_cgs/lam_cm
	nu_eV = nu_hz*eVs
	L_nu = Lnu_reduced(nu_eV,T_eV,L_bb)
	#L_nu = L_nu_bb(nu_eV,T_eV,L_bb)

	f_nu = L_nu/(4*np.pi*d**2)
	f_lam = f_nu2f_lam(f_nu,lam_AA)
	f_lam=apply_extinction(lam_AA,f_lam,EBV=EBV,EBV_mw=EBV_mw,R_v=R_v,z=z,LAW=LAW)
	return f_lam
	

def bb_F_reduced(lam_um,T_k,r_bb,EBV = 0, EBV_mw = 0,R_v = 3.1, LAW = 'MW',z=0,d = cosmo_distance_cm):
	T_k = 1e4*T_k 
	r_bb = r_bb*d*1e-10
	lam_AA = 1e4*lam_um
	A_v=R_v*EBV
	A_v_mw=3.1*Ebv_MW
	
	L_bb = 4*np.pi*r_bb**2*sigma_sb*T_k**4
	T_eV = T_k/eV2K
	lam_cm = (lam_AA/1e8)
	nu_hz = c_cgs/lam_cm
	nu_eV = nu_hz*eVs
	L_nu = Lnu_reduced(nu_eV,T_eV,L_bb)
	#L_nu = L_nu_bb(nu_eV,T_eV,L_bb)

	f_nu = L_nu/(4*np.pi*d**2)
	f_lam = f_nu2f_lam(f_nu,lam_AA)
	f_lam=apply_extinction(lam_AA,f_lam,EBV=EBV,EBV_mw=EBV_mw,R_v=R_v,z=z,LAW=LAW)
	f_lam = f_lam*1e13
	return f_lam
	
#
#
#lam_um = np.linspace(0.1,10,1000)
#T = 2e4
#R= 1e13
#d = 3.08e26/3
#t=1e-4*T
#r=1e10*R/d
#f1 = 1e-13*bb_F(lam_um,t,r)#
##vega_spec = ascii.read('/home/idoi/Dropbox/Utils/alpha_lyr_stis_004.ascii')
##solar_spec = ascii.read('/home/idoi/Dropbox/Utils/solar_spec.dat')#
#f2p = Llam_reduced(lam_um,T,R,d =d)#
#f2 = 1e-13*bb_F_reduced(lam_um,t,r)
#plt.figure()
#plt.plot(lam_um,f1,label = 'solar BB')
#plt.plot(lam_um,f2p,'k',label = 'solar LT formula BB')
#plt.plot(lam_um,f2,'r--',label = 'solar LT formula BB')
#
##plt.plot(solar_spec['col1']/1000,solar_spec['col2']*100,'b--',label = 'solar spectrum')#
#plt.xscale('log')
#plt.yscale('log')
#plt.legend(fontsize = 14)
#plt.xlabel('$\lambda\ [\mu m]$',fontsize = 14)
#plt.ylabel('$f_{\lambda}\ [{erg\ cm^{-2}\ s^{-1}\ \AA^{-1}}]$',fontsize = 14)#
#plt.show(block = False)#



def apply_extinction(lam,flam,EBV=0,EBV_mw=0, R_v=3.1,z=0,LAW=LAW):
	if EBV>0:
		if LAW == 'SMC':
			ex = SMC_Gordon03(lam*(1+z))
			ex.set_EBmV(EBV)
			A_v = ex.Av
			Alam  = A_v*ex.AlamAv 
			flux = flam*10**(-0.4*Alam)
		elif LAW == 'LMC':
			ex = LMC_Gordon03(lam*(1+z))
			ex.set_EBmV(EBV)
			A_v = ex.Av
			Alam  = A_v*ex.AlamAv 
			flux = flam*10**(-0.4*Alam)
		elif LAW == 'Cal00':
			A_v=R_v*EBV
			flux=apply(calzetti00(lam, A_v, R_v,unit='aa'), flam)
			
		else:
			A_v=R_v*EBV
			flux=apply(ccm89(lam, A_v, R_v,unit='aa'), flam)
			
	else:
		flux = flam
	if Ebv_MW>0:
		A_v_mw=3.1*Ebv_MW
		flux=apply(ccm89(lam*(1+z), A_v_mw, 3.1,unit='aa'), flux)
	return flux  


def remove_extinction(lam,flam,EBV=0,EBV_mw=0, R_v=3.1,z=0,LAW=LAW):
	if EBV>0:
		if LAW == 'SMC':
			ex = SMC_Gordon03(lam*(1+z))
			ex.set_EBmV(EBV)
			A_v = ex.Av
			Alam  = A_v*ex.AlamAv 
			flux = flam*10**(0.4*Alam)
		elif LAW == 'LMC':
			ex = LMC_Gordon03(lam*(1+z))
			ex.set_EBmV(EBV)
			A_v = ex.Av
			Alam  = A_v*ex.AlamAv 
			flux = flam*10**(0.4*Alam)
		elif LAW == 'Cal00':
			A_v=R_v*EBV
			flux=remove(calzetti00(lam, A_v, R_v,unit='aa'), flam)
			
		else:
			A_v=R_v*EBV
			flux=remove(ccm89(lam, A_v, R_v,unit='aa'), flam)
			
	else:
		flux = flam
	if Ebv_MW>0:
		A_v_mw=3.1*Ebv_MW
		flux=remove(ccm89(lam*(1+z), A_v_mw, 3.1,unit='aa'), flux)
	return flux  

def bb_fit(lam,f_lam,lims,f_lam_err=None,include_errors=True,EBV=0,Ebv_MW=Ebv_MW,R_v=3.1,z=0):
	func= lambda lam,T,r: bb_func(lam,T,r,EBV=EBV,EBV_mw=Ebv_MW, R_v=R_v, z=z)

	if include_errors:  
		popt, pcov = curve_fit(func, lam, f_lam, sigma=f_lam_err,bounds=lims,method='trf')

	else:
		popt, pcov = curve_fit(func, lam, f_lam,bounds=lims,method='trf')    

	perr = np.sqrt(np.diag(pcov))
	

	return popt, perr,pcov




def chi2(lam,f_lam,f_err,T,R,EBV=0, Ebv_MW=Ebv_MW,R_v=3.1,sys_err=0,z=0):
	t=1e-4*T
	r=1e10*R/d
	f_model=bb_func(lam,t,r,EBV=EBV,EBV_mw=Ebv_MW,R_v=R_v,z=z)
	CHI2=(f_model-f_lam)**2/(f_err**2+(f_lam*sys_err)**2)
	return CHI2

def sys_err_fit(lam,f_lam,f_err,T,R,EBV=Ebv_MW,R_v=3.1):
	def func(s):
		C=chi2(lam=lam,f_lam=f_lam,f_err=f_err,T=T,R=R,EBV=EBV,R_v=R_v,sys_err=s)
		C=np.sum(C)/len(C)-1
		return C
	sys_err=newton_krylov(func,xin=10,f_tol=1e-2)
	return sys_err



def convert_bb(T,L):
	Te3 = T/1000
	T30  = 3.1*Te3**3 - 58*Te3**2 +1630*Te3 -1730
	T15 = 0.69*Te3**3 -0.0914*Te3**2 +1150*Te3 -424
	L30 = L*(T30/T)**3
	L15 = L*(T15/T)**3
	R30 = np.sqrt(L30/(4*np.pi*sigma_sb*(T30)**4))
	R15 = np.sqrt(L15/(4*np.pi*sigma_sb*(T15)**4))
	return T30,L30,R30,T15,L15,R15

for filt in sys_shift.keys():
	cond = data['filter']==filter 
	data['flux'][cond]= data[cond]['flux']*(1 + sys_shift[filt])

#data=data[data['piv_wl']<4300]

if modify_BB_MSW:
	bb_func = bb_F_reduced
elif not modify_BB_MSW:
	bb_func = bb_F
else: 
	import ipdb; ipdb.set_trace()
	raise ValueError('the argument modify_BB_MSW must be a either True or False')

data.sort('rest_time')
#data['rest_time']=(data['jd']-first_time)/(1+z)


data_dic={}
no_data_filters=[]

for filt in np.unique(data['filter']):
		data_dic[filt]=data[data['filter']==filt]
		if len(data_dic[filt])==0:
				no_data_filters.append(filt)

del filt 
filters=np.unique(data['filter'])
time=(dates-first_time)/(1+z)

if plot_lc==1:
	plt.figure()
	for filter in np.unique(data['filter']):
		plot_error_photo(filter,colors[filter],markers[filter])   
	for date in time:     
		plt.plot([date,date],[0.8*np.min(data['flux']),1.2*np.max(data['flux'])],'k--', alpha = 0.5) 
	plt.yscale('log')
	plt.xlabel('JD')
	plt.ylabel('$f_lambda(cgs/\AA)$')
	plt.title(sn)
	plt.xlim((0,time[-1]*1.1))
	plt.show(block=False)

data_dic_interp={}
data_dic_interp_err={}

for filter in np.unique(data['filter']):

		data_dic_interp[filter]=interp_data(time,data_dic[filter],datcol='flux', out = (filter in filter_out_interp))
		data_dic_interp_err[filter]=interp_data(time,data_dic[filter],datcol='fluxerr', out = (filter in filter_out_interp))
		data_dic_interp_err[filter][data_dic_interp_err[filter] == 0] = 0.01
		if (filter not in filter_out_interp):
			idx=(time>min(data_dic[filter]['rest_time']))&(time<max(data_dic[filter]['rest_time']))
		else:
			idx=(time>min(data_dic[filter]['rest_time'])-2)&(time<max(data_dic[filter]['rest_time'])+2)
		idx=~idx
		data_dic_interp[filter][idx]=99
		data_dic_interp_err[filter][idx]=99




data_full=np.stack([data_dic_interp[filter] for filter in filters])
data_full_e=np.stack([data_dic_interp_err[filter] for filter in filters])

cond_any = [(data_full[:,i]!=99).any() for i in range(np.shape(data_full)[1])] 

data_full = data_full[:,cond_any]
data_full_e = data_full_e[:,cond_any]
time = time[cond_any]
dates = dates[cond_any]
piv_table=table.unique(data['piv_wl','filter'])
piv_dict={}
  

piv_wl=np.zeros(np.size(filters))

for i,filter in enumerate(filters):
	   piv_dict[filter]=np.mean(piv_table['piv_wl'][piv_table['filter']==filter])
	   piv_wl[i]=piv_dict[filter] 
piv_dict2 = {np.round(piv_dict[key],3):key for key in piv_dict.keys()}                
#
#piv_wl=np.array([6174.31385020061,4701.884477199233,7489.168218070939,8909.394468671806,3555.672974130721,
#        6421.166992940606,4789.313527877863,7954.929623600654,
#        3467.0320116445123,5425.378043562354,4349.641314341636,
#        2246.179150541018,2055.1544609708008, 2580.3910534803363])
#
#



lims=np.array([[temp_low*1e-4,1e10*rad_low/d],[temp_high*1e-4,1e10*rad_high/d]])


def fit_bb_sequence(time,data_full,EBV=0,Ebv_MW=Ebv_MW,R_v=3.1,plot_sed=0,z=0):
	T_bb=np.zeros_like(time)        
	r_bb=np.zeros_like(time)
	T_bb_e=np.zeros_like(time)
	r_bb_e=np.zeros_like(time)
	cov = []
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
			popt, perr,pcov=bb_fit(lam,f_lam,lims,f_lam_err=f_err,include_errors=include_errors,EBV=EBV,Ebv_MW=Ebv_MW,R_v=R_v,z=z)
		except:
			import ipdb; ipdb.set_trace()
		T_bb[i]=popt[0]*1e4
		r_bb[i]=popt[1]*d*1e-10
		T_bb_e[i]=perr[0]*1e4
		r_bb_e[i]=perr[1]*d*1e-10
		cov.append(pcov) 
		c2=chi2(lam,f_lam,f_err,T_bb[i],r_bb[i],EBV=EBV,Ebv_MW=Ebv_MW,R_v=R_v,z=z)
		chi2_vec[i]=np.sum(c2)
		dof_vec[i] = len(c2) - 2
		if np.inf in (r_bb_e[0],T_bb_e[0]):
				print("ATTENTION: there are infinite values for the error bars. The covariance matrix could not be estimated, and the fit is unreliable for these points.")

	
	if plot_sed==1:
		NN = len(time)
		if NN>=10:
			cols = 4
		elif (NN>=5)&(NN<=10):
			cols = 3
		else: 
			cols = 2
		import math
		rows = math.ceil(NN/cols) 

		aspect = cols/rows
		fig = plt.figure(figsize=(14,14//aspect))
		spec = gridspec.GridSpec(ncols=cols, nrows=rows, wspace= 0.35, hspace = 0.35)
		from matplotlib.axes import Axes
		#color_code_lam
		for i in range(len(time)):
			lam=piv_wl[data_full[:,i]!=99]*1e-4
			lam=lam/(1+z)
			f_lam=(1+z)*data_full[data_full[:,i]!=99,i]*1e13
			f_err=(1+z)*data_full_e[data_full_e[:,i]!=99,i]*1e13

			#import ipdb; ipdb.set_trace()
			ax = fig.add_subplot(spec[i])
			plt.title(r"t = {0: .2f} d".format(time[i]))
			for j in range(len(lam)):
				#import ipdb; ipdb.set_trace()
				fil_name = piv_dict2[np.round(1e4*lam[j]*(1+z),3)]
				c = colors[fil_name]
				#color_code_lam(1e4*lam[j]*(1+z))
				plt.errorbar(lam[j]*1e4,f_lam[j]*1e-13,f_err[j]*1e-13,marker='o',color=c,linestyle="None")
			l=np.linspace(0.8*min(lam),1.2*max(lam),1000)*1e4
			plt.plot(l,bb_func(l*1e-4,T_bb[i]*1e-4,r_bb[i]/(d*1e-10),EBV=EBV,EBV_mw=Ebv_MW,R_v=R_v,z=z)*1e-13)
			plt.ylim((0.5*np.min((1+z)*data_full[data_full!=99]),1.1*np.max((1+z)*data_full[data_full!=99])))

			#labels[1] = r'{0:.0f}'.format(locs[1]/10**np.round(np.log10(np.min(locs[locs!=0]))))
			#labels[-1] = r'{0:.0f}'.format(locs[-1]/10**np.round(np.log10(np.min(locs[locs!=0]))))
			plt.xlim((1000,1e4))
			#if np.max(f_lam*1e-13)/np.min(f_lam*1e-13)>7.5:
			#else: 
			#    plt.ylim((np.mean(f_lam*1e-13)/3.5,np.mean(f_lam*1e-13)*3.5))
			plt.xscale('log')
			plt.yscale('log')
			plt.tick_params(axis='both', which='major', labelsize=12)

		fig.text(0.5, 0.02, r'Rest wavelength $[\AA]$', fontsize=14)
		fig.text(0.02, 0.6, r'Flux [erg s$^{-1}$ cm$^{-2}$ $\rm \AA^{-1}$]', fontsize=14,rotation = 90)

		plt.tight_layout()
	Chi2=np.sum(chi2_vec)
	return T_bb,T_bb_e,r_bb, r_bb_e, Chi2, dof, chi2_vec,dof_vec,cov




if fit_extinction:
	CHI2_grid=np.zeros((len(Rv),len(EBVec)))
	for i in range(len(Rv)):
		for j in range(len(EBVec)):
			T_bb,T_bb_e,r_bb, r_bb_e, c2, dof,_,_,cov = fit_bb_sequence(time,data_full,EBV=EBVec[j],Ebv_MW=Ebv_MW,R_v=Rv[i],plot_sed=0,z=z)
		 
			CHI2_grid[i,j]=c2
	idx = np.argwhere(CHI2_grid==np.min(CHI2_grid))[0]
	Rv_fit=Rv[idx[0]]
	EBV_fit=EBVec[idx[1]]

	dof = np.sum(data_full>0)-4
	CHI2dof_grid = CHI2_grid/dof
	prob_grid = np.exp(-0.5*CHI2_grid)
	prob_grid = prob_grid/np.sum(prob_grid)

	#search threshold:
	S =[]
	thresh_grid = np.linspace(0,np.max(prob_grid),500)
	for thresh in thresh_grid:
		s = np.sum(prob_grid[prob_grid>thresh])
		S.append(s)
	S = np.array(S)
	#lev = np.linspace(1,0,20)
	lev = np.array([1,0.99, 0.95,0.68,0])
	indices  = [np.argwhere(np.abs(S-x) == np.abs(S-x).min())[0][0] for x in lev]
	thresh = thresh_grid[indices]
	thresh  = np.append(thresh,1.01*np.max(thresh))  
	idx_1sig = np.argwhere(np.abs(S-0.68) == np.abs(S-0.68).min())[0][0]   
	idx_2sig = np.argwhere(np.abs(S-0.95) == np.abs(S-0.95).min())[0][0]    
	idx_3sig = np.argwhere(np.abs(S-0.997) == np.abs(S-0.997).min())[0][0]    

	thresh_1sig = thresh_grid[idx_1sig]
	thresh_2sig = thresh_grid[idx_2sig]
	thresh_3sig = thresh_grid[idx_3sig]




	from matplotlib import cm
	norm = cm.colors.Normalize(vmax=(CHI2dof_grid.max()-CHI2dof_grid.min())*0.1+CHI2dof_grid.min(), vmin=CHI2dof_grid.min())
	Rvv,log_Ebvv  = np.meshgrid(Rv,np.log10(EBVec)) 



	fig = plt.figure(figsize=(7,8))
	ax = fig.add_subplot(111)
	plt.contourf(Rvv,10**(log_Ebvv),CHI2dof_grid.transpose(),150,norm=norm)
	plt.colorbar(orientation='vertical')

	plt.contour(Rvv,10**(log_Ebvv),prob_grid.transpose(),levels  = [thresh_3sig,thresh_2sig,thresh_1sig,1],colors = '#FAFAFA' ,linewidths = 2,linestyles = 'dashed')

	#plt.text(2.7392,0.105,'$1\ \sigma$',color ='#00047C',fontsize = 14)
	#plt.text(2.7392,0.0144,'$2\ \sigma$',color = '#00047C',fontsize = 14)
	#plt.text(2.7392,0.1,'$3\ \sigma$',color = '#00047C',fontsize = 14)
	plt.text(2.35,0.047,'$1\ \sigma$',color = '#FAFAFA',fontsize = 14,rotation = 40)

	plt.text(2.2,0.06,'$2\ \sigma$',color = '#FAFAFA',fontsize = 14,rotation = 30)
	plt.text(2.1,0.075,'$3\ \sigma$',color = '#FAFAFA',fontsize = 14,rotation = 30)

	ax.set_title(r'$\chi^{2}/d.o.f.$',fontsize = 16)
	#ax.set_title(r'$\frac{\chi^{2}}{d.o.f.}$')

	ax.set_ylabel(r'$E(B-V)$',fontsize = 16)
	ylims = ax.get_ylim()
	xlims = ax.get_xlim()
	#plt.text(1.2*xlims[0],2*ylims[0],'JD-t0 = 1.526 days',fontsize = 18)
	#plt.text(1.2*xlims[0],0.9*ylims[1],'(d)',fontsize = 16)
	plt.setp(ax.get_xticklabels(), fontsize=14)
	plt.setp(ax.get_yticklabels(), fontsize=14)

	#plt.yscale('log')
	#cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
	#cax.get_xaxis().set_visible(False)
	#cax.get_yaxis().set_visible(False)
	#cax.patch.set_alpha(0)
	#cax.set_frame_on(False)
	if LAW == 'LMC':
		plt.plot(Rv,EBVec[idx[1]]*np.ones_like(Rv),ls ='--',markersize=10,color='#AD0A0A')
		plt.plot(3.41,EBVec[idx[1]],'*',markersize=10,color='r')

		plt.xticks([3.41])    
	elif LAW=='SMC':
		plt.plot(Rv,EBVec[idx[1]]*np.ones_like(Rv),ls ='--',markersize=10,color='#AD0A0A')
		plt.plot(2.74,EBVec[idx[1]],'*',markersize=10,color='r')

		plt.xticks([2.74])
	else:     
		plt.plot(Rv[idx[0]],EBVec[idx[1]],'*',markersize=10,color='r')

		ax.set_xlabel(r'$R_{v}$',fontsize = 16)
		ax.tick_params(direction='inout',which='both', length=4, width=1.8)
	ax.xaxis.grid(True, which='minor')
	ax.yaxis.grid(True, which='minor')
	#plt.show(block=False)


	#fig = plt.figure(figsize=(7,8))
	#ax = fig.add_subplot(111)
	#plt.contourf(Rvv,10**(log_Ebvv),prob_grid.transpose(),levels  = thresh)
	#plt.colorbar(orientation='vertical')#
	#plt.contour(Rvv,10**(log_Ebvv),prob_grid.transpose(),levels  = [thresh_2sig,thresh_1sig,1],colors = ['k','k','k'],linewidths = [1.5,1.5,1.5],linestyles = 'dashed')#
	##sns.kdeplot(Rvv,10**(log_Ebvv),weights=prob_grid.transpose(),cmap='Blues', shade=True, shade_lowest=False)#
	#ax.set_title(r'$P(M|D)$',fontsize = 16)
	#ax.set_ylabel(r'$E(B-V)$',fontsize = 16)
	#ylims = ax.get_ylim()
	#xlims = ax.get_xlim()#
	#plt.setp(ax.get_xticklabels(), fontsize=14)
	#plt.setp(ax.get_yticklabels(), fontsize=14)#
	#plt.yscale('log')#
	#if LAW == 'LMC':
	#    plt.plot(Rv,EBVec[idx[1]]*np.ones_like(Rv),ls ='--',markersize=10,color='#AD0A0A')
	#    plt.plot(3.41,EBVec[idx[1]],'*',markersize=10,color='r')#
	#    plt.xticks([3.41])    
	#elif LAW=='SMC'
	#    plt.plot(Rv,EBVec[idx[1]]*np.ones_like(Rv),ls ='--',markersize=10,color='#AD0A0A')
	#    plt.plot(2.74,EBVec[idx[1]],'*',markersize=10,color='r')#
	#    plt.xticks([2.74])
	#else:     
	#    plt.plot(Rv[idx[0]],EBVec[idx[1]],'*',markersize=10,color='r')#
	#    ax.set_xlabel(r'$R_{v}$',fontsize = 16)
	#    ax.tick_params(direction='inout',which='both', length=4, width=1.8)#
	#plt.show(block=False)
else:
	Rv_fit=3.1
	EBV_fit=EBV_host



def color_code_lam(lam_AA):
	cmap = np.array([(2055.15446097, '#060606'), (2246.17915054, '#FF37DE'),
					 (2580.39105348, '#AE0DBB'), (3467.03201164, '#6D00C2'),
					 (3467.67049526, '#6D00C2'), (3555.67297413, '#6D00C2'),
					 (4349.64131434, '#1300FF'), (4701.8844772 , '#00BA41'),
					 (4789.31352788, '#00BA41'), (4834.88251942, '#00BA41'),
					 (5425.37804356, '#00DCA7'), (6174.3138502 , '#EA0000'),
					 (6185.45988789, '#EA0000'), (6421.16699294, '#EA0000'),
					 (7489.16821807, '#D3DE00'), (7625.8480685 , '#D3DE00'),
					 (7954.9296236 , '#D3DE00'), (8988.09262894, '#680500')])
	try:
		lamss =   np.array([float(x[0]) for x in cmap])
		colorss =   np.array([x[1] for x in cmap])
		res = colorss[lam_AA >= lamss][-1] 
	except:
		import ipdb; ipdb.set_trace()

	return res



T_bb,T_bb_e,r_bb, r_bb_e, c1 ,dof, chi2_vec, dof_vec,cov = fit_bb_sequence(time,data_full,EBV=EBV_fit,Ebv_MW=Ebv_MW,R_v=Rv_fit,plot_sed=plot_sed,z=z)
if plots:
	
	T30,L30,R30,T15,L15,R15 = convert_bb(T_bb,4*np.pi*r_bb**2*sigma_sb*T_bb**4)
	cond = T_bb<25000
	plt.figure()
	plt.errorbar(time,T_bb,T_bb_e,marker='s',color='b',linestyle="None", label = 'Fit result')
	if EBV_fit==0:
		plt.plot(time[cond],T15[cond],marker='s',color='g',linestyle="--"  ,label = 'estimated correction for EBV = 0.15 mag')
		plt.plot(time[cond],T30[cond],marker='s',color='r',linestyle="--"  ,label = 'estimated correction for EBV = 0.3 mag')
	
	plt.xlabel('Rest-frame time since explosion [d]',fontsize = 15)
	plt.ylabel(r'$T_{bb}\ [K]$',fontsize = 15)
	plt.legend(fontsize = 15)


	plt.figure()
	plt.errorbar(time,r_bb,r_bb_e,marker='s',color='b',linestyle="None", label = 'Fit result')
	if EBV_fit==0:
		plt.plot(time[cond],R15[cond],marker='s',color='g',linestyle="--"  ,label = 'estimated correction for EBV = 0.15 mag')
		plt.plot(time[cond],R30[cond],marker='s',color='r',linestyle="--"  ,label = 'estimated correction for EBV = 0.3 mag')
	plt.legend(fontsize = 15)

	plt.xlabel('Rest-frame time since explosion [d]',fontsize = 15)
	plt.ylabel(r'$r_{bb}\ [cm]$',fontsize = 15)  

	plt.figure()
	plt.errorbar(time,chi2_vec/dof_vec,marker='s',color='r',linestyle="None")

	plt.xlabel('Rest-frame time since explosion [d]',fontsize = 15)
	plt.ylabel(r'$\chi^{2}/d.o.f.$',fontsize = 15)
	plt.legend(fontsize = 15)




dates=dates.reshape((len(dates),1))

try:
	T_bb=T_bb.reshape((len(T_bb),1))
except:
	import ipdb; ipdb.set_trace()

T_bb_e=T_bb_e.reshape((len(T_bb),1))
r_bb=r_bb.reshape((len(T_bb),1))
r_bb_e=r_bb_e.reshape((len(T_bb),1))
L_bb=4*np.pi*r_bb**2*sigma_sb*T_bb**4
dLdr=2*r_bb*T_bb**4
dLdT=4*r_bb**2*T_bb**3
cov_term = [2*c[0,1]*(1e4)*(d*1e-10) for c in cov]
cov_term = np.array(cov_term)
cov_term = np.reshape(cov_term,np.shape(T_bb_e))
cov_term =cov_term*dLdr*dLdT 
#L_bb_e=4*np.pi*sigma_sb*np.sqrt((dLdr*r_bb_e)**2+(dLdT*T_bb_e)**2)
L_bb_e=4*np.pi*sigma_sb*np.sqrt((dLdr*r_bb_e)**2+(dLdT*T_bb_e)**2+cov_term)


def Integrate_SED(time,data_full,z=0,EBV=0,Ebv_MW=Ebv_MW, R_v_host=3.1,dist_mpc='z', ex = 'remove', LAW =LAW):
	if dist_mpc == 'z':
		dist_cm = cosmo_dist(z)[0]*1e6*constants.pc.cgs.value
	else:
		 dist_cm = dist_mpc*1e6*constants.pc.cgs.value

	L_bol = np.zeros((np.shape(data_full)[1],1)).flatten()
	L_bol_err = np.zeros((np.shape(data_full)[1],1)).flatten()
	for i in range(len(time)):
		lam=piv_wl[data_full[:,i]!=99]
		lam=lam/(1+z)
		f_lam=(1+z)*data_full[data_full[:,i]!=99,i]
		if ex == 'apply':
			f_lam2=apply_extinction(lam,f_lam,EBV=EBV,EBV_mw=Ebv_MW, R_v=R_v_host,z=z, LAW =LAW)
		elif ex == 'remove':
			f_lam2=remove_extinction(lam,f_lam,EBV=EBV,EBV_mw=Ebv_MW, R_v=R_v_host,z=z, LAW =LAW)
		else: 
			raise Exception("ex can be either 'apply' or 'remove'.")
		ratio  = f_lam/f_lam2
		f_lam = ratio*f_lam
		f_err=data_full_e[data_full_e[:,i]!=99,i]
		idx = np.argsort(lam)
		lam = lam[idx]
		f_lam = f_lam[idx]
		f_err = f_err[idx]
		L_bol[i] = np.trapz(f_lam,x=lam)
		L_bol_err[i] = np.sqrt(np.trapz(f_err**2,lam))
	L_bol = L_bol*4*np.pi*dist_cm**2
	L_bol_err = L_bol_err*4*np.pi*dist_cm**2     
	return L_bol,L_bol_err

def get_pseudo_bol(time,data_full,T_bb,T_bb_e,L_bb,L_bb_e,z=0,plot = False):
	IR_corr = np.zeros_like(T_bb)
	UV_corr = np.zeros_like(T_bb)
	lams1 = np.logspace(-2,5,10000)
	lams2  = {}
	lams3  = {}
	try:
		L_bol,L_bol_err=Integrate_SED(time,data_full,z=z, EBV=0,Ebv_MW=0,dist_mpc = d_mpc, ex = 'remove')
	except:
		import ipdb; ipdb.set_trace()
	for i in range(len(time)):
		try:
			lam=piv_wl[data_full[:,i]!=99]*1e-4
			lam=lam/(1+z)
			f_lam=(1+z)*data_full[data_full[:,i]!=99,i]*1e13
			f_err=(1+z)*data_full_e[data_full_e[:,i]!=99,i]*1e13
			lams2[i] = np.logspace(np.max(np.log10(lam))+4,5,10000)
			lams3[i] = np.logspace(-2,np.min(np.log10(lam))+4,10000)
			#import ipdb; ipdb.set_trace()
			bb_spec1 = bb_func(lams1*1e-4,T_bb[i]*1e-4,1, EBV=0,EBV_mw=0)
			bb_spec2 = bb_func(lams2[i]*1e-4,T_bb[i]*1e-4,1, EBV=0,EBV_mw=0)
			bb_spec3 = bb_func(lams3[i]*1e-4,T_bb[i]*1e-4,1, EBV=0,EBV_mw=0)
			
			IR_corr[i] = np.trapz(bb_spec2,lams2[i])/np.trapz(bb_spec1,lams1)
			UV_corr[i] = np.trapz(bb_spec3,lams3[i])/np.trapz(bb_spec1,lams1)
		except:
			import ipdb; ipdb.set_trace()
	L_bol = L_bol.flatten()
	L_bol_err = L_bol_err.flatten()
	UV_corr = UV_corr.flatten()
	IR_corr = IR_corr.flatten()
	L_bb = L_bb.flatten()
	L_bol_corr  = L_bol+L_bb*(IR_corr+ UV_corr)
	L_bb_e = L_bb_e.flatten()
	L_bol_corr_err  = np.sqrt(L_bol_err**2+L_bb_e**2*(IR_corr+ UV_corr)**2)
	if plot: 

		plt.figure()
		plt.errorbar(time/(1+z),L_bb,yerr = L_bb_e, capsize = 0, ls = '',color='r',marker = 'o',label='Blackbody luminosity')
		plt.errorbar(time/(1+z),L_bol,yerr = L_bol_err, capsize = 0, ls = '',color='b',marker = '*',label='SED trapezoidal integration')
		plt.errorbar(time/(1+z),L_bol_corr,yerr = L_bol_corr_err, capsize = 0, ls = '',color='g',marker = '*',label=r'Trapezoidal integration + IR/UV correction (from $L_{bb}$)',alpha = 0.5)
		#plt.errorbar(time/(1+z),L_bol_corr_2,yerr = L_bol_err, capsize = 0, ls = '',color='g',marker = '*',label='trapezoidal integration + IR/UV correction (from trapz-optical)')

		#plt.errorbar(time/(1+z),L_bb*numlum,yerr = L_bol_err, capsize = 0, ls = '',color='k',marker = '*',label='numlum')
		#plt.errorbar(time/(1+z),L_bol/numlum,yerr = L_bol_err, capsize = 0, ls = '',color='m',marker = '*',label='numlum2')
		plt.xscale('log')
		plt.yscale('log')
		plt.xlabel('Rest frame time [days]')
		plt.ylabel('Luminosity [erg/s]')
		plt.legend()
		if plots:
			plt.show(block = True)
		else:
			plt.show(block = False)
	return L_bol_corr, L_bol_corr_err, L_bol, L_bol_err, UV_corr,IR_corr

if ~fit_extinction:

	L_bol_corr, L_bol_corr_err, L_bol, L_bol_err, UV_corr,IR_corr = get_pseudo_bol(time,data_full,T_bb,T_bb_e,L_bb,L_bb_e,z = z, plot = True)
	


	if write:
			L_bol_corr = L_bol_corr.reshape(len(T_bb),1)
			L_bol_corr_err = L_bol_corr_err.reshape(len(T_bb),1)
			L_bol= L_bol.reshape(len(T_bb),1)
			L_bb = L_bb.reshape(len(T_bb),1)
			T_bb = T_bb.reshape(len(T_bb),1)
	
			header="JD,t_rest ,T, T_down, T_up, R, R_down, R_up,L,L_down,L_up, chi2/dof,Psudo_L_bol, Pseudo_L_bol_err,L_bol_w_extrap,L_bol_w_extrap_err"
	
			output_table=np.hstack([x.reshape(len(T_bb),1) for x in [dates,np.array([time]).transpose(),
									  T_bb,T_bb-T_bb_e,T_bb+T_bb_e,
									  r_bb,r_bb-r_bb_e,r_bb+r_bb_e,
									  L_bb,L_bb-L_bb_e, L_bb+L_bb_e,np.transpose(np.array([chi2_vec/dof_vec])),
									  L_bol,L_bol_err,L_bol_corr,L_bol_corr_err]])

			np.savetxt(out_path,output_table,header=header,delimiter=",")
else: 

	if write:
			header="JD,t_rest ,T, T_down, T_up, R, R_down, R_up,L,L_down,L_up, chi2/dof"
			output_table=np.hstack((dates,np.array([time]).transpose() ,T_bb,T_bb-T_bb_e,T_bb+T_bb_e,r_bb,r_bb-r_bb_e,r_bb+r_bb_e,L_bb,L_bb-L_bb_e, L_bb+L_bb_e, np.transpose(np.array([chi2_vec/dof_vec]))))
			np.savetxt(out_path,output_table,header=header,delimiter=",")


