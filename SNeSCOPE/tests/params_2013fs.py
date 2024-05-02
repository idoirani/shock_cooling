import astropy.io.ascii as ascii
import numpy as np
from SNeSCOPE.blackbody_tools import model_freq_dep_SC, fit_freq_dep_SC,model_SC, fit_SC

import os
sep = os.sep #OS dependent path separator

# path parameters (change to your local paths))
path_mat =      f'.{sep}data{sep}RSG_batch_R03_20_removed_lines_Z1.mat'
path_key  =     f'.{sep}data{sep}RSG_batch_R03_20_removed_lines_Z1_key.mat'
path_data =     f'.{sep}tests{sep}data_13fs_formatted.txt'
path_dates_sn = f'.{sep}tests{sep}13dqy_dates.txt'
path_out =      f'.{sep}output{sep}'
path_filters =  f'.{sep}Filters{sep}'
path_scripts =  f'.{sep}scripts{sep}'
path_results_table = path_out + f'results_table.txt'
path_fold = path_out
max_t = 10 #maximum time in days to attempt fitting 
min_t = 0 #minimum time in days to attempt fitting
k34 = 1 #opacity in units of 0.34 cm^2/g
Rv = 3.1 #extinction law
Rv_fit = False #if True, Rv is a free parameter
LAW = 'MW' # MW (CCM89, with UV "bump")  or Cal00 (without UV "bump"); both with variable Rv
sys_err = 0.05 #systematic error in magnitudes, added in quadrature to the photometric errors
sys_factor = 1.5 #factor by which to multiply the correlated errors. This accounts for the theoretical uncertainty in the model (see further details in Morag et al 2024)
mode = 'write' ##read|write|read and plot
z = 0.011855 #redshift
t0_init = 2456571.62 #initial time of explosion. t0 is a free parameter, but this is the initial guess
t_nd = -0.05 #time of non-detection (or, lower prior on t0). Only use if non-detection is constraining
t_first = 0.1 #first detection time. (upper limit on t0, only use if first detection is clearly a SN point. if SBO is extended, the appropriate reference is the peak of SB)
t0_vec =0.75*(np.array([t_nd,0,t_first])-0.03) #evaluate the covariance matrix at these points for t0, added to the reference time of the simulations

plot_BB = True #plot the blackbody fits for the best fit parameters
d_mpc = 50.95 #distance in Mpc
dm = 5*np.log10(d_mpc)+25 #distance modulus
sn = 'SN2013fs'  #name of the SN
ebv_host_init = 0  # initial guess for host extinction
results = {} #dictionary to store results
import pickle
covar = False #if True, use non-diagonal covariance matrix
corner_plot = True #if True, plot corner plot
lc_plot = True #if True, plot light curve
bb_plot = True  
show = False #if True, show plots
reduced = False #use reduced formula for SED only using L,T. If False, use full formula (default)
plot_sed = True #plot SED compared to model in several epochs
modified_bb = False
tight_t0 = False #use very narrow priors around t0_init
time_weighted = False #weigh early times more than late times in the fit
priors_phys  =        {'R13':np.array([0.1,20]),
				       'v85':np.array([0.3,4]),
				       'fM':np.array([0.01,10]),
				       'Menv':np.array([0.3,30])}
Ebv_prior = np.array([0,0.25])
Rv_prior = np.array([2,5])  

prior_type = ['log-uniform','log-uniform','log-uniform','log-uniform','uniform','uniform','uniform'] #prior type for each parameter (R13,v85,fM,Menv,t0, ebv, Rv)
t_tr_prior = 6 # non rectectgular lower prior on t_tr in days, time of envelope transparency. Will be calculated from each model. This parameter is used to elminate models which will push all data points outside the validity. 

#data columns - must match the columns in the data file
Date_col    = 'jd' #column name for date
M_col  = 'AB_MAG' #column name for magnitude
absmag_col  =  'absmag' #column name for absolute magnitude
M_err_col   = 'AB_MAG_ERR' #column name for absolute magnitude error
filter_col  = 'filter' #column name for filter
flux_col    = 'flux' #column name for flux
fluxerr_col = 'fluxerr' #column name for flux error
piv_wl_col  = 'piv_wl' #column name for pivot wavelength
inst_col    = 'instrument' #column name for instrument

#model and fit functions
FD = False #if True, use frequency dependent model and fit functions. If False, use frequency independent model and fit functions. Frequency dependent formula is only tested for Red Supergiant stars (Morag 2024)







#gather all variables up to this point into a dictionary
params_dic = {}
params_dic['path_merged'] = path_data
params_dic['path_dates'] = path_dates_sn
params_dic['path_mat'] = path_mat
params_dic['path_key'] = path_key
params_dic['max_t'] = max_t
params_dic['min_t'] = min_t
params_dic['k34'] = k34
params_dic['Rv'] = Rv
params_dic['Rv_fit'] = Rv_fit
params_dic['LAW'] = LAW
params_dic['sys_err'] = sys_err
params_dic['mode'] = mode
params_dic['FD'] = FD
params_dic['UV_sup'] = False
params_dic['covar'] = covar
params_dic['corner_plot'] = corner_plot
params_dic['lc_plot'] = lc_plot
params_dic['bb_plot'] = bb_plot
params_dic['plot_sed'] = plot_sed
params_dic['show'] = show
params_dic['reduced'] = reduced
params_dic['modified_bb'] = modified_bb
params_dic['time_weighted'] = time_weighted
params_dic['sys_factor'] = sys_factor
params_dic['priors_phys'] = priors_phys
params_dic['t_tr_prior'] = t_tr_prior
params_dic['tight_t0'] = tight_t0
params_dic['path_scripts'] = path_scripts
params_dic['path_out'] = path_out
params_dic['path_fold'] = path_fold
params_dic['path_results_table'] = path_results_table
params_dic['plot_BB'] = plot_BB
params_dic['d_mpc'] = d_mpc
params_dic['dm'] = dm
params_dic['sn'] = sn





#filter names, transmission curves, colors, labels, and offsets
FILTERS =   np.array(['ZTF_g'  
					 ,'ZTF_r'  
					 ,'ZTF_i'  
					 ,'u_P60'  
					 ,'g_P60'  
					 ,'r_P60'  
					 ,'i_P60'  
					 ,'LT_u'   
					 ,'LT_g'   
					 ,'LT_r'   
					 ,'LT_i'   
					 ,'LT_z'   
					 ,'LCO_u'  
					 ,'LCO_g'  
					 ,'LCO_r'  
					 ,'LCO_i'  
					 ,'LCO_r'  
					 ,'LCO_i'  
					 ,'UVM2'   
					 ,'UVW1'   
					 ,'UVW2'   
					 ,'u_swift'
					 ,'v_swift'
					 ,'b_swift'
					 '2MASS_J',
					 '2MASS_H',
					 'U',
					 'B',
					 'V',
					 'R',
					 'I',
					 'R-PTF'])

filter_transmission={'ATLAS_c':'Misc_Atlas.cyan.dat',
					 'ATLAS_o':'Misc_Atlas.orange.dat',
					 'ZTF_g':'P48_g.txt',    
					 'ZTF_r':'P48_r.txt',    
					 'ZTF_i':'ZTF_transmission_i_from_twiki.ascii',                        
					 'u_P60':'SDSS_u.txt',
					 'g_P60':'SDSS_g.txt',
					 'r_P60':'SDSS_r.txt',
					 'i_P60':'SDSS_i.txt',
					 'LT_u':'LT_u.txt',  
					 'LT_g':'LT_g.txt',  
					 'LT_r':'LT_r.txt',
					 'LT_i':'LT_i.txt',   
					 'LT_z':'LT_z.txt',
					 'LCO_u':'LCO_AA_SDSS.up.txt',
					 'LCO_g':'LCO_AA_SDSS.gp.txt',
					 'LCO_r':'LCO_AA_SDSS.rp.txt',
					 'LCO_i':'LCO_AA_SDSS.ip.txt',
					 'LCO_r':'LCO_AA_SDSS.rp.txt',
					 'LCO_i':'LCO_AA_SDSS.ip.txt',
					 'LCO_V':'LasCumbres_LasCumbres.Bessel_V.dat',
					 'LCO_B':'LasCumbres_LasCumbres.Bessel_B.dat',
					 'UVM2'   :'Swift_UVM2.rtf',
					 'UVW1'   :'Swift_UVW1.rtf',  
					 'UVW2'   :'Swift_UVW2.rtf', 
					 'u_swift':'Swift_u.rtf',
					 'v_swift':'Swift_V.rtf',  
					 'b_swift':'Swift_B.rtf',
					 'NOT_u':'NOT_u.txt',
					 'NOT_g':'NOT_g.txt',
					 'NOT_r':'NOT_r.txt',
					 'NOT_i':'NOT_i.txt',
					 'NOT_z':'NOT_z.txt',
					 'KAIT_B':   'B_kait4_shifted.txt',
					 'KAIT_V':   'V_kait4_shifted.txt',
					 'KAIT_R':   'R_kait4_shifted.txt',
					 'KAIT_I':   'I_kait4_shifted.txt',
					 'KAIT_CLEAR':   'e2v_QE_midband.csv',
					 'Ni_B':   'B_Nickel2.txt',
					 'Ni_V':   'V_Nickel2.txt',
					 'Ni_R':   'R_Nickel2.txt',
					 'Ni_I':   'I_Nickel2.txt',
					 'Ni_CLEAR':   'e2v_QE_midband.csv',
					 #'KAIT_CLEAR':'',
					 'MMIRS_J':'MMT_MMIRS.J.dat',
					 'MMIRS_H':'MMT_MMIRS.H.dat',
					 'MMIRS_Ks':'MMT_MMIRS.Ks.dat',
					 '2MASS_J':'2MASS_J.txt',
					 '2MASS_H':'2MASS_H.txt',
					 'U':'johnson_u.txt',
					 'B':'johnson_b.txt',
					 'V':'johnson_v.txt',
					 'R':'LasCumbres_LasCumbres.Bessel_V.dat',
					 'I':'LasCumbres_LasCumbres.Bessel_B.dat',
					 'R-PTF':'P48_R_T.rtf',
					 }

filter_transmission = {x:path_filters + filter_transmission[x] for x in filter_transmission.keys()}

filter_transmission_fast = {}
for x in filter_transmission.keys():
	try:
 		filter_transmission_fast[x] = np.loadtxt(filter_transmission[x])
	except:
		filter_transmission_fast[x] = np.loadtxt(filter_transmission[x],delimiter  = ',')	

filter_transmission = {x:ascii.read(filter_transmission[x]) for x in filter_transmission.keys()}




c_band ={'UVW2':'#060606'
		,'UVM2':'#FF37DE'
		,'UVW1':'#AE0DBB'
		,'u_swift':'#6D00C2'
		,'b_swift':'#1300FF'
		,'ZTF_g':'#00BA41'
		,'v_swift':'#00DCA7'
		,'u_P60': '#6D00C2'
		,'g_P60':'#00BA41'
		,'r_P60':'#EA0000'
		,'i_P60':'#D3DE00'
        ,'LCO_u':'#6D00C2'
        ,'LCO_g':'#00BA41'
        ,'LCO_r':'#EA0000'
        ,'LCO_i':'#D3DE00'
        ,'LCO_V':'#00DCA7'
        ,'LCO_B':'#1300FF'
        ,'NOT_u':'#6D00C2'
        ,'NOT_g':'#00BA41'
        ,'NOT_r':'#EA0000'
        ,'NOT_i':'#D3DE00'
        ,'NOT_z':'#680500'
        ,'KAIT_B':'#1300FF'
        ,'KAIT_V':'#00DCA7'
        ,'KAIT_R':'#EA0000'
        ,'KAIT_CLEAR':'#B2607E'
        ,'KAIT_I':'#D3DE00'
        ,'Ni_B':'#1300FF'
        ,'Ni_V':'#00DCA7'
        ,'Ni_R':'#EA0000'
        ,'Ni_CLEAR':'#B2607E'
        ,'Ni_I':'#D3DE00'
        ,'MMIRS_J':'#E79600'
        ,'MMIRS_H':'#969696'
        ,'MMIRS_Ks':'#6C3613'
		,'ZTF_r':'#EA0000'
		,'ZTF_i':'#D3DE00'
		,'LT_u':'#6D00C2'
		,'LT_z':'#680500'      
		,'LT_g':'#00BA41'
		,'LT_r':'#EA0000'
		,'LT_i':'#D3DE00'
		,'ATLAS_c':'#05E5DB'
		,'ATLAS_o':'#E56105',
		'2MASS_J':'#E79600',
		'2MASS_H':'#969696',
		'U':'#6D00C2',
		'B':'#1300FF',
		'V':'#00DCA7',
		'R':'#EA0000',
		'I':'#D3DE00',
		'R-PTF':'#EA0000',}

## band labels


lab_band = {'UVW2':'W2'
           ,'UVM2':'M2'
           ,'UVW1':'W1'
           ,'u_swift':'U'
           ,'b_swift':'b'
           ,'ZTF_g':'g'
           ,'g_P60':'g'
           ,'v_swift':'v'
           ,'r_P60':'r'
           ,'u_P60':'u'
           ,'ZTF_r':'r'
           ,'ZTF_i':'i'
           ,'i_P60':'i'
           ,'LT_z':'z'     
           ,'LT_u':'u'
           ,'LT_g':'g'  
           ,'LT_r':'r/R'
           ,'LT_i':'i/I'
         ,'KAIT_B':'B'
         ,'KAIT_V':'V'
         ,'KAIT_R':'R'
         ,'KAIT_I':'I'
         ,'KAIT_CLEAR':'Clear'
         ,'KAIT_I':''
         ,'Ni_B':'B'
         ,'Ni_V':'V'
         ,'Ni_R':'R'
         ,'Ni_CLEAR':'CLEAR'
         ,'Ni_I':'I'
         ,'MMIRS_J':'J'
         ,'MMIRS_H':'H'
         ,'MMIRS_Ks':'Ks'
		,'ATLAS_c':'c'
		,'ATLAS_o':'o'
        ,'LCO_u':'u'
        ,'LCO_g':'g'
        ,'LCO_r':'r'
        ,'LCO_i':'i'
        ,'LCO_V':'V'
        ,'LCO_B':'B'
        ,'NOT_u':'u'
        ,'NOT_g':'g'
        ,'NOT_r':'r'
        ,'NOT_i':'i'
        ,'NOT_z':'z',
		'2MASS_J':'J',
		'2MASS_H':'H',
		'U':'u',
		'B':'B',
		'V':'V',
		'R':'r/R',
		'I':'i/I',
		'R-PTF':'r/R',} 


## band offsets
offset   = {'UVW2':-4.5
		   ,'UVM2':-3.5
		   ,'UVW1':-2.5
		   ,'u_swift':-2
		   ,'u_P60':-2
		   ,'LT_u':-2
		   ,'b_swift':-1
		   ,'ZTF_g':0
		   ,'g_P60':0
		   ,'LT_g':0 
		   ,'v_swift':1
		   ,'r_P60':2
		   ,'ZTF_r':2
		   ,'LT_r':2
		   ,'ZTF_i':3
		   ,'i_P60':3
		   ,'LT_i':3
		   ,'LT_z':4 
         ,'KAIT_B':-1
         ,'KAIT_V':-1
         ,'KAIT_R':2
         ,'KAIT_I':3
         ,'KAIT_CLEAR':1
         ,'KAIT_I':3
         ,'Ni_B':-1
         ,'Ni_V':-1
         ,'Ni_R':2
         ,'Ni_CLEAR':1
         ,'Ni_I':3
         ,'MMIRS_J':5
         ,'MMIRS_H':6
         ,'MMIRS_Ks':7
		,'ATLAS_c':-0.5
		,'ATLAS_o':0.5
        ,'LCO_u':-2
        ,'LCO_g':0
        ,'LCO_r':2
        ,'LCO_i':3
        ,'LCO_V':1
        ,'LCO_B':-1
        ,'NOT_u':-2
        ,'NOT_g':0
        ,'NOT_r':2
        ,'NOT_i':3
        ,'NOT_z':4,
		'2MASS_J':5,
		'2MASS_H':6,
		'U':-2,
		'B':-1,
		'V':1,
		'R':2,
		'I':3,
		'R-PTF':2}  



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
	,'MMIRS_Ks':'*'	
	,'ATLAS_c':'x' 
	,'ATLAS_o':'x',
	'2MASS_J':'*',
	'2MASS_H':'*',
	'U':'v',
	'B':'v',
	'V':'v',
	'R':'o',
	'I':'o',
	'R-PTF':'o'}

