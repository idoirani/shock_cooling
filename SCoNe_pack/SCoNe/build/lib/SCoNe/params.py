import astropy.io.ascii as ascii
import numpy as np



path_mat =      './data/Full_batch_12_2022_Z_1_01.mat'
path_key  =     './data/Full_batch_12_2022_Z_1_01_key.mat'
path_data =     './tests/ZTF20aaynrrh_detections_ebv_corr.ascii'
path_dates_sn = './tests/ZTF20aaynrrh_dates.txt'
path_out =      './output/'
path_package =  './'
path_filters =  './Filters/'

max_t = 10
min_t = 0
k34 = 1
Rv = 3.1
Rv_fit = False
LAW = 'MW'
sys_err = 0.1
mode = 'write' ##read|write|replot
z = 0.005224
t0_init = 2458975.23
t_nd = - 3.45
t_first = 0.47
plot_BB = True
d_mpc = 14.7
dm = 5*np.log10(d_mpc)+25
sn = 'ZTF20aaynrrh' 
ebv_host_init = 0 
results = {}
import pickle
covar = True
corner_plot = True
lc_plot = True
bb_plot = True 
show = False
Rv_fit = False
modify_BB_MSW = True





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
					 ,'b_swift'])

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
					 'MMIRS_Ks':'MMT_MMIRS.Ks.dat'}

filter_transmission = {x:path_filters + filter_transmission[x] for x in filter_transmission.keys()}

filter_transmission_fast = {}
for x in filter_transmission.keys():
	try:
 		filter_transmission_fast[x] = np.loadtxt(filter_transmission[x])
	except:
		filter_transmission_fast[x] = np.loadtxt(filter_transmission[x],delimiter  = ',')	

filter_transmission = {x:ascii.read(filter_transmission[x]) for x in filter_transmission.keys()}
