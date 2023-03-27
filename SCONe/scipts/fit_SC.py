from astropy.io import ascii
import os 
import numpy as np 
import glob
import matplotlib.pyplot as plt
import sys 
import tqdm
import os




import argparse
parser = argparse.ArgumentParser(description='')
#parser.add_argument('--Date_col', type=str, help='Date column in data', default = 'jd')
#parser.add_argument('--absmag_col', type=str, help='Absolute magnitude (AB) column in data', default = 'absmag')
#parser.add_argument('--M_err_col', type=str, help='Absolute magnitude error (AB) column in data')
#parser.add_argument('--filter_col', type=str, help='Unique filter column in data, corresponding to the parameter file keys in the transmission filter dictionary (instrument column will not be used)', default = 'filter')
#parser.add_argument('--flux_col',default='flux', type=str, help='f_lambda column in data')
#parser.add_argument('--fluxerr_col',default='fluxerr', type=str, help='f_lambda error column in data')
#parser.add_argument('--piv_wl_col',default='piv_wl', type=str, help='pivot wavelength column in data')
#
parser.add_argument('--path_data', type=str, help='Path to Data')
parser.add_argument('--path_params', type=str, help='path to parameter file', default = 'filter')
parser.add_argument('--path_dates', type=str, help='path to dates file', default = 'filter')

args = parser.parse_args()
path_data=args.path_data
Path_params=args.path_params
path_dates_sn=args.path_dates

#Date_col=args.Date_col
#M_err_col=args.M_err_col
#filter_col=args.filter_col
#absmag_col=args.absmag_col
#M_err_col=args.M_err_col
#flux_col   =args.flux_col
#fluxerr_col = args.fluxerr_col
#piv_wl_col = args.piv_wl_col


params_name = Path_params.split('/')[-1].split('.py')[0]
params_dir = Path_params.split(params_name)[0]
sys.path.append(params_dir) 


exec("from {0} import *".format(params_name))


from SCONe.PhotoUtils import *
from SCONe.blackbody_tools import *


data = ascii.read(path_data)
data = data[Date_col,filter_col,absmag_col,M_err_col,flux_col,fluxerr_col,piv_wl_col,inst_col]
data[Date_col].name = 'jd'
data[absmag_col].name = 'absmag'
data[M_err_col].name = 'AB_MAG_ERR'
data[filter_col].name = 'filter'
data[flux_col].name = 'flux'
data[fluxerr_col].name = 'fluxerr'
data[piv_wl_col].name = 'piv_wl'
data[inst_col].name = 'tag'

data['t'] = data['jd'] - t0_init
data['t_rest'] = data['t']/(1+z)
data.sort('t_rest')
cond = (data['t_rest']<max_t)&(data['t_rest']>min_t)


### priors ###
Rv_prior = np.array([2,5])  
priors  =        [np.array([0.1,20]),
				  np.array([0.3,3]),
				  np.array([0.1,200]),
				  np.array([0.3,20]),
				  np.array([t_nd,t_first]),
				  np.array([0,0.25])]

if Rv_fit:
	Rv_prior = np.array([2,5])  
	priors  = np.vstack([priors,Rv_prior])  



rec_limits =True
T_rec_low = 7300    #limit on Temperature so that T is guarenteed to be > 0.7eV (upper model validity) - appropriate if EBV host <0.2 mag or if host extinction corrected
T_rec_high= 14500   #limit on Temperature so that T is guarenteed to be < 0.7eV (upper model validity) - appropriate if EBV host <0.2 mag or if host extinction corrected




if mode =='read':
	write = False
	read = True
	plot = False
elif mode == 'write':
	write = True
	read = False  
	plot = True 
elif mode == 'replot':
	write = False 
	read = False
	plot = True


if path_out[-1]!='/':
	path_out = path_out+'/'

if not os.path.exists(path_out):
	os.mkdir(path_out)
if not os.path.exists(path_out + 'blackbodies/'):
	os.mkdir(path_out + 'blackbodies/')

### run Blackbody with nominal parameters ###
path_bb = path_out+'/blackbodies/{0}_BB_fits.txt'.format(sn)
if not os.path.exists(path_bb):
	cmd = 'fit_bb_extinction --data "{2}" --out_path "{3}" --write True --plots {8} --name {0} --EBV_host {1} --z {4} --d_mpc {5} --dates "{6}" --t0 {7: .2f} --modify_BB_MSW {9}'.format(sn,ebv_host_init,path_data,path_bb,z,d_mpc,path_dates_sn,t0_init,plot_BB, modify_BB_MSW)
	#cmd = cmd + " --Date_col {0} --absmag_col {1} --M_err_col {2} --filter_col {3} --flux_col {4} --fluxerr_col {5} --piv_wl_col {6}".format(Date_col,absmag_col,M_err_col,filter_col,flux_col,fluxerr_col,piv_wl_col)
	cmd = cmd + " --path_params {0}".format(Path_params)
	
	os.system(cmd)

bb_fits = ascii.read(path_bb)

### use BB to get recombination time limits ###
if rec_limits: 
	T_up = np.max(bb_fits['T'])
	T_down = np.min(bb_fits['T'])
	if T_up>T_rec_high:
		t_low = bb_fits[bb_fits['T']>14500]['t_rest'][-1]
	else:
		t_low = -np.inf
	if T_down<T_rec_low:
		t_high = bb_fits[bb_fits['T']<7300]['t_rest'][0]
	else:
		t_high = np.inf

	rec_time_lims = [t_low,t_high]
else: 
	rec_time_lims = [-np.inf,np.inf] 




import scipy.io
key_mat = scipy.io.loadmat(path_key)
key_mat = key_mat['key'][0]
key_dic = {}
key_dic['names']   =    key_mat[0][0][0]
key_dic['ind_valid'] = np.argwhere(['TOPS' not in key_dic['names'][i][0] for i in range(len(key_dic['names']))] ).flatten()
key_dic['ind_notvalid'] = np.argwhere(['TOPS' in key_dic['names'][i][0] for i in range(len(key_dic['names']))] ).flatten()




if write:
	if covar:
		resids_cov,resids,t_max = construct_covariance_MG_v2(data[cond],path_mat,path_key,dic_transmission=filter_transmission,model_func = model_freq_dep_SC,valid_inds = key_dic['ind_valid'])
		#resids_cov[np.isnan(resids_cov)] = 0   
		cov_obs = np.diagflat(np.array(data[(data['t_rest']<=t_max)&cond]['AB_MAG_ERR'])**2+ sys_err**2)   
		cov = resids_cov + cov_obs  
		#data_resid = data_resid['resid'][data_resid['t_rest']<tmax]
		#cov[np.isnan(cov)] = 0
		u,d,v = np.linalg.svd(resids_cov)
		A = u[:,:3] @ np.diag(d[:3]) @ v[:3,:]   
		if (np.linalg.eig(A+cov_obs)[0]<0).any():
			A = u[:,:2] @ np.diag(d[:2]) @ v[:2,:]   
		cov_est = A+cov_obs
		inv_cov = np.linalg.inv(cov_est)
	else: 
		cov_obs = np.diagflat(np.array(data[data['t_rest']<=t_max]['AB_MAG_ERR'])**2+ sys_err**2)   
		cov = cov_obs  
		inv_cov = np.linalg.inv(cov_obs)
	if Rv_fit:
		Rv = 'fit'
	mean, quantiles,dresults = fit_freq_dep_SC(data[cond],filter_transmission_fast,inv_cov = inv_cov,k34 = k34,Rv=Rv,sys_err =sys_err, LAW = LAW,plot_corner=False,priors =priors,rec_time_lims = rec_time_lims)
	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])
	best =  samples[weights == np.max(weights)][0]
	if not Rv_fit:
		R13,v85,fM,Menv,t0,ebv = best
	else:
		R13,v85,fM,Menv,t0,ebv,Rv = best

	obj = model_freq_dep_SC(R13,v85,fM,Menv,t0,k34, ebv = ebv, Rv = Rv, LAW = LAW, distance = 3.0856e19)

	cmd = 'python "{8}/fit_bb_extinction.py" --data "{2}" --out_path "{3}blackbodies/{0}_BB_from_SC.txt" --write True --plots {9} --name {0} --EBV_host {1} --z {4} --d_mpc {5} --dates "{6}" --t0 {7: .2f} --modify_BB_MSW {10}'.format(sn,obj.ebv,path_data,path_out,z,d_mpc,path_dates_sn,t0_init+t0,path_package,False, modify_BB_MSW)
	os.system(cmd)
	
	Temps = ascii.read(path_out+'blackbodies/{0}_BB_from_SC.txt'.format(sn))
	results[sn] = [best,obj,Temps,dresults]
	exist = True
	if write:
		#write    
		ob = results[sn]
		filehandler = open(path_out + '{0}_fit_bb_obj.pkl'.format(sn),'wb') 
		pickle.dump(ob,filehandler)

elif read:
	#read
	fname = path_out + '{0}_fit_bb_obj.pkl'.format(sn)
	exist = os.path.isfile(fname)
	if exist:
		filehandler = open(fname,'rb') 
		object = pickle.load(filehandler)
		best,obj,Temps,dresults = object 
		results[sn] = object
elif (not read)&(not write):
	#read
	fname = path_out + '{0}_fit_bb_obj.pkl'.format(sn)
	exist = sn in results.keys()
	if exist:
		best,obj,Temps,dresults = results[sn] 
if exist:
	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])
	best =  samples[weights == np.max(weights)][0]
	if not Rv_fit:
		R13,v85,fM,Menv,t0,ebv = best
	else:
		R13,v85,fM,Menv,t0,ebv,Rv = best
	if plot:
		if corner_plot:
			lab_vec = [r'$R_{13}$',r'$v_{s,8.5}$',r'$f_{\rho}M$',r'$M_{env}$',r'$t_{0}$',r'$E(B-V)$']
			if Rv_fit:
					lab_vec = lab_vec.append(r'$R_{v}$')
			cfig, caxes = dyplot.cornerplot(dresults,labels=lab_vec,label_kwargs={'fontsize':14}, color = '#0042CF',show_titles = True)
			plt.title(sn)
			figname1 = path_out + '{0}_fit_corner.pdf'.format(sn)
			cfig.savefig(figname1, dpi=200, format='pdf')
			if show:
				plt.show(block = False)
			plt.close(cfig)
		if bb_plot:
			fig_bb, ax_bb = plot_bb_with_model(obj,Temps,0)
			plt.title(sn)
			rec_low = rec_time_lims[0]
			rec_high = rec_time_lims[1]
			ylim = ax_bb[0].get_ylim()
			xlim = ax_bb[0].get_xlim()
			if (rec_high<np.inf):
				xlim = xlim[0],max(xlim[1]*1.4,1.1*rec_high)
				ax_bb[0].text(rec_low+0.3*(rec_high-rec_low),2e4,r'$t_{rec}$')
			else:
				xlim = xlim[0],xlim[1]*1.4
				ax_bb[0].text(rec_low+0.3*(xlim[1]-rec_low),2e4,r'$t_{rec}$')
			ax_bb[0].set_xlim(xlim)
			ax_bb[0].plot([rec_low,rec_low],[ylim[0],ylim[1]],'k--',alpha = 0.5)
			ax_bb[0].plot([rec_high,rec_high],[ylim[0],ylim[1]],'k--',alpha = 0.5)
			ax_bb[0].arrow(float(rec_low),2e4,0.1*float(rec_low),0, head_width=1000, head_length=0.07*float(rec_low), fc='k', ec='k')
			ax_bb[0].arrow(float(rec_high),2e4,-0.1*float(rec_high),0, head_width=1000, head_length=0.07*float(rec_high), fc='k', ec='k')
			ax_bb[0].plot([xlim[0],xlim[1]],[0.7*eV2K]*2,'k--',alpha = 0.5)
			ax_bb[0].text(xlim[0] +0.01*(xlim[1]-xlim[0]) ,0.7*eV2K,'0.7 eV')
			figname2 = path_out + '{0}_fit_bb.pdf'.format(sn)
			fig_bb.savefig(figname2, dpi=200, format='pdf')
			if show:
				plt.show(block = False)
			plt.close(fig_bb)
		if lc_plot:
			fig_lc,ax_lc = plot_lc_with_model(data,obj,t0, c_band, lab_band, offset)
			str2 = r'Fit parameters: $R_{13}=$' + '{0:.2f}'.format(R13) + r', $v_{*,8.5}=$' + '{0:.2f}'.format(v85) + r', $M_{env}=$' + '{0:.2f}'.format(Menv) + r', $E(B-V)=$' + '{0:.2f}'.format(ebv)
			ax_lc.text(0.05,0.015,str2,transform=ax_lc.transAxes)
			plt.title(sn)
			figname3 = path_out + '{0}_fit_lc.pdf'.format(sn)
			fig_lc.savefig(figname3, dpi=200, format='pdf')
			if show:
				plt.show(block = False)
			plt.close(fig_lc)
print('##### Finished sucessfully! #####')