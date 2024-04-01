from astropy.io import ascii
import os 
import numpy as np 
import glob
import matplotlib.pyplot as plt
import sys 
import tqdm
import os

sep = os.sep #OS dependent path separator


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
Path_params=args.path_params


#Date_col=args.Date_col
#M_err_col=args.M_err_col
#filter_col=args.filter_col
#absmag_col=args.absmag_col
#M_err_col=args.M_err_col
#flux_col   =args.flux_col
#fluxerr_col = args.fluxerr_col
#piv_wl_col = args.piv_wl_col



params_name = Path_params.split(sep)[-1].split('.py')[0]
params_dir = Path_params.split(params_name)[0]
sys.path.append(params_dir) 


exec("from {0} import *".format(params_name))
path_dates_sn=params_dic['path_dates']
path_data=params_dic['path_merged']


from SCONe.PhotoUtils import *
from SCONe.blackbody_tools import *


data = ascii.read(path_data)
data = data[Date_col,filter_col,absmag_col, M_col,M_err_col,flux_col,fluxerr_col,piv_wl_col,inst_col]
data[Date_col].name = 'jd'
data[absmag_col].name = 'absmag'
data[M_col].name = 'AB_MAG'
data[M_err_col].name = 'AB_MAG_ERR'
data[filter_col].name = 'filter'
data[flux_col].name = 'flux'
data[fluxerr_col].name = 'fluxerr'
data[piv_wl_col].name = 'piv_wl'
data[inst_col].name = 'instrument'

data['t'] = data['jd'] - t0_init
data['t_rest'] = data['t']/(1+z)
data.sort('t_rest')
cond = (data['t_rest']<max_t)&(data['t_rest']>min_t)

### priors ###
if not tight_t0:
	t0_priors = np.array([t_nd-0.03,t_first-0.03])
elif tight_t0:
	t0_err = np.sqrt(meta['t0_err'][meta['name']==sn][0]**2+ 0.1**2)
	t0_priors = np.array([-t0_err,t0_err])

priors  =        [priors_phys['R13'],
				  priors_phys['v85'],
				  priors_phys['fM'],
				  priors_phys['Menv'],
				  t0_priors,
				  Ebv_prior]

if Rv_fit:
	priors  = np.vstack([priors,Rv_prior])  



rec_limits =True
T_rec_low = 5500    #limit on Temperature so that T is guarenteed to be > 0.7eV (upper model validity) - appropriate if EBV host <0.2 mag or if host extinction corrected
T_rec_high= 10700   #limit on Temperature so that T is guarenteed to be < 0.7eV (upper model validity) - appropriate if EBV host <0.2 mag or if host extinction corrected




if mode =='read':
	write = False
	read = True
	plot = False
elif mode == 'write':
	write = True
	read = False  
	plot = True 
elif mode == 'read and plot':
	write = False 
	read = True
	plot = True


if path_out[-1]!=sep:
	path_out = path_out+sep

if not os.path.exists(path_out):
	os.mkdir(path_out)
if not os.path.exists(path_out + f'blackbodies{sep}'):
	os.mkdir(path_out + f'blackbodies{sep}')

### run Blackbody with nominal parameters ###
path_bb = path_out+f'blackbodies{sep}'+'{0}_BB_fits.txt'.format(sn)
if not os.path.exists(path_bb):
	cmd = 'python ' + path_scripts + 'fit_bb_extinction.py --data "{2}" --out_path "{3}" --write True --plots {8} --name {0} --EBV_host {1} --z {4} --d_mpc {5} --dates "{6}" --t0 {7: .2f} --modify_BB_MSW {9}'.format(sn,ebv_host_init,path_data,path_bb,z,d_mpc,path_dates_sn,t0_init,plot_BB, modified_bb)
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
path_scripts = params_dic['path_scripts']

def write_action(sn,data,cond,priors,rec_time_lims,t0_vec,d_mpc,z,t0_init,t_first,t_nd,path_dates_sn,path_scripts):
	data.sort('t_rest')
	inv_cov,cov_est = generate_full_covariance(data[cond],path_mat,path_key,filter_transmission_fast,sys_err = sys_err,covar = covar, reduced = reduced,valid_inds = key_dic['ind_valid'],t0_vec = t0_vec)    
	if Rv_fit:
		Rv = 'fit'
	else:
		Rv = params_dic['Rv']
	if FD:
		mean, quantiles,dresults = fit_freq_dep_SC(data[cond],filter_transmission_fast,inv_cov = inv_cov,k34 = k34,Rv=Rv,sys_err =sys_err, LAW = LAW,plot_corner=False,priors =priors,prior_type = prior_type,rec_time_lims = rec_time_lims,t_tr_min = t_tr_prior, reduced = reduced,time_weighted = time_weighted)
	else:
		mean, quantiles,dresults = fit_SC(data[cond],filter_transmission = filter_transmission_fast,k34 = k34,Rv=Rv,sys_err =sys_err, LAW = LAW,plot_corner=False,priors =priors)
	try:
		
		samples = dresults.samples  # samples
		weights = np.exp(dresults.logwt - dresults.logz[-1])
		best =  samples[weights == np.max(weights)][0]
		if not Rv_fit:
			R13,v85,fM,Menv,t0,ebv = best
		else:
			R13,v85,fM,Menv,t0,ebv,Rv = best
		if FD:
			obj = model_freq_dep_SC(R13,v85,fM,Menv,t0,k34,filter_transmission = filter_transmission_fast, ebv = ebv, Rv = Rv, LAW = LAW, distance = 3.0856e19, reduced = reduced)
		else:
			obj = model_SC(R13,v85,fM,Menv,t0,k34,filter_transmission = filter_transmission_fast, ebv = ebv, Rv = Rv, LAW = LAW, distance = 3.0856e19)
		results[sn] = [best,obj,dresults]
		exist = True
		if path_scripts[-1]!=sep: 
			path_scripts = path_scripts + sep
		cmd = 'python "{8}fit_bb_extinction.py" --data "{2}" --out_path "{3}blackbodies{11}{0}_BB_from_SC.txt" --write True --plots {9} --name {0} --EBV_host {1} --z {4} --d_mpc {5} --dates "{6}" --t0 {7: .2f} --modify_BB_MSW {10} --path_params {12}'.format(sn,obj.ebv,path_data,path_out,z,d_mpc,path_dates_sn,t0_init+t0,path_scripts,False, False,sep, Path_params)
		import os
		os.system(cmd)
		Temps = ascii.read(path_out+'blackbodies{1}{0}_BB_from_SC.txt'.format(sn,sep))
		results[sn] = [best,obj,Temps,dresults]
		exist = True
		#write    
		ob = results[sn]
		filehandler = open(path_out + '{0}_fit_bb_obj.pkl'.format(sn),'wb') 
		pickle.dump(ob,filehandler)
	except Exception as err:
		print(err)
		import ipdb; ipdb.set_trace()
	return results,exist,best,obj,Temps,dresults
def read_action(sn):
	#read
	fname = path_fold + '{0}_fit_bb_obj.pkl'.format(sn)
	import os
	exist = os.path.isfile(fname)
	if exist:
		filehandler = open(fname,'rb') 
		object = pickle.load(filehandler)
		best,obj,Temps,dresults = object 
		results[sn] = object	
	return results,exist,best,obj,Temps,dresults
		
def corner_plot_action(sn,dresults,show = show, path_fold = path_fold):
	lab_vec = [r'$R_{13}$',r'$v_{s,8.5}$',r'$f_{\rho}M$',r'$M_{env}$',r'$t_{0}$',r'$E(B-V)$']
	if Rv_fit:
			lab_vec.append(r'$R_{v}$')
	cfig, caxes = dyplot.cornerplot(dresults,labels=lab_vec,label_kwargs={'fontsize':14}, color = '#0042CF',show_titles = True)
	#plt.title(sn)
	plt.text(0.5, 0.8, sn, horizontalalignment='center',verticalalignment='center', transform=cfig.transFigure)
	figname1 = path_fold + '{0}_fit_corner.pdf'.format(sn)
	cfig.savefig(figname1, dpi=200, format='pdf')
	if show:
		plt.show(block = False)
	else:
		plt.close(cfig)	
	pass
def bb_plot_action(sn,obj,Temps,t_low,t_high,show = show):
	fig_bb, ax_bb = plot_bb_with_model(obj,Temps,0)
	plt.title(sn)
	rec_low  = t_low 
	rec_high = t_high
	ylim = ax_bb[0].get_ylim()
	xlim = ax_bb[0].get_xlim()
	if rec_high<np.inf:
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
	figname2 = path_fold + '{0}_fit_bb.pdf'.format(sn)
	fig_bb.savefig(figname2, dpi=200, format='pdf')
	if show:
		plt.show(block = False)
	else:
		plt.close(fig_bb)	
	pass
def plot_lc_action(sn,data,obj, c_band,samples,weights,show = show,size = 50):
	best =  samples[weights == np.max(weights)][0]
	randind = np.random.choice(len(samples), size=size, replace=True, p=weights/weights.sum())
	random = samples[randind]
	if not Rv_fit:
		R13,v85,fM,Menv,t0,ebv = best
	else:
		R13,v85,fM,Menv,t0,ebv,Rv = best
	lab_d = {x:'' for x in lab_band.keys()}
	
	fig_lc,ax_lc = plot_lc_with_model(data,obj,t0, c_band,lab_d , offset)

	print('Plotting random models from posterior')
	for i in tqdm.tqdm(range(len(random))):
		if not Rv_fit:
			R13_rand,v85_rand,fM_rand,Menv_rand,t0_rand,ebv_rand = random[i]
			Rv_rand =  3.1
		else:
			R13_rand,v85_rand,fM_rand,Menv_rand,t0_rand,ebv_rand,Rv_rand = random[i]
		obj_rand = model_freq_dep_SC(R13_rand,v85_rand,fM_rand,Menv_rand,t0_rand,k34,filter_transmission = filter_transmission_fast, ebv = ebv_rand, Rv = Rv_rand, LAW = LAW, distance = 3.0856e19, reduced = reduced)
		filt_list = np.unique(data['filter'])
		_,ax_lc = plot_lc_model_only(obj_rand,filt_list,obj_rand.t_down,obj_rand.t_up,obj_rand.t0-t0, c_band, lab_d,offset,validity = True, fig = fig_lc, ax = ax_lc,linewidth = 2, alpha = 0.8/len(random))
	ax_lc.set_xlim((-0.2*(obj.t_up),1.1*obj.t_up))
	randind_full = np.random.choice(len(samples), size=1000, replace=True, p=weights/weights.sum())
	random_full = samples[randind_full]
	R13_mean, v85_mean, fM_mean, Menv_mean, t0_mean, ebv_mean = np.mean(random_full[:,:6],axis = 0)
	if Rv_fit:
		Rv_mean = np.mean(random_full[:,6])
	
	#obj_mean = model_freq_dep_SC(R13_mean,v85_mean,fM_mean,Menv_mean,t0_mean,k34,filter_transmission = filter_transmission_fast, ebv = ebv_mean, Rv = Rv_mean, LAW = LAW, distance = 3.0856e19, reduced = reduced)
	#_,ax_lc = plot_lc_model_only(obj_mean,filt_list,obj_mean.t_down,obj_mean.t_up,obj_mean.t0-t0, c_band, lab_d,offset,validity = True, fig = fig_lc, ax = ax_lc,linewidth = 2, alpha = 1,ls = '--')
	
	R13_std, v85_std, fM_std, Menv_std, t0_std, ebv_std = np.std(random_full[:,:6],axis = 0)
	R13_err_up  , v85_err_up  , _, Menv_err_up  , t0_err_up  , ebv_err_up   = np.percentile(random_full[:,:6],84,axis = 0) - best[:6]
	R13_err_down, v85_err_down, _, Menv_err_down, t0_err_down, ebv_err_down = best[:6] - np.percentile(random_full[:,:6],16,axis = 0)
	R13_up  , v85_up  , _, _, _, _= np.percentile(random_full[:,:6],84,axis = 0)
	R13_down, v85_down, _, _, _, _= np.percentile(random_full[:,:6],16,axis = 0)
		
	ylim = ax_lc.get_ylim()
	ylim = (np.max(obj.mags_single([obj.t_down,obj.t_up],'UVW2')['UVW2'])-offset['UVW2']+1,ylim[1])	
	if Rv_fit:
		Rv_err_up =np.percentile(random_full[:,6],84,axis = 0) - best[6]
		Rv_err_down = best[6] - np.percentile(random_full[:,6],16,axis = 0)
	#str2 = r'$R_{13}=' + '{0:.1f}'.format(R13) + r'^{+' + '{0:.1f}'.format(R13_err_up) + '}'  + r'_{-' + '{0:.1f}'.format(R13_err_down) + '}$'
	#str2 = str2 + r', $v_{*,8.5}=' + '{0:.1f}'.format(v85) +r'^{+' + '{0:.1f}'.format(v85_err_up) + '}'  + r'_{-' + '{0:.1f}'.format(v85_err_down) + '}$'
	str2 = r'$R_{13}=' + '{0:.1f}$ ({1:.1f},{2:.1f})'.format(R13,R13_down,R13_up)
	str2 = str2 + r', $v_{*,8.5}=' + '{0:.1f}$ ({1:.1f},{2:.1f})'.format(v85,v85_down,v85_up)
	#str3 = r'$M_{env}=' + '{0:.1f}'.format(Menv) +r'^{+' + '{0:.1f}'.format(Menv_err_up) + '}'  + r'_{-' + '{0:.1f}'.format(Menv_err_down) + '}$'
	#str3 = str3 + r', $E(B-V)=' + '{0:.2f}'.format(ebv)+r'^{+' + '{0:.1f}'.format(ebv_err_up) + '}'  + r'_{-' + '{0:.1f}'.format(ebv_err_down) + '}$'
	
	str3 = r'$E(B-V)=' + '{0:.2f}'.format(ebv)+r'^{+' + '{0:.2f}'.format(ebv_err_up) + '}'  + r'_{-' + '{0:.2f}'.format(ebv_err_down) + '}$'
	if Rv_fit:
		str3 = str3 + r', $R_V='+ '{0:.1f}'.format(Rv) +r'^{+' + '{0:.1f}'.format(Rv_err_up) + '}'  + r'_{-' + '{0:.1f}'.format(Rv_err_down) + '}$'
	ax_lc.text(0.05,0.04,str2,transform=ax_lc.transAxes,fontsize = 14)  
	ax_lc.text(0.05,0.01,str3,transform=ax_lc.transAxes,fontsize = 14)  
	ax_lc.tick_params(axis='both', which='major', labelsize=14)
	#create minor ticks 
	from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
	ax_lc.xaxis.set_minor_locator(AutoMinorLocator())
	ax_lc.yaxis.set_minor_locator(AutoMinorLocator())
	
	plt.title(sn,fontsize = 15)
	t_trans , _, _,_ = break_params_new(obj.R13, obj.v85,obj.fM,obj.Menv,obj.k34)
	t_trans = t_trans/24

	ax_lc.plot([t_trans,t_trans],[ylim[0],ylim[1]],'r--',alpha = 0.3)
	ax_lc.plot([obj.t_down,obj.t_down],[ylim[0],ylim[1]],'k--',alpha = 0.3)
	ax_lc.set_ylim(ylim)
	xlim = ax_lc.get_xlim()
	minx = -0.17*max(1.1*obj.t_up,5)
	xlab_pos = -0.15*max(1.1*obj.t_up,5)
	xlim = (minx,max(1.1*obj.t_up,5))
	ax_lc.set_xscale('log')
	ax_lc.set_xlim((0.02,1.1*obj.t_up))
	figname3log = path_fold + '{0}_fit_lc_logt.pdf'.format(sn)
	fig_lc.savefig(figname3log, dpi=200, format='pdf')

	ax_lc.set_xscale('linear')
	ax_lc.set_xlim(xlim)
	time_2 = np.logspace(-2,np.log10(np.max(data['t_rest'])),30)
	filt_list = np.unique(data['filter'])
	mags =   obj.mags(time_2,filt_list=filt_list)

	figname3 = path_fold + '{0}_fit_lc.pdf'.format(sn)
	for band in np.unique(data['filter']):
		if lab_band[band]!='':
			if np.sign(-offset[band]) == 1:
				string = lab_band[band]+' +{0}'.format(-offset[band])
			elif np.sign(-offset[band]) == -1:
				string = lab_band[band]+' -{0}'.format(-offset[band])
			elif np.sign(-offset[band]) == 0:
				string = lab_band[band]
			cond_dic = (time_2<=max(1.2*obj.t_up,5))
			ax_lc.text(xlab_pos,np.mean(mags[band][cond_dic]-offset[band]),string, color =c_band[band] ) 
	fig_lc.savefig(figname3, dpi=200, format='pdf')

	if show:
		plt.show(block = False)
	else:
		plt.close(fig_lc)	
	pass
def plot_sed_action(sn,data,d_mpc, c_band,samples,weights,show = show):
	fig_sed = plt.figure(figsize=(20,15))
	fignamesed = path_fold + '{0}_fit_sed.pdf'.format(sn)
	plot_SED_sequence(data,samples,weights,d_mpc,filter_transmission = filter_transmission_fast,c_band = c_band, fig_sed = fig_sed,save_path = fignamesed,sn = sn)
	if show:
		plt.show(block = False)
	else:
		plt.close(fig_sed)		
def fit_action(sn,results= results):
	if write:
		result_d,exist,best,obj,Temps,dresults = write_action(sn,data,cond,priors,rec_time_lims,t0_vec,d_mpc,z,t0_init,t_first,t_nd,path_dates_sn,path_scripts)
		results.update(result_d)
	elif read:
		result_d,exist,best,obj,Temps,dresults = read_action(sn)
		results.update(result_d)
	elif (not read)&(not write):
		best,obj,Temps,dresults	= replot_action(sn,results)
		
	if exist:
		samples = dresults.samples  # samples
		weights = np.exp(dresults.logwt - dresults.logz[-1])
	
		if plot:
			print('plot')
			if corner_plot:
				corner_plot_action(sn,dresults,show = show, path_fold = path_fold)
			if bb_plot:
				bb_plot_action(sn,obj,Temps,t_low,t_high,show = show)
			if lc_plot:
				plot_lc_action(sn,data,obj, c_band,samples,weights,show = show)
			if plot_sed:
				plot_sed_action(sn,data,d_mpc, c_band,samples,weights,show = show)
	return results		
results=fit_action(sn,results)


t_transp = {}
validity_dic = {}
for sn in results.keys():
	[best,obj,Temps,dresults] = results[sn]
	if not Rv_fit:
		R13,v85,fM,Menv,t0,ebv = best
	else:
		R13,v85,fM,Menv,t0,ebv,Rv = best
	t_tr  = 19.5*np.sqrt(Menv*k34/v85)
	t07 = 6.86*R13**(0.56)*v85**(0.16)*k34**(-0.61)*fM**(-0.06) 
	t_transp[sn] = (t_tr/2,t07) 
	validity_dic[sn] = validity2(R13, v85,fM,k34,Menv)[1]

path_results_table = path_fold + 'Results_table.txt'
#create a table of the best fit results and errors for all SNe 
names = ('SN','R13','R13_u','R13_l','v85','v85_u','v85_l','fM','fM_u','fM_l','Menv','Menv_u','Menv_l','t0','t0_u','t0_l','ebv','ebv_u','ebv_l','Rv','Rv_u','Rv_l','vbo9','vbo9_u','vbo9_l','rhobo9','rhobo9_u','rhobo9_l','t_tr','t07')
dtypes = ('S20',float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float)
Results_table = table.Table(names = names,dtype = dtypes)
for sn in results.keys():
	[best,obj,Temps,dresults] = results[sn]

	samples = dresults.samples  # samples
	weights = np.exp(dresults.logwt - dresults.logz[-1])  # normalized weights
	if not Rv_fit:
		R13,v85,fM,Menv,t0,ebv = np.average(samples, axis=0, weights=weights)
		Rv = 3.1 #default value
	else:
		R13,v85,fM,Menv,t0,ebv,Rv = np.average(samples, axis=0, weights=weights)
	bo_beta = np.array(list(map(lambda i: bo_params_from_phys(samples[i,2],samples[i,1],samples[i,0],k034=1)[0],range(np.shape(samples)[0]))))
	bo_rho9 = np.array(list(map(lambda i: bo_params_from_phys(samples[i,2],samples[i,1],samples[i,0],k034=1)[1],range(np.shape(samples)[0]))))

	v_bo9   = np.average(bo_beta*3e10/1e9,weights = weights)
	rho_bo9 = np.average(bo_rho9,weights = weights)
	samples_ext = np.concatenate((samples,np.array([bo_beta*3e10/1e9]).T,np.array([bo_rho9]).T),axis = 1)
	# Compute 10%-90% quantiles.
	quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
				 for samps in samples.T]
	quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights)
				 for samps in samples_ext.T]	
	errors = np.array(quantiles).T
	eR13  = list(errors[:,0])
	ev85  = list(errors[:,1])
	efM   = list(errors[:,2])
	eMenv = list(errors[:,3])
	et0   = list(errors[:,4])
	eebv  = list(errors[:,5])
	if Rv_fit:
		eRv = errors[:,6]
		e_v_bo9   = list(errors[:,7])
		e_rho_bo9 = list(errors[:,8])
	else:
		eRv = [-1.0,-1.0]
		e_v_bo9   = list(errors[:,6])
		e_rho_bo9 = list(errors[:,7])
	Results_table.add_row((sn,R13,eR13[1],eR13[0],v85,ev85[1],ev85[0],fM,efM[1],efM[0],Menv,eMenv[1],eMenv[0],t0,et0[1],et0[0],ebv,eebv[1],eebv[0],Rv,eRv[1],eRv[0],v_bo9,e_v_bo9[1],e_v_bo9[0],rho_bo9,e_rho_bo9[1],e_rho_bo9[0],t_transp[sn][0],t_transp[sn][1]))
Results_table.write(path_results_table,format='ascii.fixed_width',delimiter=' ',overwrite=True)


print('##### Finished sucessfully! #####')