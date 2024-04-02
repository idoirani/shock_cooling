# SNeSCOPE - SuperNovae Shock Cooling Observations Parameter Estimator
## Documentation
SNeSCOPE is a python package for modeling supernova light curves using the analytic models of Morag et al 2023 and Morag et al 2024. 

## Installation



to install, git clone the repository. Then run 
#python setup.py intsall

To run pre exsiting script, modify the parameter files (e.g., see in tests). You will modify paths, SN parameters (e.g., distance, redshift, extinction), package options, and the photometric filters used while observing (an extensive folder is added, but you can always add more. Make sure to update the plotting parameters accordingly). 

Then run the script, e.g. 

#python .\scripts\fit_SC.py --path_params .\tests\params_2020jfo.py


### Package contents
 


### Python version
* `python 3`

### Required python packages
* `numpy`
* `dynesty`
* `matplotlib`
* `scipy`
* 'astropy'
* 'numba'
* 'astropy'
* 'os'


## Credit


## Credit

If you are using SNeSCOPE, please cite the following papers: 

light curve fitter: Irani et al 2024, "the early UV light curves of Type II SNe"
analytic model for T,L evolution: Morag et al 2023 "Shock cooling emission from explosions of red super-giants: I. A numerically calibrated analytic model"
deviations from blackbody: Morag et al 2024, "Shock cooling emission from explosions of red super-giants: II. An analytic model of deviations from blackbody emission"
variable validity domain: Soumagnac et al 2020 "SN 2018fif: The Explosion of a Large Red Supergiant Discovered in Its Infancy by the Zwicky Transient Facility"
If using Piro 2021 "Shock Cooling Emission from Extended Material Revisited"





