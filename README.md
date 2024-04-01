# SNeSCOPE - SuperNovae Shock Cooling Observations Parameter Estimator
## Documentation
SNeSCOPE is a python package for modeling supernova light curves using the analytic models of Morag et al 2023 and Morag et al 2024. 

## Installation



to install, git clone and then 
python setup.py intsall

To run pre exsiting script, modify the parameter files (e.g., see in tests). You will modify paths, SN parameters (e.g., distance, redshift, extinction), package options, and the photometric filters used while observing (an extensive folder is added, but you can always add more. Make sure to update the plotting parameters accordingly). 

Then run the script, e.g. 

python .\scripts\fit_SC.py --path_params .\tests\params_2020jfo.py


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

If you are using SOPRANOS, please cite the following papers: 



