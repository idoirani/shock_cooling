# SNeSCOPE - Shock Cooling Optimization for superNovae
## Documentation
SNeSCOPE is a python package for modeling supernova light curves using the analytic models of Morag Sapir and Waxman (2022). 

## Installation



install package with 
pip install SNeSCOPE

or 

git clone and then 
python setup.py intsall

if you want to use the pre-existing script, able to run from the command line, add to your .bashrc file (no spaces):

alias fit_bb_extinction='python  /path/to/scipts/fit_bb_extinction.py'
alias fit_SC='python  /path/to/scipts/fit_SC.py'

and then run from the command line
source .bashrc


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

If you are using SNeSCOPE, please cite the following papers: 

light curve fitter: Irani et al 2024, "the early UV light curves of Type II SNe"
analytic model for T,L evolution: Morag et al 2023 "Shock cooling emission from explosions of red super-giants: I. A numerically calibrated analytic model"
deviations from blackbody: Morag et al 2024, "Shock cooling emission from explosions of red super-giants: II. An analytic model of deviations from blackbody emission"
variable validity domain: Soumagnac et al 2020 "SN 2018fif: The Explosion of a Large Red Supergiant Discovered in Its Infancy by the Zwicky Transient Facility"
If using Piro 2021 "Shock Cooling Emission from Extended Material Revisited"


