# SNeSCOPE - SuperNovae Shock Cooling Observations Parameter Estimator
## Documentation
SNeSCOPE is a python package for modeling supernova light curves using the analytic models of Morag et al 2023 and Morag et al 2024. 

## Installation



to install, git clone the repository. Then run from within the SNeSCOPE directory (where the script setup.py is) 
``` 
python setup.py install
``` 
To run pre exsiting script, modify the parameter files (e.g., see in tests). You will modify paths, SN parameters (e.g., distance, redshift, extinction), package options, and the photometric filters used while observing (an extensive folder is added, but you can always add more. Make sure to update the plotting parameters accordingly). 

If you encounter errors at this point, make sure you have the correct paths and consider moving from relative to absolute paths. 

Then run the script fit_SC.py, e.g., from the SNeSCOPE directory:
``` 
python ./scripts/fit_SC.py --path_params ./tests/params_2020jfo.py
```
or (if using windows)
``` 
python .\scripts\fit_SC.py --path_params .\tests\params_2020jfo.py
```

As the script is currently written, it requires a data file containing magnituds,  fluxes, instruments and filters as is shown in the example test files. The script uses a non-rectangular prior on the recombination time of Hydrogen at a photospheric temperature of roughly t_rec 0.7 eV~ 8000K. Given the typical deviations from blackbody and given a mild amount of host extinction (up to E(B-V) = 0.2), a blackboyd fit assuming E(B-V) = 0 mag at 8000K can give anywhere between 5000 and 10000 K.   I recommend to first fix the extinction to lower than 0.2, using the color curves of Irani 2024. Then add a dates files. This is a list of times (JD) where E(B-V) = 0 blackbody fits are made, and then used as priors. 

The script will run the fitter in the following steps: 
- blackbody fits
- prepare covariance matrix for likelihood function using the light curve sampling provided
- fit the data
- get the results. re-fit a blackbody using the fit host-extinciton
- save the results as a pickle object
- plot
- calculate physical parameters and save to a table (for a full description of these parameters, see Irani 2024, and Morag 2024, and references therein).

  

There are 4 plots which are provided. 
1) Light curve fits. These include a vertical line indicating the early validity time of the model (the time the breakout pulse is over and homologous expansion is reached). I also provide these plots with a logarithmic axis, which are more convenient for high cadence sampling of the early light curves.
2) Blackbody fits compared to the model predictions. These are compared to the new blackbody fits, using the fit E(B-V) (and Rv, if fitted). A good fit should also fit the blackbody temperature and radius reasonable well, although some deviations are expected as the SED is not a perfect blackbody. 
3) corner plots
4) SED plot - These can be used to evaluate the observed and model SED at various epochs. The different lines are both the blackbody and the frequency dependent formulas smapled from the posterior. Keep in mind sometimes, due to a low mass envelope or fast V*, the model can be no longer valid at the lowest temperature plotted in these plots. 


### Python version
* `python 3`

### Required python packages
* `numpy`
* `dynesty`
* `matplotlib`
* `scipy`
* 'astropy`
* 'numba`
* 'astropy`
* 'os`
* 'pandas`

## Support

I'm happy to provide support in setting up the package and interpreting its results. You can contact me at idoirani@gmail.com.

## Credit

If you are using SNeSCOPE, please cite the following papers: 

light curve fitter: Irani et al 2024, "the early UV light curves of Type II SNe"
analytic model for T,L evolution: Morag et al 2023 "Shock cooling emission from explosions of red super-giants: I. A numerically calibrated analytic model"
deviations from blackbody: Morag et al 2024, "Shock cooling emission from explosions of red super-giants: II. An analytic model of deviations from blackbody emission"
variable validity domain: Soumagnac et al 2020 "SN 2018fif: The Explosion of a Large Red Supergiant Discovered in Its Infancy by the Zwicky Transient Facility"
If using Piro 2021 "Shock Cooling Emission from Extended Material Revisited"

<a href="https://doi.org/10.5281/zenodo.10909915"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.10909915.svg" alt="DOI"></a>




