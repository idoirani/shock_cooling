# SNeSCOPE - SuperNovae Shock Cooling Observations Parameter Estimator
## Documentation
SNeSCOPE is a python package for modeling supernova light curves using the analytic models of Morag et al 2023 and Morag et al 2024. 

## Installation



to install, git clone the repository. Then run from within the SNeSCOPE directory (where the script setup.py is) 
``` 
python setup.py install
```
Alternatively, use pip
``` 
pip install SNeSCOPE
```



## add simulation data files
Use the following link to download the MG simulatiosn used for constructing the covariance matrix for the likelihood function: 
https://www.dropbox.com/scl/fi/kawt98t2lri5j90o61u60/RSG_batch_R03_20_removed_lines_Z1.mat?rlkey=1hnpbgiayol7jtu4cn4895slg&dl=0
and for the keys:
https://www.dropbox.com/scl/fi/o22vj00kjskp9gk6ae2h1/RSG_batch_R03_20_removed_lines_Z1_key.mat?rlkey=kop5iq609h10pgkgsnycuvjyx&dl=0
If these links do not work for any reason, the files can be found in the zenodo repository linked in the bottom of this readme

In addition to these, the filter transmission data I use is collected in the Filters folder avilable in this repository, in the zenodo repository, and in the following link: 


## running the script
To run pre exsiting script, modify the parameter files (e.g., see in tests). You will modify paths, SN parameters (e.g., distance, redshift, extinction), package options, and the photometric filters used while observing (an extensive folder is added, but you can always add more. Make sure to update the plotting parameters accordingly). 

If you encounter errors at this point, make sure you have the correct paths and consider moving from relative to absolute paths. 

Then run the script fit_SC.py, e.g., from the SNeSCOPE directory (for a unix based system):
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
* `astropy`
* `numba`
* `astropy`
* `os`
* `pandas`

## Support

I'm happy to provide support in setting up the package and interpreting its results. You can contact me at idoirani@gmail.com.

## Credit
<a href="https://doi.org/10.5281/zenodo.10909915"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.10909915.svg" alt="DOI"></a>

If you are using SNeSCOPE, please cite the following papers: 

light curve fitter: **Irani et al 2024**, "the early UV light curves of Type II SNe"
```
@ARTICLE{2023arXiv231016885I,
       author = {{Irani}, Ido and {Morag}, Jonathan and {Gal-Yam}, Avishay and {Waxman}, Eli and {Schulze}, Steve and {Sollerman}, Jesper and {Hinds}, K-Ryan and {Perley}, Daniel A. and {Chen}, Ping and {Strotjohann}, Nora L. and {Yaron}, Ofer and {Zimmerman}, Erez A. and {Bruch}, Rachel and {Ofek}, Eran O. and {Soumagnac}, Maayane T. and {Yang}, Yi and {Groom}, Steven L. and {Masci}, Frank J. and {Riddle}, Reed and {Bellm}, Eric C. and {Hale}, David},
        title = "{The Early Ultraviolet Light-Curves of Type II Supernovae and the Radii of Their Progenitor Stars}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - High Energy Astrophysical Phenomena},
         year = 2023,
        month = oct,
          eid = {arXiv:2310.16885},
        pages = {arXiv:2310.16885},
          doi = {10.48550/arXiv.2310.16885},
archivePrefix = {arXiv},
       eprint = {2310.16885},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv231016885I},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
``` 


analytic model for T,L evolution: **Morag et al 2023** "Shock cooling emission from explosions of red super-giants: I. A numerically calibrated analytic model"
``` 
@ARTICLE{2023MNRAS.522.2764M,
       author = {{Morag}, Jonathan and {Sapir}, Nir and {Waxman}, Eli},
        title = "{Shock cooling emission from explosions of red supergiants - I. A numerically calibrated analytic model}",
      journal = {\mnras},
     keywords = {shock waves, supernovae: general, Astrophysics - High Energy Astrophysical Phenomena},
         year = 2023,
        month = jun,
       volume = {522},
       number = {2},
        pages = {2764-2776},
          doi = {10.1093/mnras/stad899},
archivePrefix = {arXiv},
       eprint = {2207.06179},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.2764M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
``` 
deviations from blackbody: **Morag et al 2024**, "Shock cooling emission from explosions of red super-giants: II. An analytic model of deviations from blackbody emission"
``` 
@ARTICLE{2024MNRAS.528.7137M,
       author = {{Morag}, Jonathan and {Irani}, Ido and {Sapir}, Nir and {Waxman}, Eli},
        title = "{Shock cooling emission from explosions of red supergiants: II. An analytic model of deviations from blackbody emission}",
      journal = {\mnras},
     keywords = {radiation: dynamics, radiative transfer, shock waves, transients: supernovae, Astrophysics - High Energy Astrophysical Phenomena, Astrophysics - Solar and Stellar Astrophysics},
         year = 2024,
        month = mar,
       volume = {528},
       number = {4},
        pages = {7137-7155},
          doi = {10.1093/mnras/stae374},
archivePrefix = {arXiv},
       eprint = {2307.05598},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.7137M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
``` 

Variable validity domain: **Soumagnac et al 2020** "SN 2018fif: The Explosion of a Large Red Supergiant Discovered in Its Infancy by the Zwicky Transient Facility"
``` 
@ARTICLE{2020ApJ...902....6S,
       author = {{Soumagnac}, Maayane T. and {Ganot}, Noam and {Irani}, Ido and {Gal-yam}, Avishay and {Ofek}, Eran O. and {Waxman}, Eli and {Morag}, Jonathan and {Yaron}, Ofer and {Schulze}, Steve and {Yang}, Yi and {Rubin}, Adam and {Cenko}, S. Bradley and {Sollerman}, Jesper and {Perley}, Daniel A. and {Fremling}, Christoffer and {Nugent}, Peter and {Neill}, James D. and {Karamehmetoglu}, Emir and {Bellm}, Eric C. and {Bruch}, Rachel J. and {Burruss}, Rick and {Cunningham}, Virginia and {Dekany}, Richard and {Golkhou}, V. Zach and {Graham}, Matthew J. and {Kasliwal}, Mansi M. and {Konidaris}, Nicholas P. and {Kulkarni}, Shrinivas R. and {Kupfer}, Thomas and {Laher}, Russ R. and {Masci}, Frank J. and {Riddle}, Reed and {Rigault}, Mickael and {Rusholme}, Ben and {van Roestel}, Jan and {Zackay}, Barak},
        title = "{SN 2018fif: The Explosion of a Large Red Supergiant Discovered in Its Infancy by the Zwicky Transient Facility}",
      journal = {\apj},
     keywords = {Supernovae, Type II supernovae, Astronomy data modeling, Observational astronomy, Ultraviolet transient sources, Transient sources, 1668, 1731, 1859, 1145, 1854, 1851, Astrophysics - High Energy Astrophysical Phenomena},
         year = 2020,
        month = oct,
       volume = {902},
       number = {1},
          eid = {6},
        pages = {6},
          doi = {10.3847/1538-4357/abb247},
archivePrefix = {arXiv},
       eprint = {1907.11252},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020ApJ...902....6S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
``` 

If using **Piro 2021** "Shock Cooling Emission from Extended Material Revisited"
``` 
@ARTICLE{2021ApJ...909..209P,
       author = {{Piro}, Anthony L. and {Haynie}, Annastasia and {Yao}, Yuhan},
        title = "{Shock Cooling Emission from Extended Material Revisited}",
      journal = {\apj},
     keywords = {Radiative transfer, Supernovae, 1335, 1668, Astrophysics - High Energy Astrophysical Phenomena},
         year = 2021,
        month = mar,
       volume = {909},
       number = {2},
          eid = {209},
        pages = {209},
          doi = {10.3847/1538-4357/abe2b1},
archivePrefix = {arXiv},
       eprint = {2007.08543},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

``` 

In addition, consider citing the following papers: 
Astropy: <a href="https://doi.org/10.1051/0004-6361/201322068">Astropy collaboration 2013</a>
Astropy: <a href="https://doi.org/10.3847/1538-3881/aabc4f">Astropy collaboration 2018</a>
Matpotlib: <a href="https://doi.org/10.1109/MCSE.2007.55">(hunter et al 2007)</a>
Numpy: <a href="https://doi.org/10.48550/arXiv.cs/0502072">(Oliphant et al 2006)</a> 
Scipy: <a href="https://doi.org/10.1038/s41592-019-0686-2">(Virtanen et al 2020)</a> 
extinction: <a href="https://doi.org/10.5281/zenodo.804967">(Barbary et al 2016)</a> 
dynesty: <a href="https://doi.org/10.1063/1.1835238,">Skilling et al 2004</a>,<a href="https://doi.org/10.1214/06-BA127">Skilling et al 2006</a>,<a href="https://doi.org/10.1111/j.1365-2966.2009.14548.x">Feroz et al 2009</a>,<a href="https://doi.org/10.1007/s11222-018-9844-0">Higson et al 2019</a>,<a href="https://doi.org/10.1093/mnras/staa278">Speagle et al 2020</a>

Spanish Virtual Observatory filter profile service <a href="10.5479/ADS/bib/2012ivoa.rept.1015R">(Rodrigo et al, 2012)</a>








