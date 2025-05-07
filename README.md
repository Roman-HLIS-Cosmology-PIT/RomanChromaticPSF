# Chromatic Effects on PSF and Shear Measurement for Roman

## Introduction

The software and data were used for the research that is described in this paper:
 - Chromatic Effects on the PSF and Shear Measurement for the Roman Space Telescope High-Latitude Wide Area Survey; Berlfein et al. (2025), submitted - [ADS entry](https://ui.adsabs.harvard.edu/abs/2025arXiv250500093B/abstract)


This repository includes software to:
 - Generate Roman-like image simulations of the [OpenUniverse2024 catalog](https://irsa.ipac.caltech.edu/data/theory/openuniverse2024/overview.html) or the [cosmoDC2 catalog](https://portal.nersc.gov/project/lsst/cosmoDC2/_README.html) using [GalSim](https://github.com/GalSim-developers/GalSim).
 - Generate catalog-level magnitude noise using [PhotErr](https://github.com/jfcrenshaw/photerr).
 - Perform shear measurement on galaxy images using [AnaCal](https://github.com/mr-superonion/AnaCal).
 - Correct for SED-dependent chromatic effects on the PSF.

Note that in order to generate the actual image simulations, the relevant full extragalactic and stellar catalogs are not provided here and must be downloaded. We provide a subset of the catalog, with the relevant object IDs, true and noisy magnitudes, along with their errors, for the Roman (YJHFW) and LSST (ugrizy) bands. 


## Using this software

The set of notebooks and scripts can be run after cloning the repo:
```
git clone https://github.com/FedericoBerlfein/RomanChromaticPSF.git
```

## Guide to repository contents
Below is an overview of the contents of this repository. More details on each folder can be found in each folder's respective README. 

- ``Gal_props_catnoise``: This folder contains the catalog-level noisy and true magnitudes, magnitude errors, object IDs, and redshift, for every object used in the paper. 
- ``PaperFigures``: All the figures used in the paper.
- ``SED_fit_coeff``: The scripts used to fit the SEDs, as well as the results of the fits.
- ``gen_sims``: The scripts used to generate the image simulations and do the shape measurement.
- ``notebooks``: Some of the notebooks needed to either generate some of the intermediate data, analyze results, and plot relevant figures.
- ``shear_measurement_cosmoDC2/diffsky``: The measured shapes from the image simulations for different PSFs or correction schemes.
- ``sim_images``: Empty directory given that simulated images are too heavy to upload to Github.  Nevertheless, this is where there simulated images would live after running relevant scripts in ``gen_sims``.



## Contact us

In case you need help or have questions about the code, feel free to contact Federico Berlfein (fberlfei "at" andrew.cmu.edu) or use the [issues](https://github.com/FedericoBerlfein/RomanChromaticPSF/issues) on this repository.


## License

The code has been publicly released; it is available under the terms of our [LICENSE](LICENSE).

If you make use of the software, please cite the paper listed at the top of this README, and provide a link to this code repository.