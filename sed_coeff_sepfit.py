import argparse
import numpy as np
import sys, os
import math
import logging
import time
import galsim
import galsim.roman as roman
import datetime
import fpfs
import pickle
import h5py 

import matplotlib.pyplot as plt

from astropy.io import fits
from matplotlib.colors import LogNorm
from astropy.table import Table
from astropy.visualization import simple_norm
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
import matplotlib.colors as colors

from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
import photerr
import anacal
import psutil
from sklearn.linear_model import RANSACRegressor, LinearRegression


def get_gal_obj (gal_data,i, filter_name ):
    flux_fact = (galsim.SED._h * galsim.SED._c)
    gal_id = str(np.array(gal_data['galaxy_id'])[i])
    bandpass = roman_filters[filter_name]
    redshift = np.array(gal_data['redshift'])[i]
    bulge_hlr = np.array(gal_data['spheroidHalfLightRadiusArcsec'])[i]
    disk_hlr = np.array(gal_data['diskHalfLightRadiusArcsec'])[i]
    disk_shear1, disk_shear2 = np.array(gal_data['diskEllipticity1'])[i], np.array(gal_data['diskEllipticity2'])[i]
    bulge_shear1, bulge_shear2 = np.array(gal_data['spheroidEllipticity1'])[i], np.array(gal_data['spheroidEllipticity2'])[i]
    f_sed = data[gal_id[:9]][gal_id][()].copy()
    bulge_lookup = galsim.LookupTable( x = wave_list/10, f = f_sed[0])
    disk_lookup = galsim.LookupTable( x = wave_list/10, f = f_sed[1])
    knots_lookup = galsim.LookupTable( x = wave_list/10, f = f_sed[2])
    bulge_sed = galsim.SED(bulge_lookup, wave_type = 'nm', flux_type='fnu', redshift = redshift)
    disk_sed = galsim.SED(disk_lookup, wave_type = 'nm', flux_type='fnu', redshift = redshift)
    knots_sed = galsim.SED(knots_lookup, wave_type = 'nm', flux_type='fnu', redshift = redshift)
    bulge = galsim.Sersic(4, half_light_radius=bulge_hlr).shear(g1 =bulge_shear1 , g2 = bulge_shear2)
    disk = galsim.Sersic(1, half_light_radius=disk_hlr).shear(g1 =disk_shear1 , g2 = disk_shear2)
    obj = bulge*bulge_sed + disk*(disk_sed + knots_sed)
    obj = obj.withFlux(np.array(flux_data['roman_flux_' + filter_name])[i]/flux_fact, bandpass)
    tot_sed =  f_sed[0] + f_sed[1] + f_sed[2] 
    tot_lookup = galsim.LookupTable( x = wave_list/10, f = tot_sed)
    tot_sed = galsim.SED(tot_lookup, wave_type = 'nm', flux_type='fnu', redshift = redshift)
    tot_sed = tot_sed.withFlux(np.array(flux_data['roman_flux_' + filter_name])[i]/flux_fact, bandpass)
    return obj, np.array(flux_data['roman_flux_' + filter_name])[i]/flux_fact, tot_sed


def get_star_obj(l, filter_name , num_stars = 40):
    bp500 = galsim.Bandpass(galsim.LookupTable([499, 500, 501], [0, 1, 0]),
                                             wave_type='nm').withZeropoint('AB')
    bandpass = roman_filters[filter_name]
    mag_min, mag_max = get_mag_range(filter_name)
    star_obj = star_data[l*num_stars:(l+1)*num_stars]
    star_seds = []
    #star_flux= star_flux_data[l*num_stars:(l+1)*num_stars]
    for j in range(num_stars):
        star_sed_file = stellar_root + np.array(star_obj['sed_filepath'])[j]
        star_sed = pd.read_csv(star_sed_file, compression='gzip', header=1, sep=' ', quotechar='"')
        wavelength = np.array(star_sed)[:,0]
        SED = np.array(star_sed)[:,1]
        lookup_table = galsim.LookupTable(wavelength, SED)
        sed =  galsim.SED(lookup_table, 'nm', 'flambda')#.withFluxDensity(target_flux_density=1.0, wavelength=500))

        magnorm = np.array(star_obj['magnorm'])[j]
        #MW_av = star_obj[j: j+1]['MW_av']
        #MW_rv = star_obj[j: j+1]star_data['MW_rv']
        flux_500 = np.exp(-0.9210340371976184 * magnorm)
        sed = sed.withMagnitude(0, bp500)
        sed = sed*flux_500
        star_mag = sed.calculateMagnitude(bandpass)
        #SED = ((wavelength)*5e-9 + 1e-3)/(wavelength)*(galsim.SED._h* galsim.SED._c)
        #lookup_table = galsim.LookupTable(wavelength, SED)
        #sed =  galsim.SED(lookup_table, 'nm', 'flambda')
        if star_mag < mag_max and star_mag > mag_min:
            star_seds.append(sed)
    # The rng for photon shooting should be different for each filter.
    if len(star_seds) == 0:
            star_seds.append(sed)
    return star_seds

def get_mag_range(filter_name):
    if filter_name == 'Z087':
         return 17.85, 21.37
    if filter_name == 'Y106':
         return 17.9, 21.2
    if filter_name == 'J129':
        return 17.9, 21.2
    if filter_name == 'H158':
        return 17.9, 21.2
    if filter_name == 'F184':
        return 17.2, 20.8
    if filter_name == 'K213':
        return 17.0, 20.6
    if filter_name == 'W146':
        return 19.1, 21.2


def get_coeff(indices, filters):
    star_coeff = {}
    gal_coeff = {}
    for filter_name in filters:
        filt_star_coeff = np.zeros(len(indices))
        filt_gal_coeff = np.zeros(len(indices))
        bandpass = roman_filters[filter_name]
        for l in range(len(indices)):
        #for l in range(10):
            i = indices[l]
            reverse_idx = int(((9999 - l)/250))*(250) + (l%250)
            obj, flux, sed_f = get_gal_obj(gal_data,i, filter_name )
            star_objs = get_star_obj(reverse_idx,filter_name, num_stars = 40)
            
            obj_wave = sed_f._orig_spec.x
            obj_f = sed_f._orig_spec.f
            obj_phot_sed = sed_f._flux_to_photons_fnu(obj_f, obj_wave)
            obj_wave = obj_wave * (1 + sed_f.redshift)
            obj_flux = sed_f.calculateFlux(bandpass)
            
            #native_wave = star_objs[0]._orig_spec.x
            obj_look = galsim.LookupTable(obj_wave,obj_phot_sed )
            #wave_cut = np.logical_and(native_wave > bandpass.blue_limit - 10,native_wave < bandpass.red_limit + 10 )
            #obj_phot_sed = obj_look(native_wave[wave_cut])
            #obj_wave = native_wave[wave_cut]
            native_wave = np.linspace(bandpass.blue_limit, bandpass.red_limit, 1000)
            obj_phot_sed = obj_look(native_wave)
            obj_wave = native_wave.copy()#[wave_cut]
            sed_f = galsim.SED(obj_look, wave_type = 'nm', flux_type='fphotons', redshift = 0)

            native_wave = star_objs[0]._orig_spec.x
            native_f = star_objs[0]._orig_spec.f
            star_avg_sed = star_objs[0]._flux_to_photons_flam(native_f, native_wave)
            star_avg = star_objs[0] / len(star_objs)

            for j in range(1, len(star_objs)):
                native_wave = star_objs[j]._orig_spec.x
                native_f = star_objs[j]._orig_spec.f
                star_avg_sed += star_objs[j]._flux_to_photons_flam(native_f, native_wave)
                star_avg += star_objs[j] / len(star_objs)

            star_avg_sed /= len(star_objs)
            star_avg_flux = star_avg.calculateFlux(bandpass)
            
            # Filter the data within the specified x-limits
            obj_mask = (obj_wave >= bandpass.blue_limit ) & (obj_wave <= bandpass.red_limit)
            native_mask = (native_wave >= bandpass.blue_limit - 10) & (native_wave <= bandpass.red_limit + 10)

            # Filtered data
            bp_0 = bandpass.effective_wavelength
            filtered_obj_wave = obj_wave[obj_mask] - bp_0
            filtered_obj_phot_sed = obj_phot_sed[obj_mask] / obj_flux
            #filtered_obj_phot_sed = smooth_gaussian(filtered_obj_phot_sed, 1e4)
            filtered_native_wave = native_wave[native_mask] - bp_0
            filtered_star_avg_sed = star_avg_sed[native_mask] / star_avg_flux

            # Reshape data for sklearn (needs 2D arrays)
            filtered_obj_wave_2d = filtered_obj_wave.reshape(-1, 1)
            filtered_native_wave_2d = filtered_native_wave.reshape(-1, 1)

            # RANSAC linear fit for the filtered object data
            ransac_obj = RANSACRegressor(LinearRegression(), min_samples=len(filtered_obj_phot_sed), residual_threshold=1e5, random_state = 0)
            ransac_obj.fit(filtered_obj_wave_2d, filtered_obj_phot_sed, sample_weight = bandpass(obj_wave))
            obj_fit = ransac_obj.predict(filtered_obj_wave_2d)
            obj_slope, obj_intercept = ransac_obj.estimator_.coef_[0], ransac_obj.estimator_.intercept_

            # RANSAC linear fit for the filtered star data
            ransac_star = RANSACRegressor(LinearRegression(), min_samples=len(filtered_star_avg_sed), residual_threshold=1e5, random_state = 0)
            ransac_star.fit(filtered_native_wave_2d, filtered_star_avg_sed)
            star_fit = ransac_star.predict(filtered_native_wave_2d)
            star_slope, star_intercept = ransac_star.estimator_.coef_[0], ransac_star.estimator_.intercept_

            #ransac_obj.estimator_.coef_[0] = (true_prof[filter_name]/(basis_prof[filter_name]))[0] + star_slope
            filt_gal_coeff[l] = obj_slope
            filt_star_coeff[l] = star_slope
        star_coeff[filter_name] = filt_star_coeff
        gal_coeff[filter_name] = filt_gal_coeff
    return star_coeff, gal_coeff

def get_pos():
    #pickle_in = open('10000Gal_NoNoise_J129_RealisticSEDs_g10_g20.pickle', "rb")
    with open('10000Gal_NoNoise_J129_RealisticSEDs2_g10_g20.pickle', "rb") as pickle_in:
        # Load only necessary parts if possible
        data = pickle.load(pickle_in)
        dict_J129 = data['gal_df'][['SCA', 'x_pos', 'y_pos', 'redshift', 'indices']]
    #dict_J129 = pickle.load(pickle_in)['gal_df'][['SCA', 'x_pos', 'y_pos']]
    pickle_in.close()
    SCAs = dict_J129['SCA'].to_numpy()
    x_pos = (dict_J129['x_pos']).to_numpy()
    y_pos = (dict_J129['y_pos']).to_numpy()
    z = (dict_J129['redshift']).to_numpy()
    indices = (dict_J129['indices']).to_numpy()
    del dict_J129
    del data
    return SCAs, x_pos, y_pos, z, indices
    



use_filters = 'ZYJHFW'
roman_filters = roman.getBandpasses(AB_zeropoint=True)

# Get the names of the ones we will use here.
filters = [filter_name for filter_name in roman_filters if filter_name[0] in use_filters]


blue_limit, red_limit, eff_wave = np.zeros(6), np.zeros(6), np.zeros(6)
for i in range(6):

    bandpass = roman_filters[filters[i]]
    eff_wave[i] = bandpass.effective_wavelength
    blue_limit[i], red_limit[i] = bandpass.blue_limit, bandpass.red_limit
    sampling_rate = (eff_wave[i]/1e3)*7.9/10
    #print(filters[i] + ': %.2f'%sampling_rate )
    

## galaxy catalog
gal_data = pd.read_parquet('galaxy_10050.parquet', engine='pyarrow')
flux_data = pd.read_parquet('galaxy_flux_10050.parquet', engine='pyarrow')

z = gal_data['redshift']
zp=roman.getBandpasses()['J129'].zeropoint
mag=-2.5*np.log10(flux_data['roman_flux_J129'])+zp

mag_cut = mag < 26#np.logical_and(mag < 24.5, z < 1)

gal_data = gal_data[mag_cut]
flux_data = flux_data[mag_cut]
list_len = len(flux_data)

#import h5py 
filename = "galaxy_sed_10050.hdf5"

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    #print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    #print(type(f[a_group_key])) 

    # If a_group_key is a group name, 
    # this gets the object names in the group and returns as a list
    data = list(f[a_group_key])

    # If a_group_key is a dataset name, 
    # this gets the dataset values and returns as a list
    data = list(f[a_group_key])
    # preferred methods to get dataset values:
    ds_obj = f[a_group_key]      # returns as a h5py dataset object
    #ds_arr = f[a_group_key][()]  # returns as a numpy array
filename = "galaxy_sed_10050.hdf5"
f = h5py.File(filename, "r")
data = f['galaxy']
wave_list = f['meta']['wave_list'][()]


lst_files = os.listdir('stellar_SED/')
lst_files = [s for s in lst_files if s[12] == '1']
comb_star_data = pd.read_parquet('stellar_SED/' + lst_files[0], engine='pyarrow')
for file in lst_files[1:]:
    temp_star_data = pd.read_parquet('stellar_SED/' + file, engine='pyarrow')
    comb_star_data = pd.concat([comb_star_data,temp_star_data], ignore_index=True)
    
kurucz_mask = np.zeros(len(comb_star_data), dtype = bool)
for i in range(len(comb_star_data)):
    sed_file = comb_star_data['sed_filepath']
    if sed_file[i][8] == 'k':
        kurucz_mask[i] = True
        
star_data= comb_star_data[kurucz_mask]
stellar_root = '/hildafs/projects/phy200017p/share/tq_share/'
stellar_dir = ['gizis_SED' , 'kurucz' , 'mlt' , 'old_mlt' , 'phoSimMLT' , 'wDs']


filter_name = 'W146'
SCAs, x_pos, y_pos, z, indices = get_pos()
star_coeff, gal_coeff = get_coeff(indices, [filter_name])
coeff = {'star_coeff':star_coeff, 'gal_coeff':gal_coeff }

pickle_out = open("SEDcoeff_sepfit_Final_Sims_10000Gal_" + filter_name ,"wb")
pickle.dump(coeff, pickle_out)
pickle_out.close()
