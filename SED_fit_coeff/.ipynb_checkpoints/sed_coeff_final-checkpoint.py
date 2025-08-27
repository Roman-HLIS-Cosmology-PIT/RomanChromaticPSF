import argparse
import numpy as np
import sys, os
import math
import logging
import time
import galsim
import galsim.roman as roman
import datetime
import pickle
import h5py 
import pandas as pd



from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import h5py 
import gzip



def get_mags_avg(gal_props ,err_model, indx_seed = 9999, use_filters = None, point = False):
    obs_mags = {}
    true_mags = {}
    err_mags = {}
    if use_filters is None:
        use_filters = filters
    for filter_name in use_filters:
        #print(filter_name)
        rng = np.random.default_rng(indx_seed)
        band = roman_filters[filter_name]
        band_prop_b = gal_props[filter_name]
        if point is False:
            a_tot,b_tot, mag_true_tot = band_prop_b['a'], band_prop_b['b'], band_prop_b['mag_true']
            d = {'major':a_tot, 'minor': b_tot, filter_name:mag_true_tot }
        else:
            mag_true_tot = band_prop_b['mag_true']
            d = {filter_name:mag_true_tot }    
        df = pd.DataFrame(data=d)
        df_all_obs = err_model(df, random_state = rng)
        obs_mag, mag_err = df_all_obs[filter_name].to_numpy(), df_all_obs[filter_name + '_err'].to_numpy()
        
        for i in range(len(mag_err)):
            if obs_mag[i] >= 100:
                #print(i)
                #mag_err[i] = 50
                gal_props_test = {}
                if point is False:
                    a_bulge,b_bulge, mag_true_bulge = np.array([a_tot[i]]), np.array([b_tot[i]]), np.array([mag_true_tot[i]])
                    gal_props_test[filter_name] = {'a': a_bulge, 'b': b_bulge, 'mag_true':mag_true_bulge }
                else:
                    mag_true_bulge = np.array([mag_true_tot[i]])
                    gal_props_test[filter_name] = {'mag_true':mag_true_bulge }
                true_mags_, obs_mags_, err_mags_ = get_mags_avg(gal_props_test, err_model,indx_seed = indx_seed+1, use_filters = [filter_name])
                mag_err[i] = err_mags_[filter_name][0]
                obs_mag[i] = obs_mags_[filter_name][0]
        
        obs_mags[filter_name] = obs_mag
        err_mags[filter_name] = mag_err
        true_mags[filter_name] = mag_true_tot
    return true_mags, obs_mags, err_mags




def get_gal_obj_cd2 (i, filter_name ):
    bp500 = galsim.Bandpass(galsim.LookupTable([499, 500, 501], [0, 1, 0]),
                                             wave_type='nm').withZeropoint('AB')
    gal_id = str(np.array(gal_data['id'])[i])
    bandpass = roman_filters[filter_name]
    redshift = np.array(gal_data['redshift'])[i]
    f_sed_disk_name = sed_dict['disk_sed'][gal_id].copy()
    f_sed_bulge_name = sed_dict['bulge_sed'][gal_id].copy()
    f_disk_magnorm = sed_dict['disk_magnorm'][gal_id].copy()
    f_bulge_magnorm = sed_dict['bulge_magnorm'][gal_id].copy()
    
    f_sed_disk = sed_f_dict[f_sed_disk_name]
    f_sed_bulge = sed_f_dict[f_sed_bulge_name]
    
    disk_lk = galsim.LookupTable( x = waves, f = f_sed_disk)
    bulge_lk = galsim.LookupTable( x = waves, f = f_sed_bulge)
    disk_sed =  galsim.SED(disk_lk, 'nm', 'flambda', redshift = redshift).withMagnitude(f_disk_magnorm,bp500 )
    bulge_sed =  galsim.SED(bulge_lk, 'nm', 'flambda', redshift = redshift).withMagnitude(f_bulge_magnorm,bp500 )
    
    f_sed = disk_sed._orig_spec.f + bulge_sed._orig_spec.f 
    tot_sed =  f_sed
    tot_lookup = galsim.LookupTable( x = waves, f = tot_sed)
    tot_sed = galsim.SED(tot_lookup, wave_type = 'nm', flux_type='flambda', redshift = redshift)
    tot_sed = tot_sed.withFlux(np.array(flux_data['roman_flux_' + filter_name])[i], bandpass)
    return np.array(flux_data['roman_flux_' + filter_name])[i], tot_sed

def get_gal_obj (i, filter_name ):
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
    obj = obj.withFlux(np.array(flux_data['roman_flux_' + filter_name])[i], bandpass)
    tot_sed =  f_sed[0] + f_sed[1] + f_sed[2] 
    tot_lookup = galsim.LookupTable( x = wave_list/10, f = tot_sed)
    tot_sed = galsim.SED(tot_lookup, wave_type = 'nm', flux_type='fnu', redshift = redshift)
    tot_sed = tot_sed.withFlux(np.array(flux_data['roman_flux_' + filter_name])[i], bandpass)
    return np.array(flux_data['roman_flux_' + filter_name])[i], tot_sed


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
        if star_snr[filter_name][l*num_stars + j] > 100 and star_mag > mag_min:
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
        return 19.1, 22.4
    
def m5_scenario(mode = 'reference', sky = 'dark'):
    if mode == 'reference':
        if sky == 'dark':
            m5 = {'Y106': 25.58, 'J129': 25.57, 'H158': 25.56, 'F184': 25.01 ,'K213': 23.96}
        if sky == 'bright':
            m5 = {'Y106': 25.32 ,'J129': 25.31 ,'H158': 25.31, 'F184': 24.85,  'K213': 23.94}
    elif mode == 'fast':
        if sky == 'dark':
            m5 = {'Y106': 25.10,  'J129': 25.09,  'H158': 25.08,  'F184': 24.53, 'K213': 23.67}
        if sky == 'bright':
            m5 = {'Y106': 24.92, 'J129': 24.91, 'H158': 24.91, 'F184': 24.42 ,'K213': 23.65}
    return m5

def Hcut_SN18(scenario = 'A'):
    if scenario == 'A':
        return 24.96
    elif scenario == 'B':
        return 23.93
    elif scenario == 'C':
        return 24.31
    elif scenario == 'D':
        return 24.49
    elif scenario == 'E':
        return 24.31
    elif scenario == 'F':
        return 24.31
    
def scenario_mode(scenario = 'A'):
    if scenario == 'A':
        return 'reference', 'dark'
    elif scenario == 'B':
        return 'fast', 'bright'
    elif scenario == 'C':
        return 'fast', 'bright'
    elif scenario == 'D':
        return 'fast', 'dark'
    elif scenario == 'E':
        return 'fast', 'bright'
    elif scenario == 'F':
        return 'fast', 'bright'


def get_coeff(filters):
    coeff0 = {}
    coeff = {}
    coeff2 = {}
    coeff3 = {}
    flux_all = {}
    
    coeff0_s = {}
    coeff_s = {}
    coeff2_s = {}
    coeff3_s = {}
    flux_all_s = {}
    for filter_name in filters:
        filt_coeff0 = np.zeros(num_train)
        filt_coeff = np.zeros(num_train)
        filt_coeff2 = np.zeros(num_train)
        filt_coeff3 = np.zeros(num_train)
        filt_flux = np.zeros(num_train)
        
        filt_coeff0_s = np.zeros(num_train)
        filt_coeff_s = np.zeros(num_train)
        filt_coeff2_s = np.zeros(num_train)
        filt_coeff3_s = np.zeros(num_train)
        filt_flux_s = np.zeros(num_train)
        bandpass = roman_filters[filter_name]
        for l in range(num_train):
        #for l in range(10):
            i = l
            reverse_idx = l
            flux, sed_f = get_gal_obj(gal_data,i, filter_name )
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
            native_wave = np.linspace(bandpass.blue_limit, bandpass.red_limit, 2000)
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
            polynomial_features = PolynomialFeatures(degree=1)
            X_poly = polynomial_features.fit_transform(filtered_obj_wave_2d)
            ransac_obj = RANSACRegressor(LinearRegression(), min_samples=len(filtered_obj_phot_sed), residual_threshold=1e5, random_state = 0)
            ransac_obj.fit(X_poly, filtered_obj_phot_sed, sample_weight = bandpass(obj_wave))
            obj_fit = ransac_obj.predict(X_poly)
            obj_2nd, obj_slope, obj_intercept =  ransac_obj.estimator_.coef_[1], ransac_obj.estimator_.coef_[1], ransac_obj.estimator_.intercept_

            # RANSAC linear fit for the filtered star data
            polynomial_features = PolynomialFeatures(degree=1)
            X_poly = polynomial_features.fit_transform(filtered_native_wave_2d)
            ransac_star = RANSACRegressor(LinearRegression(), min_samples=len(filtered_star_avg_sed), residual_threshold=1e5, random_state = 0)
            ransac_star.fit(X_poly, filtered_star_avg_sed, sample_weight = bandpass(native_wave[native_mask]))
            star_fit = ransac_star.predict(X_poly)
            star_2nd,star_slope, star_intercept =  ransac_star.estimator_.coef_[1], ransac_star.estimator_.coef_[1], ransac_star.estimator_.intercept_

            #ransac_obj.estimator_.coef_[0] = (true_prof[filter_name]/(basis_prof[filter_name]))[0] + star_slope
            filt_coeff0[l] = obj_intercept #- star_intercept
            filt_coeff[l] = obj_slope #- star_slope
            filt_coeff2[l] = obj_2nd #- star_2nd
            filt_flux[l] = obj_flux
            
            filt_coeff0_s[l] =  star_intercept
            filt_coeff_s[l] =  star_slope
            filt_coeff2_s[l] =  star_2nd
            filt_flux_s[l] = star_avg_flux
            #filt_coeff3[l] = obj_3rd - star_3rd
        coeff0[filter_name] = filt_coeff0
        coeff[filter_name] = filt_coeff
        coeff2[filter_name] = filt_coeff2
        flux_all[filter_name] = filt_flux
        
        coeff0_s[filter_name] = filt_coeff0_s
        coeff_s[filter_name] = filt_coeff_s
        coeff2_s[filter_name] = filt_coeff2_s
        flux_all_s[filter_name] = filt_flux_s
        #coeff3[filter_name] = filt_coeff3
    return [coeff0, coeff, coeff2 ],[coeff0_s, coeff_s, coeff2_s], [flux_all, flux_all_s] #, coeff3

def get_pos():
    #pickle_in = open('10000Gal_NoNoise_J129_RealisticSEDs_g10_g20.pickle', "rb")
    with open('sim_images/10000Gal_NoNoise/J129/gal_info.pickle', "rb") as pickle_in:
        # Load only necessary parts if possible
        data = pickle.load(pickle_in)
        dict_J129 = data[['SCA', 'x_pos', 'y_pos', 'redshift', 'indices']]
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



from concurrent.futures import ProcessPoolExecutor

def process_single_galaxy(l, filter_name, bandpass):
    i = l
    reverse_idx = l

    # Get galaxy and star objects for the current galaxy index
    if is_cosmoDC2:
        flux, sed_f = get_gal_obj_cd2(i, filter_name)
    else:
        flux, sed_f = get_gal_obj(i, filter_name)
    
    star_objs = get_star_obj(0, filter_name, num_stars=40)
    if reverse_idx < 10000:
        star_objs = get_star_obj(reverse_idx, filter_name, num_stars=40)
    #if is_deep:
    #    if reverse_idx >= 10000:
    #        star_objs = get_star_obj(0, filter_name, num_stars=40)
    #else:
    #    star_objs = get_star_obj(reverse_idx, filter_name, num_stars=40)
    # Object SED calculation
    obj_wave = sed_f._orig_spec.x
    obj_f = sed_f._orig_spec.f
    obj_phot_sed = sed_f._flux_to_photons_fnu(obj_f, obj_wave)
    if is_cosmoDC2:
        obj_phot_sed = sed_f._flux_to_photons_flam(obj_f, obj_wave)
    obj_wave = obj_wave * (1 + sed_f.redshift)
    obj_flux = sed_f.calculateFlux(bandpass)

    obj_look = galsim.LookupTable(obj_wave, obj_phot_sed)
    native_wave = np.linspace(bandpass.blue_limit, bandpass.red_limit, 2000)
    obj_phot_sed = obj_look(native_wave)
    obj_wave = native_wave.copy()

    # Star SED calculation and averaging
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

    # Filter the data within the wavelength limits
    obj_mask = (obj_wave >= bandpass.blue_limit) & (obj_wave <= bandpass.red_limit)
    native_mask = (native_wave >= bandpass.blue_limit) & (native_wave <= bandpass.red_limit)

    # Filtered data
    bp_0 = bandpass.effective_wavelength
    filtered_obj_wave = obj_wave[obj_mask] - bp_0
    filtered_obj_phot_sed = obj_phot_sed[obj_mask] / obj_flux
    filtered_native_wave = native_wave[native_mask] - bp_0
    filtered_star_avg_sed = star_avg_sed[native_mask] / star_avg_flux

    # Reshape data for sklearn (requires 2D arrays)
    filtered_obj_wave_2d = filtered_obj_wave.reshape(-1, 1)
    filtered_native_wave_2d = filtered_native_wave.reshape(-1, 1)

    # RANSAC linear fit for object data
    polynomial_features = PolynomialFeatures(degree=1)
    X_poly = polynomial_features.fit_transform(filtered_obj_wave_2d)
    ransac_obj = RANSACRegressor(LinearRegression(), min_samples=len(filtered_obj_phot_sed), residual_threshold=1e5, random_state=0)
    ransac_obj.fit(X_poly, filtered_obj_phot_sed, sample_weight=bandpass(obj_wave))
    obj_intercept, obj_slope, obj_2nd = ransac_obj.estimator_.intercept_, ransac_obj.estimator_.coef_[1], ransac_obj.estimator_.coef_[1]

    # RANSAC linear fit for star data
    X_poly_star = polynomial_features.fit_transform(filtered_native_wave_2d)
    ransac_star = RANSACRegressor(LinearRegression(), min_samples=len(filtered_star_avg_sed), residual_threshold=1e5, random_state=0)
    ransac_star.fit(X_poly_star, filtered_star_avg_sed, sample_weight=bandpass(native_wave[native_mask]))
    star_intercept, star_slope, star_2nd = ransac_star.estimator_.intercept_, ransac_star.estimator_.coef_[1], ransac_star.estimator_.coef_[1]

    return obj_intercept, obj_slope, obj_2nd, obj_flux, star_intercept, star_slope, star_2nd, star_avg_flux


def get_coeff_parallel(filters):
    coeff0 = {}
    coeff = {}
    coeff2 = {}
    flux_all = {}

    coeff0_s = {}
    coeff_s = {}
    coeff2_s = {}
    flux_all_s = {}

    for filter_name in filters:
        filt_coeff0 = np.zeros(num_train)
        filt_coeff = np.zeros(num_train)
        filt_coeff2 = np.zeros(num_train)
        filt_flux = np.zeros(num_train)

        filt_coeff0_s = np.zeros(num_train)
        filt_coeff_s = np.zeros(num_train)
        filt_coeff2_s = np.zeros(num_train)
        filt_flux_s = np.zeros(num_train)

        bandpass = roman_filters[filter_name]

        # Parallel processing over galaxies using ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            results = executor.map(process_single_galaxy, range(num_train), [filter_name] * num_train, [bandpass] * num_train)

        # Collect the results
        for l, result in enumerate(results):
            obj_intercept, obj_slope, obj_2nd, obj_flux, star_intercept, star_slope, star_2nd, star_avg_flux = result
            filt_coeff0[l], filt_coeff[l], filt_coeff2[l], filt_flux[l] = obj_intercept, obj_slope, obj_2nd, obj_flux
            filt_coeff0_s[l], filt_coeff_s[l], filt_coeff2_s[l], filt_flux_s[l] = star_intercept, star_slope, star_2nd, star_avg_flux

        # Store the results for each filter
        coeff0[filter_name], coeff[filter_name], coeff2[filter_name], flux_all[filter_name] = filt_coeff0, filt_coeff, filt_coeff2, filt_flux
        coeff0_s[filter_name], coeff_s[filter_name], coeff2_s[filter_name], flux_all_s[filter_name] = filt_coeff0_s, filt_coeff_s, filt_coeff2_s, filt_flux_s

    return [coeff0, coeff, coeff2], [coeff0_s, coeff_s, coeff2_s], [flux_all, flux_all_s]


def get_galSED(file_name):
    with gzip.open(file_name, 'rb') as f:
        file_content = f.read()
    # Split the content by lines
    lines = file_content.splitlines()

    # Extract the data lines (skip the first line which is the header)
    data_lines = lines[1:]

    # Create a pandas DataFrame from the data lines
    # Use a space as the delimiter (split each line into two columns)
    data = [line.split() for line in data_lines]

    # Create a DataFrame and define column names
    df = pd.DataFrame(data, columns=['Wavelength (nm)', 'F_lamA (normalized erg/cm2/s/A)'])
    wave = pd.to_numeric(df['Wavelength (nm)'])
    f_sed = pd.to_numeric(df['F_lamA (normalized erg/cm2/s/A)'])
    return np.array(wave), np.array(f_sed)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run SED fits.')
parser.add_argument('--scenario', type=str, default='A', help='Name of survey scenario.')


args = parser.parse_args()
scenario = args.scenario


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
    


# get noisy fluxes
use_filters = ['Y106', 'J129', 'H158', 'F184', 'W146']
mode, sky  = scenario_mode(scenario)
is_cosmoDC2 = True
is_deep =  True

num_ = "10000"
cat_file = 'roman_gal_obsmags_diffsky_10307_' + num_ +'cut'
if is_deep:
    num_ = "40000"
    cat_file = 'roman_gal_obsmags_diffsky_10307_deep_' + num_ +'cut'
if is_cosmoDC2:
    cat_file = 'roman_gal_obsmags_cosmodc2_10067_' + num_ + 'cut'
    if is_deep:
        cat_file = 'roman_gal_obsmags_cosmodc2_10067_deep_' + num_ + 'cut'
        
pickle_out = open("Gal_props_catnoise/" + cat_file,"rb")
roman_mag_info = pickle.load(pickle_out)
pickle_out.close()

gal_ids = roman_mag_info['gal_id']
z = roman_mag_info['z']
obs_mags =  roman_mag_info['obs_mag'][scenario]
mag_cut,rand_idx = roman_mag_info['cut'], roman_mag_info['rand_ind']
if is_deep:
    mag_cut,rand_idx = mag_cut[scenario],rand_idx[scenario] 

pickle_out = open("Gal_props_catnoise/roman_star_obsmags_diffsky","rb")
roman_star_mag_info = pickle.load(pickle_out)
pickle_out.close()
star_snr = roman_star_mag_info['snr'][scenario]


# read catalog and make cuts    
#data_dir = '/hildafs/projects/phy200017p/share/euclid_sim/input_catalog/roman_rubin_cats_v1.1.2_faint/'
ext = ''
#data_dir = 'cosmoDC2/'
if is_cosmoDC2:
    data_dir = 'cosmoDC2/'
    ext = '_subset_fullsed'
    cat = 'cosmoDC2'
    healpix_file = '10067'
else:
    data_dir = '/hildafs/projects/phy200017p/share/euclid_sim/input_catalog/roman_rubin_cats_v1.1.2_faint/'
    ext = ''
    cat = 'diffsky'
    healpix_file = '10307' ## 10307 for diffsky and 10067 for cosmoDC2
gal_data = pd.read_parquet(data_dir + 'galaxy_'+ healpix_file + ext +'.parquet', engine='pyarrow')
flux_data = pd.read_parquet(data_dir +'galaxy_flux_'+ healpix_file + ext + '.parquet', engine='pyarrow')

gal_data = gal_data[mag_cut].reset_index(drop=True)
flux_data = flux_data[mag_cut].reset_index(drop=True)

indx_seed = 1234
#np.random.seed(indx_seed)
#train_indices = np.random.choice(np.arange(list_len), num_g, replace = False)
gal_data = gal_data.iloc[rand_idx].reset_index(drop=True)
flux_data = flux_data.iloc[rand_idx].reset_index(drop=True)
list_len = len(flux_data)

num_train = list_len


## reading SEDs

if not is_cosmoDC2:
    filename = data_dir +"galaxy_sed_" + healpix_file + ".hdf5"
    f = h5py.File(filename, "r")
    data = f['galaxy']
    wave_list = f['meta']['wave_list'][()]
    
if is_cosmoDC2:
    filename = 'cosmoDC2/' +"sed_fit_"+ healpix_file +".h5"
    f = h5py.File(filename, "r")
    gal_id = f['galaxy_id'][()].astype(str)
    disk_sed_idx = f['disk_sed'][()]
    bulge_sed_idx = f['bulge_sed'][()]
    disk_sed_name = f['sed_names'][()][disk_sed_idx]
    bulge_sed_name = f['sed_names'][()][bulge_sed_idx]
    disk_magnorm = f['disk_magnorm'][()][3]
    bulge_magnorm = f['bulge_magnorm'][()][3]

    disk_sed_name = np.array([s.decode('utf-8') for s in disk_sed_name])
    bulge_sed_name = np.array([s.decode('utf-8') for s in bulge_sed_name])

    #disk_magnorm[disk_magnorm == np.inf] = 200
    #bulge_magnorm[bulge_magnorm == np.inf] = 200

    disk_dict = dict(zip(gal_id, disk_sed_name))
    bulge_dict = dict(zip(gal_id, bulge_sed_name))
    disk_dict_magnorm = dict(zip(gal_id, disk_magnorm))
    bulge_dict_magnorm = dict(zip(gal_id, bulge_magnorm ))
    sed_dict = {}
    sed_dict['disk_sed'] = disk_dict
    sed_dict['bulge_sed'] = bulge_dict
    sed_dict['disk_magnorm'] = disk_dict_magnorm
    sed_dict['bulge_magnorm'] = bulge_dict_magnorm

    lst_files = os.listdir('galaxySED/')
    file_names = []
    gal_SEDs = [] 
    for file_ in lst_files:
        file_name = os.path.join('galaxySED/',file_)
        file_names.append(file_name)
        wave, f_sed = get_galSED(file_name)
        if len(wave) == 6900: 
            f_sed = f_sed[1:]
            wave = wave[1:]
        gal_SEDs.append(f_sed)
    waves = wave
    sed_f_dict = dict(zip(file_names, gal_SEDs))




# read star catalog
lst_files = os.listdir('stellar_SED/')
lst_files = [s for s in lst_files if s[12] == '1']
lst_flux_files = []
for lst_file in lst_files:
    lst_flux_files.append(lst_file[:11] +'_flux' + lst_file[11:])
comb_star_data = pd.read_parquet('stellar_SED/' + lst_files[0], engine='pyarrow')
comb_star_flux_data = pd.read_parquet('rubin_roman_star_files/' + lst_flux_files[0], engine='pyarrow')
for file in lst_files[1:]:
    temp_star_data = pd.read_parquet('stellar_SED/' + file, engine='pyarrow')
    comb_star_data = pd.concat([comb_star_data,temp_star_data], ignore_index=True)
for file in lst_flux_files[1:]:
    temp_star_data = pd.read_parquet('rubin_roman_star_files/' + file, engine='pyarrow')
    comb_star_flux_data = pd.concat([comb_star_flux_data,temp_star_data], ignore_index=True)
    
kurucz_mask = np.zeros(len(comb_star_data), dtype = bool)
for i in range(len(comb_star_data)):
    sed_file = comb_star_data['sed_filepath']
    if sed_file[i][8] == 'k':
        kurucz_mask[i] = True
        
star_data= comb_star_data[kurucz_mask]
star_flux_data= comb_star_flux_data[kurucz_mask]
stellar_root = '/hildafs/projects/phy200017p/share/tq_share/'
stellar_dir = ['gizis_SED' , 'kurucz' , 'mlt' , 'old_mlt' , 'phoSimMLT' , 'wDs']


#plot_mags(true_mags, obs_mags, err_mags)


#filter_name = 'Y106'
#SCAs, x_pos, y_pos, z, indices = get_pos()
#coeff, coeff_s, fluxes= get_coeff([filter_name])
#coeff, coeff_s, fluxes= get_coeff(use_filters)
coeff, coeff_s, fluxes= get_coeff_parallel(use_filters)

pickle_out = open("final_run/10000Gal_1ordfit_"+scenario+ "_YJHFW_"+ cat + "_deep_" + healpix_file ,"wb")
pickle.dump({'gal': coeff, 'star:': coeff_s, 'gal_fluxes':fluxes[0], 'star_fluxes':fluxes[1],
             'gal_id':gal_ids, 'z':z }, pickle_out)
pickle_out.close()



