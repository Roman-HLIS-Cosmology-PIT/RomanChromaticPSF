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


from astropy.io import fits
from matplotlib.colors import LogNorm
from astropy.table import Table
from astropy.visualization import simple_norm
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
import matplotlib.colors as colors

import photerr
import anacal
import psutil
from sklearn.decomposition import PCA
import scipy.integrate as integrate
import cv2
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from copy import deepcopy
from scipy.optimize import curve_fit



def fitShear(x, m , c):
    return x*(1 + m ) + c

def radial_profile(image, nbins = None, offset = 0):
    # Calculate the center of the image
    center =  np.array(image.shape) // 2.0#- 0.5 #+ 0.5*roman.pixel_scale/4 + offset

    # Create a grid of coordinates
    y, x = np.indices(image.shape, dtype = np.float64)
  
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Get unique radii and calculate average intensity for each radius
    radii = np.unique(r[r < 35])
    if nbins is None:
        intensity = np.zeros(len(radii))
        intensity[radii < 35] = np.array([np.mean(image[r == radius]) for radius in radii[radii < 35]])

        return radii, intensity
    
def create_radial_image(radial_bins, image_shape, radial_profile, offset = 0):
    """
    Create a 2D image based on radial bins, image dimensions, and radial profile.

    Parameters:
        radial_bins (array): Radial bins.
        image_shape (tuple): Dimensions of the image (height, width).
        radial_profile (array): Radial profile values.

    Returns:
        array: 2D image representing the radial profile.
    """
    # Calculate the center of the image
    center = np.array(image_shape) // 2.0
    y, x = np.indices(image_shape, dtype = np.float64)
    
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Interpolate radial profile values onto the image grid
    radial_image = np.interp(r, radial_bins, radial_profile, left=0.0, right=0.0)

    return radial_image
def radial_profile_spin2(image, nbins = 25, offset = 0):
    # Calculate the center of the image
    center = np.array(image.shape) // 2# - 0.5 #+ 0.5*roman.pixel_scale/4 + offset
    #center = np.array(np.unravel_index(image.argmax(), image.shape))

    # Create a grid of coordinates
    y, x = np.indices(image.shape)
    theta = np.arctan2((y - center[0]), (x - center[1]))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    radii = np.unique(r)
    # Use logspace to create logarithmically spaced radial bins
    bins = np.linspace(0.5, center[0]/2, num=nbins)
    #bins = np.hstack([radii[:10], bins])
    bins = np.unique(bins)
    
    # Use np.histogram to calculate the radial profile
    radial_sum_r, _ = np.histogram(r, bins=bins, weights=image*np.cos(2*theta))
    radial_sum_im, _ = np.histogram(r, bins=bins, weights=-image*np.sin(2*theta))
    radial_count, _ = np.histogram(r, bins=bins)

    # Avoid division by zero
    radial_profile_r = radial_sum_r/ radial_count
    radial_profile_r[radial_count == 0] = 0#np.divide(radial_sum_r, radial_count, where=radial_count != 0)
    radial_profile_im = radial_sum_im/radial_count
    radial_profile_im[radial_count == 0] = 0#np.divide(radial_sum_im, radial_count, where=radial_count != 0)
    
    bins = (bins[:-1] + bins[1:]) / 2

    return bins, 2*radial_profile_r, 2*radial_profile_im


def read_fits(filename):
    images = []
    with fits.open(filename + '.fits') as hdul:
        primary_image = hdul[0].data
        #print("Primary image shape:", primary_image.shape)
        images.append(primary_image)
        # Access the subsequent HDUs (additional images)
        for i in range(1, len(hdul)):
            image = hdul[i].data
            images.append(image)
    return np.array(images)


def get_data(root_dir, shear_str = '02', batch = '0_999', return_psf = True, filters = None):
   
    avg_true_psf, avg_star_psf = {}, {}
    gal_stamps = {}
    if filters is None:
        filters = ['Y106', 'J129', 'H158', 'F184', 'W146']
    for filter_name in filters:
        gal_file_name = root_dir + filter_name + '/g1' + shear_str + '_g20/galStamps_' + batch 
        star_psf_file_name = root_dir + filter_name + '/StarPSF/' + 'PSF_' + batch 
        true_psf_file_name = root_dir + filter_name + '/TruePSF/' + 'PSF_' + batch
        if return_psf:
            avg_true_psf[filter_name] = read_fits(true_psf_file_name)
            avg_star_psf[filter_name] = read_fits(star_psf_file_name)
        gal_stamps[filter_name] = read_fits(gal_file_name)

    return gal_stamps, avg_true_psf, avg_star_psf



def measureShear(avg_star_psf, gal_stamps, num_train = 0, end_idx= 10000, sigma_arcsec_list = None,
                nx = 440, ny = 440, norm = True, use_filters = None, cut = None):
    shear_corr_e1_all_0, shear_corr_e2_all_0, shear_corr_Resp_all_0 = [], [], []
    if cut is None:
        cut = np.ones(end_idx - num_train, dtype = bool)
    for j in range(len(use_filters)):
        l = use_filters[j]
        #print('Measuring Shear for Filter: ' + filters[j])
        img_acc,img_acc_star  = [], []
        shape_diff, size_diff = [], []
        shape_star, size_star = [], []
        shear_corr_e1, shear_corr_e2, shear_corr_Resp = [], [], []
        counter = 0

       
        sigma_arcsec = sigma_arcsec_list[j]

        scale = roman.pixel_scale/4
        avg_psf_task = avg_star_psf[l][0]/np.sum(avg_star_psf[l][0])
        fpTask = anacal.fpfs.FpfsMeasure(avg_psf_task, pix_scale = scale, sigma_arcsec=sigma_arcsec,
                nord = 4, det_nrot = 4)
        
        #if l == 'Z087':
        #    nx,ny = 128, 128
        #else: 
        #nx,ny = 440, 440
        num_objects = 4
        scale = roman.pixel_scale/4
        indX = np.arange(int(nx/2), nx*1, nx)
        indY = np.arange(int(ny/2), ny, ny)
        inds = np.meshgrid(indY, indX, indexing="ij")
        coords = np.vstack(inds).T
        coords = [(cc[0], cc[1], True) for cc in coords]
        
        
        
        for i in range (end_idx - num_train):
            if cut[i]:
                test_img = avg_star_psf[l][i + num_train]
                if  norm:
                    test_img = test_img/np.sum(test_img)

                corr_img = test_img


                gal_data = gal_stamps[l][i+ num_train ]
                gal_list = [gal_data[:, k* nx: (k+1)*nx] for k in range(4)]
                psf_list = [corr_img]*4
                #fpTask  =  fpfs.image.measure_source(corr_img, pix_scale = scale, sigma_arcsec=sigma_arcsec)
                #mms =  fpTask.measure(gal_data,  coords)
                #fpTask = anacal.fpfs.FpfsMeasure(corr_img, pix_scale = scale, sigma_arcsec=sigma_arcsec,
                #    nord = 4, det_nrot = 4)
                for n in range(4):
                    mms =  fpTask.run(gal_array = gal_list[n], psf_array= psf_list[n],  det = coords)
                #mms2 = np.vstack([
                #fpTask.run(
                #    gal_array=gal_list[i],
                #    psf_array=psf_list[i]
                #) for i in range(4)
                #])
                    mms = fpTask.get_results(mms)
                    ells=   anacal.fpfs.catalog.m2e(mms,const=10)
                    resp_e1= ells['fpfs_R1E']#np.average(ells['fpfs_R1E'])
                    resp_e2= ells['fpfs_R2E']
                    shear_e1=ells['fpfs_e1']
                    shear_e2=ells['fpfs_e2']
                    shear_corr_Resp.extend(resp_e1)
                    #shear_corr_Resp_e2.extend(resp_e2)
                    shear_corr_e1.extend(shear_e1)
                    shear_corr_e2.extend(shear_e2)
        shear_corr_Resp_all_0.append(np.array(shear_corr_Resp))
        #shear_corr_Resp_e2_all_0.append(shear_corr_Resp_e2)
        shear_corr_e1_all_0.append(np.array(shear_corr_e1))
        shear_corr_e2_all_0.append(np.array(shear_corr_e2))
    return shear_corr_e1_all_0, shear_corr_e2_all_0, shear_corr_Resp_all_0

def get_shear(psf,start_idx = 0,end_idx = 1000,
             sigma_arcsec_list = None, nx = 440, ny = 440, norm = True,
             use_filters = None, cut = None):
    
        
    #no shear
    print('Measuring Shear: g1 = 0, g2 = 0')
    shear_corr_e1_all_0, shear_corr_e2_all_0, shear_corr_Resp_all_0= measureShear(psf, gal_stamps_0,
                                            start_idx,end_idx = end_idx,
                                            sigma_arcsec_list =sigma_arcsec_list,nx = nx, ny = ny, norm = norm,
                                            use_filters = use_filters, cut = cut)

    # shear g1 = 0.02
    print('Measuring Shear: g1 = 0.02, g2 = 0')
    shear_corr_e1_all_02, shear_corr_e2_all_02, shear_corr_Resp_all_02= measureShear(psf, gal_stamps_02,
                                             start_idx,end_idx= end_idx,
                                             sigma_arcsec_list =sigma_arcsec_list, nx = nx, ny = ny, norm = norm,
                                            use_filters = use_filters, cut = cut) 

    # shear g1 = -0.02
    print('Measuring Shear: g1 = -0.02, g2 = 0')
    shear_corr_e1_all_n02, shear_corr_e2_all_n02, shear_corr_Resp_all_n02= measureShear(psf, gal_stamps_n02,
                                            start_idx,end_idx= end_idx,
                                            sigma_arcsec_list =sigma_arcsec_list, nx = nx, ny = ny, norm = norm,
                                            use_filters = use_filters, cut = cut)
    
    dict_shear = {'g1_0':[shear_corr_e1_all_0, shear_corr_Resp_all_0],
                 'g1_02':[shear_corr_e1_all_02, shear_corr_Resp_all_02],
                 'g1_n02':[shear_corr_e1_all_n02, shear_corr_Resp_all_n02]}
    return dict_shear
def get_color(mags):
    color = {}

    color['Z087'] = mags['Z087']- mags['Y106']
    color['Y106'] = mags['Y106']- mags['J129']
    color['J129'] = mags['J129']- mags['H158']
    color['H158'] = mags['H158']- mags['F184']
    color['F184'] = mags['F184']- mags['K213']
    color['W146'] = mags['W146']- mags['F184']
    return color
def flux2color(flux, filters):
    zp = {}
    color = {}
    for filter_name in filters:
        band = roman_filters[filter_name]
        zp[filter_name] =band.zeropoint
    for j in range(5):
        filter_b, filter_r = roman_filters[filters[j]], roman_filters[filters[j + 1]]
        zp_b, zp_red = filter_b.zeropoint, filter_r.zeropoint
        color[filters[j]] = (-2.5*np.log10(flux[filters[j]]/flux[filters[j+1]]) + (zp_b - zp_red)  )
    filter_b, filter_r = roman_filters['F184'], roman_filters['W146']
    zp_b, zp_red = filter_b.zeropoint, filter_r.zeropoint
    color['W146'] = (-2.5*np.log10(flux['F184']/flux['W146']) + (zp_b - zp_red)  )
    return color

def flux2mag(flux, filters):
    zp = {}
    mag = {}
    for j in range(len(filters)):
        bp = roman_filters[filters[j]]
        zp = bp.zeropoint
        mag[filters[j]] = -2.5*np.log10(flux[filters[j]]) + zp  
    return mag

def get_pos(dir_ , use_filters):
    #pickle_in = open('10000Gal_NoNoise_J129_RealisticSEDs_g10_g20.pickle', "rb")
    #with open('sim_images/10000Gal_NoNoise_scenarios/J129/gal_info.pickle', "rb") as pickle_in:
    with open( dir_ + 'J129/gal_info.pickle', "rb") as pickle_in:
        # Load only necessary parts if possible
        data = pickle.load(pickle_in)
        dict_J129 = data[['SCA', 'x_pos', 'y_pos', 'redshift']]
    #dict_J129 = pickle.load(pickle_in)['gal_df'][['SCA', 'x_pos', 'y_pos']]
    pickle_in.close()
    SCAs = dict_J129['SCA'].to_numpy()
    x_pos = (dict_J129['x_pos']).to_numpy()
    y_pos = (dict_J129['y_pos']).to_numpy()
    z = (dict_J129['redshift']).to_numpy()
    #indices = (dict_J129['indices']).to_numpy()
    del dict_J129
    del data

    flux = {}
    avg_star_flux = {}
    for filter_name in use_filters:
        with open(dir_+ filter_name+'/gal_info.pickle', "rb") as pickle_in:
            # Load only necessary parts if possible
            data = pickle.load(pickle_in)
            dict_filt = data[['flux', 'avg_star_flux']]
            flux[filter_name] = (dict_filt['flux']).to_numpy()
            avg_star_flux[filter_name] = (dict_filt['avg_star_flux']).to_numpy()
    
    return SCAs, x_pos, y_pos, z, flux, avg_star_flux 


def get_psf_basis(res_mult = 5, nx = 440, ny = 440):
    imgs_basis, rad_prof_basis, rad_prof2_r_basis,rad_prof2_im_basis  = {}, {}, {}, {}
    for filter_name in filters:
        filt_img_basis, filt_prof_basis = [], []
        filt_prof2_r_basis, filt_prof2_im_basis = [], []
        #nx,ny = 440, 440
        #if filter_name == 'Z087':
        #    nx,ny = 128, 128
        #else: 
        #    nx,ny = 440, 440
        for i in range(1,19):
            prof_img_t_1, prof_t_1,img_t_1 =  firstOrderPSF_s0 (filter_name, 
                                              nwaves = 12, SCA = i, order = 1, units = 'fphotons', nx = nx, ny=ny)
            filt_img_basis.append(img_t_1)
            filt_prof_basis.append(prof_t_1[1])
            test_img = cv2.resize(img_t_1 , dsize=(nx*res_mult, ny*res_mult), interpolation=cv2.INTER_CUBIC)
            rad_prof_spin2 = radial_profile_spin2(test_img, nbins = 50)
            filt_prof2_r_basis.append(rad_prof_spin2[1])
            filt_prof2_im_basis.append(rad_prof_spin2[2])
        imgs_basis[filter_name] = filt_img_basis
        rad_prof_basis[filter_name] = filt_prof_basis
        rad_prof2_r_basis[filter_name] = filt_prof2_r_basis
        rad_prof2_im_basis[filter_name] = filt_prof2_im_basis
    return imgs_basis,rad_prof_basis, rad_prof2_r_basis, rad_prof2_im_basis


def get_psf_basis_2nord(res_mult = 5, nx = 128, ny = 128, order = 2):
    imgs_basis, rad_prof_basis, rad_prof2_r_basis,rad_prof2_im_basis  = {}, {}, {}, {}
    for filter_name in filters:
        filt_img_basis, filt_prof_basis = [], []
        filt_prof2_r_basis, filt_prof2_im_basis = [], []
        nx,ny = 440, 440
        for i in range(1,19):
            prof_img_t_1, prof_t_1,img_t_1 =  secondOrderPSF_s0 (filter_name, 
                                              nwaves = 12, SCA = i, order = order, units = 'fphotons', nx = nx, ny=ny)
            filt_img_basis.append(img_t_1)
            filt_prof_basis.append(prof_t_1[1])
            test_img = cv2.resize(img_t_1 , dsize=(nx*res_mult, ny*res_mult), interpolation=cv2.INTER_CUBIC)
            rad_prof_spin2 = radial_profile_spin2(test_img, nbins = 50)
            filt_prof2_r_basis.append(rad_prof_spin2[1])
            filt_prof2_im_basis.append(rad_prof_spin2[2])
        imgs_basis[filter_name] = filt_img_basis
        rad_prof_basis[filter_name] = filt_prof_basis
        rad_prof2_r_basis[filter_name] = filt_prof2_r_basis
        rad_prof2_im_basis[filter_name] = filt_prof2_im_basis
    return imgs_basis,rad_prof_basis, rad_prof2_r_basis, rad_prof2_im_basis


def firstOrderPSF_s0 (filter_name, x = None,y= None, nwaves = 10, SCA = 10,scale = roman.pixel_scale/4, 
                      nx= 440, ny = 440, order = 1, legendre = False, units = 'fphotons',image_pos = None):
    flux_fact = (galsim.SED._h * galsim.SED._c)
    offset = galsim.PositionD(0.5 , 0.5)
    if x is not None and y is not None:
        image_pos = galsim.PositionD(x,y)
    bandpass = roman_filters[filter_name]
    bp_width = bandpass.red_limit - bandpass.blue_limit
    bp_w = bandpass.effective_wavelength #+  bp_offset(filter_name)
    #print((bp_w - 1000)/100)
    #bp_w += bp_width*0.001*(1 +(bp_w - 1000)/1000 )
    wave_arr = np.linspace(bandpass.blue_limit  ,bandpass.red_limit  , 1000 )
    #wave_arr2 = np.linspace(bandpass.blue_limit -0.9 ,bandpass.red_limit + 0.9  , 100 )
    psf = roman.getPSF(SCA, filter_name, SCA_pos = image_pos, n_waves=nwaves,  pupil_bin=8)
    #x_wave = ((wave_arr - bp_w))#/wave_arr
    x_wave = wave_arr
    if units == 'fnu':
        x_wave =x_wave/galsim.SED._c*wave_arr**2
    f1  = galsim.LookupTable(wave_arr, (x_wave)**order) 
    if legendre is True:
        if order == 3:
            f1  = galsim.LookupTable(wave_arr, 1/2*(5*x_wave**3 - 3*x_wave ))
    first_o_sed = galsim.SED(f1, 'nm', units)
    gauss = galsim.Gaussian(sigma=1e-8)
    pixel_response = galsim.Pixel(roman.pixel_scale)
    eff_psf_1o = galsim.Convolve(psf, pixel_response*first_o_sed)
    psfData_1o = eff_psf_1o.drawImage(bandpass, nx=nx, ny=ny, scale=scale, method = 'no_pixel', offset = offset)
    img = psfData_1o.array
    rad_prof = radial_profile(img)
    rad_image = create_radial_image(rad_prof[0], img.shape, rad_prof[1])
    
    
    x_wave = np.full(len(x_wave),bp_w)
    if units == 'fnu':
        x_wave =x_wave/galsim.SED._c*wave_arr**2
    f1  = galsim.LookupTable(wave_arr, (x_wave)**order) 
    if legendre is True:
        if order == 3:
            f1  = galsim.LookupTable(wave_arr, 1/2*(5*x_wave**3 - 3*x_wave ))
    first_o_sed = galsim.SED(f1, 'nm', units)
    gauss = galsim.Gaussian(sigma=1e-8)
    eff_psf_1o = galsim.Convolve(psf, pixel_response*first_o_sed)
    psfData_1o = eff_psf_1o.drawImage(bandpass, nx=nx, ny=ny, scale=scale, method = 'no_pixel', offset = offset)
    img_const = psfData_1o.array
    rad_prof_const = radial_profile(img_const)
    rad_image_const = create_radial_image(rad_prof_const[0], img.shape, rad_prof_const[1])
    
    rad_image = rad_image - rad_image_const
    new_rad_prof = [rad_prof[0], rad_prof[1] - rad_prof_const[1]]
    img = img - img_const
    
    return rad_image , new_rad_prof, img#psfData_1o.array


def secondOrderPSF_s0 (filter_name, x = None,y= None, nwaves = 10, SCA = 10,scale = roman.pixel_scale/4, 
                      nx= 128, ny = 128, order = 2, legendre = False, units = 'fphotons'):
    flux_fact = (galsim.SED._h * galsim.SED._c)
    offset = galsim.PositionD(0.5 , 0.5)
    image_pos = None
    if x is not None and y is not None:
        image_pos = galsim.PositionD(x,y)
    bandpass = roman_filters[filter_name]
    bp_width = bandpass.red_limit - bandpass.blue_limit
    bp_w = bandpass.effective_wavelength #+  bp_offset(filter_name)
    #print((bp_w - 1000)/100)
    #bp_w += bp_width*0.001*(1 +(bp_w - 1000)/1000 )
    wave_arr = np.linspace(bandpass.blue_limit  ,bandpass.red_limit  , 1000 )
    #wave_arr2 = np.linspace(bandpass.blue_limit -0.9 ,bandpass.red_limit + 0.9  , 100 )
    psf = roman.getPSF(SCA, filter_name, SCA_pos = image_pos, n_waves=nwaves,  pupil_bin=8)
    #x_wave = ((wave_arr - bp_w))#/wave_arr
    x_wave = (wave_arr- bp_w)
    if units == 'fnu':
        x_wave =x_wave/galsim.SED._c*wave_arr**2
    f1  = galsim.LookupTable(wave_arr, (x_wave)**order) 
    if legendre is True:
        if order == 3:
            f1  = galsim.LookupTable(wave_arr, 1/2*(5*x_wave**3 - 3*x_wave ))
    first_o_sed = galsim.SED(f1, 'nm', units)
    gauss = galsim.Gaussian(sigma=1e-8)
    pixel_response = galsim.Pixel(roman.pixel_scale)
    eff_psf_1o = galsim.Convolve(psf, pixel_response*first_o_sed)
    psfData_1o = eff_psf_1o.drawImage(bandpass, nx=nx, ny=ny, scale=scale, method = 'no_pixel', offset = offset)
    img = psfData_1o.array
    rad_prof = radial_profile(img)
    rad_image = create_radial_image(rad_prof[0], img.shape, rad_prof[1])
    
    rad_image = rad_image #- rad_image_const
    new_rad_prof = [rad_prof[0], rad_prof[1]] #- rad_prof_const[1]]
    img = img #- img_const
    
    return rad_image , new_rad_prof, img#psfData_1o.array

def get_mags_avg(gal_props ,err_model, indx_seed = 9999, use_filters = None, point = False):
    obs_mags = {}
    true_mags = {}
    err_mags = {}
    if use_filters is None:
        use_filters = filters
    for filter_name in use_filters:
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
            if obs_mag[i] >= 50:
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

def get_idx(batch):
    if batch == '0_999':
        return 0
    return int(batch[:4])

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
    


## Roman filters
use_filters = 'ZYJHFKW'
roman_filters = roman.getBandpasses(AB_zeropoint=True)

# Get the names of the ones we will use here.
filters = [filter_name for filter_name in roman_filters if filter_name[0] in use_filters]

filters_copy = filters.copy()
filters_copy[5] = filters_copy[6]
filters_copy[6] = filters[5]
filters = filters_copy

blue_limit, red_limit, eff_wave = np.zeros(7), np.zeros(7), np.zeros(7)
for i in range(len(filters)):

    bandpass = roman_filters[filters[i]]
    eff_wave[i] = bandpass.effective_wavelength
    blue_limit[i], red_limit[i] = bandpass.blue_limit, bandpass.red_limit
    sampling_rate = (eff_wave[i]/1e3)*7.9/10
    

img_dir = 'sim_images/10000Gal_NoNoise_diffsky/'
#img_dir = 'sim_images/10000Gal_NoNoise_cosmoDC2_final/'
#use_filters = ['Y106', 'J129', 'H158', 'F184', 'W146']
use_filters = ['H158']
scenarios = ['A', 'B', 'C', 'D', 'E', 'F']
SCAs, x_pos, y_pos, z_test, gal_flux, star_flux = get_pos(img_dir, use_filters)




scenarios = ['A']
coeff_diff_slope = {}
#for scenario  in scenarios:
pickle_in = open("SED_fit_coeff/final_run/10000Gal_1ordfit_A_YJHFW_diffsky_10307", "rb")
#pickle_in = open("SED_fit_coeff/final_run/10000Gal_1ordfit_A_YJHFW_cosmoDC2_10067", "rb")
coeffs = pickle.load(pickle_in)
pickle_in.close()
coeff_diff_slope = {}
coeff_gal = {}
coeff_star = {}
flux_star = {}
flux_gal = {}
coeff_diff_0th = {}
coeff_diff_2nd = {}
coeff_diff_3rd = {}
for filter_ in use_filters:
    coeffs_gal_slope = coeffs['gal'][1][filter_]
    coeffs_star_slope = coeffs['star:'][1][filter_]
    coeff_diff_slope[filter_] = coeffs_star_slope - coeffs_gal_slope
    coeff_gal[filter_] = coeffs['gal'][1][filter_]
    coeff_star[filter_] = coeffs['star:'][1][filter_]
    flux_star[filter_] = coeffs['star_fluxes'][filter_]
    flux_gal[filter_] = coeffs['gal_fluxes'][filter_]
    coeff_diff_0th[filter_] = coeffs['star:'][0][filter_]- coeffs['gal'][0][filter_]
    coeff_diff_2nd[filter_] = coeffs['star:'][2][filter_]*coeffs['gal'][0][filter_] - coeffs['star:'][0][filter_]*coeffs['gal'][2][filter_]
    coeff_diff_2nd[filter_] = coeff_diff_2nd[filter_]
coeff_diff = coeff_diff_slope.copy()


#read basis
imgs_basis,rad_prof_basis, rad_prof2_r_basis, rad_prof2_im_basis = get_psf_basis()
fwhm_dict = {'Y106': 0.131, 'J129': 0.139, 'H158': 0.153, 'F184': 0.169, 'W146': 0.148}
fwhm = []
for filt_ in use_filters:
    fwhm.append(fwhm_dict[filt_])
fwhm = np.array(fwhm)



#test_photoz = pd.read_parquet('photoz_samples/roman_pz/output/test_scenario_A_fzb.pq', engine='pyarrow')
#photoz_test = np.array(test_photoz['flexzboost_median'])


#calculate shear for each batch
batches = ['0_999', '1000_1999', '2000_2999','3000_3999','4000_4999','5000_5999','6000_6999','7000_7999', '8000_8999', '9000_9999']


out_dir = 'shear_measurement_diffsky/'
#out_dir = 'shear_measurement_cosmoDC2_final/'

z_bins = [0.0, 0.65, 1.0, 1.4, 1.9, 3.5]
z_bins2 = [0, 0.45, 0.65, 0.8, 1.0,1.15, 1.35, 1.55, 1.85, 2.25, 3.1]
def get_corr_psf(coeffs,avg = False,zavg = False):
    corr_psf = {}
    for filter_name in use_filters:
        print(filter_name)
        corr_imgs = []
        coeff_z_mean = []
        for j in range(0,5):
            mask = np.logical_and(z_test > z_bins[j], z_test < z_bins[j + 1])
            flux_mean = np.mean(coeffs[filter_name][mask])
            coeff_z_mean.append(flux_mean)
        print(coeff_z_mean)
        for i in range(len(avg_star_psf_0[filter_name])):
            for j in range(0,5):
                mask = np.logical_and(z_test[i+ add_idx] > z_bins[j], z_test[i+ add_idx] < z_bins[j + 1])
                if mask:
                    mean_filt = coeff_z_mean[j]
            star_img_test =avg_star_psf_0[filter_name][i].copy()
            star_img_test/= np.sum(star_img_test)
    
            corr_coeff = coeffs[filter_name][i + add_idx]
            if avg:
                corr_coeff = np.mean(coeffs[filter_name])
            if zavg:
                corr_coeff = mean_filt
                
            corr = imgs_basis[filter_name][SCAs[i+ add_idx] - 1]*corr_coeff
            corr_img = star_img_test - corr
            corr_imgs.append(corr_img)
        corr_psf[filter_name] = corr_imgs
    return corr_psf

for batch in batches:
    add_idx = get_idx(batch)
    gal_stamps_n02, avg_true_psf_n02, avg_star_psf_n02 = get_data(img_dir, shear_str = 'n02', batch = batch, 
                                                                  return_psf = False, filters = use_filters)
    gal_stamps_02, avg_true_psf_02, avg_star_psf_02 = get_data(img_dir,shear_str = '02', batch = batch, 
                                                               return_psf = False, filters = use_filters)
    gal_stamps_0, avg_true_psf_0, avg_star_psf_0 = get_data(img_dir,shear_str = '0', batch = batch, filters = use_filters)

   
    dict_shear_star = get_shear(avg_star_psf_0, end_idx = 1000, sigma_arcsec_list = fwhm*1.15, use_filters = use_filters)
    pickle_out = open(out_dir + "StarPSF/Shear_" + batch + '.pickle',"wb")
    pickle.dump(dict_shear_star, pickle_out)
    pickle_out.close()


    dict_shear_true = get_shear(avg_true_psf_0, end_idx = 1000, sigma_arcsec_list = fwhm*1.15, use_filters = use_filters)
    pickle_out = open(out_dir + "TruePSF/Shear_" + batch + '.pickle',"wb")
    pickle.dump(dict_shear_true, pickle_out)
    pickle_out.close()
    
    
    corr_psf = get_corr_psf(coeff_diff ,avg = False)
    dict_shear_avgcorr = get_shear(corr_psf, end_idx = 1000, sigma_arcsec_list = fwhm*1.15, use_filters = use_filters)
    pickle_out = open(out_dir + "TrueCorrPSF_test/Shear_" + batch + '.pickle',"wb")
    pickle.dump(dict_shear_avgcorr, pickle_out)
    pickle_out.close()
    
    corr_psf = get_corr_psf(coeff_diff ,avg = True)
    dict_shear_avgcorr = get_shear(corr_psf, end_idx = 1000, sigma_arcsec_list = fwhm*1.15, use_filters = use_filters)
    pickle_out = open(out_dir + "TrueAvgCorrPSF/Shear_" + batch + '.pickle',"wb")
    pickle.dump(dict_shear_avgcorr, pickle_out)
    pickle_out.close()
    
    corr_psf = get_corr_psf(coeff_diff , zavg = True)
    dict_shear_avgcorr = get_shear(corr_psf, end_idx = 1000, sigma_arcsec_list = fwhm*1.15, use_filters = use_filters)
    pickle_out = open(out_dir + "TrueAvgCorrPSF_zbin/Shear_" + batch + '.pickle',"wb")
    pickle.dump(dict_shear_avgcorr, pickle_out)
    pickle_out.close()
    

    # analytical pred. Need to change files for different prediction coeffs.
    pickle_in = open("SED_fit_coeff/final_run/10000Gal_pred_A_H_diffsky_10307" ,"rb")
    #pickle_in = open("SED_fit_coeff/final_run/10000Gal_pred_A_H_cosmoDC2_10067" ,"rb")
    pred_coeff = pickle.load(pickle_in)
    pickle_in.close()
    
    keys_ = list(pred_coeff.keys())
    for key in keys_:
        corr_psf = get_corr_psf({'H158': pred_coeff[key]})
        dict_shear_avgcorr = get_shear(corr_psf, end_idx = 1000, sigma_arcsec_list = fwhm*1.15, use_filters = use_filters)
        pickle_out = open(out_dir + "Analyticpred_H/Shear_"+ key + '_' + batch + ".pickle","wb")
        pickle.dump(dict_shear_avgcorr, pickle_out)
        pickle_out.close()
    
    keys_ = list(pred_coeff.keys())
    for key in keys_:
        corr_psf = get_corr_psf({'H158': pred_coeff[key]}, zavg = True)
        dict_shear_avgcorr = get_shear(corr_psf, end_idx = 1000, sigma_arcsec_list = fwhm*1.15, use_filters = use_filters)
        pickle_out = open(out_dir + "Analyticpred_H_zbin/Shear_"+ key + '_' + batch + ".pickle","wb")
        pickle.dump(dict_shear_avgcorr, pickle_out)
        pickle_out.close()
    
    

    # som pred. Need to change files for different prediction coeffs.
    pickle_in = open("SED_fit_coeff/final_run/10000Gal_SOMpred_A_H_diffsky_10307" ,"rb")
    #pickle_in = open("SED_fit_coeff/final_run/10000Gal_SOMpred_A_H_cosmodc2_10067" ,"rb")
    #pickle_in = open("SED_fit_coeff/final_run/10000Gal_SOMpred_A_H_cosmodc2_10067_55pSEDCut" ,"rb")
    #pickle_in = open("SED_fit_coeff/final_run/10000Gal_SOMpred_A_H_cosmodc2_10067_25pSEDCut" ,"rb")
    #pickle_in = open("SED_fit_coeff/final_run/10000Gal_SOMpred_A_H_diffsky_traincd2_10037" ,"rb")
    pred_coeff = pickle.load(pickle_in)
    pickle_in.close()

    keys_ = list(pred_coeff.keys())
    for key in keys_:
        corr_psf = get_corr_psf({'H158': pred_coeff[key]})
        dict_shear_avgcorr = get_shear(corr_psf, end_idx = 1000, sigma_arcsec_list = fwhm*1.15, use_filters = use_filters)
        pickle_out = open(out_dir + "SOMpred_H/Shear_"+ key + '_' + batch + ".pickle","wb")
        pickle.dump(dict_shear_avgcorr, pickle_out)
        pickle_out.close()
    
    keys_ = list(pred_coeff.keys())
    for key in keys_:
        corr_psf = get_corr_psf({'H158': pred_coeff[key]}, zavg = True)
        dict_shear_avgcorr = get_shear(corr_psf, end_idx = 1000, sigma_arcsec_list = fwhm*1.15, use_filters = use_filters)
        pickle_out = open(out_dir + "SOMpred_H_zbin/Shear_"+ key + '_' + batch + ".pickle","wb")
        pickle.dump(dict_shear_avgcorr, pickle_out)
        pickle_out.close()
    
    
    



    