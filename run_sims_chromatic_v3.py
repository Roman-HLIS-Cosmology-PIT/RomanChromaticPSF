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

import matplotlib.pyplot as plt

from astropy.io import fits
from matplotlib.colors import LogNorm
from astropy.table import Table
from astropy.visualization import simple_norm
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
import matplotlib.colors as colors






use_filters = 'ZYJHFWK'
roman_filters = roman.getBandpasses(AB_zeropoint=True)

# Get the names of the ones we will use here.
filters = [filter_name for filter_name in roman_filters if filter_name[0] in use_filters]

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

# star catalog
#star_data = pd.read_parquet('pointsource_10050.parquet', engine='pyarrow')
#star_flux_data = pd.read_parquet('stellar_SED/pointsource_flux_10050.parquet', engine='pyarrow')

#kurucz_mask = np.zeros(len(star_data), dtype = bool)
#for i in range(len(star_data)):
#    sed_file = star_data['sed_filepath']
#    if sed_file[i][8] == 'k':
#        kurucz_mask[i] = True
        
#star_data = star_data[kurucz_mask]
#star_flux_data = star_flux_data[kurucz_mask]
#file_name = open('star_seds.pickle', 'rb')
#star_SEDs = pickle.load(file_name)
#file_name.close()

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
#seed_indices = 9999
#np.random.seed(seed_indices)
#random_indices = np.random.random(np.arange(list_len), num_objects, replace = False)

## reading SEDs
import h5py 
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

## stellar catalog
stellar_root = '/hildafs/projects/phy200017p/share/tq_share/'
stellar_dir = ['gizis_SED' , 'kurucz' , 'mlt' , 'old_mlt' , 'phoSimMLT' , 'wDs']


wDs_root = stellar_root + 'starSED/' + 'wDs/'
kurucz_root = stellar_root + 'starSED/' + 'kurucz/'
lst_wDs = os.listdir(stellar_root + 'starSED/'+'wDs')
wDs_seds = [wDs_root + s for s in lst_wDs]
lst_kurucz = os.listdir(stellar_root + 'starSED/'+'kurucz')
kurucz_seds = [kurucz_root + s for s in lst_kurucz]
all_seds = kurucz_seds # np.hstack([wDs_seds, kurucz_seds])



vega_sed = galsim.SED('vega.txt', 'nm', 'flambda')
vega_sed = vega_sed.withFluxDensity(target_flux_density=1.0, wavelength=500)
vega_sed#.wave_list
#vega_sed._orig_spec.f
vega_500_idx = np.argmin(abs(vega_sed.wave_list - 500))


#seed = 1234
#np.random.seed(seed)
#numSeds = 1000
#rand_idx = np.random.randint(0, len(all_seds), numSeds)
#stellar_SEDs = []
#for idx in rand_idx:
#    star_sed = pd.read_csv(all_seds[idx], compression='gzip', header=1, sep=' ', quotechar='"')
#    wavelength = np.array(star_sed)[:,0]
#    SED = np.array(star_sed)[:,1]#[wavelength <= 5000]
#    #wavelength = wavelength[wavelength <= 5000]
#    lookup_table = galsim.LookupTable(wavelength, SED)
#    stellar_SEDs.append(galsim.SED(lookup_table, 'nm', 'flambda').withFluxDensity(target_flux_density=1.0, wavelength=500))



def shearEst(num_gal = 1000, filter_name = 'H158', add_seed = 0, scale = roman.pixel_scale, SCA = 7, nwaves = 10, nx = 64, ny = 64, shear_value = 0.02,sigma_arcsec = 0.175, stellar_SEDs = None, indx_seed = 9999, mag_min=17 , mag_max =22, out_file = 'out' , start_idx = 0):
    shear_chrom_e1, shear_chrom_e2, shear_chrom_Resp_e1,shear_chrom_Resp_e2 , PSF_chrom_e1, PSF_chrom_e2, PSF_chrom_R = [], [], [], [], [], [], []
    #shear_achrom_e1, shear_achrom_e2, shear_achrom_Resp, PSF_achrom_e1, PSF_achrom_e2, PSF_achrom_R = [], [], [], [], [], [] 
    shear_star_e1, shear_star_e2, shear_star_Resp_e1,shear_star_Resp_e2,  PSF_star_e1, PSF_star_e2, PSF_star_R = [], [], [], [], [], [], []
    shear_star_norm_e1, shear_star_norm_e2, shear_star_norm_Resp_e1,shear_star_norm_Resp_e2, PSF_star_e1, PSF_star_e2, PSF_star_R = [], [], [], [], [], [], []
    stamps, psf_stamps, psf_star_stamps, psf_star_avg_stamps = [], [], [], []
    bad_images = []
    z = []
    gal_ids, star_ids, gal_star_ids = [], [], []
    star_flux, flux = [], []
    avg_star_flux = []
    #cat1 = galsim.COSMOSCatalog(sample='25.2')
    SCAs, x_pos, y_pos = [],[], []
    y_bandpass = roman_filters['Y106']
    flux_fact = (galsim.SED._h * galsim.SED._c)
    filter_name_col = filter_name
    if filter_name == 'W149':
        filter_name_col = 'W146'
    flux_fact = (galsim.SED._h * galsim.SED._c)
    num_stars = 40
    bandpass = roman_filters[filter_name]
    np.random.seed(indx_seed)
    random_indices = np.random.choice(np.arange(list_len), num_gal, replace = False)
    #random_star_indices = np.random.choice(np.arange(list_len), num_gal*num_stars, replace = False)
    bp500 = galsim.Bandpass(galsim.LookupTable([499, 500, 501], [0, 1, 0]),
                                             wave_type='nm').withZeropoint('AB')
    #for l in range(start_idx, len(random_indices)):
    #for l in range(start_idx, 5000):
    #for l in range(8750, len(random_indices)):
    for l in range(len(random_indices)):
        i = random_indices[l]
        #print(i)
        seed = i + add_seed#np.random.randint(0, 81499, 1)[0]
        np.random.seed(seed)
        gal_id = str(np.array(gal_data['galaxy_id'])[i])
        gal_ids.append(gal_id)
        
        ## make_galaxy
        redshift = np.array(gal_data['redshift'])[i]
        shear1, shear2 = np.array(gal_data['shear1'])[i], np.array(gal_data['shear2'])[i]
        bulge_hlr = np.array(gal_data['spheroidHalfLightRadiusArcsec'])[i]
        disk_hlr = np.array(gal_data['diskHalfLightRadiusArcsec'])[i]
        disk_shear1, disk_shear2 = np.array(gal_data['diskEllipticity1'])[i], np.array(gal_data['diskEllipticity2'])[i]
        bulge_shear1, bulge_shear2 = np.array(gal_data['spheroidEllipticity1'])[i], np.array(gal_data['spheroidEllipticity2'])[i]
        f_sed = data[gal_id[:9]][gal_id][()]
        bulge_lookup = galsim.LookupTable( x = wave_list, f = f_sed[0])
        disk_lookup = galsim.LookupTable( x = wave_list, f = f_sed[1])
        knots_lookup = galsim.LookupTable( x = wave_list, f = f_sed[2])
        bulge_sed = galsim.SED(bulge_lookup, wave_type = 'Ang', flux_type='fnu', redshift = redshift)
        disk_sed = galsim.SED(disk_lookup, wave_type = 'Ang', flux_type='fnu', redshift = redshift)
        knots_sed = galsim.SED(knots_lookup, wave_type = 'Ang', flux_type='fnu', redshift = redshift)
        bulge = galsim.Sersic(4, half_light_radius=bulge_hlr).shear(g1 =bulge_shear1 , g2 = bulge_shear2)
        disk = galsim.Sersic(1, half_light_radius=disk_hlr).shear(g1 =disk_shear1 , g2 = disk_shear2)
        obj = bulge*bulge_sed + disk*(disk_sed + knots_sed)
        obj = obj.withFlux(np.array(flux_data['roman_flux_' + filter_name_col])[i]/flux_fact, bandpass)
        z.append(redshift)
 
        
        ## make stars
        #random_star_idx = np.random.choice(np.arange(len(stellar_SEDs)), num_stars, replace = False)
        #if stellar_SEDs is None:
        #    star_sed = galsim.SED('vega.txt', 'nm', 'flambda')
        #else:
            #star_sed = stellar_SEDs[int(i/10.0)]
        #    star_seds = np.array(stellar_SEDs)[random_star_idx]
        
        #np.random.seed(seed)
        #random_star_idx = np.random.choice(np.arange(len(stellar_SEDs)), num_stars, replace = False)
        #star_seds = []
        #if stellar_SEDs is None:
        #    star_seds.append(galsim.SED('vega.txt', 'nm', 'flambda'))
        #else:
        #star_seds = star_SEDs[l*num_stars:(l+1)*num_stars]
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
            if star_mag < mag_max and star_mag > mag_min:
                star_seds.append(sed)
                star_ids.append(str(np.array(star_obj['id'])[j]))
                gal_star_ids.append(gal_id)
        # The rng for photon shooting should be different for each filter.
        if len(star_seds) == 0:
                star_seds.append(sed)
                star_ids.append(str(np.array(star_obj['id'])[j]))
                gal_star_ids.append(gal_id)
        phot_rng = galsim.UniformDeviate(seed + ord(filter_name[0])*num_gal)


        #mu_x = 1.e5
        #sigma_x = 2.e5
        #mu = np.log(mu_x**2 / (mu_x**2+sigma_x**2)**0.5)
        #sigma = (np.log(1 + sigma_x**2/mu_x**2))**0.5
        #gd = galsim.GaussianDeviate(obj_rng, mean=mu, sigma=sigma)
        #flux = np.exp(gd())

        # Normalize the SED to have this flux in the Y band.
        #star_sed = star_sed.withFlux(flux, y_bandpass)
        # Pick a random position in the image to draw it.
        #pos_rng = galsim.UniformDeviate(int(i/10.0) + 1e6 + add_seed)
        pos_rng = galsim.UniformDeviate(i + 1e6 + add_seed)
        x = pos_rng() * roman.n_pix
        y = pos_rng() * roman.n_pix
        image_pos = galsim.PositionD(x,y)
        x_pos.append(x)
        y_pos.append(y)
        
        
        #galsim.DeltaFunction()
        SCA = np.random.randint(1, 19, 1)[0]
        psfInt = roman.getPSF(SCA, filter_name, SCA_pos = image_pos, n_waves=nwaves,  pupil_bin=8)
        eff_wave = bandpass.effective_wavelength
        #psf_achrom = roman.getPSF(SCA, filter_name,  wavelength= eff_wave,  pupil_bin=8)
        #pixel_response = galsim.Pixel(scale)*obj.SED
        #pixel_response_star = galsim.Pixel(scale)*star_sed
        #pixel_response = galsim.Pixel(roman.pixel_scale)*obj.SED
        #eff_psf = galsim.Convolve(psfInt, pixel_response) 
        #eff_psf_star = galsim.Convolve(psfInt, pixel_response_star) 
        #eff_psf_achrom = galsim.Convolve(psf_achrom, galsim.Pixel(scale)) 
        eff_psf = galsim.Convolve(psfInt, galsim.DeltaFunction()*obj.sed)
        offset = galsim.PositionD(0.5, 0.5)
        psfData = eff_psf.drawImage(bandpass, nx=nx, ny=ny, scale=scale, method = 'auto', offset = offset)#.array
        #psfData_achrom = eff_psf_achrom.drawImage(nx=nx, ny=ny, scale=scale, method = 'no_pixel')#.array
        # measure psf moments
        mom_true = galsim.hsm.FindAdaptiveMom(psfData, strict = False)
        PSF_chrom_R.append(mom_true.moments_sigma*scale)
        PSF_chrom_e1.append(mom_true.observed_shape.e1)
        PSF_chrom_e2.append(mom_true.observed_shape.e2)
        psfData_star, stars_flux = [], []
        for j in range(len(star_seds)):
            eff_psf_star = galsim.Convolve(psfInt, galsim.DeltaFunction()*star_seds[j])
            psf_star = eff_psf_star.drawImage(bandpass, nx=nx, ny=ny, scale=scale, method = 'auto', offset = offset)#.array
            stars_flux.append(np.sum(psf_star.array))
            psfData_star.append(psf_star.array)
            
        #mom_star = galsim.hsm.FindAdaptiveMom(psfData_star, strict = False)
        #PSF_star_R.append(mom_star.moments_sigma*scale)
        #PSF_star_e1.append(mom_star.observed_shape.e1)
        #PSF_star_e2.append(mom_star.observed_shape.e2)
        
        
        psfData = psfData.array#/np.sum(psfData.array)
        #psfData_achrom = psfData_achrom.array
        #psfData_star = psfData_star.array#/np.sum(psfData_star.array)
        #psfData_scaled = psfData_scaled.array#/np.sum(psfData_scaled.array)
        
        # Four Galaxies to cancel spin-2 and spin-4 ansiotropies
        # spin-2 is shape noise in diagnonal elements of shear response matrix
        # spin-4 is shape noise in diagnonal and of-diagnoal elements of shear response matrix, 
        # but an order of magnitude smaller than spin-2
        psf_stamps.append(psfData)
        psf_star_stamps.extend(psfData_star)
        star_flux.extend(stars_flux)
        SCAs.append(SCA)
        try:
            num_objects = 4
            for j in range(num_objects):
                #theta = np.random.rand()*np.pi * 2.0 * galsim.radians
                if j == 0:
                    ang = (np.random.uniform(0.0, np.pi * 2.0)) * galsim.radians
                else:
                    ang = np.pi / 4 * galsim.radians
                obj = obj.rotate(ang)
                obj_rot = obj.shear(g1=shear_value, g2=0.0)
                final = galsim.Convolve(obj_rot, psfInt)
                gal_data_rot = final.drawImage(bandpass, nx=nx, ny=ny, scale=scale, offset = offset).array
                if j == 0:
                    galaxy_data = gal_data_rot
                else:
                    galaxy_data = np.hstack([galaxy_data,gal_data_rot ])
            flux.append(np.sum(gal_data_rot))
            stamps.append(galaxy_data)
            #psf_scaled_stamps.append(psfData_scaled)
            
            # Now we measure shear

            # fake detection
            #indX = np.arange(int(nx/2), nx*num_objects, nx)
            #indY = np.arange(int(ny/2), ny, ny)
            #inds = np.meshgrid(indY, indX, indexing="ij")
            #coords = np.vstack(inds).T


            #fpTask  =  fpfs.image.measure_source(psfData, pix_scale = scale, sigma_arcsec=sigma_arcsec)
            #mms =  fpTask.measure(galaxy_data, coords)
            #mms = fpTask.get_results(mms)
            #ells=   fpfs.catalog.fpfs_m2e(mms,const=2000)
            #resp_e1= ells['fpfs_R1E']
            #resp_e2= ells['fpfs_R2E']
            #shear_e1=ells['fpfs_e1']
            #shear_e2=ells['fpfs_e2']
            #shear_chrom_Resp_e1.extend(resp_e1)
            #shear_chrom_Resp_e2.extend(resp_e2)
            #shear_chrom_e1.extend(shear_e1)
            #shear_chrom_e2.extend(shear_e2)
            #print('Input shear is: %.6f' %shear_value)
            #print('Estimated shear (chromatic PSF) is: %.6f' %shear_g1)


            psf_star_avg = np.average(psfData_star, axis = 0)
            psf_star_avg_stamps.append(psf_star_avg)
            avg_star_flux.append(np.sum(psf_star_avg))
            #fpTask  =  fpfs.image.measure_source(psf_star_avg, pix_scale = scale, sigma_arcsec=sigma_arcsec)
            #mms =  fpTask.measure(galaxy_data, coords)
            #mms = fpTask.get_results(mms)
            #ells=   fpfs.catalog.fpfs_m2e(mms,const=2000)
            #resp_e1= ells['fpfs_R1E']
            #resp_e2= ells['fpfs_R2E']
            #shear_e1=ells['fpfs_e1']
            #shear_e2=ells['fpfs_e2']
            #shear_star_Resp_e1.extend(resp_e1)
            #shear_star_Resp_e2.extend(resp_e2)
            #shear_star_e1.extend(shear_e1)
            #shear_star_e2.extend(shear_e2)
            
            #norm_psf = np.array(psfData_star)/(np.array(stars_flux)[:, np.newaxis][:, np.newaxis])
            #psf_star_avg = np.average(norm_psf, axis = 0)
            #fpTask  =  fpfs.image.measure_source(psf_star_avg, pix_scale = scale, sigma_arcsec=sigma_arcsec)
            #mms =  fpTask.measure(galaxy_data, coords)
            #mms = fpTask.get_results(mms)
            #ells=   fpfs.catalog.fpfs_m2e(mms,const=2000)
            #resp_e1= ells['fpfs_R1E']
            #resp_e2= ells['fpfs_R2E']
            #shear_e1=ells['fpfs_e1']
            #shear_e2=ells['fpfs_e2']
            #shear_star_norm_Resp_e1.extend(resp_e1)
            #shear_star_norm_Resp_e2.extend(resp_e2)
            #shear_star_norm_e1.extend(shear_e1)
            #shear_star_norm_e2.extend(shear_e2)
            bad_images.append(0)
        except:
            bad_images.append(-1)
            flux.append(-1)
            stamps.append([-1])

        num_steps = 250
        if (l+1) % num_steps == 0 and l != 0 :
            shear_chrom_e1, shear_chrom_e2, shear_chrom_Resp_e1,shear_chrom_Resp_e2,  PSF_chrom_e1, PSF_chrom_e2, PSF_chrom_R  = np.array(shear_chrom_e1), np.array(shear_chrom_e2), np.array(shear_chrom_Resp_e1),np.array(shear_chrom_Resp_e2),  np.array(PSF_chrom_e1), np.array(PSF_chrom_e2), np.array(PSF_chrom_R )
            #shear_achrom_e1, shear_achrom_e2, shear_achrom_Resp, PSF_achrom_e1, PSF_achrom_e2, PSF_achrom_R  = np.array(shear_achrom_e1), np.array(shear_achrom_e2), np.array(shear_achrom_Resp), np.array(PSF_achrom_e1), np.array(PSF_achrom_e2), np.array(PSF_achrom_R )
            shear_star_e1, shear_star_e2, shear_star_Resp_e1,shear_star_Resp_e2, PSF_star_e1, PSF_star_e2, PSF_star_R  = np.array(shear_star_e1), np.array(shear_star_e2), np.array(shear_star_Resp_e1),np.array(shear_star_Resp_e2),  np.array(PSF_star_e1), np.array(PSF_star_e2), np.array(PSF_star_R )

            shear_star_norm_e1, shear_star_norm_e2, shear_star_norm_Resp_e1,shear_star_Resp_norm_e2  = np.array(shear_star_norm_e1), np.array(shear_star_norm_e2), np.array(shear_star_norm_Resp_e1),np.array(shear_star_norm_Resp_e2)
            #shear_scaled_e1, shear_scaled_e2, shear_scaled_Resp, PSF_scaled_e1, PSF_scaled_e2, PSF_scaled_R  = np.array(shear_scaled_e1), np.array(shear_scaled_e2), np.array(shear_scaled_Resp), np.array(PSF_scaled_e1), np.array(PSF_scaled_e2), np.array(PSF_scaled_R )

            flux = np.array(flux)
            avg_star_flux = np.array(avg_star_flux)
            z = np.array(z)
            gal_ids = np.array(gal_ids)
            
            start_idx = int((l+1) / num_steps - 1)*num_steps #+1
            end_idx = l + 1
            if (l+1) == num_steps:
                start_idx = 0
                end_idx = l + 1
            
            
            #print(len(gal_ids), len(stamps), len(psf_stamps), len(bad_images))
            #print(len(x_pos), len(random_indices[start_idx:end_idx], ))
            #print(len(SCAs), len(x_pos), len(y_pos), len(flux))
            #print(len(random_indices[start_idx:end_idx]), len(PSF_chrom_e1), len(PSF_chrom_e2), len(PSF_chrom_R))

            #gal_dict = {'galaxy_id': gal_ids, 'stamps': stamps, 'psf_stamps': psf_stamps,'bad_images': bad_images, 'redshift': z,                 'x_pos': x_pos, 'y_pos': y_pos,'flux': flux, 'SCA': SCAs, 'indices': random_indices[start_idx:end_idx],                                            'eff_psf_e1': PSF_chrom_e1, 'eff_psf_e2': PSF_chrom_e2, 'eff_psf_R': PSF_chrom_R,
                        #'psf_star_avg_stamps':psf_star_avg_stamps, 'avg_star_flux': avg_star_flux}
                        
            gal_dict = {'galaxy_id': gal_ids, 'stamps': stamps, 'psf_stamps': psf_stamps,'bad_images': bad_images, 'redshift': z,                 'x_pos': x_pos, 'y_pos': y_pos,'flux': flux, 'SCA': SCAs, 'indices': random_indices[start_idx:end_idx],                                            'eff_psf_e1': PSF_chrom_e1, 'eff_psf_e2': PSF_chrom_e2, 'eff_psf_R': PSF_chrom_R,
                        'psf_star_avg_stamps':psf_star_avg_stamps, 'avg_star_flux': avg_star_flux}


            #print(len(star_ids), len(psf_star_stamps), len(psf_star_avg_stamps), len(star_flux))
            #print(len(avg_star_flux))
            #print(len(SCAs), len(x_pos), len(y_pos), len(flux))
            #print(len(random_indices[start_idx:end_idx]), len(PSF_chrom_e1), len(PSF_chrom_e2), len(PSF_chrom_R))
            
            #star_dict = { 'star_id':star_ids, 'psf_star_stamps': psf_star_stamps,
                         #'star_flux': star_flux ,'gal_host_id':  gal_star_ids}
            star_dict = {} ## for second run
            shear_dict = {'gal': {'e1': shear_chrom_e1, 'e2': shear_chrom_e2, 'R_e1': shear_chrom_Resp_e1, 'R_e2': shear_chrom_Resp_e2 },
                         'star': {'e1': shear_star_e1, 'e2': shear_star_e2, 'R_e1': shear_star_Resp_e1, 'R_e2': shear_star_Resp_e2 }}

            df_gal = pd.DataFrame( gal_dict)
            df_star = pd.DataFrame(star_dict)
            df_shear = pd.DataFrame(shear_dict)

            if (l + 1) == num_steps:
                dict_sims = {'gal_df': df_gal, 'star_df': df_star, 'shear_df': df_shear }
                file_name = open(out_file, 'wb')
                pickle.dump(dict_sims, file_name)
                file_name.close()
            else:
                file_name = open(out_file, 'rb')
                dict_sims = pickle.load(file_name)
                file_name.close()
                df_gal = pd.concat([df_gal, dict_sims['gal_df']])
                df_star = pd.concat([df_star, dict_sims['star_df']])
                df_shear = pd.concat([df_shear, dict_sims['shear_df']])
                dict_sims = {'gal_df': df_gal, 'star_df': df_star, 'shear_df': df_shear }
                file_name = open(out_file, 'wb')
                pickle.dump(dict_sims, file_name)
                file_name.close()
            df_gal, df_star, df_shear = None, None, None
            dict_sims = None
                
            shear_chrom_e1, shear_chrom_e2, shear_chrom_Resp_e1,shear_chrom_Resp_e2 , PSF_chrom_e1, PSF_chrom_e2, PSF_chrom_R = [], [], [], [], [], [], []
            shear_star_e1, shear_star_e2, shear_star_Resp_e1,shear_star_Resp_e2,  PSF_star_e1, PSF_star_e2, PSF_star_R = [], [], [], [], [], [], []
            shear_star_norm_e1, shear_star_norm_e2, shear_star_norm_Resp_e1,shear_star_norm_Resp_e2, PSF_star_e1, PSF_star_e2, PSF_star_R = [], [], [], [], [], [], []
            stamps, psf_stamps, psf_star_stamps, psf_star_avg_stamps = [], [], [], []
            bad_images = []
            z = []
            gal_ids, star_ids, gal_star_ids = [], [], []
            star_flux, flux = [], []
            avg_star_flux = []
            
            gal_ids, star_ids, gal_star_ids = [], [], []
            SCAs, x_pos, y_pos = [],[], []
            
            
    
    
    #return dict_sims



filter_name = 'K213'

if filter_name == 'Z087':
    sigma_arcsec = 0.073*1.15
    nwaves = 6
    mag_min, mag_max = 17.85, 21.37
if filter_name == 'Y106':
    sigma_arcsec =  0.087*1.15
    nwaves = 10
    mag_min, mag_max = 17.9, 21.2
if filter_name == 'J129':
    sigma_arcsec =  0.105*1.15
    mag_min, mag_max = 17.9, 21.2
    nwaves = 10
if filter_name == 'H158':
    sigma_arcsec =  0.127*1.15
    nwaves = 5
    mag_min, mag_max = 17.9, 21.2
if filter_name == 'F184':
    sigma_arcsec = 0.151*1.15
    nwaves = 5
    mag_min, mag_max = 17.2, 20.8
if filter_name == 'W146':
    sigma_arcsec = 0.12*1.15#0.105*1.15
    nwaves = 12
    mag_min, mag_max = 19.1, 21.2
    
if filter_name == 'K213':
    sigma_arcsec = 0.175*1.15
    nwaves = 10
    mag_min, mag_max = 17.0, 20.6
    
print(filter_name)
import time
start_time = time.time()
file_out = '10000Gal_NoNoise_' + filter_name + '_RealisticSEDs2_g10_g20.pickle'
shearEst(num_gal = 10000, filter_name = filter_name, scale = roman.pixel_scale/4, 
                       nwaves = nwaves, nx = 64*2, ny = 64*2, shear_value = 0.00,
                       sigma_arcsec =sigma_arcsec, mag_min = mag_min, mag_max = mag_max, out_file = file_out, start_idx = 0 )
end_time = time.time()
print(end_time - start_time )
