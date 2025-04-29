import argparse
import numpy as np
import sys, os
import math
import logging
import time
import galsim
import galsim.roman as roman
import datetime
#import fpfs
import pickle

import matplotlib.pyplot as plt

from astropy.io import fits
from matplotlib.colors import LogNorm
from astropy.table import Table
from astropy.visualization import simple_norm
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
import matplotlib.colors as colors
import h5py 
import gzip


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


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run simulations for specified filter and shear.')
parser.add_argument('--filter_name', type=str, required=True, help='Name of the filter to use.')
parser.add_argument('--shear_value', type=float, default=0.0, help='Value of the shear to apply.')
parser.add_argument('--scale', type=float, default=roman.pixel_scale/4, help='Pixel scale to draw.')
parser.add_argument('--nx', type=int, default=440, help='Stamp size width.')
parser.add_argument('--ny', type=int, default=440, help='Stamp size height.')
parser.add_argument('--num_gal', type=float, default=10000, help='Number of galaxies to draw.')
parser.add_argument('--start_idx', type=int, default=0, help='Index to start drawing.')
parser.add_argument('--end_index', type=int, default=10000, help='Index to start drawing.')
parser.add_argument('--drawPSF', type=bool, default=False, help='To draw and save PSF.')
parser.add_argument('--drawGal', type=bool, default=False, help='To draw and save PSF.')
parser.add_argument('--saveInfo', type=bool, default=False, help='To draw and save PSF.')

args = parser.parse_args()

filter_name = args.filter_name
shear_value = args.shear_value
drawPSF = args.drawPSF
drawGal = args.drawGal
saveInfo = args.saveInfo
scale = args.scale
nx, ny = args.nx, args.ny
num_gal = args.num_gal
start_idx = args.start_idx
end_index = args.end_index



use_filters = 'ZYJHFWK'
roman_filters = roman.getBandpasses(AB_zeropoint=True)

# Get the names of the ones we will use here.
filters = [filter_name for filter_name in roman_filters if filter_name[0] in use_filters]

# get noisy fluxes
# read cuts, magsm redshift                                                        
scenario = 'A'
pickle_out = open("../Gal_props_catnoise/roman_gal_obsmags_cosmodc2_10067_10000cut","rb")
roman_mag_info = pickle.load(pickle_out)
pickle_out.close()

gal_ids = roman_mag_info['gal_id']
z = roman_mag_info['z']
obs_mags =  roman_mag_info['obs_mag'][scenario]
mag_cut,rand_idx = roman_mag_info['cut'], roman_mag_info['rand_ind']



## galaxy catalog
data_dir = 'cosmoDC2/'
gal_data = pd.read_parquet(data_dir + 'galaxy_10067_subset_fullsed.parquet', engine='pyarrow')
flux_data = pd.read_parquet(data_dir +'galaxy_flux_10067_subset_fullsed.parquet', engine='pyarrow')

gal_data = gal_data[mag_cut].reset_index(drop=True)
flux_data = flux_data[mag_cut].reset_index(drop=True)

gal_data = gal_data.iloc[rand_idx].reset_index(drop=True)
flux_data = flux_data.iloc[rand_idx].reset_index(drop=True)
list_len = len(flux_data)

## read SED 

filename = 'cosmoDC2/' +"sed_fit_10067.h5"
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
#pickle_out = open("cosmoDC2/galaxy_sed_10067_emissionlines.pickle","rb")
#f = pickle.load(pickle_out)
#pickle_out.close()

#data = f['galaxy']
#wave_list = f['meta']['wave_list']


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


pickle_out = open("Gal_props_catnoise/roman_star_obsmags_diffsky","rb")
roman_star_mag_info = pickle.load(pickle_out)
pickle_out.close()
star_snr = roman_star_mag_info['snr'][scenario]


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
def write_fits(images, fileout):
    primary_hdu = fits.PrimaryHDU(images[0])
    # Create ImageHDU objects for the rest of the images
    hdu_list = [primary_hdu] + [fits.CompImageHDU(img, compression_type='RICE_1') for img in images[1:]]
    # Create an HDUList object to contain the HDUs
    hdul = fits.HDUList(hdu_list)
    # Write the HDUList to a FITS file
    hdul.writeto(fileout + '.fits', overwrite=True)

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
    return images



def sim(num_gal = 1000, filter_name = 'H158', add_seed = 0, scale = roman.pixel_scale, SCA = 7, nwaves = 10, nx = 64, ny = 64, shear_value = 0.02,sigma_arcsec = 0.175, stellar_SEDs = None, indx_seed = 9999, mag_min=17 , mag_max =22, out_file = 'out' , start_index = 0,true_psf_dir= None, star_psf_dir= None, drawPSF = False ,  root_dir = None, end_index = 10000, drawGal = False,
       saveInfo = False):
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
    #random_indices = np.random.choice(np.arange(list_len), num_gal, replace = False)
    #random_star_indices = np.random.choice(np.arange(list_len), num_gal*num_stars, replace = False)
    bp500 = galsim.Bandpass(galsim.LookupTable([499, 500, 501], [0, 1, 0]),
                                             wave_type='nm').withZeropoint('AB')
    #for l in range(start_idx, len(random_indices)):
    #for l in range(start_idx, 5000):
    #for l in range(9000, len(random_indices)):
    #for l in range(start_idx, len(random_indices)):
    for l in range(start_index, end_index):
    #for l in range(start_idx, 100):
    #for l in range(1000):
        i = l
        #print(i)
        seed = i + add_seed#np.random.randint(0, 81499, 1)[0]
        np.random.seed(seed)
        gal_id = str(np.array(gal_data['id'])[i])
        gal_ids.append(gal_id)
        
        ## make_galaxy
        redshift = np.array(gal_data['redshift'])[i]
        bulge_hlr = np.array(gal_data['spheroidHalfLightRadiusArcsec'])[i]
        disk_hlr = np.array(gal_data['diskHalfLightRadiusArcsec'])[i]
        disk_shear1, disk_shear2 = np.array(gal_data['diskEllipticity1'])[i], np.array(gal_data['diskEllipticity2'])[i]
        bulge_shear1, bulge_shear2 = np.array(gal_data['spheroidEllipticity1'])[i], np.array(gal_data['spheroidEllipticity2'])[i]
        
        
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
        
        bulge = galsim.Sersic(4, half_light_radius=bulge_hlr).shear(g1 =bulge_shear1 , g2 = bulge_shear2)
        disk = galsim.Sersic(1, half_light_radius=disk_hlr).shear(g1 =disk_shear1 , g2 = disk_shear2)
        obj = bulge*bulge_sed + disk*(disk_sed)
        if f_disk_magnorm >= 100:
            obj = bulge*bulge_sed
        if f_bulge_magnorm >= 100:
            obj = disk*disk_sed
           
        obj_flux = np.array(flux_data['roman_flux_' + filter_name_col])[i]
        obj = obj.withFlux(obj_flux, bandpass)
        z.append(redshift)
 
        
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
            if star_snr[filter_name][l*num_stars + j] > 100  and star_mag > mag_min:
                star_seds.append(sed)
                star_ids.append(str(np.array(star_obj['id'])[j]))
                gal_star_ids.append(gal_id)
        # The rng for photon shooting should be different for each filter.
        if len(star_seds) == 0:
                star_seds.append(sed)
                star_ids.append(str(np.array(star_obj['id'])[j]))
                gal_star_ids.append(gal_id)
        phot_rng = galsim.UniformDeviate(seed + ord(filter_name[0])*num_gal)


        pos_rng = galsim.UniformDeviate(i + 1e6 + add_seed)
        x = pos_rng() * roman.n_pix
        y = pos_rng() * roman.n_pix
        image_pos = galsim.PositionD(x,y)
        x_pos.append(x)
        y_pos.append(y)
        
        
        #galsim.DeltaFunction()
        SCA = np.random.randint(1, 19, 1)[0]
        psfInt = roman.getPSF(SCA, filter_name, SCA_pos = image_pos, n_waves=nwaves,  pupil_bin=8)
        pixel_response = galsim.Pixel(roman.pixel_scale)
        eff_psf = galsim.Convolve(psfInt, pixel_response) 
        gal_sed_psf = galsim.DeltaFunction()*obj.sed 
        gal_eff_psf = galsim.Convolve(gal_sed_psf, eff_psf)
        offset = galsim.PositionD(0.5, 0.5)
        if drawPSF:
            psfData = gal_eff_psf.drawImage(bandpass, nx=nx, ny=ny, scale=scale, method = 'no_pixel', offset = offset)#.array
            psfData_star, stars_flux = [], []
            #for j in range(len(star_seds)):
            #    star_sed_psf = galsim.DeltaFunction()*star_seds[j]
            #    eff_psf_star = galsim.Convolve(star_sed_psf, eff_psf)
            #    psf_star = eff_psf_star.drawImage(bandpass, nx=nx, ny=ny, scale=scale, method = 'no_pixel', offset = offset)#.array
            avg_star_seds = star_seds[0]
            for j in range(1, len(star_seds)):
                avg_star_seds += star_seds[j]
            avg_star_seds /= len(star_seds)
            star_sed_psf = galsim.DeltaFunction()*avg_star_seds
            eff_psf_star = galsim.Convolve(star_sed_psf, eff_psf)
            psf_star = eff_psf_star.drawImage(bandpass, nx=nx, ny=ny, scale=scale, method = 'no_pixel', offset = offset)#.array
            stars_flux.append(np.sum(psf_star.array))
            psfData_star.append(psf_star.array)
        
        
            psfData = psfData.array#/np.sum(psfData.array)
        
        
        # Four Galaxies to cancel spin-2 and spin-4 ansiotropies
        # spin-2 is shape noise in diagnonal elements of shear response matrix
        # spin-4 is shape noise in diagnonal and of-diagnoal elements of shear response matrix, 
        # but an order of magnitude smaller than spin-2
            psf_stamps.append(psfData)
            psf_star_stamps.extend(psfData_star)
            star_flux.extend(stars_flux)
        SCAs.append(SCA)
        
        if drawGal:
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
                    final = galsim.Convolve(obj_rot,eff_psf)
                    gal_data_rot = final.drawImage(bandpass, nx=nx, ny=ny, scale=scale, offset = offset, method = 'no_pixel').array
                    if j == 0:
                        galaxy_data = gal_data_rot
                    else:
                        galaxy_data = np.hstack([galaxy_data,gal_data_rot ])
                #flux.append(np.sum(gal_data_rot))
                stamps.append(galaxy_data)


                bad_images.append(0)
            except:
                ###
                num_objects = 4
                for j in range(num_objects):
                    #theta = np.random.rand()*np.pi * 2.0 * galsim.radians
                    if j == 0:
                        ang = (np.random.uniform(0.0, np.pi * 2.0)) * galsim.radians
                    else:
                        ang = np.pi / 4 * galsim.radians
                    obj = obj.rotate(ang)
                    obj_rot = obj.shear(g1=shear_value, g2=0.0)
                    big_fft_params = galsim.GSParams(maximum_fft_size=12300)
                    final = galsim.Convolve(obj_rot,eff_psf, gsparams=big_fft_params)
                    gal_data_rot = final.drawImage(bandpass, nx=nx, ny=ny, scale=scale, offset = offset, method = 'no_pixel').array
                    if j == 0:
                        galaxy_data = gal_data_rot
                    else:
                        galaxy_data = np.hstack([galaxy_data,gal_data_rot ])
                ###
                stamps.append(galaxy_data)
                bad_images.append(0)
            
        if drawPSF:
            psf_star_avg = np.average(psfData_star, axis = 0)
            psf_star_avg_stamps.append(psf_star_avg)
            #avg_star_flux.append(np.sum(psf_star_avg))
        
        flux.append(obj_flux)
        star_sed_flux = 0
        for j in range(len(star_seds)):
            star_sed_flux += star_seds[j].calculateFlux(bandpass)
        avg_star_flux.append(star_sed_flux/len(star_seds))

        num_steps = 1000
        if (l+1) % num_steps == 0 and l != 0 :

            flux = np.array(flux)
            avg_star_flux = np.array(avg_star_flux)
            z = np.array(z)
            gal_ids = np.array(gal_ids)
            
            start_idx = int((l+1) / num_steps - 1)*num_steps #+1
            end_idx = l + 1
            if (l+1) == num_steps:
                start_idx = 0
                end_idx = l + 1
            
            
            gal_per_group = 1000
            min_l = int(l/gal_per_group)*gal_per_group
            max_l =  min_l + gal_per_group - 1
            gal_file = out_file + 'galStamps_' + str(min_l) + '_' + str(max_l )
            truePSF_file = true_psf_dir + 'PSF_' + str(min_l) + '_' + str(max_l )
            starPSF_file = star_psf_dir + 'PSF_' + str(min_l) + '_' + str(max_l )
            table_file = root_dir + '/gal_info.pickle'
                        
            if saveInfo:
                # removed bad_images
                gal_dict = {'galaxy_id': gal_ids, 'redshift': z,'x_pos': x_pos, 'y_pos': y_pos,
                                'flux': flux, 'SCA': SCAs,  'avg_star_flux': avg_star_flux}

                df_gal = pd.DataFrame(gal_dict)
            

                if (l + 1) == num_steps:    
                    file_name = open(table_file, 'wb')
                    pickle.dump(df_gal, file_name)
                    file_name.close()
                else:
                    file_name = open(table_file, 'rb')
                    dict_sims = pickle.load(file_name)
                    file_name.close()
                    df_gal = pd.concat([dict_sims, df_gal])
                    file_name = open(table_file, 'wb')
                    pickle.dump(df_gal, file_name)
                    file_name.close()
            
            if (l + 1) == (num_steps +min_l):
                if drawGal:
                    write_fits(stamps, gal_file)
                if drawPSF:
                    write_fits(psf_stamps, truePSF_file)
                    write_fits(psf_star_avg_stamps, starPSF_file)
                    
            else:
                if drawGal:
                    saved_stamps = read_fits(gal_file)
                    write_fits(saved_stamps + stamps, gal_file)
                if drawPSF:
                    saved_truePSF = read_fits(truePSF_file)
                    saved_starPSF = read_fits(starPSF_file)
                    write_fits(saved_truePSF + psf_stamps, truePSF_file)
                    write_fits(saved_starPSF + psf_star_avg_stamps, starPSF_file)
                    
                
                
            stamps, psf_stamps, psf_star_stamps, psf_star_avg_stamps = [], [], [], []
            bad_images = []
            z = []
            gal_ids, star_ids, gal_star_ids = [], [], []
            star_flux, flux = [], []
            avg_star_flux = []
            
            gal_ids, star_ids, gal_star_ids = [], [], []
            SCAs, x_pos, y_pos = [],[], []
            
            
    
#filter_name = 'J129'

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
    mag_min, mag_max = 19.1, 22.4
    
if filter_name == 'K213':
    sigma_arcsec = 0.175*1.15
    nwaves = 10
    mag_min, mag_max = 17.0, 20.6
    
print(filter_name)
import time
start_time = time.time()
shear_str = '0'
if shear_value == 0.02:
    shear_str = '02'
if shear_value == -0.02:
    shear_str = 'n02'
    
root_dir = '../sim_images/10000Gal_NoNoise_cosmoDC2_final/' + filter_name
dir_out = root_dir + '/g1' + shear_str + '_g20/'
ensure_directory_exists(dir_out)
true_psf_dir = root_dir + '/TruePSF/'
star_psf_dir = root_dir + '/StarPSF/'
print(shear_str, drawPSF)
if drawPSF:
    ensure_directory_exists(true_psf_dir)
    ensure_directory_exists(star_psf_dir)
    
sim(num_gal = num_gal, filter_name = filter_name, scale = scale, 
                       nwaves = nwaves, nx = nx, ny = ny, shear_value = shear_value,
                       sigma_arcsec =sigma_arcsec, mag_min = mag_min, mag_max = mag_max, out_file = dir_out,
                       start_index = start_idx, true_psf_dir= true_psf_dir, star_psf_dir= star_psf_dir, drawPSF = drawPSF,
                       root_dir = root_dir, end_index = end_index, drawGal = drawGal, saveInfo = saveInfo)
end_time = time.time()
print(end_time - start_time )







