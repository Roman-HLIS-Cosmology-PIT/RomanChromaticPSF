import numpy as np
import sys, os
import math
import logging
import time
import galsim
import galsim.roman as roman
import datetime
import fpfs

import matplotlib.pyplot as plt
%matplotlib inline

from astropy.io import fits
from matplotlib.colors import LogNorm
from astropy.table import Table


def runSim(filters, nobj, seed =12345, SCA = 7,  chrom = True, xdim = roman.n_pix, ydim = roman.n_pix):
    
    # list of outputs: individual images before convolving with PSF, index of object type and positions. List of images for different bands
    obj_list = []
    obj_types = []
    obj_pos = []
    full_images = []  # List of images for different bands
    use_filters = filters
    nobj = nobj
    seed = seed
    seed1 = galsim.BaseDeviate(seed).raw()
    use_SCA = SCA
    xdim = xdim
    ydim = ydim


    # Use a logger to output some information about the run.
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logger = logging.getLogger("demo13")
    logging_levels = { 0: logging.CRITICAL,
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    level = logging_levels[args.verbosity]
    logger.setLevel(level)

    # Read in the Roman filters, setting an AB zeropoint appropriate for Roman
    roman_filters = roman.getBandpasses(AB_zeropoint=True)
    logger.debug('Read in Roman imaging filters.')

    # Get the names of the filters to use
    filters = [filter_name for filter_name in roman_filters if filter_name[0] in use_filters]
    logger.debug('Using filters: %s',filters)

    # Bandpass for flux normalization of stars
    y_bandpass = roman_filters['Y106']

    # read cosmos catalogs

    cat1 = galsim.COSMOSCatalog(sample='25.2', area=roman.collecting_area, exptime=roman.exptime)
    cat2 = galsim.COSMOSCatalog(sample='23.5', area=roman.collecting_area, exptime=roman.exptime)  
  
    logger.info('Read in %d galaxies from I<25.2 catalog'%cat1.nobjects)
    logger.info('Read in %d galaxies from I<23.5 catalog'%cat2.nobjects)

    # Use the vega SED for stars
    vega_sed = galsim.SED('vega.txt', 'nm', 'flambda')

    # Pick a ra, dec target position and date. Use one form Galsim example.
    ra_targ = galsim.Angle.from_hms('16:01:41.01257')
    dec_targ = galsim.Angle.from_dms('66:48:10.1312')
    targ_pos = galsim.CelestialCoord(ra=ra_targ, dec=dec_targ)
    date = datetime.datetime(2025, 5, 16)

    # WCS for this particular SCA, position and date
    wcs_dict = roman.getWCS(world_pos=targ_pos, SCAs=use_SCA, date=date)
    wcs = wcs_dict[use_SCA]

    # loop through the filters to draw objects.
    for ifilter, filter_name in enumerate(filters):

        logger.info('Beginning work for {0}.'.format(filter_name))

        bandpass = roman_filters[filter_name]

        # Create chromatic or achromatic psf
        logger.info('Building PSF for SCA %d, filter %s.'%(use_SCA, filter_name))
        eff_wave = roman_filters[filter_name].effective_wavelength
        if chrom:
            psf = roman.getPSF(use_SCA, filter_name, n_waves=20, wcs=wcs, pupil_bin=8)
        else:
            psf = roman.getPSF(use_SCA,filter_name,  wavelength= eff_wave, wcs=wcs, pupil_bin=8)

        # Set up the full image for the galaxies/stars
        full_image = galsim.ImageF(xdim, ydim, wcs = wcs)

        # Seperate sky image for getting the noise of full image
        sky_image = galsim.ImageF(xdim, ydim,  wcs = wcs)

        # rng for image-level stuff
        image_rng = galsim.UniformDeviate(seed1 + ifilter * nobj)


        # First we get the amount of zodaical light for a position corresponding to the center of SCA
        SCA_cent_pos = wcs.toWorld(sky_image.true_center)
        sky_level = roman.getSkyLevel(bandpass, world_pos=SCA_cent_pos)
        sky_level *= (1.0 + roman.stray_light_fraction)

        wcs.makeSkyImage(sky_image, sky_level)

        # Add the expected thermal backgrounds in this band.
        sky_image += roman.thermal_backgrounds[filter_name]*roman.exptime

        # Draw the objects into the image, keeping same positions for all filters.
        for i_obj in range(nobj):
            logger.info('Drawing image for object {} in band {}'.format(i_obj, filter_name))

            obj_rng = galsim.UniformDeviate(seed + 1 + 10**6 + i_obj)
            # The rng for photon shooting should be different for each filter.
            phot_rng = galsim.UniformDeviate(seed1 + 1 + i_obj + ifilter*nobj)

            # deals probability of being faint, bright galaxy, or star
            p = obj_rng()

            # Pick a random position in the image to draw it.
            x = obj_rng() * roman.n_pix
            y = obj_rng() * roman.n_pix
            image_pos = galsim.PositionD(x,y)
            image_pos_psf = galsim.PositionD(x,y)
            logger.debug('Position = %s',image_pos)
            
            if ifilter == 0:
                obj_pos.append((x,y))

            # 80% faint galaxies, 10% stars, 10% bright galaxies
            if p < 0.8:
                # Faint galaxy
                #print('Faint')
                logger.debug('Faint galaxy')

                # Select a random galaxy from the catalog1, which has fainter objects.
                obj = cat1.makeGalaxy(chromatic=True, gal_type='parametric', rng=obj_rng)
                logger.debug('galaxy index = %s',obj.index)

                # Rotate the galaxy randomly
                theta = obj_rng() * 2 * np.pi * galsim.radians
                logger.debug('theta = %s',theta)
                obj = obj.rotate(theta)

                #test_stamp = obj.drawImage(bandpass, center=image_pos, wcs=wcs.local(image_pos),
                #                    method='phot', rng=phot_rng)

                #mom_test = galsim.hsm.FindAdaptiveMom(test_stamp, strict = False)
                #print(mom_test.observed_shape.g1, mom_test.observed_shape.g2)

                #test_stamp_psf = psf_achrom.drawImage(center=image_pos_psf, wcs=wcs.local(image_pos_psf))

                #est_shear = galsim.hsm.EstimateShear(test_stamp,test_stamp_psf, shear_est = 'KSB', strict = False)
                #print(est_shear.corrected_g1, est_shear.corrected_g1)
                if ifilter == 0:
                    obj_types.append('Faint galaxy')
                    obj_list.append(obj)

            elif p < 0.9:
                # Star
                logger.debug('Star')

                # Use a log-normal distribution for the stellar fluxes.
                mu_x = 1.e5
                sigma_x = 2.e5
                mu = np.log(mu_x**2 / (mu_x**2+sigma_x**2)**0.5)
                sigma = (np.log(1 + sigma_x**2/mu_x**2))**0.5
                gd = galsim.GaussianDeviate(obj_rng, mean=mu, sigma=sigma)
                flux = np.exp(gd())
                logger.debug('flux = %s',flux)

                # Normalize the SED to have this flux in the Y band.
                sed = vega_sed.withFlux(flux, y_bandpass)

                obj = galsim.DeltaFunction() * sed
                
                if ifilter == 0:
                    obj_types.append('Star')
                    obj_list.append(obj)

            else:
                # Bright galaxy

                logger.debug('Bright galaxy')

                # Select a random galaxy from the catalog2, which has brighter objects.
                obj = cat2.makeGalaxy(chromatic=True, gal_type='parametric', rng=obj_rng)
                logger.debug('galaxy index = %s',obj.index)
                

                # To make bigger galaxies scale up the area by a factor of 2, and the flux by a factor of 4, more noticeable on final image
                obj = obj.dilate(2) * 4

                # Rotate the galaxy randomly
                theta = obj_rng() * 2 * np.pi * galsim.radians
                logger.debug('theta = %s',theta)
                obj = obj.rotate(theta)

                #test_stamp = obj.drawImage(bandpass, center=image_pos, wcs=wcs.local(image_pos),
                #                   method='phot', rng=phot_rng)
                #mom_test = galsim.hsm.FindAdaptiveMom(test_stamp, strict = False)
                #print(mom_test.observed_shape.g1, mom_test.observed_shape.g2, mom_test.moments_sigma)
                
                if ifilter == 0:
                    obj_types.append('Bright Galaxy')
                    obj_list.append(obj)

            # Convolve the chromatic object with the (chromatic/achromatic) PSF.
            final = galsim.Convolve(obj, psf)
            stamp = final.drawImage(bandpass, center=image_pos, wcs=wcs.local(image_pos),
                                    method='phot', rng=phot_rng)
            
            # Code below attempts at measuring shear
            #mom_test = galsim.hsm.FindAdaptiveMom(stamp, strict = False)
            #print('After Convolution', mom_test.observed_shape.g1, mom_test.observed_shape.g2, mom_test.moments_sigma)
            #stamp_psf = psf_achrom.drawImage(bandpass, center=image_pos, wcs=wcs.local(image_pos),
            #                        method='phot', rng=phot_rng)

            #stamp_psf = psf_achrom.drawImage(center=image_pos_psf, wcs=wcs.local(image_pos_psf))
            #galsim.hsm.EstimateShear(stamp, stamp_psf, shear_est = 'Linear', strict = False)

            #measure shear
            #psfData = stamp_psf.array
            #gal_data = stamp.array
            #fpTask = fpfs.image.measure_source(psfData, sigma_arcsec=0.6, pix_scale = roman.pixel_scale)
            #mms =  fpTask.measure(gal_data)
            #mms = fpTask.get_results(mms)
            #ells=   fpfs.catalog.fpfs_m2e(mms,const=2000)
            #resp=np.average(ells['fpfs_R1E'])
            #shear_g1=np.average(ells['fpfs_e1'])/resp
            #shear_g2=np.average(ells['fpfs_e2'])/resp

            #m_bias1.append(shear_g1/mom_test.observed_shape.g1)
            #m_bias2.append(shear_g1/mom_test.observed_shape.g2)
            #print('Mult Bias:', shear_g1, shear_g2)
            #shear = galsim.hsm.EstimateShear(stamp, stamp_psf, shear_est = 'KSB', strict = False)
            #print(shear.corrected_g1, shear.corrected_g2, shear.corrected_e1, shear.corrected_e2)
          
            # Find overlapping bounds between the large image and the individual stamp.
            bounds = stamp.bounds & full_image.bounds

            # Add stamp to full sky image
            full_image[bounds] += stamp[bounds]

        logger.info('All objects have been drawn for filter %s.',filter_name)

        logger.info('Adding the noise and detector non-idealities.')

        # photon number may not be integer, so quantize to make fluxes integers
        full_image.quantize()

        # Add the sky image.  Galaxies already have Poisson noise due to photon, but the sky image doesn't.
        poisson_noise = galsim.PoissonNoise(image_rng)
        sky_image_realized = sky_image.copy()
        sky_image_realized.addNoise(poisson_noise)
        full_image += sky_image_realized
        
        # To add all detector effects:
        #roman.allDetectorEffects(full_image, rng = image_rng)
        
        # Add invidual detector effects:

        # 1) Reciprocity failure:
        roman.addReciprocityFailure(full_image)
        logger.debug('Included reciprocity failure in {0}-band image'.format(filter_name))

        # 2) Adding dark current to the image:
        dark_current = roman.dark_current*roman.exptime
        dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(image_rng, dark_current))
        full_image.addNoise(dark_noise)
        sky_image += dark_current # (also want to subtract this expectation value along with sky)

        # 3) Applying a quadratic non-linearity:
        roman.applyNonlinearity(full_image)
        logger.debug('Applied nonlinearity to {0}-band image'.format(filter_name))

        # 4) Including Interpixel capacitance:
        roman.applyIPC(full_image)
        logger.debug('Applied interpixel capacitance to {0}-band image'.format(filter_name))

        # 5) Adding noise. For now no noise, so commented out:
        #read_noise = galsim.GaussianNoise(image_rng, sigma=roman.read_noise)
        #full_image.addNoise(read_noise)
        #logger.debug('Added readnoise to {0}-band image'.format(filter_name))

        # We divide by the gain to convert from e- to ADU. 
        full_image /= roman.gain
        sky_image /= roman.gain

        # Once again quantize values to integers
        full_image.quantize()
        sky_image.quantize()
        
        # Subtract background
        full_image -= sky_image

        logger.debug('Subtracted background for {0}-band image'.format(filter_name))

        logger.info('Completed {0}-band image.'.format(filter_name))
        full_images.append(full_image)
        dict_sim = {'images': full_images, 'obj_list':obj_list, 'obj_types': obj_types, 'obj_pos': obj_pos}
        
        return dict_sim

#example run
filters = 'Z' 
nobj = 1000
sim_dict = runSim(filters, nobj)

#plot image
fig = plt.figure(figsize=(8, 8))
plt.imshow(sim_dict['images'][0].array, cmap='gray', vmin = -50, vmax = 100)
plt.colorbar()
