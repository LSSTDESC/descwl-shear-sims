"""
simple example with ring test (rotating intrinsic galaxies)
"""
import os
import numpy as np
from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.galaxies import WLDeblendGalaxyCatalog   # one of the galaxy catalog classes
from descwl_shear_sims.stars import StarCatalog                 # star catalog class
from descwl_shear_sims.psfs import make_ps_psf,make_fixed_psf   # for making a power spectrum PSF
from descwl_shear_sims.sim import get_se_dim                    # convert coadd dims to SE dims


ifield=0
itest=0

rng = np.random.RandomState(ifield)

coadd_dim = 400
buff   = 50
rotate = False
dither = False
psf_vary=False


nrot= 4
g1_list=[0.02,-0.02]
band_list=['r', 'i', 'z']
rot_list=[np.pi/nrot*i for i in range(nrot)]
nshear=len(g1_list)


if itest==0:
    args={
            'cosmic_rays':False,
            'bad_columns':False,
            'star_bleeds':False,
    }
    star_catalog=None
elif itest==1:
    args={
            'cosmic_rays':False,
            'bad_columns':False,
            'star_bleeds':False,
    }
    star_catalog = StarCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        density=(ifield%1000)/10+1,
        layout='random_circle',
    )
elif itest==2:
    args={
            'cosmic_rays':True,
            'bad_columns':True,
            'star_bleeds':True,
    }
    star_catalog = StarCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        density=(ifield%1000)/10+1,
        layout='random_circle',
    )
else:
    raise ValueError('itest must be 0, 1 or 2 !!!')

os.makedirs('outputs/test%d' %itest,exist_ok=True)


# galaxy catalog; you can make your own
galaxy_catalog = WLDeblendGalaxyCatalog(
    rng=rng,
    coadd_dim=coadd_dim,
    buff=buff,
    layout='random_circle',
)


if psf_vary:
    # this is the single epoch image sized used by the sim, we need
    # it for the power spectrum psf
    se_dim = get_se_dim(coadd_dim=coadd_dim, rotate=rotate, dither=dither)
    psf = make_ps_psf(rng=rng, dim=se_dim)
else:
    psf = make_fixed_psf(psf_type='moffat')


for irot in range(nrot):
    for ishear in range(nshear):
        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            star_catalog=star_catalog,
            coadd_dim=coadd_dim,
            g1=g1_list[ishear],
            g2=0.00,
            psf=psf,
            dither=dither,
            rotate=rotate,
            bands=band_list,
            noise_factor=0.,
            theta0=rot_list[irot],
            **args
        )
        for bb in band_list:
            sim_data['band_data'][bb][0].\
                writeFits('outputs/test%d/field%04d_shear%d_rot%d_%s.fits' \
                %(itest,ifield,ishear,irot,bb))
