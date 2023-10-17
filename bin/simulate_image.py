#!/usr/bin/env python
#
# simple example with ring test (rotating intrinsic galaxies)
# Copyright 20230916 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
import gc
import glob
import json
import os
import pickle
from argparse import ArgumentParser
from configparser import ConfigParser, ExtendedInterpolation

import fitsio
import fpfs
import numpy as np
import schwimmbad

from descwl_shear_sims.galaxies import \
    WLDeblendGalaxyCatalog  # one of the galaxy catalog classes
from descwl_shear_sims.psfs import (  # for making a power spectrum PSF
    make_fixed_psf, make_ps_psf)
from descwl_shear_sims.shear import ShearRedshift
# convert coadd dims to SE dims
from descwl_shear_sims.sim import get_se_dim, make_sim

band_list = ["g", "r", "i", "z"]
nband = len(band_list)


class Worker:
    def __init__(self, config_name):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        # layout of the simulation (random, random_disk, or hex)
        # see details in descwl-shear-sims
        self.layout = cparser.get("simulation", "layout")
        # image root directory
        self.img_root = cparser.get("simulation", "img_root")
        # Whether do rotation or dithering
        self.rotate = cparser.getboolean("simulation", "rotate")
        self.dither = cparser.getboolean("simulation", "dither")
        # version of the PSF simulation: 0 -- fixed PSF; 1 -- variational PSF
        self.psf_version = cparser.getint("simulation", "psf_version")
        # length of the exposure
        self.coadd_dim = cparser.getint("simulation", "coadd_dim")
        # buffer length to avoid galaxies hitting the boundary of the exposure
        self.buff = cparser.getint("simulation", "buff")
        # number of rotations for ring test
        self.nrot = cparser.getint("simulation", "nrot")
        # number of redshiftbins
        self.nzbin = cparser.getint("simulation", "nzbin")
        self.rot_list = [np.pi / self.nrot * i for i in range(self.nrot)]
        self.test_name = cparser.get("simulation", "test_name")
        self.shear_mode_list = json.loads(
            cparser.get("simulation", "shear_mode_list")
        )
        self.z_bounds = json.loads(
            cparser.get("simulation", "z_bounds")
        )
        self.nshear = len(self.shear_mode_list)
        return

    def run(self, ifield=0):
        print("Simulating for field: %d" % ifield)
        rng = np.random.RandomState(ifield)
        scale = 0.2

        if self.psf_version == 0:
            # basic test
            kargs = {
                "cosmic_rays": False,
                "bad_columns": False,
                "star_bleeds": False,
                "draw_method": "auto",
            }
            star_catalog = None
            psf = make_fixed_psf(psf_type="moffat")  # .shear(e1=0.02, e2=-0.02)
            psf_fname = "%s/PSF_%s_32.fits" % (self.img_root, self.test_name)
            if not os.path.isfile(psf_fname):
                psf_data = psf.shift(
                    0.5 * scale,
                    0.5 * scale
                ).drawImage(nx=64, ny=64, scale=scale).array
                fitsio.write(psf_fname, psf_data)

        elif self.psf_version == 1:
            # spatial varying PSF
            kargs = {
                "cosmic_rays": False,
                "bad_columns": False,
                "star_bleeds": False,
                "draw_method": "auto",
            }
            star_catalog = None
            # this is the single epoch image sized used by the sim, we need
            # it for the power spectrum psf
            se_dim = get_se_dim(
                coadd_dim=self.coadd_dim, rotate=self.rotate, dither=self.dither
            )
            psf = make_ps_psf(rng=rng, dim=se_dim)
            psf_fname = "%s/PSF_%s.pkl" % (self.img_root, self.test_name)
            if not os.path.isfile(psf_fname):
                with open(psf_fname, "wb") as f:
                    pickle.dump(
                        {"psf": psf},
                        f,
                    )
        else:
            raise ValueError("psf_version must be 0 or 1")

        img_dir = "%s/%s" % (self.img_root, self.test_name)
        os.makedirs(img_dir, exist_ok=True)
        nfiles = len(glob.glob(
            "%s/image-%05d_g1-*" % (img_dir, ifield)
        ))
        if nfiles == self.nrot * self.nshear * nband:
            print("We aleady have all the images for this subfield.")
            return

        # galaxy catalog; you can make your own
        galaxy_catalog = WLDeblendGalaxyCatalog(
            rng=rng,
            coadd_dim=self.coadd_dim,
            buff=self.buff,
            layout=self.layout,
        )
        print("Simulation has galaxies: %d" % len(galaxy_catalog))
        for shear_mode in self.shear_mode_list:
            for irot in range(self.nrot):

                shear_obj = ShearRedshift(
                    mode=shear_mode,
                    z_bounds=self.z_bounds,
                    g_dist="g1", # need to enable users to set this value
                )
                sim_data = make_sim(
                    rng=rng,
                    galaxy_catalog=galaxy_catalog,
                    star_catalog=star_catalog,
                    coadd_dim=self.coadd_dim,
                    shear_obj=shear_obj,
                    psf=psf,
                    dither=self.dither,
                    rotate=self.rotate,
                    bands=band_list,
                    noise_factor=0.0,
                    theta0=self.rot_list[irot],
                    **kargs
                )
                # write galaxy images
                for band_name in band_list:
                    gal_fname = "%s/image-%05d_g1-%d_rot%d_%s.fits" % (
                        img_dir,
                        ifield,
                        shear_mode,
                        irot,
                        band_name,
                    )
                    mi = sim_data["band_data"][band_name][0].getMaskedImage()
                    gdata = mi.getImage().getArray()
                    fpfs.io.save_image(gal_fname, gdata)
                    del mi, gdata, gal_fname
                del sim_data
                gc.collect()
        del galaxy_catalog, psf
        return


if __name__ == "__main__":
    parser = ArgumentParser(description="simulate blended images")
    parser.add_argument(
        "--min_id",
        default=0,
        type=int,
        help="minimum id number, e.g. 0",
    )
    parser.add_argument(
        "--max_id",
        default=5000,
        type=int,
        help="maximum id number, e.g. 4000",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="configure file name",
    )
    #
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
    )
    group.add_argument(
        "--mpi",
        dest="mpi",
        default=False,
        action="store_true",
        help="Run with MPI.",
    )
    cmd_args = parser.parse_args()
    min_id = cmd_args.min_id
    max_id = cmd_args.max_id
    pool = schwimmbad.choose_pool(mpi=cmd_args.mpi, processes=cmd_args.n_cores)
    idlist = list(range(min_id, max_id))
    worker = Worker(cmd_args.config)
    for r in pool.map(worker.run, idlist):
        pass
    pool.close()
