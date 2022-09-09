from __future__ import print_function, absolute_import, division
from builtins import range  # overload range to ensure python3 style
from six import iteritems
import sys
import numpy as np
from tempfile import mkstemp
from shutil import move, copy
import os
import warnings
import astropy.units as u
import astropy.constants as constants
from astropy.coordinates import Galactic, Galactocentric, ICRS, LSR, SkyCoord
from astropy.coordinates import CartesianRepresentation
from . import magellanicstream

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

SOLAR_MASS = 1.989e33
CM_PER_MPC = 3.085678e24

UnitMass_in_g = 1e10*SOLAR_MASS; # 10^10 solar masses
UnitLength_in_cm = CM_PER_MPC/1000; # 1 kpc
UnitVelocity_in_cm_per_s = 1e5; # 1 km/sec

UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s;
UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2);

class Conversions(object):
    """
    Holds useful conversions for Gadget units
    -> from Gadget2 files copyright Volker Springel
    """
    def __init__(self,
                 UnitMass_in_g=1.989e43,  # 1.e10 solar masses
                 UnitVelocity_in_cm_per_s=1e5,  # 1 km/s
                 UnitLength_in_cm=3.085678e21):  # 1 kpc
        #constants
        self.ProtonMass = 1.6725e-24
        self.Boltzmann = 1.38066e-16
        self.Hydrogen_MassFrac = 0.76
        self.SolarMetallicity = 0.02
        #set unit system
        self.UnitMass_in_g = UnitMass_in_g
        self.UnitVelocity_in_cm_per_s = UnitVelocity_in_cm_per_s
        self.UnitLength_in_cm = UnitLength_in_cm
        #derived
        self.UnitTime_in_s = self.UnitLength_in_cm / self.UnitVelocity_in_cm_per_s
        self.UnitTime_in_Gyr = self.UnitTime_in_s / (3600.*24.*365.*10.**9.)
        self.UnitDensity_in_cgs = self.UnitMass_in_g / self.UnitLength_in_cm**3.
        self.UnitEnergy_in_cgs = self.UnitMass_in_g * self.UnitLength_in_cm**2. / self.UnitTime_in_s**2.
        self.G = 6.672e-8 / self.UnitLength_in_cm**3. * self.UnitMass_in_g * self.UnitTime_in_s**2.
        self.Hubble = 3.2407789e-18  * self.UnitTime_in_s

def transform_coords(pos, vel=None, return_coord="magellanic", return_vel=False):

    def _transform_positions(pos, vel, coord):
        gal_coords = SkyCoord(Galactocentric(x=pos[:, 0]*u.kpc, y=pos[:, 1]*u.kpc, z=pos[:, 2]*u.kpc,
                                    v_x=vel[:, 0]*u.km/u.s, v_y=vel[:, 1]*u.km/u.s, v_z=vel[:, 2]*u.km/u.s,
                                    z_sun=5.0*u.pc,galcen_distance=8.15*u.kpc,
                                    representation_type=CartesianRepresentation))

        ## LSR Radial Velocity
        transformed_lsr = gal_coords.transform_to(LSR)
        radial_vel = transformed_lsr.radial_velocity.value

        ## Galactic Coordinates
        if (coord == "galactic"):
            transformed = gal_coords.transform_to(Galactic)
            x = np.remainder(transformed.l.value+360,360) # shift RA values
            x[x > 180] -= 360    # scale conversion to [-180, 180]
            x=-x    # reverse the scale: East to the left
            dist = transformed.distance.value
            galactic = np.empty((len(x), 3))
            galactic[:,0] = dist
            galactic[:,1] = x
            galactic[:,2] = transformed.b.value

            return galactic, radial_vel

        ## Magellanic Coordinates
        if (coord == "magellanic"):
            transformed = gal_coords.transform_to(magellanicstream.MagellanicStream)
            lam = transformed.MSLongitude.value
            lam[lam > 180] -= 360
            Beta = transformed.MSLatitude.value
            dist = transformed.distance.value
            magellanic = np.empty((len(lam), 3))
            magellanic[:,0] = dist
            magellanic[:,1] = lam
            magellanic[:,2] = Beta

            return magellanic, radial_vel

        ## RA/Dec
        if (coord == "equatorial"):
            transformed = gal_coords.transform_to(ICRS)
            ra = transformed.ra.value
            dec = transformed.dec.value
            dist = transformed.distance.value
            equatorial = np.empty((len(ra), 3))
            equatorial[:,0] = dist
            equatorial[:,1] = ra
            equatorial[:,2] = dec

            return equatorial, radial_vel


        raise Exception("Unrecognized coord: {}".format(coord))

    ### Begin public function ###

    pos = np.array(pos)
    if (len(pos) == 0):
        raise Exception("No positions supplied.")
    inputndim = 2
    if (np.ndim(pos) == 1):
        inputndim = 1
        pos = np.array([pos])

    if (vel is None):
        if (return_vel):
            raise Exception("return_vel set to True, but no input velocities supplied.")
        vel = np.zeros_like(pos)
    
    coords, radial_vel = _transform_positions(pos, vel, return_coord)

    if (inputndim == 1):
        coords = coords[0]
    if return_vel:
        return coords, radial_vel
    else:
        return coords


# def calc_u(temp=10000):
#     """
#     Calculate internal energy using temperature and hydrogen fraction
#     This assumes full ionization (copied from Makegalaxy/effmodel.c)
#     """
#     Xh = 0.76
#     gamma = 5./3.

#     meanweight = 4/(8 - 5*(1-Xh))
#     u4 = 1/meanweight*1/(gamma-1)*(constants.k_B.cgs/constants.m_p.cgs)*temp*u.K

#     return u4.value * UnitMass_in_g / UnitEnergy_in_cgs

def calc_u(temp=10000):
    xe = 1

    Xh = 0.76
    gamma = 5./3.

    mu = (1 + Xh /(1-Xh)) / (1 + Xh/(4*(1-Xh)) + xe)*constants.m_p.cgs.value
    # mu = 4./(1 + 3*Xh + 4*Xh*xe)*constants.m_p.cgs.value
#     temp = (gamma - 1)*U/constants.k_B.cgs.value*mu

    return temp*constants.k_B.cgs.value/mu/(gamma - 1)/1e10

def set_ne(U, temp=10000):
    """
    Set electron density to match internal energy and desired temperature
    """
    ## internal units assumed

    energy = U*1e+10 # erg/g
    temp = temp # Kelvin

    Xh = 0.76
    gamma = 5./3.
    mu = (temp * constants.k_B.cgs.value) / ((gamma - 1)*energy)

    xe = (4*constants.m_p.cgs.value/mu - 1 - 3*Xh)*(1/(4.*Xh))

    return xe

def list_snapshots(snapids, folder, base):

    if isinstance(snapids, int):
        snaps = range(snapids+1)
        convert = lambda s: folder+base+str(s).zfill(3)
        snaps = map(convert, snaps)
    elif hasattr(snapids, '__iter__'):
        if all(isinstance(x, int) for x in snapids):
            if len(snapids) == 1:
                snaps = range(snapids[0]+1)
            elif len(snapids) == 2:
                snaps = range(snapids[0], snapids[1]+1)
            elif len(snapids) == 3:
                snaps = range(snapids[0], snapids[1], snapids[2])
                if snapids[1] == snaps[-1]+snapids[2]:
                    snaps.append(snapids[1])
            else:
                snaps = snapids
            convert = lambda s: folder+base+str(s).zfill(3)
            snaps = map(convert, snaps)
        elif all(isinstance(x, str) for x in snapids):
            snaps = snapids
    elif isinstance(snapids, str):
        snaps = snapids

    return snaps

def make_settings(**kwargs):
    #default settings
    settings = {'panel_mode': "xy",
                'log_scale': True,
                'in_min': -1,
                'in_max': 2.2,
                'com': False,
                'first_only': False,
                'gal_num': -1,
                'colormap': 'gnuplot',
                'xlen': 30,
                'ylen': 30,
                'zlen': 30,
                'colorbar': 'None',
                'NBINS': 512,
                'plotCompanionCOM': False,
                'plotPotMin': False,
                'parttype': 'stars',
                'filename': '',
                'outputname': '',
                'snapbase': 'snap_',
                'offset': [0, 0, 0],
                'im_func': None,
                'halo_center_method':'pot'}

    for name, val, in iteritems(kwargs):
        if name not in settings.keys():
            warnings.warn("WARNING! %s is not a default setting" % name, RuntimeWarning)
        settings[name] = val

        #Temporary backwards compatible check for first_only
        if name == 'first_only':
            warnings.warn("first_only is being deprecated", DeprecationWarning)
            if val:
                print("first_only is set, setting gal_num = 0")
                settings['gal_num'] = 0

    return settings


def check_args(base_val, *args):
    # This function is mostly broken and likely unneccassary
    # Done this way because of https://hynek.me/articles/hasattr/
    y = getattr(base_val, '__iter__', None)
    if y is None:  # ensure we can loop over these
        base_val = [base_val]

    new_args = []

    for i, val in enumerate(args):
        y = getattr(val, '__iter__', None)
        if y is None:
            val = [val]*len(base_val)
            new_args.append(val)
        elif len(val) != len(base_val):
            raise IndexError("IF YOU PROVIDE A LIST FOR ONE ARGUMENT YOU MUST PROVIDE ONE FOR ALL OF THEM")

    return base_val, new_args


def read_offsets(fname):
    (angle,
     major,
     minor,
     ecc,
     axis_ratio,
     xCenter,
     yCenter) = np.loadtxt(fname,
                           skiprows=2,
                           unpack=True,
                           comments='Pot')
    measurements = {}
    measurements['axes_ratios'] = axis_ratio
    measurements['angles'] = angle
    measurements['majors'] = major
    measurements['minors'] = minor
    measurements['xCenters'] = xCenter
    measurements['yCenters'] = yCenter
    measurements['eccs'] = ecc
    return measurements


def replace(file_path, pattern, subst, tag=None):
    #Create temp file
    fh, abs_path = mkstemp()
    new_file = open(abs_path, 'w')
    old_file = open(file_path)
    for line in old_file:
        if tag:
            if len(line.split()) > 0 and line.split()[0] == tag:
                new_file.write(line.replace(pattern, subst))
            else:
                new_file.write(line)
        else:
            new_file.write(line.replace(pattern, subst))

    #close temp file
    new_file.close()
    os.close(fh)
    old_file.close()
    #Remove original file
    os.remove(file_path)
    #Move new file
    move(abs_path, file_path)

# Taken from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap