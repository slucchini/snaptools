import pygad as pg
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import astropy.units as u
import astropy.constants as constants
from astropy.coordinates import Galactic, Galactocentric, ICRS, LSR, SkyCoord
from astropy.coordinates import CartesianRepresentation
from . import magellanicstream

def loadSnap(snapName,snapbase="snapshot",outfolder="output"):
    splitName = snapName.split('-')
    galaxies = splitName[0][:-2]
    sim_type = splitName[0][-2:]
    run = int(splitName[1])
    snum = int(splitName[2])

    if (sim_type == "01"):
        sim_type = "01_nogas"
    elif (sim_type == "02"):
        sim_type = "02_mw_corona"
    elif (sim_type == "03"):
        sim_type = "03_lmc_halo"
    elif (sim_type == "04"):
        sim_type = "04_warm_and_hot"
    else:
        raise Exception("Unrecognized simulation type {}".format(sim_type))

    if (galaxies == "LS"):
        galaxies = "LMC_SMC"
    elif (galaxies == "MLS"):
        galaxies = "MW_LMC_SMC"
    elif (galaxies == "SP"):
        galaxies = "Second_Passage"
    elif (galaxies == "DI"):
        galaxies = "Direct_Infall"
    else:
        raise Exception("Unrecognized simulation category {}".format(galaxies))

    snappath = "/Volumes/LucchiniResearchData2/HPC_Backup/home/working/{0}/{1}/{1}_run{2}/{3}/{4}_{5:03}.hdf5".format(sim_type,galaxies,run,outfolder,snapbase,snum)
    s = pg.Snap(snappath)
    return s

def shift_to_com(s, particle_mass=None, return_com=False):
    if (particle_mass is None):
        raise Excception("No particle mass supplied")
    inds = []
    # If multiple masses passed in, loop through
    try:
        for m in particle_mass:
            inds.extend(np.where(s['mass'] == m)[0])
    except:
        inds = np.where(s['mass'] == particle_mass)[0]
    com = pg.analysis.center_of_mass(s[inds])
    pg.Translation(-com).apply(s)

    if (return_com):
        return s, com
    else:
        return s

def split_galaxies(s):
    indices = []
    mass = s['mass']
    unq = np.unique(mass)
    masses = np.argsort([np.sum(mass == u)*u for u in unq])[::-1]
    #invert the order so that the largest galaxy is first, as was the case in the old code
    for j, m in enumerate(masses):
        indices.append(np.where(mass == unq[m])[0])
    return indices

def get_show_ids_masses(s,masses=None,negate=False):
    if (masses is None):
        raise Exception("No masses supplied")

    show_ids = []
    # If multiple masses passed in, loop through
    try:
        for m in masses:
            show_ids.extend(np.where(s['mass'] == m)[0])
    except:
        show_ids = np.where(s['mass'] == masses)[0]

    if (negate):
        show_negated = np.ones(len(s['mass']),dtype=bool)
        show_negated[show_ids] = 0
        return show_negated
    return np.int_(show_ids)

def get_show_ids(s,inds=None,verbose=False):
    gal_inds = np.array(split_galaxies(s))

    if (inds is None):
        inds = np.ones(len(gal_inds),bool)

    show_ids = []
    for g in gal_inds[inds]:
        show_ids = np.concatenate((show_ids,g.tolist()))
    show_ids = np.int_(show_ids)

    return show_ids

def get_show_ids_old(s,show='all',sl=None,verbose=False):
    gal_inds = split_galaxies(s.gas)
    if (verbose):
        print([len(g) for g in gal_inds])

    # Only use the desired particles
    if (sl is None):
        sl = slice(None,None,None)
        if (show == 'lmc'):
            sl = slice(-2,-1,None)
        if (show == 'smc'):
            sl = slice(-1,None,None)
        if (show == 'mcs'):
            sl = slice(-2,None,None)
        elif (show == 'warm'):
            sl = slice(-3,None,None)
        elif (show == 'warm-only'):
            sl = slice(-3,-2,None)
        elif (show == 'hot'):
            sl = slice(-4,None,None)
    show_ids = []
    for g in gal_inds[sl]:
        show_ids = np.concatenate((show_ids,g.tolist()))
    show_ids = np.int_(show_ids)

    return show_ids

def plot_components(sub,plot_size=800):
    basepath = os.path.dirname(sub.filename)+"/"
    # s0 = pg.Snap(basepath+"{}_000.hdf5".format("snapshot"))
    # gal_ids0 = np.array(split_galaxies(s0))
    gal_ids0 = np.array(split_galaxies(sub))

    sz = plot_size
    fig,ax = plt.subplots(1,len(gal_ids0),figsize=(4*len(gal_ids0),4))

    for i,g in enumerate(gal_ids0):
        ax[i].hist2d(sub['pos'][g][:,1],sub['pos'][g][:,2],bins=300,range=[[-sz,sz],[-sz,sz]],norm=LogNorm())
        ax[i].set_title("i = {}".format(i))
        ax[i].set_aspect(1)
    plt.show()

    return gal_ids0

def gas_temp_old(s):

    xe = 1
    U = []
    if ('ne' in s.gas.available_blocks()):
        xe = s.gas['ne']
    if ('u' in s.gas.available_blocks()):
        U = s.gas['u']*1e10
    else:
        raise Exception("Snapshot doesn't have Internal Energy data (U)")

    Xh = 0.76
    gamma = 5./3.

    mu = 4./(1 + 3*Xh + 4*Xh*xe)*constants.m_p.cgs.value
    temp = (gamma - 1)*U/constants.k_B.cgs.value*mu

    return temp

def gas_temp(s):

    k_Boltzmann = constants.k_B.to("J/K").value
    gamma = 5./3.
    helium_mass_fraction = 0.76

    InternalEnergy = np.array(s.gas['u'].in_units_of("(m/s)**2"))
    ElectronAbundance = 1.
    if ('ne' in s.gas.available_blocks()):
        ElectronAbundance = s.gas['ne']

    y_helium = helium_mass_fraction / (4*(1-helium_mass_fraction))
    mu = (1 + 4*y_helium) / (1+y_helium+ElectronAbundance)

    mean_molecular_weight = mu*constants.m_p.to("kg").value

    temp = mean_molecular_weight * (gamma-1) * InternalEnergy / k_Boltzmann

    return temp

def _prep_snap(s, x, y, r, lsr_vels=None, show='mcs', min_dist=0, max_dist=None, min_vel=0, max_pix_dist=None, ne='all', temp='all', temp_cutoff=2e4):
    gaspos = np.zeros((len(x),3))
    gaspos[:,0] = x
    gaspos[:,1] = y
    gaspos[:,2] = r
    s['pos'] = pg.UnitArr(gaspos, 'ckpc h_0**-1')
    s['counter'] = pg.units.UnitArr(np.ones(len(s['pos'])), 'kpc')

    show_ids = show
    if (isinstance(show_ids,str)):
        show_ids = get_show_ids(s,show=show)
    mask = pg.ExprMask("r > '{} kpc'".format(min_dist))
    if (max_dist is not None):
        mask = (mask) & (pg.ExprMask("r < '{} kpc'".format(max_dist)))
    if (lsr_vels is None) and (min_vel > 0):
        print("WARNING: min_vel > 0, but no lsr_vels supplied. Ignoring min_vel.")
    elif (lsr_vels is not None) and (min_vel > 0):
        s['lsr_vel'] = pg.units.UnitArr(lsr_vels, 'km/s')
        vel_mask = pg.ExprMask("lsr_vel > '{} km/s'".format(min_vel))
        # mask = (~mask) & ((~mask) | (~vel_mask))
        mask = (mask) & (vel_mask)

    if (max_pix_dist is not None):
        delta = (max(x) - min(x))/2.
        s['pix_dist'] = pg.units.UnitArr(np.linalg.norm(list(zip(x-delta,y-delta)),axis=1), 'kpc')
        mask2 = pg.ExprMask("pix_dist < '{} kpc'".format(max_pix_dist))
        sub = s[show_ids][mask][mask2]
    else:
        sub = s[show_ids][mask]

    if (ne == 'ionized'):
        sub = sub[sub['nh'] < 0.5]
    elif (ne == 'neutral'):
        sub = sub[sub['nh'] > 0.5]

    if (temp != 'all'):
        if (temp == 'hot'):
            sub = sub[sub['temp'] > temp_cutoff]
        if (temp == 'cold'):
            sub = sub[sub['temp'] < temp_cutoff]

    return sub

def get_data(s, x, y, r, extent, lsr_vels=None, Npx=500, show='mcs', min_dist=0, max_dist=None, min_vel=0, max_pix_dist=None, ne='all', temp='all', temp_cutoff=2e4):

    sub = _prep_snap(s.gas, x, y, r, lsr_vels=lsr_vels, show=show, min_dist=min_dist, max_dist=max_dist, min_vel=min_vel, max_pix_dist=max_pix_dist, ne=ne, temp=temp, temp_cutoff=temp_cutoff)

    units_dens = 'g/cm**2'
    m_dens = pg.binning.SPH_to_2Dgrid(sub, qty='rho', extent=extent, Npx=Npx, xaxis=1, yaxis=0)
    # return sub
    m_dens = m_dens.in_units_of(pg.units.Unit(units_dens), subs=s)
    dens_data = np.log10(np.array(m_dens))

    # return sub
    return dens_data

def get_vel_data(s, x, y, r, v, extent, Npx=500, show='mcs', min_dist=0, max_pix_dist=None, ne='all'):

    args = dict(
        extent=extent,
        field=False,
        av=None,
        Npx=Npx,
        sph=True,
        xaxis=1,
        yaxis=0
    )

    s.gas['lsrvel'] = pg.units.UnitArr(v,'km/s')
    sub = _prep_snap(s.gas, x, y, r, show=show, min_dist=min_dist, max_pix_dist=max_pix_dist, ne=ne)

    units_vel = 'km/s'
    m_vel = pg.binning.map_qty(sub, qty='lsrvel', reduction='mean', **args)
    m_vel = m_vel.in_units_of(pg.units.Unit(units_vel), subs=s.gas)
    # m_tot = pg.binning.map_qty(sub, qty='counter', reduction=None, **args)
    vel_data = np.array(m_vel)

    return vel_data

def get_em_data(s, x, y, r, extent, lsr_vels=None, Npx=500, show='mcs', min_dist=0, min_vel=0, max_pix_dist=None):

    sub = _prep_snap(s.gas, x, y, r, lsr_vels=lsr_vels, show=show, min_dist=min_dist, min_vel=min_vel, max_pix_dist=max_pix_dist)

    units_dens = '1/cm**3'
    m_dens = pg.binning.SPH_to_2Dgrid(sub, qty='ne^2', extent=extent, Npx=Npx, xaxis=1, yaxis=0)
    m_dens = m_dens.in_units_of(pg.units.Unit(units_dens), subs=s.gas)
    dens_data = np.log10(np.array(m_dens))

    # return sub
    return dens_data


def get_temp_data(s, x, y, r, extent, Npx=500, show='mcs', min_dist=0, max_pix_dist=None, ne='all'):

    sub = _prep_snap(s, x, y, r, show=show, min_dist=min_dist, max_pix_dist=max_pix_dist, ne=ne)

    units_temp = 'K'
    m_temp = pg.binning.map_qty(sub, extent=extent, field=False, qty='temp',
                             av=None, reduction='mean', Npx=Npx,
                             xaxis=1, yaxis=0)
    m_temp = m_temp.in_units_of(pg.units.Unit(units_temp), subs=s.gas)
    temp_data = np.log10(np.array(m_temp))

    return temp_data


def transform_coords(pos, vel=None, return_coord="magellanic", return_vel=False):

    def _transform_positions(pos, vel, coord):
        gal_coords = Galactocentric(x=pos[:, 0]*u.kpc, y=pos[:, 1]*u.kpc, z=pos[:, 2]*u.kpc,
                                    v_x=vel[:, 0]*u.km/u.s, v_y=vel[:, 1]*u.km/u.s, v_z=vel[:, 2]*u.km/u.s,
                                    representation=CartesianRepresentation)

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

    if (vel is None):
        if (return_vel):
            raise Exception("return_vel set to True, but no input velocities supplied.")
        vel = np.zeros(np.shape(pos))

    coords, radial_vel = _transform_positions(pos, vel, return_coord)

    if return_vel:
        return coords, radial_vel
    else:
        return coords

def rotate_point(xyz, angles, axes):
    """
    Rotate a point counterclockwise in plane specified by axes
    Default is rotation about z-axis
    """
    if (len(angles) != len(axes)):
        raise Exception('Need the same number of angles and axes. Got {} angles and {} axes'.format(len(angles),len(axes)))
    for i,ax in enumerate(axes):
        co = np.cos(angles[i] / 180.0 * np.pi)
        si = np.sin(angles[i] / 180.0 * np.pi)

        xyz[:, ax[0]], xyz[:, ax[1]] = co * xyz[:, ax[0]] - si * xyz[:, ax[1]],\
                                            si * xyz[:, ax[0]] + co * xyz[:, ax[1]]
    return xyz
