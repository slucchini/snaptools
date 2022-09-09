import numpy as np
# import matplotlib as mpl, matplotlib.pyplot as plt
from snaptools import snapshot
import astropy.units as u
from astropy import constants

PART_NAMES = ['gas', 'halo', 'stars', 'bulge', 'sfr', 'other']

# solar metallicity
solarZ = [2.00e-02, 2.5e-01, 3.26e-03, 1.32e-03, 8.65e-03, 2.22e-03, 9.31e-04, 1.08e-03, 6.44e-04, 1.01e-04, 1.73e-03]

datablocks = {"pos": "Coordinates",
              "vel": "Velocities",
              "pot": "Potential",
              "masses": "Masses",
              "ids": "ParticleIDs",
              "U": "InternalEnergy",
              "RHO": "Density",
              "VOL": "Volume",
              "CMCE": "Center-of-Mass",
              "AREA": "Surface Area",
              "NFAC": "Number of faces of cell",
              "NE": "ElectronAbundance",
              "NH": "NeutralHydrogenAbundance",
              "HSML": "SmoothingLength",
              "SFR": "StarFormationRate",
              "AGE": "StellarFormationTime",
              "Z": "Metallicity",
              "ACCE": "Acceleration",
              "VEVE": "VertexVelocity",
              "FACA": "MaxFaceAngle",
              "COOR": "CoolingRate",
              "MACH": "MachNumber",
              "DMHS": "DM Hsml",
              "DMDE": "DM Density",
              "PTSU": "PSum",
              "DMNB": "DMNumNgb",
              "NTSC": "NumTotalScatter",
              "SHSM": "SIDMHsml",
              "SRHO": "SIDMRho",
              "SVEL": "SVelDisp",
              "GAGE": "GFM StellarFormationTime",
              "GIMA": "GFM InitialMass",
              "GZ": "GFM Metallicity",
              "GMET": "GFM Metals",
              "GMRE": "GFM MetalsReleased",
              "GMAR": "GFM MetalMassReleased"}

"""
Combine two snapshots together
args:
    s1 - first snapshot
    s2 - second snapshot
kwargs:
    take_header_from - default: 'first' which snapshot to use as the header.
                       options: 'first', 'second'
    part_names1 - keys for particle types
    part_names2 - ^
    part_names_out - ^
"""
def combine_snaps(s1, s2,
                  take_header_from='first',
                  part_names1=['gas', 'halo', 'stars',
                               'bulge', 'sfr', 'other'],
                  part_names2=['gas', 'halo', 'stars',
                               'bulge', 'sfr', 'other'],
                  part_names_out=['gas', 'halo', 'stars',
                                  'bulge', 'sfr', 'other'],
                  metallicity=None):

    from builtins import range  # overload range to ensure python3 style
    from six import iteritems

    def combine(array1, array2):
        assert array1.ndim == array2.ndim

        if array1.ndim > 1:
            return np.vstack((array1, array2))
        else:
            return np.concatenate((array1, array2))

    def combine_misc_blocks(s1, s2, ptype, block):
        total = []
        if (block in s1.misc[ptype]):
            total = s1.misc[ptype][block]
        else:
            total = np.zeros(len(s1.pos[ptype]))
        if (block in s2.misc[ptype]):
            total = combine(total, s2.misc[ptype][block])
        else:
            total = combine(total, np.zeros(len(s2.pos[ptype])))
        return total

    snap = snapshot.Snapshot()
    # Grab the number of particles in galaxy 1
    # this will be the starting point for IDs in galaxy 2
    max_id = max((max(s1.ids[p]) for p in part_names1 if p in s1.ids))


    ### Copy misc info first
    # Copy gas U and NE
    # If no NE, use U to calculate
    snap.misc = {}
    if (('gas' in s1.misc) and ('gas' in s2.misc)):
        snap.misc['gas'] = {}
        if (('U' in s1.misc['gas']) and ('U' in s2.misc['gas'])):
            snap.misc['gas']['U'] = combine(s1.misc['gas']['U'], s2.misc['gas']['U'])
        else:
            badsnap = 1
            if ('U' in s1.misc['gas']):
                badsnap = 2
            raise Exception("Snap {} doesn't have InternalEnergy data".format(badsnap))

        totalNE = []
        if ('NE' in s1.misc['gas']):
            totalNE = s1.misc['gas']['NE']
        else:
            totalNE = set_ne(s1.misc['gas']['U'], temp=1000)
        if ('NE' in s2.misc['gas']):
            totalNE = combine(totalNE,s2.misc['gas']['NE'])
        else:
            totalNE = combine(totalNE,set_ne(s2.misc['gas']['U'], temp=1000))
        snap.misc['gas']['NE'] = totalNE


        if (('Z' in s1.misc['gas']) & ('Z' in s2.misc['gas'])):
            totalZ = combine(s1.misc['gas']['Z'],s2.misc['gas']['Z'])
            snap.misc['gas']['Z'] = totalZ
        elif ((metallicity is not None) & (('Z' in s1.misc['gas']) | ('Z' in s2.misc['gas']))):
            if ('Z' in s1.misc['gas']):
                newZ = np.array(solarZ*len(s2.pos['gas'])).reshape(-1,len(solarZ))*metallicity
                newZ[:,1] = solarZ[1]
                totalZ = combine(s1.misc['gas']['Z'],newZ)
            elif ('Z' in s2.misc['gas']):
                newZ = np.array(solarZ*len(s1.pos['gas'])).reshape(-1,len(solarZ))*metallicity
                newZ[:,1] = solarZ[1]
                totalZ = combine(newZ,s2.misc['gas']['Z'])
            snap.misc['gas']['Z'] = totalZ

        # others
        for k in np.unique(np.concatenate((list(s1.misc['gas'].keys()),list(s2.misc['gas'].keys())))):
            if ((k == 'NE') or (k == 'U') or (k == 'Z')):
                continue
            snap.misc['gas'][k] = combine_misc_blocks(s1, s2, 'gas', k)
        # if (('Z' in s1.misc['gas']) or ('Z' in s2.misc['gas'])):
        #     snap.misc['gas']['Z'] = combine_misc_blocks(s1, s2, 'gas', 'Z')
    elif (('gas' in s1.misc) or ('gas' in s2.misc)):
        swgas = s1
        badsnap = 1
        if ('gas' in s2.misc):
            badsnap = 2
            swgas = s2
        snap.misc['gas'] = swgas.misc['gas']

        if ('U' not in snap.misc['gas']):
            raise Exception("Snap {} doesn't have Internal Energy data.".format(badsnap))
        if ('NE' not in swgas.misc['gas']):
            snap.misc['gas']['NE'] = set_ne(snap.misc['gas']['U'], temp=1000)

    # stars
    if (('stars' in s1.misc.keys()) and ('stars' in s2.misc.keys())):
        # Z
        if (('Z' in s1.misc['stars']) or ('Z' in s2.misc['stars'])):
            snap.misc['stars'] = {}
            snap.misc['stars']['Z'] = combine_misc_blocks(s1, s2, 'stars', 'Z')
        # AGE
        if (('AGE' in s1.misc['stars']) or ('AGE' in s2.misc['stars'])):
            if 'stars' not in snap.misc.keys():
                snap.misc['stars'] = {}
            snap.misc['stars']['AGE'] = combine_misc_blocks(s1, s2, 'stars', 'AGE')
    elif ('stars' in s1.misc.keys()):
        snap.misc['stars'] = s1.misc['stars']
    elif ('stars' in s2.misc.keys()):
        snap.misc['stars'] = s2.misc['stars']

    ### Copy the main info next, using numbers from each header ###

    n1 = s1.header['nall']
    n2 = s2.header['nall']

    for s, part_names in zip([s1, s2], [part_names1, part_names2]):
        for attr_name, attr in iteritems(s.__dict__):
            x = getattr(attr, 'states', None)
            if (x is None) and (attr_name not in datablocks):  # only want lazy-dict things
                continue

            if attr_name not in snap.__dict__:
                snap.__dict__[attr_name] = {}  # make a regular dict for ease here

            for p, p_o in zip(part_names, part_names_out):
                if p in s.__dict__[attr_name]:
                    if p in snap.__dict__[attr_name]:
                        snap.__dict__[attr_name][p] = combine(snap.__dict__[attr_name][p],
                                                              s.__dict__[attr_name][p])
                    else:
                        snap.__dict__[attr_name][p] = s.__dict__[attr_name][p]

    # Add on the highest id number from the first galaxy
    # ids for second galaxy will start at max(id_gal1)
    # That means that DM ids for gal2 will be all higher than gal2
    for i, p in enumerate(part_names2):
        try:
            snap.ids[p][n1[i]:] += max_id
        except KeyError:
            continue

    # wait until the end to copy the header info

    if take_header_from == 'first':
        snap.header = s1.header.copy()
    elif take_header_from == 'second':
        snap.header = s2.header.copy()

    head1 = s1.header
    head2 = s2.header

    snap.header['npart'] = head1['npart'] + head2['npart']
    snap.header['nall'] = head1['nall'] + head2['nall']
    try:
        snap.header['nall_highword'] = head1['nall_highword'] + head2['nall_highword']
    except:
        snap.header['nall_highword'] = head1['nall'] + head2['nall']

    snap.header['massarr'] = [0, 0, 0, 0, 0, 0]

    return snap


"""
Set electron density to match internal energy and desired temperature
"""
def set_ne(U, temp=10000):
    ## internal units assumed

    energy = U*1e+10*(u.erg/u.g)
    temp = temp*u.K

    Xh = 0.76
    gamma = 5./3.
    mu = (temp * constants.k_B.cgs) / ((gamma - 1)*energy)

    xe = (4*constants.m_p.cgs/mu - 1 - 3*Xh)*(1/(4.*Xh))

    return xe.value

"""
Calculate internal energy using temperature and hydrogen fraction
This assumes full ionization (copied from Makegalaxy/effmodel.c)
"""
def calc_u(temp=10000):
    SOLAR_MASS = 1.989e33
    CM_PER_MPC = 3.085678e24

    UnitMass_in_g = 1e10*SOLAR_MASS; # 10^10 solar masses
    UnitLength_in_cm = CM_PER_MPC/1000; # 1 kpc
    UnitVelocity_in_cm_per_s = 1e5; # 1 km/sec

    UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s;
    UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2);

    Xh = 0.76
    gamma = 5./3.

    meanweight = 4/(8 - 5*(1-Xh))
    u4 = 1/meanweight*1/(gamma-1)*(constants.k_B.cgs/constants.m_p.cgs)*temp*u.K

    return u4.value * UnitMass_in_g / UnitEnergy_in_cgs


"""
Shift entire snapshot to newpos moving at newvel
"""
def move_galaxy(snap, newpos, newvel):
    for i,p in enumerate(PART_NAMES):
        if (snap.header["npart"][i] == 0): continue
        if (np.all(newpos != None)):
            for j in range(3):
                snap.pos[p][:,j] += newpos[j]
        if (np.all(newvel != None)):
            for j in range(3):
                snap.vel[p][:,j] += newvel[j]
    return snap

"""
Finds COM of galaxy using particle type ptype. Moves snap to be centered on COM
"""
def recenter_galaxy(snap, ptype, gal_id=0):
    com = snap.measure_com(ptype,snap.split_galaxies(ptype))
    return move_galaxy(snap, -com[gal_id], None)


def move_com(snap, newpos, newvel):
    if ('stars' in snap.pos.keys()):
        snap = recenter_galaxy(snap, "stars")
    elif ('halo' in snap.pos.keys()):
        snap = recenter_galaxy(snap, "halo")
    else:
        raise Exception("No stars or DM to recenter on...")
    return move_galaxy(snap, newpos, newvel)

"""
Centers the snapshot on the center of mass location of the galaxy with more particles
"""
def center_on_galaxy(snap, ptype):
    galids = snap.split_galaxies(ptype)
    coms = snap.measure_com(ptype, galids)
    largestCom = coms[0]
    largestGal = galids[0]
    for i,galid in enumerate(galids):
        if (len(galid) > len(largestGal)):
            largestCom = coms[i]
            largestGal = galid
    return move_galaxy(snap, -1*largestCom, [0,0,0])

"""
Rotations
"""
def rotate_galaxy(snap, angles, axes):
    """
    Rotate each particle type of a galaxy
    Specify sequence of angles (in degrees) and axes to rotate by
    Axes defined by plane of rotation (axes=[0,1] will rotate about z-axis)
    """
    for i,p in enumerate(PART_NAMES):
        if snap.header['npart'][i] > 0:
            snap.pos[p] = rotate_point(snap.pos[p], angles, axes)
            snap.vel[p] = rotate_point(snap.vel[p], angles, axes)
    return snap

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
