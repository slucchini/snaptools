from __future__ import print_function, absolute_import, division
from builtins import range  # overload range to ensure python3 style
from six import iteritems
from collections import defaultdict
import numpy as np
import os
from .snapshot import Snapshot
import h5py
from . import utils

"""
This file contains the Snapshot classes
These inherit modules from their superclass Snapshot
Next two classes are for particular filetypes
"""


# Convienent names for header attributes
# Add more here if you need more.
# Will use the name in the HDF5 file if not specified
HEAD_ATTRS = {'NumPart_ThisFile': 'npart',
              'NumPart_Total': 'nall',
              'NumPart_Total_HighWord': 'nall_highword',
              'MassTable': 'massarr',
              'Time': 'time',
              'Redshift': 'redshift',
              'BoxSize': 'boxsize',
              'NumFilesPerSnapshot': 'filenum',
              'Omega0': 'omega0',
              'OmegaLambda': 'omega_l',
              'HubbleParam': 'hubble',
              'Flag_Sfr': 'sfr',
              'Flag_Cooling': 'cooling',
              'Flag_StellarAge': 'stellar_age',
              'Flag_Metals': 'metals',
              'Flag_Feedback': 'feedback',
              'Flag_DoublePrecision': 'double'}
# Defines convienent standard names for misc. data blocks
# Todo: add function for arbitrary datablocks

DATABLOCKS = {"Coordinates": "pos",
              "Velocities": "vel",
              "ParticleIDs": "ids",
              "Potential": "pot",
              "Masses": "masses",
              "InternalEnergy": "U",
              "Density": "RHO",
              "Volume": "VOL",
              "Center-of-Mass": "CMCE",
              "Surface Area": "AREA",
              "Number of faces of cell": "NFAC",
              "ElectronAbundance": "NE",
              "NeutralHydrogenAbundance": "NH",
              "SmoothingLength": "HSML",
              "StarFormationRate": "SFR",
              "StellarFormationTime": "AGE",
              "Metallicity": "Z",
              "Acceleration": "ACCE",
              "VertexVelocity": "VEVE",
              "MaxFaceAngle": "FACA",
              "CoolingRate": "COOR",
              "MachNumber": "MACH",
              "DM Hsml": "DMHS",
              "DM Density": "DMDE",
              "PSum": "PTSU",
              "DMNumNgb": "DMNB",
              "NumTotalScatter": "NTSC",
              "SIDMHsml": "SHSM",
              "SIDMRho": "SRHO",
              "SVelDisp": "SVEL",
              "GFM StellarFormationTime": "GAGE",
              "GFM InitialMass": "GIMA",
              "GFM Metallicity": "GZ  ",
              "GFM Metals": "GMET",
              "GFM MetalsReleased": "GMRE",
              "GFM MetalMassReleased": "GMAR"}


MISC_DATABLOCKS = DATABLOCKS  # backwards compatibility

def load_dataset(filenames, group, variable):
    dataset = None

    for filename in filenames:
        with h5py.File(filename) as f:
            if dataset is None:
                dataset = f[group][variable][()]
            else:
                dataset = np.append(dataset, f[group][variable][()], axis=0)

    return dataset


class SnapLazy(Snapshot):
    """
    lazydict implementation of HDF5 snapshot
    """

    def __init__(self, fname, **kwargs):
        pass

    def init(self, fname, **kwargs):
        from functools import partial
        from . import lazydict
        part_names = ['gas',
                      'halo',
                      'stars',
                      'bulge',
                      'sfr',
                      'other']
        self.part_names = part_names

        self.settings = utils.make_settings(**kwargs)
        self.bin_dict = None

        if not isinstance(fname, list):
            fname = [fname]

        self.filename = fname

        #load header only
        with h5py.File(fname[0], 'r') as s:
            self.header = {}
            #header_keys = s['Header'].attrs.keys()
            for head_key, head_val in iteritems(s['Header'].attrs):
                if head_key in HEAD_ATTRS.keys():
                    self.header[HEAD_ATTRS[head_key]] = head_val
                else:
                    self.header[head_key] = head_val
            # setup loaders
            for i, part in enumerate(part_names):
                if self.header['nall'][i] > 0:
                    for key in s['PartType%d' % i].keys():

                        try:
                            attr_name = DATABLOCKS[key]
                        except KeyError:
                            attr_name = key

                        if attr_name not in self.__dict__.keys():
                            self.__dict__[attr_name] = lazydict.MutableLazyDictionary()
                        self.__dict__[attr_name][part] = partial(load_dataset, self.filename,
                                                                 "PartType%d" % i, key)

            if any(self.header['massarr']):
                wmass, = np.where(self.header['massarr'])
                for i in wmass:
                    part = part_names[i]
                    npart = self.header['nall'][i]
                    mass = self.header['massarr'][i]
                    if 'masses' not in self.__dict__.keys():
                        self.__dict__['masses'] = lazydict.MutableLazyDictionary()

                    # here we are keeping things lazy by defining a function
                    # that will make our array only when needed
                    # the inner lambda function takes two arguments - a number of particles (n) and a mass (m)
                    # the outer function (partial) freezes the arguments for the current particle type
                    # the outer function is necessary or else arguments will be overwritten by other types
                    self.masses[part] = partial(lambda n, m: np.ones(n)*m, npart, mass)


    def save(self, fname, userblock_size=0):
        """
        Save a snapshot object to an hdf5 file. Overload base case
        Note: Must have matching header and data.
        Todo: Gracefully handle mismatches between header and data
        """
        import h5py

        # A list of header attributes, their key names, and data types
        head_attrs = {'npart': (np.int32, 'NumPart_ThisFile'),
                      'nall': (np.uint32, 'NumPart_Total'),
                      'nall_highword': (np.uint32, 'NumPart_Total_HighWord'),
                      'massarr': (np.float64, 'MassTable'),
                      'time': (np.float64, 'Time'),
                      'redshift': (np.float64, 'Redshift'),
                      'boxsize': (np.float64, 'BoxSize'),
                      'filenum': (np.int32, 'NumFilesPerSnapshot'),
                      'omega0': (np.float64, 'Omega0'),
                      'omega_l': (np.float64, 'OmegaLambda'),
                      'hubble': (np.float64, 'HubbleParam'),
                      'sfr': (np.int32, 'Flag_Sfr'),
                      'cooling': (np.int32, 'Flag_Cooling'),
                      'stellar_age': (np.int32, 'Flag_StellarAge'),
                      'metals': (np.int32, 'Flag_Metals'),
                      'feedback': (np.int32, 'Flag_Feedback'),
                      'double': (np.int32, 'Flag_DoublePrecision')}
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

        part_names = ['gas',
                      'halo',
                      'stars',
                      'bulge',
                      'sfr',
                      'other']
        # Open the file
        with h5py.File(fname, 'w', userblock_size=userblock_size) as f:
            # First write the header
            grp = f.create_group('Header')
            for key, val in iteritems(self.header):
                # If we have a name and dtype, use those
                if key in head_attrs.keys():
                    grp.attrs.create(head_attrs[key][1], val,
                                     dtype=head_attrs[key][0])
                # Otherwise simply use the name we read in
                else:
                    grp.attrs.create(key, val)
            # create the groups for all particles in the snapshot
            grps = [f.create_group('PartType{:d}'.format(i))  if n > 0 else None
                    for i, n in enumerate(self.header['nall'])]
            # iterate through datablocks first

            for attr_name, attr in iteritems(self.__dict__):
                x = getattr(attr, 'states', None)
                if (x is None) and (attr_name not in datablocks):  # only want lazy-dict things
                    continue
                for p, val in iteritems(attr):  # then through particle types

                    i = part_names.index(p)
                    try:
                        dset = grps[i].create_dataset(datablocks[attr_name], val.shape,
                                                      dtype=val.dtype)
                    except KeyError:
                        dset = grps[i].create_dataset(attr_name, val.shape, dtype=val.dtype)

                    dset[:] = val


class SnapHDF5(Snapshot):
    """
    Snapshots in HDF5
    snap = SnapHDF5('mycoolsnapshot.hdf5')
    Note: Must have matching header and data.
    Todo: Gracefully handle mismatches between header and data
    """
    def __init__(self, fname, **kwargs):
        """
        This method is purposefully empty. __new__ method of parent class will call init().
        """
        pass

    def init(self, fname, **kwargs):
        """Read from an HDF5 file
        """
        self.settings = utils.make_settings(**kwargs)
        self.bin_dict = None
        self.filename = os.path.abspath(fname)
        self.folder = os.path.dirname(fname)+"/"

        with h5py.File(fname, 'r') as s:
            self.header = {}
            #header_keys = s['Header'].attrs.keys()
            for head_key, head_val in iteritems(s['Header'].attrs):
                if head_key in HEAD_ATTRS.keys():
                    self.header[HEAD_ATTRS[head_key]] = head_val
                else:
                    self.header[head_key] = head_val

            part_names = ['gas',
                          'halo',
                          'stars',
                          'bulge',
                          'sfr',
                          'other']
            self.pos = {}
            self.vel = {}
            self.ids = {}
            self.masses = {}
            self.pot = {}
            self.misc = {}
            for i, n in enumerate(self.header['npart']):
                if n > 0:
                    group = 'PartType%s' % i
                    part_name = part_names[i]
                    for key in s[group].keys():
                        if key == 'Coordinates':
                            self.pos[part_name] = s[group]['Coordinates'][()]
                        elif key == 'Velocities':
                            self.vel[part_name] = s[group]['Velocities'][()]
                        elif key == 'ParticleIDs':
                            self.ids[part_name] = s[group]['ParticleIDs'][()]
                        elif key == 'Potential':
                            self.pot[part_name] = s[group]['Potential'][()]
                        elif key == 'Masses':
                            self.masses[part_name] = s[group]['Masses'][()]
                        # If we find a misc. key then add it to the misc variable (a dict)
                        elif key in MISC_DATABLOCKS.keys():
                            if part_name not in self.misc.keys():
                                self.misc[part_name] = {}
                            self.misc[part_name][MISC_DATABLOCKS[key]] = s[group][key][()]
                        # We have an unidentified key, throw it in with the misc. keys
                        else:
                            if part_name not in self.misc.keys():
                                self.misc[part_name] = {}
                            self.misc[part_name][key] = s[group][key][()]
                    # If we never found the masses key then make one
                    if part_name not in self.masses.keys():
                        self.masses[part_name] = (np.ones(n) * self.header['massarr'][i])


class SnapBinary(Snapshot):
    def __init__(self, fname, **kwargs):
        pass

    def init(self, fname, **kwargs):
        self.settings = utils.make_settings(**kwargs)
        self.bin_dict = None
        self.filename = fname

        f = open(fname, 'rb')
        blocksize = np.fromfile(f, dtype=np.int32, count=1)
        if blocksize[0] == 8:
            swap = 0
            format = 2
        elif blocksize[0] == 256:
            swap = 0
            format = 1
        else:
            blocksize.byteswap(True)
            if blocksize[0] == 8:
                swap = 1
                format = 2
            elif blocksize[0] == 256:
                swap = 1
                format = 1
            else:
                print("incorrect file format encountered when " +\
                      "reading header of", fname)
        self.header = {}
        if format == 2:
            f.seek(16, os.SEEK_CUR)
        part_names = ['gas',
                      'halo',
                      'stars',
                      'bulge',
                      'sfr',
                      'other']
        npart = np.fromfile(f, dtype=np.int32,
                            count=6)
        self.header['npart'] = npart
        massarr = np.fromfile(f, dtype=np.float64,
                              count=6)
        self.header['massarr'] = massarr
        self.header['time'] = (np.fromfile(f, dtype=np.float64,
                                           count=1))[0]
        self.header['redshift'] = (np.fromfile(f, dtype=np.float64,
                                               count=1))[0]
        self.header['sfr'] = (np.fromfile(f, dtype=np.int32,
                                          count=1))[0]
        self.header['feedback'] = (np.fromfile(f, dtype=np.int32,
                                               count=1))[0]
        nall = np.fromfile(f, dtype=np.int32,
                           count=6)
        self.header['nall'] = nall
        self.header['cooling'] = (np.fromfile(f, dtype=np.int32,
                                              count=1))[0]
        self.header['filenum'] = (np.fromfile(f, dtype=np.int32,
                                              count=1))[0]
        self.header['boxsize'] = (np.fromfile(f, dtype=np.float64,
                                              count=1))[0]
        self.header['omega0'] = (np.fromfile(f, dtype=np.float64,
                                              count=1))[0]
        self.header['omega_l'] = (np.fromfile(f, dtype=np.float64,
                                              count=1))[0]
        self.header['hubble'] = (np.fromfile(f, dtype=np.float64,
                                             count=1))[0]
        self.header['double'] = 0
        if swap:
            self.header['npart'].byteswap(True)
            self.header['massarr'].byteswap(True)
            self.header['time'] = self.time.byteswap()
            self.header['redshift'] = self.redshift.byteswap()
            self.header['sfr'] = self.sfr.byteswap()
            self.header['feedback'] = self.feedback.byteswap()
            self.header['nall'].byteswap(True)
            self.header['cooling'] = self.cooling.byteswap()
            self.header['filenum'] = self.filenum.byteswap()
            self.header['boxsize'] = self.boxsize.byteswap()
            self.header['omega0'] = self.omega_m.byteswap()
            self.header['omega_l'] = self.omega_l.byteswap()
            self.header['hubble'] = self.hubble.byteswap()
        np.fromfile(f, dtype=np.int32, count=1)
        np.fromfile(f, dtype=np.int32, count=25)
        NPARTS = np.sum(self.header['npart'])
        self.pos = {}
        positions = np.fromfile(f, dtype=np.float32,
                                count=NPARTS*3).reshape(NPARTS, 3)
        self.vel = {}
        np.fromfile(f, dtype=np.int32, count=2)
        velocities = np.fromfile(f, dtype=np.float32,
                                 count=NPARTS*3).reshape(NPARTS, 3)
        self.ids = {}
        np.fromfile(f, dtype=np.int32, count=2)
        ids = np.fromfile(f, dtype=np.int32,
                          count=NPARTS)
        NPREV = 0
        for i,g in enumerate(part_names):
            NGROUP = self.header['npart'][i]
            if NGROUP:
                self.pos[g] = positions[NPREV:NPREV+NGROUP, :]
                self.vel[g] = velocities[NPREV:NPREV+NGROUP, :]
                self.ids[g] = ids[NPREV:NPREV+NGROUP]
                NPREV += NGROUP
        self.pot = {}
        self.masses = {}
        self.misc = {}

        np.fromfile(f, dtype=np.int32, count=2)

        #if massarr[parttype] > 0:
        #    np.ones(nall[parttype], dtype=np.float)*massarr[parttype]

        for i in range(len(npart)):
            if npart[i] == 0:
                continue
            if massarr[i] == 0:
                self.masses[part_names[i]] = np.fromfile(f, dtype=np.float32, count=npart[i])
            else:
                self.masses[part_names[i]] = np.ones(npart[i], dtype=np.float32)*massarr[i]

        if not any(massarr):
            np.fromfile(f, dtype=np.int32, count=2)

        NGAS = self.header['npart'][0]
        if NGAS:
            self.misc['gas'] = {}
            self.misc['gas']['U'] = np.fromfile(f, dtype=np.float32, count=NGAS)
            np.fromfile(f, dtype=np.int32, count=2)
            self.misc['gas']['RHO'] = np.fromfile(f, dtype=np.float32, count=NGAS)
            np.fromfile(f, dtype=np.int32, count=2)
            self.misc['gas']['HSML'] = np.fromfile(f, dtype=np.float32, count=NGAS)
            np.fromfile(f, dtype=np.int32, count=2)

        f.close()
        # if not any(massarr):
        #     masses = np.fromfile(f, dtype=np.float32,
        #                          count=NPARTS)
        #     np.fromfile(f, dtype=np.int32, count=2)
        #     potentials = np.fromfile(f, dtype=np.float32,
        #                              count=NPARTS)
        # else:
        #     potentials = np.fromfile(f, dtype=np.float32,
        #                              count=NPARTS)
        #
        # if NGAS:
        #     if massarr[0] != 0:
        #         self.masses['gas'] = np.zeros(NGAS) + \
        #             self.header['massarr'][0]
        #         self.pot['gas'] = potentials[0:NGAS]
        #     else:
        #         self.masses['gas'] = masses[0:NGAS]
        #         self.pot['gas'] = potentials[0:NGAS]
        #
        # if NHALO:
        #     if massarr[1] != 0:
        #         self.masses['halo'] = np.zeros(NHALO) + \
        #             self.header['massarr'][1]
        #         self.pot['halo'] = potentials[NGAS:NGAS+NHALO]
        #     else:
        #         self.masses['halo'] = masses[NGAS:NGAS+NHALO]
        #         self.pot['halo'] = potentials[NGAS:NGAS+NHALO]
        #
        # if NSTARS:
        #     if massarr[2] != 0:
        #         self.masses['stars'] = np.zeros(NSTARS) + \
        #             self.header['massarr'][2]
        #         self.pot['stars'] = potentials[NGAS+NHALO:NGAS+NHALO+NSTARS]
        #     else:
        #         self.masses['stars'] = masses[NGAS+NHALO:NGAS+NHALO+NSTARS]
        #         self.pot['stars'] = potentials[NGAS+NHALO:NGAS+NHALO+NSTARS]
        # f.close()
