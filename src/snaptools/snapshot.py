from __future__ import print_function, absolute_import, division
from builtins import range  # overload range to ensure python3 style
from six import iteritems
from . import utils
from . import combine_gals as cg
import astropy.units as u
import astropy.constants as constants
import pNbody as pnb, numpy as np, os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


"""
Base class for snapshots. If called with a filename, will return one of two supported subclasses.
If called with no filename, it will return an empty object.
This class contains all snapshot methods. These methods should act directly on a snapshot.
"""

"""
Loads a snapshot from the LucchiniResearchData drive based on an info string
formatted as "{gals}{sim-type}-{run}-{snap}".
    Ex: LS03-14-55
    Ex: MLS04-7-27
"""
def loadSnap(info, snapbase="snapshot", outfolder="output", datadrive=3):
    base, run, snapnum = info.split("-")
    # run = int(run)
    snapnum = int(snapnum)
    gals = base[:-2]
    sim_type_num = base[-2:]

    if (sim_type_num == "00"):
        sim_type = "00_isolated"
    elif (sim_type_num == "01"):
        sim_type = "01_nogas"
    elif (sim_type_num == "02"):
        sim_type = "02_mw_corona"
    elif (sim_type_num == "03"):
        sim_type = "03_lmc_halo"
    elif (sim_type_num == "04"):
        sim_type = "04_warm_and_hot"
    elif (sim_type_num == "05"):
        sim_type = "05_test_problems"
    else:
        raise Exception("Unrecognized simulation type {}".format(sim_type_num))

    extra_path = ""
    if (gals == "LS"):
        category = "LMC_SMC"
    elif (gals == "MLS"):
        category = "MW_LMC_SMC"
    elif (gals == "SP"):
        category = "Second_Passage"
    elif (gals == "DI"):
        category = "Direct_Infall"
    elif (gals == "B"):
        category = "Batch"
    elif (gals == "CC"):
        category = "Cloud_Crushing"
        extra_path = "/runs"
    elif ((gals == "LMC") | (gals == "SMC") | (gals == "MW")):
        category = gals
        extra_path = "/stability"
    else:
        raise Exception("Unrecognized simulation category {}".format(gals))

    while (datadrive > 0):
        snappath = "/Volumes/LucchiniResearchData{0}/HPC_Backup/home/working/{1}/{2}{3}/{2}_run{4}/{5}/{6}_{7:03}.hdf5".format(datadrive,sim_type,category,extra_path,run,outfolder,snapbase,snapnum)
        if (os.path.exists(snappath)):
            return Snapshot(snappath)
        snappath = "/Volumes/LucchiniResearchData{0}/HPC_Backup/scratch/{1}/{2}{3}/{2}_run{4}/{5}/{6}_{7:03}.hdf5".format(datadrive,sim_type,category,extra_path,run,outfolder,snapbase,snapnum)
        if (os.path.exists(snappath)):
            return Snapshot(snappath)

        datadrive -= 1

    raise Exception("File not found on any data drives attached: {}".format(snappath))



class Snapshot(object):

    def __new__(cls, filename=None, lazy=False):
        """
        Factory method for calling proper subclass or empty object
        """
        if filename is not None:

            from . import snapshot_io
            import h5py

            multi = False

            if os.path.exists(filename):
                curfilename = filename
            elif os.path.exists(filename + ".hdf5"):
                curfilename = filename + ".hdf5"
            elif os.path.exists(filename + ".0.hdf5"):
                if lazy:
                    # allow multi-part files only in the lazy eval mode
                    import glob
                    filelist = list(sorted(glob.glob(filename+".[0-9].hdf5")))
                    multi = True

                curfilename = filename + ".0.hdf5"
            else:
                raise IOError("[error] file not found : %s" % filename)

            if h5py.is_hdf5(curfilename):
                if lazy:
                    snapclass = super(Snapshot, cls).__new__(snapshot_io.SnapLazy)
                    if multi:
                        snapclass.init(filelist)  # replaces standard __init__ method
                    else:
                        snapclass.init(curfilename)
                    return snapclass
                else:
                    snapclass = super(Snapshot, cls).__new__(snapshot_io.SnapHDF5)
                    snapclass.init(curfilename)  # replaces standard __init__ method
                    return snapclass
            else:
                snapclass = super(Snapshot, cls).__new__(snapshot_io.SnapBinary)
                snapclass.init(curfilename)  # replaces standard __init__ method
                return snapclass
        else:
            return super(Snapshot, cls).__new__(cls)

    def __init__(self):
        """
        Create empty snapshot object
        """
        self.filename = None
        self.pos = {}
        self.vel = {}
        self.ids = {}
        self.masses = {}
        self.pot = {}
        self.misc = {}
        self.header = {'npart': np.array([0, 0, 0, 0, 0, 0]),
                         'nall': np.array([0, 0, 0, 0, 0, 0]),
                         'nall_highword': np.array([0, 0, 0, 0, 0, 0]),
                         'massarr': np.array([0., 0., 0., 0., 0.]),
                         'time': 0.0,
                         'redshift': 0.0,
                         'boxsize': 0.0,
                         'filenum': 1,
                         'omega0': 0.0,
                         'omega_l': 0.0,
                         'hubble': 1.0,
                         'sfr': 0,
                         'cooling': 0,
                         'stellar_age': 0,
                         'metals': 0,
                         'feedback': 0,
                         'double': 0,
                         'Flag_IC_Info': 0}
        self.settings = utils.make_settings()
        self.bin_dict = None

    def set_settings(self, **kwargs):
        """
        Set the settings used by analysis and plotting tools.
        """
        for name, val in iteritems(kwargs):
            if name not in self.settings.keys():
                print("WARNING! {} is not a default setting!".format(name))
            self.settings[name] = val

    def get_mass_arr_from_s0(self,ptype,snapbase="snapshot"):
        """
        Only accepts a single ptype
        """
        s0 = Snapshot(os.path.dirname(self.filename)+"/{}_000.hdf5".format(snapbase))
        gal_inds = s0.split_galaxies(ptype)

        return [s0.masses[ptype][g[0]] for g in gal_inds]

    def split_galaxies(self, ptype, mass_list=None):
        """
        Split galaxies based on particles that have the same mass
        Args:
            ptype: A string or iterable of particle types
        kwargs:
            mass_list: masses of particles in each galaxy
                        should be size NxM where N is number of gals and M is number length of ptype
        """
        if (getattr(ptype, '__iter__', None) is None) or (isinstance(ptype, (str, bytes))):
            ptype = [ptype]

        indices = []

        nlast = 0

        for i, p in enumerate(ptype):
            mass = self.masses[p]
            if mass_list is not None:
                for j, m in enumerate(mass_list):
                    if i == 0:
                        indices.append(np.where(mass == m)[0])
                        nlast = len(mass)
                    else:
                        indices[j] = np.append(indices[j], np.where(mass == m)[0] + nlast)
                        nlast = len(mass)

            else:
                unq = np.unique(mass)
                if i == 0:
                    ngals = len(unq)

                # Add negatives to make list same size as galaxy list.
                #Assume that larger galaxies have more parts for now
                while len(unq) < ngals:
                        unq = np.insert(unq, 0, -1.0)

                masses = np.argsort([np.sum(mass == u)*u for u in unq])[::-1]
                #invert the order so that the largest galaxy is first, as was the case in the old code
                for j, m in enumerate(masses):
                    if i == 0:
                        indices.append(np.where(mass == unq[m])[0])
                        nlast = len(mass)
                    else:
                        indices[j] = np.append(indices[j], np.where(mass == unq[m])[0] + nlast)
                        nlast = len(mass)

        return indices

    def get_show_ids(self,ptype,inds=None,verbose=False):
        gal_inds = np.array(self.split_galaxies(ptype))

        if (inds is None):
            inds = np.ones(len(gal_inds),bool)

        show_ids = []
        for g in gal_inds[inds]:
            show_ids = np.concatenate((show_ids,g.tolist()))
        show_ids = np.int_(show_ids)

        return show_ids

    def get_show_ids_masses(self,ptype,masses=None,negate=False):
        if (masses is None):
            raise Exception("No masses supplied")

        show_ids = []
        # If multiple masses passed in, loop through
        try:
            for m in masses:
                show_ids.extend(np.where(self.masses[ptype] == m)[0])
        except:
            show_ids = np.where(self.masses[ptype] == masses)[0]

        if (negate):
            show_negated = np.ones(len(self.masses[ptype]),dtype=bool)
            show_negated[show_ids] = 0
            return show_negated
        return np.int_(show_ids)

    def get_show_ids_mranges(self,ptype,mlims=None,bins=None):
        if (mlims is None):
            raise Exception("No mass limits supplied")
        if (bins is None):
            raise Exception("No mass bin selections supplied")

        massranges = [[10**mlims[i],10**mlims[i+1]] for i in bins]

        show_ids = []
        for mr in massranges:
            show_ids.extend(np.where((self.masses[ptype] > mr[0]) & (self.masses[ptype] < mr[1]))[0])
        return np.array(show_ids)

    def split_galaxies_mranges(self,ptype):
        n,bins = np.histogram(np.log10(self.masses[ptype]),bins=100)
        
        maxs = np.where((np.roll(n,1) <= n) & (np.roll(n,-1) <= n) == True)[0]
        if maxs[0] == 0:
            maxs = maxs[1:]
        if maxs[-1] == len(n)-1:
            maxs = maxs[:-1]
        mins = np.where((np.roll(n,1) >= n) & (np.roll(n,-1) >= n) == True)[0]
        maxs = bins[maxs]
        mins = bins[mins]
        mlims = [bins[0]-0.1]
        for i,m in enumerate(maxs):
            if (mins[0] > m):
                continue
            mask = np.where((bins > mlims[-1]) & (bins < mins[mins < m][-1]))
            if ((len(mask) == 0) | (np.sum(n[mask]) < 100)):
                continue
            mlims.append(mins[mins < m][-1])
        mlims.append(mins[mins < bins[-1]][-1])
        mlims.append(bins[-1]+0.1)
        mlims = np.array(mlims)

        return mlims

    def get_components(self,ptype,uses0=True):
        basepath = os.path.dirname(self.filename)+"/"
        if (uses0):
            try:
                s0 = Snapshot(basepath+"snapshot_000.hdf5")
            except:
                print("Warn: Unable to load snapshot_000")
                s0 = None
        else:
            s0 = None

        if (len(np.unique(self.masses[ptype])) < 10):
            gal_ids0 = np.array(self.split_galaxies(ptype),dtype='object')
            if (len(gal_ids0) == 1):
                gal_ids0 = gal_ids0.astype(int)

            return gal_ids0
        else:
            n,bins = np.histogram(np.log10(self.masses[ptype]),bins=100)
            if ((s0 is not None) and (ptype in s0.masses.keys())):
                n0,bins0 = np.histogram(np.log10(s0.masses[ptype]),bins=100)
            else:
                n0 = n
                bins0 = bins
            
            # max0 = bins0[argrelextrema(n0, np.greater)[0]]
            # mins = bins[argrelextrema(n, np.less)[0]]
            max0 = np.where((np.roll(n0,1) <= n0) & (np.roll(n0,-1) <= n0) == True)[0]
            if max0[0] == 0:
                max0 = max0[1:]
            if max0[-1] == len(n0)-1:
                max0 = max0[:-1]
            mins = np.where((np.roll(n,1) >= n) & (np.roll(n,-1) >= n) == True)[0]
            max0 = bins0[max0]
            mins = bins[mins]
            mlims = [bins[0]-0.1]
            for i,m in enumerate(max0):
                if (mins[0] > m):
                    continue
                mask = np.where((bins > mlims[-1]) & (bins < mins[mins < m][-1]))
                if ((len(mask) == 0) | (np.sum(n[mask]) < 100)):
                    continue
                mlims.append(mins[mins < m][-1])
            mlims.append(mins[mins < bins0[-1]][-1])
            mlims.append(bins[-1]+0.1)
            mlims = np.array(mlims)

            return mlims

    def plot_components(self,ptype,plot=True,plot_size=800,xy=[1,2],mlims=None,uses0=True):
        basepath = os.path.dirname(self.filename)+"/"
        if (uses0):
            try:
                s0 = Snapshot(basepath+"snapshot_000.hdf5")
            except:
                print("Warn: Unable to load snapshot_000")
                s0 = None
        else:
            s0 = None
        sz = plot_size

        if (len(np.unique(self.masses[ptype])) < 10):
            gal_ids0 = np.array(self.split_galaxies(ptype),dtype='object')
            if (len(gal_ids0) == 1):
                gal_ids0 = gal_ids0.astype(int)

            fig,ax = plt.subplots(1,len(gal_ids0),figsize=(4*len(gal_ids0),4))
            if (len(gal_ids0) == 1):
                ax = [ax]

            for i,g in enumerate(gal_ids0):
                ax[i].hist2d(self.pos[ptype][g][:,xy[0]],self.pos[ptype][g][:,xy[1]],
                            bins=300,range=[[-sz,sz],[-sz,sz]],norm=LogNorm())
                ax[i].set_title("i = {}".format(i))
                ax[i].set_aspect(1)
            plt.show()

            return gal_ids0
        else:
            n,bins,_ = plt.hist(np.log10(self.masses[ptype]),bins=100)
            if ((s0 is not None) and (ptype in s0.masses.keys())):
                n0,bins0,_ = plt.hist(np.log10(s0.masses[ptype]),bins=100,alpha=0.5)
            else:
                n0 = n
                bins0 = bins
            if (mlims is None):
                # max0 = bins0[argrelextrema(n0, np.greater)[0]]
                # mins = bins[argrelextrema(n, np.less)[0]]
                max0 = np.where((np.roll(n0,1) <= n0) & (np.roll(n0,-1) <= n0) == True)[0]
                if max0[0] == 0:
                    max0 = max0[1:]
                if max0[-1] == len(n0)-1:
                    max0 = max0[:-1]
                mins = np.where((np.roll(n,1) >= n) & (np.roll(n,-1) >= n) == True)[0]
                max0 = bins0[max0]
                mins = bins[mins]
                mlims = [bins[0]-0.1]
                for i,m in enumerate(max0):
                    if (mins[0] > m):
                        continue
                    mask = np.where((bins > mlims[-1]) & (bins < mins[mins < m][-1]))
                    if ((len(mask) == 0) | (np.sum(n[mask]) < 100)):
                        continue
                    mlims.append(mins[mins < m][-1])
                mlims.append(mins[mins < bins0[-1]][-1])
                mlims.append(bins[-1]+0.1)
                mlims = np.array(mlims)
            
            for ml in mlims:
                plt.axvline(ml,c='k',alpha=0.5,ls='--',lw=0.7)
            plt.show()

            mass = self.masses[ptype]
            fig,ax = plt.subplots(1,len(mlims)-1,figsize=(4*len(mlims)-1,4))

            for i in range(len(mlims)):
                if (i == 0):
                    continue
                mask = (mass > 10**mlims[i-1]) & (mass < 10**mlims[i])
                ax[i-1].hist2d(self.pos[ptype][:,xy[0]][mask],self.pos[ptype][:,xy[1]][mask],bins=300,norm=LogNorm(),range=[[-sz,sz],[-sz,sz]])
                ax[i-1].set_title("[{:.3f},{:.3f}]".format(mlims[i-1],mlims[i]))

            plt.show()

            return mlims

    def get_mass_per_particle(self):
        ptypes = list(self.masses.keys())
        mpp = {}
        for p in ptypes:
            gal_ids = self.split_galaxies(p)
            mpp[p] = [self.masses[p][g][0] for g in gal_ids]
        return mpp

    def measure_com(self, ptype, indices_list):
        """
        Measure the centers of mass of galaxies in the snapshot
        Args:
            ptype: A string or list of particle types to draw positions from
            indices_list: An iterable item containing indices of each galaxy.
                          If multiple ptypes are given then the indices should
                          index the combined position vector in the proper order.
        """

        if ((getattr(ptype, '__iter__', None) is not None) and
            (not isinstance(ptype, (str, bytes)))):
            pos = np.append(*[self.pos[k] for k in ptype], axis=0)
        else:
            pos = self.pos[ptype]

        centers = []
        for indices in indices_list:  # change this logic
            #First check to see if we have a list lists or just one galaxy
            y = getattr(indices, '__iter__', None)
            if y is None:
                centers = np.mean(pos[indices_list], axis=0)
                break

            centers.append(np.mean(pos[indices], axis=0))

        return np.array(centers)


    def tree_potential_center(self, ptypes=['halo'], offset=[0.0, 0.0, 0.0], gal_num=0):
        """
        Use a tree to find the center of the potential
        kwargs:
            ptypes: list-like container of particle types
            offset: offset of simulation
            gal_num: which galaxy to test. By default will test the first galaxy.
        """
        import pNbody as pnb

        if gal_num < 0:
            idgal = list(range(len(self.ids[ptypes[0]])))
        else:
            idgal = self.split_galaxies(ptypes[0], mass_list=None)[gal_num]

        allpos = self.pos[ptypes[0]][idgal, :]
        allmasses = self.masses[ptypes[0]][idgal]

        for ptype in ptypes[1:]:
            # check to make sure we have particles
            if (ptype in self.pos.keys()) and (len(self.pos[ptype]) > 0):

                if gal_num < 0:
                    idgal = list(range(len(self.ids[ptype])))
                else:
                    idgal = self.split_galaxies(ptype, mass_list=None)[gal_num]

                # construct a list of positions and masses
                allpos = np.concatenate((self.pos[ptype][idgal, :],
                                         allpos), axis=0)

                allmasses = np.concatenate((self.masses[ptype][idgal],
                                            allmasses), axis=0)

        # use pnbody to run a force tree
        nb = pnb.Nbody(pos=allpos, mass=allmasses, verbose=0)
        pot = nb.TreePot(nb.pos, eps=0.1)
        #take the center of mass of the 100 most bound halo particles
        #Why only halo particles? Because we want the center of the halo not of some other component
        own_most_bound = np.argsort(pot)[:100]
        return allpos[own_most_bound, :].mean(axis=0) - np.array(offset)


    def potential_center(self, mass_arr=None):

        pt = 'halo'
        gal_ids = self.split_galaxies(pt,mass_list=mass_arr)
        output = []

        for g in gal_ids:
            pos = np.array(self.pos[pt][g],dtype=np.float32)
            vel = np.array(self.vel[pt][g],dtype=np.float32)
            mass = np.array(self.masses[pt][g],dtype=np.float32)

            nb = pnb.Nbody(pos=pos, mass=mass, verbose=0)
            pot = nb.TreePot(pos, eps=0.1)
            own_most_bound = np.argsort(pot)[:100]
            output.append(pos[own_most_bound, :].mean(axis=0))
            output.append(vel[own_most_bound, :].mean(axis=0))
        # output.append(self.header['time'])
        return output


    def get_stripped_particles(self, show_ids, show_ids_gas=None):
        """
        show_ids is a dictionary with keys matching each particle type and values that 
        mask the particles to be used to calculate the full potential (e.g. all LMC particles)
        
        Returns a dictionary with lists of the stripped and not stripped particle masks.
        """

        stripping = {}

        allpos = np.zeros((0,3))
        allmasses = np.zeros((0))
        allvel = np.zeros((0,3))
        allmask = []
        ptypes = self.masses.keys()
        for pt in ptypes:
            if (pt not in show_ids):
                raise Exception("{} not in show_ids".format(pt))
            allpos = np.concatenate((allpos,self.pos[pt][show_ids[pt]]))
            allmasses = np.concatenate((allmasses,self.masses[pt][show_ids[pt]]))
            allvel = np.concatenate((allvel,self.vel[pt][show_ids[pt]]))
            # msk = np.zeros_like(self.masses[pt][show_ids[pt]],dtype=bool)
            # if (pt == 'gas'):
            #     msk[show_ids['gas']] = True
            # allmask.append(msk)

        # make sure that there are no particles at the same location
        sorted_arr, unique_ids = np.unique(allpos,return_index=True,axis=0)
        dup_pc = 1 - len(unique_ids)/len(allpos)
        if (dup_pc > 0.05):
            print("WARNING: {:.1f}% of particles have duplicated positions.".format(dup_pc*100))

        if (show_ids_gas is None):
            show_ids_gas = show_ids['gas']

        gas_pos = self.pos['gas'].astype('float32')[show_ids_gas]
        gas_vel = self.vel['gas'][show_ids_gas]
        gas_masses = self.masses['gas'][show_ids_gas]
        # use pnbody to run a force tree
        nb = pnb.Nbody(pos=allpos[unique_ids], mass=allmasses[unique_ids], verbose=0)
        pot = nb.TreePot(gas_pos, eps=0.1)

        own_most_bound = np.argsort(pot)[:100]
        center = gas_pos[own_most_bound, :].mean(axis=0)
        vel_center = gas_vel[own_most_bound, :].mean(axis=0)

        tot_energy = 0.5*np.linalg.norm(gas_vel - vel_center, axis=1)**2 + pot*43007
        stripped = np.where(tot_energy > 0)[0]
        not_stripped = np.where(tot_energy < 0)[0]
        stripping['stripped'] = stripped
        stripping['not_stripped'] = not_stripped

        return stripping


    def gas_temp(self):

        xe = 1
        U = []
        if ('gas' in self.misc.keys()):
            if ('NE' in self.misc['gas'].keys()):
                xe = self.misc['gas']['NE']
            if ('U' in self.misc['gas'].keys()):
                U = self.misc['gas']['U']*1e+10
            else:
                raise Exception("Snapshot doesn't have Internal Energy data (U)")

        Xh = 0.76
        gamma = 5./3.

        mu = (1 + Xh /(1-Xh)) / (1 + Xh/(4*(1-Xh)) + xe)*constants.m_p.cgs.value
        # mu = 4./(1 + 3*Xh + 4*Xh*xe)*constants.m_p.cgs.value
        temp = (gamma - 1)*U/constants.k_B.cgs.value*mu

        return temp

    def to_cube(self,
                filename='snap',
                theta=0,
                lengthX=15,
                lengthY=15,
                BINS=512,
                parttype='stars',
                first_only=False,
                com=False,
                write=True):

        """
        Write snapshot to a fits cube (ppv)

        Kwargs:
            filename: Filename stub to save to disk.
                      Will append '_cube.fits'.
        """

        if write:
            from astropy.io import fits
        theta *= (np.pi / 180.)
    #    head=rs.snapshot_header(filename)
        pos2 = self.pos[parttype]
        vel2 = self.vel[parttype]
        mass2 = self.masses[parttype]

        if first_only:
            com1, com2, gal1id, gal2id = self.center_of_mass(parttype)
            pos2 = pos2[gal1id, :]
            vel2 = vel2[gal1id, :]
            mass2 = mass2[gal1id]
            if com:
                pos2 -= com1

        if theta:  # first check to see if this is even necessary
            rotation_matrix = [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]]
            for i in range(len(pos2[:, 0])):
                pos_vector = np.array([[pos2[i, 0]],
                                       [pos2[i, 1]],
                                       [pos2[i, 2]]])
                vel_vector = np.array([[vel2[i, 0]],
                                       [vel2[i, 1]],
                                       [vel2[i, 2]]])
                pos_rotated = np.dot(rotation_matrix, pos_vector)
                vel_rotated = np.dot(rotation_matrix, vel_vector)
                pos2[i, :] = pos_rotated.T[0]
                vel2[i, :] = vel_rotated.T[0]

        px2 = pos2[:, 0]
        py2 = pos2[:, 1]
        velz = vel2[:, 2]
        galvel = np.median(velz)
        velz -= galvel
        #    pz2=pos2[:,zax]
        H, Edges = np.histogramdd((px2, py2, velz),
                                  range=((-lengthX, lengthX),
                                         (-lengthY, lengthY),
                                         (-200, 200)),
                                  weights=mass2 * 1E10,
                                  bins=(BINS, BINS, 100),
                                  normed=False)


        if write:
            hdu = fits.PrimaryHDU()
            hdu.header['BITPIX'] = -64
            hdu.header['NAXIS'] = 3
            hdu.header['NAXIS1'] = 512
            hdu.header['NAXIS2'] = 512
            hdu.header['NAXIS3'] = 100
            hdu.header['CTYPE3'] = 'VELOCITY'
            #hdu.header['CTYPE3'] = 'VELO-LSR'
            hdu.header['CRVAL3'] = 0.0000000000000E+00
            hdu.header['CDELT3'] = 0.4000000000000E+04
            hdu.header['CRPIX3'] = 0.5000000000000E+02
            hdu.header['CROTA3'] = 0.0000000000000E+00
            hdu.data = H.T
            hdu.writeto(filename + '_cube.fits', clobber=True)
        else:
            return H, Edges


    def to_fits(self,
                filename='snap',
                theta=0,
                lengthX=15,
                lengthY=15,
                BINS=512,
                first_only=False,
                com=False,
                parttype='stars'):
        """
        Write snapshot to a fits map
        """

        from astropy.io import fits
        theta *= (np.pi / 180.)
    #    head=rs.snapshot_header(filename)

        pos2 = self.pos[parttype]
        mass2 = self.masses[parttype]

        if first_only:
            com1, com2, gal1id, gal2id = self.center_of_mass(parttype)
            pos2 = pos2[gal1id, :]
            mass2 = mass2[gal1id]
            if com:
                pos2 -= com1

        if theta:  # first check to see if this is even necessary
            rotation_matrix = [[np.cos(theta), 0, np.sin(theta)],
                               [0, 1, 0],
                               [-np.sin(theta), 0, np.cos(theta)]]
            for i in range(len(pos2[:, 0])):
                vector = np.array([[pos2[i, 0]], [pos2[i, 1]], [pos2[i, 2]]])
                rotated = np.dot(rotation_matrix, vector)
                pos2[i, :] = rotated.T[0]

        px2 = pos2[:, 0]
        py2 = pos2[:, 1]
        # pz2=pos2[:,zax]

        Z2, x, y = np.histogram2d(px2, py2,
                                  range=[[-lengthX, lengthX],
                                         [-lengthY, lengthY]],
                                  weights=mass2 * 1E10,
                                  bins=BINS,
                                  normed=False)

        fits.writeto(filename + '_map.fits', Z2, clobber=True)


    def quick_plot(self,plot_size=None):

        fig,axes = plt.subplots(3,2,figsize=(10,12))

        ptypes = ['halo','gas','stars']
        for i,ax in enumerate(axes):
            if (ptypes[i] in self.pos.keys()):
                ax[0].hist2d(self.pos[ptypes[i]][:,0],self.pos[ptypes[i]][:,2],bins=200,range=plot_size,norm=LogNorm())
                ax[0].set_xlabel("x (kpc)")
                ax[0].set_ylabel("z (kpc)")
                ax[0].set_title(ptypes[i])
                ax[1].hist2d(self.pos[ptypes[i]][:,1],self.pos[ptypes[i]][:,2],bins=200,range=plot_size,norm=LogNorm())
                ax[1].set_xlabel("y (kpc)")
                ax[1].set_ylabel("z (kpc)")
        
        plt.show()


    def save(self, fname, userblock_size=0):
        import h5py
        """
        Save a snapshot object to an hdf5 file.
        Note: Must have matching header and data.
        Todo: Gracefully handle mismatches between header and data
        """
        # A list of header attributes, their key names, and data types, and default values
        head_attrs = {'npart': (np.int32, 'NumPart_ThisFile'),
                      'nall': (np.uint32, 'NumPart_Total'),
                      'nall_highword': (np.uint32, 'NumPart_Total_HighWord', [0,0,0,0,0,0]),
                      'massarr': (np.float64, 'MassTable', [0,0,0,0,0,0]),
                      'time': (np.float64, 'Time', 0.0),
                      'redshift': (np.float64, 'Redshift', 1.0),
                      'boxsize': (np.float64, 'BoxSize'),
                      'filenum': (np.int32, 'NumFilesPerSnapshot', 1),
                      'omega0': (np.float64, 'Omega0', 1.0),
                      'omega_l': (np.float64, 'OmegaLambda', 0.0),
                      'hubble': (np.float64, 'HubbleParam', 1.0),
                      'sfr': (np.int32, 'Flag_Sfr', True),
                      'cooling': (np.int32, 'Flag_Cooling', True),
                      'stellar_age': (np.int32, 'Flag_StellarAge', True),
                      'metals': (np.int32, 'Flag_Metals', True),
                      'feedback': (np.int32, 'Flag_Feedback', True),
                      'double': (np.int32, 'Flag_DoublePrecision', True)}
        misc_datablocks = {"U": "InternalEnergy",
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
        for hval in head_attrs.keys():
            if (hval not in self.header):
                if (len(head_attrs[hval]) >= 3):
                    self.header[hval] = head_attrs[hval][2]
                else:
                    print("Missing header value: {}".format(hval))
                    return
        
        shift_ids = False
        for pt in self.ids.keys():
            if (0 in self.ids[pt]):
                shift_ids = True
        if (shift_ids):
            for pt in self.ids.keys():
                self.ids[pt] += 1
        
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
            for i, n in enumerate(self.header['nall']):
                if n > 0:
                    p = part_names[i]
                    grp = f.create_group('PartType'+str(i))
                    dset = grp.create_dataset('Coordinates',
                                              self.pos[p].shape
                                              )
                    dset[:] = self.pos[p]
                    dset = grp.create_dataset('Velocities',
                                              self.vel[p].shape
                                              )
                    dset[:] = self.vel[p]
                    dset = grp.create_dataset('ParticleIDs',
                                              self.ids[p].shape,
                                              dtype='i4')
                    dset[:] = self.ids[p]
                    dset = grp.create_dataset('Masses',
                                              self.masses[p].shape
                                              )
                    dset[:] = self.masses[p]

                    if p in self.pot:
                        dset = grp.create_dataset('Potential',
                                                  self.pot[p].shape
                                                 )
                        dset[:] = self.pot[p]

                    if p in self.misc.keys():  # Check for any misc data
                        for k in self.misc[p].keys():  # Loop through all misc data we have
                            if k.rstrip() in misc_datablocks.keys():
                                name = misc_datablocks[k]
                            else:
                                name = k
                            dset = grp.create_dataset(name,
                                                      self.misc[p]
                                                      [k].shape
                                                      )
                            dset[:] = self.misc[p][k]

    """
    Builds an HDF5 snapshot from raw position and velocity Data
    snapData - dictionary of positions, velocities, and masses of all particles types
    """
    @staticmethod
    def saveAsSnap(fname, snapData, headerOverwrite=None, userblock_size=0):
        import h5py
        """
        Save a snapshot object to an hdf5 file.
        Note: Must have matching header and data.
        Todo: Gracefully handle mismatches between header and data
        """
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
        misc_datablocks = {"U": "InternalEnergy",
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

        # Build the header
        npart = []
        for p in part_names:
            if (p in snapData):
                npart.append(len(snapData[p].pos))
            else:
                npart.append(0)
        header = {'npart': np.array(npart, dtype="int32"),
                     'nall': np.array(npart, dtype="uint32"),
                     'nall_highword': np.array([0, 0, 0, 0, 0, 0], dtype="uint32"),
                     'massarr': np.array([0., 0., 0., 0., 0., 0.]),
                     'time': 0.0,
                     'redshift': 0.0,
                     'boxsize': 10000.0,
                     'filenum': 1,
                     'omega0': 0.27,
                     'omega_l': 0.73,
                     'hubble': 1.0,
                     'sfr': 1,
                     'cooling': 1,
                     'stellar_age': 1,
                     'metals': 1,
                     'feedback': 1,
                     'double': 0,
                     'Flag_IC_Info': 0}
        
        if (headerOverwrite is not None):
            for k in headerOverwrite.keys():
                header[k] = headerOverwrite[k]

        # Open the file
        with h5py.File(fname, 'w', userblock_size=userblock_size) as f:
            # First write the header
            grp = f.create_group('Header')
            for key, val in iteritems(header):
                # If we have a name and dtype, use those
                if key in head_attrs.keys():
                    grp.attrs.create(head_attrs[key][1], val,
                                     dtype=head_attrs[key][0])
                # Otherwise simply use the name we read in
                else:
                    grp.attrs.create(key, val)
            runningID = 0
            for i, n in enumerate(header['nall']):
                if n > 0:
                    p = part_names[i]
                    print("Working on {}".format(p))
                    grp = f.create_group('PartType'+str(i))
                    dset = grp.create_dataset('Coordinates',
                                              snapData[p].pos.shape
                                              )
                    dset[:] = snapData[p].pos
                    dset = grp.create_dataset('Velocities',
                                              snapData[p].vel.shape
                                              )
                    dset[:] = snapData[p].vel
                    dset = grp.create_dataset('ParticleIDs',
                                              np.array(range(runningID,runningID+n)).shape,
                                              dtype='i4')
                    print(np.array(range(runningID,runningID+n)))
                    dset[:] = np.array(range(runningID,runningID+n))
                    runningID += n
                    dset = grp.create_dataset('Masses',
                                              snapData[p].masses.shape
                                              )
                    dset[:] = snapData[p].masses

                    # if p in snapData.pot:
                    #     dset = grp.create_dataset('Potential',
                    #                               snapData.pot[p].shape
                    #                              )
                    #     dset[:] = snapData.pot[p]

                    if snapData[p].misc != None:  # Check for any misc data
                        for k in snapData[p].misc.keys():  # Loop through all misc data we have
                            if k.rstrip() in misc_datablocks.keys():
                                name = misc_datablocks[k]
                            else:
                                name = k
                            dset = grp.create_dataset(name, snapData[p].misc[k].shape)
                            dset[:] = snapData[p].misc[k]

    def write_csv(self, gal_num=-1, ptypes=['stars'], stepsize=100, columns=['pos', 'vel']):
        """
        Write a csv version of the snapshot. Helpful for paraview or sharing simple versions with collaborators.
        Particles are sorted by ID number, so that the same particles are always in the same row of the file.
        kwargs:
            gal_num: Galaxy number to save. -1 for all galaxies.
            ptypes: Particle types to save.
            stepsize: Save every nth particle.
            columns: Which properties to save. Properties must be in every particle type you request.
        """
        if gal_num < 0:
            idgal = list(range(len(self.ids[ptypes[0]])))
        else:
            idgal = self.split_galaxies(ptypes[0], mass_list=None)[gal_num]

        all_data = {}
        allids = self.ids[ptypes[0]][idgal]
        for column in columns:
            # Get an arbitrary name, first try in non-misc properties, then try in misc props
            try:
                all_data[column] = self.__dict__[column][ptypes[0]][idgal]
            except KeyError:
                all_data[column] = self.misc[ptypes[0]][column][idgal]

            for ptype in ptypes[1:]:
                # check to make sure we have particles
                if (ptype in self.pos.keys()) and (len(self.pos[ptype]) > 0):
                    if gal_num < 0:
                        idgal = list(range(len(self.ids[ptype])))
                    else:
                        idgal = self.split_galaxies(ptype, mass_list=None)[gal_num]
                #first construc list of all ids
                allids = np.concatenate((self.ids[ptype][idgal], allids), axis=0)
                # construct lists of other properties
                try:
                    all_data[column] = np.concatenate((self.__dict__[column][ptypes[0]][idgal],
                                                       all_data[column]), axis=0)
                except KeyError:
                    all_data[column] = np.concatenate((self.misc[ptypes[0]][column][idgal],
                                                       all_data[column]), axis=0)

        s = np.argsort(allids)

        header = []
        for column_name, column_data in iteritems(all_data):
            if len(column_data.shape) == 1:
                header.append(column_name)
            else:
                header.append("{0}x,{0}y,{0}z".format(column_name))
        header = ','.join(header)

        with open(self.filename+".csv", "w") as f:
            f.write(header+'\n')
            for si in s[::stepsize]:
                column_text = ''
                for _, column_data in iteritems(all_data):
                    if len(column_data.shape) > 1:
                        column_text += "{:3.3f},{:3.3f},{:3.3f},".format(*column_data[si, :])
                    else:
                        column_text += "{:g},".format(column_data[si])
                #Strip trailing comma and add newline break
                f.write(column_text[:-1]+"\n")

    def __repr__(self):
        if not self.header:  # empty dict evaluates to False
            return "Empty Snapshot"
        else:
            if self.filename is None:
                return str(self.header)
            else:
                if isinstance(self.filename, list):
                    return """Multi-part snapshot files: {:s}
--------------------------------------------------
Header: {:s}""".format(str(self.filename), str(self.header))
                else:
                    return """Snapshot file - {:s}
-------------------------------------------------
Header: {:s}""".format(self.filename, str(self.header))


class ptypeInfo:
    def __init__(self, pos, vel, masses, misc = None):
        self.pos = pos
        self.vel = vel
        self.masses = masses
        self.misc = misc
