from __future__ import print_function, absolute_import, division
from builtins import range  # overload range to ensure python3 style
from six import iteritems
import os, warnings
from . import utils
from . import snapshot
from . import compute_positions as cp
from multiprocess import Pool
import numpy as np, re, h5py
from astropy.coordinates import SkyCoord,Galactocentric
import astropy.units as u
from .magellanicstream import MagellanicStream as MScoord
from tqdm.notebook import tqdm


"""
Loads a simulation from the LucchiniResearchData drive based on an info string
formatted as "{gals}{sim-type}-{run}".
    Ex: LS03-14
    Ex: MLS04-7
"""
def loadSim(info, snapbase="snapshot_", outfolder="output", datadrive=3):

    if (info == "Stephen"):
        snapbase = "snap_"
        folder = "/Volumes/LucchiniResearchData2/Stephen_Data/largerLMC_largerSMC_moregas/With_MW/output_first_passage_rotate/"
        return Simulation(folder, snapbase=snapbase)

    base, run = info.split("-")
    # run = int(run)
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
        folder = "/Volumes/LucchiniResearchData{0}/HPC_Backup/home/working/{1}/{2}{3}/{2}_run{4}/{5}/".format(datadrive,sim_type,category,extra_path,run,outfolder)
        if (os.path.exists(folder)):
            return Simulation(folder, snapbase=snapbase)
        folder = "/Volumes/LucchiniResearchData{0}/HPC_Backup/scratch/{1}/{2}{3}/{2}_run{4}/{5}/".format(datadrive,sim_type,category,extra_path,run,outfolder)
        if (os.path.exists(folder)):
            return Simulation(folder, snapbase=snapbase)
        datadrive -= 1

    raise Exception("Folder not found on any data drives attached: {}".format(folder))

class Simulation(object):
    """
    This class holds a folder with snapshots belonging to a single simulation
    and makes it easy to apply functions over all of those snapshots (plotting,
    measuring quantities, or printing info). Initialize by supplying a folder.
    """
    def __init__(self, folder, snaps=None, snapbase='snapshot_', snapext='hdf5'):
        """
        Default behavior is to get all the snapshots in a folder, more complicated
        behavior is governed by utils.list_snapshots()
        """
        if (folder[-1] != '/'):
            folder += '/'
        self.folder = os.path.realpath(folder)
        self.snapbase = snapbase
        # Default behavior is to collect everything that matches the snapbase
        if snaps is None:
            self.snaps = [folder+f for f in os.listdir(folder)
                            if re.search("^"+snapbase+"[0-9]{3}.*\.%s$" % snapext, f)]
            self.snaps = np.sort(self.snaps)
        #otherwise will use only certain range or given numbers
        else:
           self.snaps = utils.list_snapshots(snaps, folder, snapbase)

        self.nsnaps = len(self.snaps)
        self.settings = utils.make_settings()

    def present_day(self,return_last=True):
        cpfile = self.folder+"/computed_positions.hdf5"

        if (not os.path.exists(cpfile)):
            raise Exception("{}/computed_positions.hdf5 not found".format(self.folder))
        with h5py.File(cpfile) as f:
            lmc_pos = f['Disk/lmc_pos'][:]
            mw_pos = f['Disk/mw_pos'][:]

        lmc_pos_cent = lmc_pos - mw_pos
        lmc_pos_mc = SkyCoord(Galactocentric(x=lmc_pos_cent[:,0]*u.kpc,y=lmc_pos_cent[:,1]*u.kpc,
                                             z=lmc_pos_cent[:,2]*u.kpc)) \
                        .transform_to(MScoord).MSLongitude.value
        lmc_pos_mc[lmc_pos_mc > 180] -= 360
        lmc_dist = np.linalg.norm(lmc_pos,axis=1)
        t0index = np.where((lmc_pos_mc > -5) & (lmc_dist < 100))[0]
        # t0index = np.argmin(np.abs(lmc_pos_mc))
        if (len(t0index) == 0):
            if (return_last):
                warnings.warn("Doesn't reach present day position. Returning last snap.")
                return len(lmc_pos)-1
            else:
                raise Exception("Doesn't reach present day position.")
        t0index = t0index[0]

        return t0index

    def has_computed_positions(self):
        return os.path.exists(self.folder+'/computed_positions.hdf5')

    def compute_positions(self,lmconly=False,multiprocessing=False,overwrite=True,verbose=True):
        """
        Generates computed_positions.hdf5 file with LMC, SMC, and MW pos and vel at all sim times
        """
        cp.run(self,lmconly=lmconly,multiprocessing=multiprocessing,overwrite=overwrite,verbose=verbose)

    def get_computed_positions(self,mw=True,lmc=True,smc=True,times=True):
        if (self.has_computed_positions()):
            return cp.get_computed_positions(self,mw,lmc,smc,times)
        else:
            raise Exception("No computed postions file found.")

    def measure_centers_of_mass(self):
        """
        Get the center of masses of the galaxies in the snapshots.
        """
        def center_of_mass(snapname):
            try:
                snap = snapshot.Snapshot(snapname, lazy=True)
                com1s, com2s, idgals, idgal2s = snap.center_of_mass('stars')
                return com1s
            except KeyboardInterrupt:
                pass

        return np.array(self.apply_function(center_of_mass))


    def measure_separation(self, baseGal=0, includeOrigin=False, mass_list=None):
        """
        Get relative velocity and radial separation between N galaxies.
        Measures relative to the galaxy given in baseGal
        """
        def centers(snapname):
            try:
                snap = snapshot.Snapshot(snapname) #, lazy=True)
                gals_indices = snap.split_galaxies('stars', mass_list=mass_list)
                coms = snap.measure_com('stars', gals_indices)
                covs = np.empty((len(gals_indices), 3))
                for i, gal_indices in enumerate(gals_indices):
                    covs[i, :] = snap.vel['stars'][gal_indices, :].mean(axis=0)
                if includeOrigin:
                    coms = np.insert(coms,0,[0,0,0],axis=0)
                    covs = np.insert(covs,0,[0,0,0],axis=0)

                time = snap.header['time']
                return [coms, covs, time]
            except KeyboardInterrupt:
                pass

        cents = self.apply_function(centers)
        nsnaps = len(self.snaps)

        Ngals = len(cents[0][0])

        distances = np.empty((nsnaps, Ngals))
        velocities = np.empty((nsnaps, Ngals))
        times = np.empty(nsnaps)

        for i, cent in enumerate(cents):
            reference_pos = cent[0][baseGal, :]
            reference_vel = cent[1][baseGal, :]
            times[i] = cent[2]

            for j, (other_pos, other_vel) in enumerate(zip(cent[0], cent[1])):
                # if (np.all(other_pos == reference_pos) and np.all(other_vel == reference_vel)): continue
                distances[i, j] = np.linalg.norm(other_pos - reference_pos)
                velocities[i, j] = np.linalg.norm(other_vel - reference_vel)

        keepCols = np.delete(np.array(range(Ngals)),baseGal)
        distances = distances[:,keepCols]
        velocities = velocities[:,keepCols]

        return distances, velocities, times


    def apply_function(self, function, multiprocessing=True, *args):
        """
        Map a user supplied function over the snapshots.
        Uses pathos.multiprocessing (https://github.com/uqfoundation/pathos.git).
        """
        if (multiprocessing):
            pool = Pool()

            try:
                val = pool.map(function, self.snaps)
                return val
            except KeyboardInterrupt:
                print('got ^C while pool mapping, terminating the pool')
                pool.terminate()
                print('pool is terminated')
            except Exception as e:
                print('got exception: %r, terminating the pool' % (e,))
                pool.terminate()
                print('pool is terminated')
        else:
            val = []
            for s in tqdm(self.snaps):
                val.append(function(s))
            return val


    def print_settings(self):
        """
        Print the current settings
        """
        for key, val in iteritems(self.settings):
            print("{0}: {1}".format(key, val))


    def set_settings(self, **kwargs):
        """
        Set simulation-wide settings
        """
        for name, val in iteritems(kwargs):
            if name not in self.settings.keys():
                print("WARNING! {:s} is not a default setting!".format(name))
            self.settings[name] = val


    def get_snapshot(self, num=None, lazy=False):
        """
        Return snapshot
        """
        if num is None:
            for i, s in enumerate(self.snaps):
                print("snap %i: %s" % (i, s))
            num = int(raw_input('Select snapshot:'))
        try:
            if lazy:
                return snapshot.Snapshot(self.snaps[num], lazy=lazy)
            else:
                return snapshot.Snapshot(self.snaps[num])
        except IndexError:
            print('Not a valid snapshot')
            self.get_snapshot(None)
        except TypeError:
            return None


    def __repr__(self):
        return "Simulation located at {0} with {1} snapshots".format(self.folder, self.nsnaps)
