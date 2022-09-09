import numpy as np, pNbody as pnb, datetime, os, h5py
from snaptools import simulation, snapshot
from functools import partial

def today():
    return datetime.date.today().strftime('%B%d')

def find_pos_pot(snapname, mass_arr=None):
    
    try:
        snap = snapshot.Snapshot(snapname)
    except:
        raise RuntimeException("Failure on snapshot: {}".format(snapname))
    
    pt = 'halo'
    gal_ids = snap.split_galaxies(pt,mass_list=mass_arr)
    output = []
    
    for g in gal_ids:
        pos = np.array(snap.pos[pt][g],dtype=np.float32)
        vel = np.array(snap.vel[pt][g],dtype=np.float32)
        mass = np.array(snap.masses[pt][g],dtype=np.float32)

        nb = pnb.Nbody(pos=pos, mass=mass, verbose=0)
        pot = nb.TreePot(pos, eps=0.1)
        own_most_bound = np.argsort(pot)[:100]
        output.append(pos[own_most_bound, :].mean(axis=0))
        output.append(vel[own_most_bound, :].mean(axis=0))
    output.append(snap.header['time'])
    return output

def save_to_file(f, folder, times, mw_mass, lmc_mass, smc_mass, mw_pos, lmc_pos, smc_pos, mw_vel, lmc_vel, smc_vel):
    """
    Write positions to file
    """
    grp = f.create_group('Header')
    # grp.attrs.create('Name', np.string_(name))
    grp.attrs.create('Created', np.string_(today()))
    grp.attrs.create('Folder', np.string_(folder))
    grp.attrs.create('LMC_mass', lmc_mass)
    grp.attrs.create('SMC_mass', smc_mass)
    grp.attrs.create('MW_mass', mw_mass)

    subgrp = f.create_group('Disk')
    dset = subgrp.create_dataset('time', times.shape)
    dset[:] = times
    
    dset = subgrp.create_dataset('mw_pos', mw_pos.shape)
    dset[:] = mw_pos
    dset = subgrp.create_dataset('lmc_pos', lmc_pos.shape)
    dset[:] = lmc_pos
    dset = subgrp.create_dataset('smc_pos', smc_pos.shape)
    dset[:] = smc_pos
    
    dset = subgrp.create_dataset('mw_vel', mw_vel.shape)
    dset[:] = mw_vel
    dset = subgrp.create_dataset('lmc_vel', lmc_vel.shape)
    dset[:] = lmc_vel
    dset = subgrp.create_dataset('smc_vel', smc_vel.shape)
    dset[:] = smc_vel

def run(sim,lmconly,overwrite=True,verbose=False):

    folder = sim.folder+"/"

    snap = sim.get_snapshot(0)

    gal_ids_halo = snap.split_galaxies('halo')
    mw_mass = 0
    lmc_mass = 0
    smc_mass = 0
    if (len(gal_ids_halo) >= 3):
        mw_mass, lmc_mass, smc_mass = [snap.masses['halo'][g[0]] for g in gal_ids_halo[:3]]
    if (len(gal_ids_halo) == 2):
        if (lmconly):
            mw_mass, lmc_mass = [snap.masses['halo'][g[0]] for g in gal_ids_halo]
        else:
            lmc_mass, smc_mass = [snap.masses['halo'][g[0]] for g in gal_ids_halo]
    
    if (verbose):
        print(mw_mass,lmc_mass,smc_mass)
    
    save_file = "{}computed_positions.hdf5".format(folder)

    sim = simulation.Simulation(folder)

    if (lmconly):
        marr = [lmc_mass,mw_mass]
    else:
        marr = [lmc_mass,smc_mass]
        if (mw_mass != 0):
            marr = [lmc_mass,smc_mass,mw_mass]

    _find_pos = partial(find_pos_pot, mass_arr=marr)

    # only run the rest of the analysis if we need to
    if not os.path.isfile(save_file) or overwrite:  # if the file doesn't exist or if we want to overwrite the files

        #disk
        positions_stars = sim.apply_function(_find_pos)

        mw_pos = np.zeros((len(positions_stars), 3))
        mw_vel = np.zeros((len(positions_stars), 3))
        lmc_pos = np.zeros((len(positions_stars), 3))
        lmc_vel = np.zeros((len(positions_stars), 3))
        smc_pos = np.zeros((len(positions_stars), 3))
        smc_vel = np.zeros((len(positions_stars), 3))
        times = np.zeros(len(positions_stars))


        for j, p in enumerate(positions_stars):
            lmc_pos[j, :] = p[0]
            lmc_vel[j, :] = p[1]
            if (lmconly):
                mw_pos[j,:] = p[2]
                mw_vel[j,:] = p[3]
            else:
                smc_pos[j, :] = p[2]
                smc_vel[j, :] = p[3]
                if (len(marr) > 2):
                    mw_pos[j, :] = p[4]
                    mw_vel[j, :] = p[5]
            times[j] = p[-1]

        # save to disk
        with h5py.File(save_file, 'w') as f:
            save_to_file(f, folder, times, mw_mass, lmc_mass, smc_mass, mw_pos, lmc_pos, smc_pos, mw_vel, lmc_vel, smc_vel)

    if (verbose):
        print("Saved to: {}".format(save_file))
        print("Present Day = snapshot_{:03}".format(sim.present_day()))
