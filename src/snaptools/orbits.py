import astropy.coordinates as coord
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import G
from scipy.special import erf
from scipy.signal import argrelextrema
from truncatedhernquist.potential import TruncatedHernquistPotential

import gala.coordinates as gc
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic

G_gal = G.decompose(galactic).value

# Analytic MW from Barnes+ 2002 (ref'd in Wang+ 2019, Hammer+ 2015)
# MW1: a = 6 kpc, rho0 = 0.7379 Msun/pc^3
# MW2: a = 6 kpc, rho0 = 0.4919 Msun/pc^3
class BarnesPotential(gp.PotentialBase):
    a = gp.PotentialParameter("a")
    rho0 = gp.PotentialParameter("rho0")
    ndim = 3

    def _energy(self, xyz, t):
        a = self.parameters['a'].value*u.kpc
        rho0 = self.parameters['rho0'].value*u.Msun/u.pc**3
        x,y,z = xyz.T
        r = np.sqrt(x**2+y**2+z**2)
        const = -2.0*np.pi/3.0*G*rho0*a**3
        return const*(a+2*r)/(a+r)**2
    
    def _gradient(self, xyz, t):
        a = self.parameters['a'].value*u.kpc
        rho0 = self.parameters['rho0'].value*u.Msun/u.pc**3

        x,y,z = xyz.T
        r = np.sqrt(x**2+y**2+z**2)

        const = -2.0*np.pi/3.0*G*rho0*a**3
        grad = const*2.0/(a+r)**2*(1 - (a+2*r)/(a+r))
        return grad

###
### constant coulomb log ###
###
# def chandrasekhar_acc(t, w, mw_potential, Msat, ln_Lambda, v_disp):
#     """
#     Eqn. 8.2, 8.7 in Binney & Tremaine (2008)
#     """
#     x = np.ascontiguousarray(w[:3].T)
#     v = w[3:]
    
#     # This looks ugly because, for speed, I access some low-level Gala/C stuff..
#     dens = mw_potential._density(x, t=np.array([0.]))
#     v_norm = np.sqrt(np.sum(v**2, axis=0))
    
#     X = v_norm / (np.sqrt(2) * v_disp)
#     fac = erf(X) - 2*X/np.sqrt(np.pi) * np.exp(-X**2)
    
#     dv_dynfric = - 4*np.pi * G_gal**2 * Msat * dens * ln_Lambda / v_norm**3 * fac * v
    
#     return -dv_dynfric

# def F(t, raw_w, nbody, chandra_kwargs):
#     w = gd.PhaseSpacePosition.from_w(raw_w, units=nbody.units)
#     nbody.w0 = w
    
#     wdot = np.zeros((2 * w.ndim, w.shape[0]))
    
#     # Compute the mutual N-body acceleration:
#     wdot[3:] = nbody._nbody_acceleration()
    
#     # Only compute DF for non-MW particles (the 1:):
#     wdot[3:, 1:] += chandrasekhar_acc(t, raw_w[:, 1:], **chandra_kwargs)
    
#     wdot[:3] = w.v_xyz.decompose(nbody.units).value
    
#     return wdot

def get_vdisp(r,vmax):
    return vmax.decompose(galactic).value*1.4394*r**0.354/(1+1.1756*r**0.725)

###
### r-dependent coulomb log ###
###
def chandrasekhar_acc(t, w, mw_potential, Msat, vmax, rs, LCa=(0.0,1.22,1.0), mw=True):
    """
    Eqn. 8.2, 8.7 in Binney & Tremaine (2008)
    """

    if ((mw_potential is None) or (np.all(Msat == 0))):
        return 0.0

    x0 = np.ascontiguousarray(w[:3].T)[0] 
    x = np.ascontiguousarray(w[:3].T)[-1:]
    v0 = w[3:,0]
    v = w[3:,-1]
    v = np.array([v[i] - v0[i] for i in range(3)])

    dens = mw_potential._density((x - x0),t=np.array([0.]))
    v_norm = np.sqrt(np.sum(v**2, axis=0))
    v_disp = get_vdisp(np.linalg.norm((x-x0),axis=1),vmax)

    X = v_norm / (np.sqrt(2) * v_disp)
    fac = erf(X) - 2*X/np.sqrt(np.pi) * np.exp(-X**2)
    
    # from Patel et al (2016) https://arxiv.org/pdf/1609.04823.pdf
    r = np.array([np.linalg.norm(xi - x0) for xi in x])
    if (LCa is not None):
        L,C,alpha = LCa
        ln_Lambda = np.array([max(L,np.log(r[i]/(C*rs[i]))**alpha) for i in range(len(r))])
    else:
        ln_Lambda = np.array([np.log(r[i])/(1.44*rs[i]) for i in range(len(r))])
    
    if mw:
        dv_dynfric = - 4*np.pi * G_gal**2 * Msat * dens * ln_Lambda / v_norm**3 * fac * v
    else:
        # ln_Lambda = np.array([[0.3]*len(r)])
        ln_Lambda = 0.3
        dv_dynfric = - 0.428 * ln_Lambda * G_gal * Msat / r**2 * v / v_norm

    return dv_dynfric

###
### ram pressure ###
###
def rampressure_acc(t, w, hydro_cpt, area, Msat):

    x0 = np.ascontiguousarray(w[:3].T)[0] 
    x = np.ascontiguousarray(w[:3].T)[1:]
    v0 = w[3:,0]
    v = w[3:,1:]
    v = np.array([v[i] - v0[i] for i in range(3)])
    # vnorm = np.linalg.norm(v,axis=0)
    # vunit = np.array([v.T[i]/vnorm[i] for i in range(len(vnorm))])
    
    # This looks ugly because, for speed, I access some low-level Gala/C stuff..
    dens = hydro_cpt._density((x-x0), t=np.array([0.]))
    v2 = np.sum(v**2, axis=0)
    vnorm = np.sqrt(v2)
    
    press = dens*v2*v/vnorm
    force = press*area
    acc = force/Msat

    return -1*acc

def soften(r):
    softeningLength = 0.1*u.kpc
    return r*(softeningLength**2 + r**2)**(-1.5) # kpc^-2

def get_tidal_r(t,w,host_potential,sat_potential):
    x0 = np.ascontiguousarray(w[:3])[:,0]*u.kpc
    x = np.ascontiguousarray(w[:3])[:,1:]*u.kpc
    x0 = np.reshape(x0,(3,1))
    deltavec = x - x0
    delta = np.linalg.norm(deltavec,axis=0)
    if (0 in delta): raise Exception("oops delta = 0")
    mass1 = np.array([np.sum([p[k].parameters['m'].value for k in p.keys()]) for p in sat_potential])*u.Msun
    mass2 = host_potential.mass_enclosed(deltavec*u.kpc)
    tidal_r = (delta*(mass1/(2*mass2))**(1/3.)).value

    return tidal_r

def tidal_mass_loss(t,w,host_potential,sat_potential):
    tidal_r = get_tidal_r(t,w,host_potential,sat_potential)
    
    sat_Mleft = np.array([sat_potential[i]['halo'].mass_enclosed(np.array([tidal_r[i],0,0])*u.kpc) for i in range(len(sat_potential))])

    return sat_Mleft

def disk_ram_pressure(t, w):
    global disk_interaction

    area = np.pi*5**2
    dens = 247100 # Msun/kpc^3 = 1e-2 cm^-3
    thickness = 1 # kpc
    diskmass = 1e9 # Msun
    rmax = 5

    dv = np.zeros_like(w[3:,1:])

    for i in range(len(w.T)-1):
        x0 = np.ascontiguousarray(w[:3].T)[i]
        x = np.ascontiguousarray(w[:3].T)[i+1:]
        rr = np.linalg.norm(x-x0,axis=1)
        rxy = np.linalg.norm((x-x0)[:,:2],axis=1)
        zz = (x-x0)[:,2]
        for j in range(len(x)):
            # if (rr[j] < rmax):
            if ((rxy[j] < rmax) & (zz < thickness)):
                v0 = w[3:,i]
                v = w[3:,i+j+1]
                v = np.array([v[k] - v0[k] for k in range(3)])
                v2 = np.sum(v**2, axis=0)
                vnorm = np.sqrt(v2)
                dv[:,i+j] = -dens*area*v2/diskmass*v/vnorm

    return dv

def F(t, raw_w, nbody, chandra_kwargs):

    w = gd.PhaseSpacePosition.from_w(raw_w, units=nbody.units)
    nbody.w0 = w
    
    wdot = np.zeros((2 * w.ndim, w.shape[0]))
    
    # Compute the mutual N-body acceleration:
    wdot[3:] = nbody._nbody_acceleration()
    
    # wdot[3:, 0] += chandrasekhar_acc_mw(t, raw_w[:, 0:], mw_potential=chandra_kwargs['lmc_potential'], 
    #                             Msat=1e12,vmax=chandra_kwargs['vmax'][0])

    # Compute MW DF (only on non-MW particles)
    xi = chandra_kwargs['xi']
    if (chandra_kwargs['mw_potential'] is not None):
        wdot[3:, 1] += xi*chandrasekhar_acc(t, raw_w[:, 0:2], mw_potential=chandra_kwargs['mw_potential'], Msat=chandra_kwargs['Msats'][:,0], vmax=chandra_kwargs['vmax'][0],rs=np.array([23.1]),LCa=chandra_kwargs['LCa'],mw=True)
        # Compute DF on SMC from MW and LMC
        if (chandra_kwargs['smc_potential'] is not None):
            wdot[3:, 2] += xi*chandrasekhar_acc(t, raw_w[:, 0:3], mw_potential=chandra_kwargs['mw_potential'], Msat=chandra_kwargs['Msats'][:,1], vmax=chandra_kwargs['vmax'][0],rs=np.array([8.6]),LCa=None,mw=True)

            wdot[3:, 2] += xi*chandrasekhar_acc(t, raw_w[:, 1:], mw_potential=chandra_kwargs['lmc_potential'],
                                        Msat=chandra_kwargs['Msats'][:,1],vmax=chandra_kwargs['vmax'][1],rs=np.array([8.6]),LCa=None,mw=False)
    else:
        if (chandra_kwargs['smc_potential'] is not None):
            # Compute SMC DF (on LMC)
            # wdot[3:, 0] += chandrasekhar_acc(t, raw_w[:, 0:], mw_potential=chandra_kwargs['smc_potential'], 
            #                     Msat=chandra_kwargs['Msats'][0][0],vmax=chandra_kwargs['vmax'][2],
            #                     rs=chandra_kwargs['rs'][0],LCa=chandra_kwargs['LCa_mw'])
            
            # Compute LMC DF (only on SMC)
            wdot[3:, 1:] += xi*chandrasekhar_acc(t, raw_w[:, 0:], mw_potential=chandra_kwargs['lmc_potential'],
                                        Msat=chandra_kwargs['Msats'][0][1:],vmax=chandra_kwargs['vmax'][1],
                                        xi=chandra_kwargs['xi'],rs=chandra_kwargs['rs'][1:],LCa=None,mw=False)
    # wdot[3:, 2:] += tidal_acc(t, raw_w[:, 1:], chandra_kwargs['lmc_potential'], chandra_kwargs['Msmc'])

    wdot[:3] = w.v_xyz.decompose(nbody.units).value

    ### Ram Pressure ###
    # if ((chandra_kwargs['hydro_component'] is not None) and (chandra_kwargs['hydro_factor'] != 0)):
    #     wdot[3:,1:] += chandra_kwargs['hydro_factor']*rampressure_acc(t, raw_w[:,0:],chandra_kwargs['hydro_component'],chandra_kwargs['hydro_area'],chandra_kwargs['hydro_Msat'])
    if (chandra_kwargs['hydro_factor'] != 0):
        if (chandra_kwargs['mw_potential'] is not None):
            for cpt in chandra_kwargs['hydro_component'][0]:
                wdot[3:,1:] += chandra_kwargs['hydro_factor']*rampressure_acc(t, raw_w[:,0:],cpt,chandra_kwargs['hydro_area'],chandra_kwargs['hydro_Msat'])
            if (chandra_kwargs['smc_potential'] is not None):
                for cpt in chandra_kwargs['hydro_component'][1]:
                    wdot[3:,2:] += chandra_kwargs['hydro_factor']*rampressure_acc(t, raw_w[:,1:],cpt,chandra_kwargs['hydro_area'][0][1],chandra_kwargs['hydro_Msat'][0][1])
        else:
            for cpt in chandra_kwargs['hydro_component'][1]:
                wdot[3:,1:] += chandra_kwargs['hydro_factor']*rampressure_acc(t, raw_w[:,0:],cpt,chandra_kwargs['hydro_area'][0][1],chandra_kwargs['hydro_Msat'][0][1])
    
    ### Disk ram pressure ###
    
    # wdot[3:,1:] += chandra_kwargs['disk_ram_pressure']*disk_ram_pressure(t,raw_w)

    ### Tidal Mass Loss ###
    # from MW
    # if chandra_kwargs['massloss']:
    #     if (chandra_kwargs['smc_potential'] is not None):
    #         sat_Mleft1 = tidal_mass_loss(t,raw_w[:,0:3],chandra_kwargs['mw_potential'],[chandra_kwargs['lmc_potential'],chandra_kwargs['smc_potential']])
    #         sat_Mleft2 = tidal_mass_loss(t,raw_w[:,1:3],chandra_kwargs['lmc_potential'],[chandra_kwargs['smc_potential']])
    #         sat_Mleft = [sat_Mleft1[0],min(sat_Mleft1[1],sat_Mleft2[0])]
    #     else:
    #         sat_Mleft = [tidal_mass_loss(t,raw_w,chandra_kwargs['mw_potential'],[chandra_kwargs['lmc_potential']])]
        
    #     for i in range(len(sat_Mleft)):
    #         nbody.particle_potentials[i+1]['halo'] = gp.HernquistPotential(m=sat_Mleft[i],c=nbody.particle_potentials[i+1]['halo'].parameters['c'],units=galactic)
    if chandra_kwargs['massloss']:
        if (chandra_kwargs['mw_potential'] is not None):
            if (chandra_kwargs['smc_potential'] is not None):
                sat_r1 = get_tidal_r(t,raw_w[:,0:3],chandra_kwargs['mw_potential'],[chandra_kwargs['lmc_potential'],chandra_kwargs['smc_potential']])
                sat_r2 = get_tidal_r(t,raw_w[:,1:3],chandra_kwargs['lmc_potential'],[chandra_kwargs['smc_potential']])
                sat_r = [sat_r1[0],min(sat_r1[1],sat_r2[0])]
            else:
                sat_r = [get_tidal_r(t,raw_w,chandra_kwargs['mw_potential'],[chandra_kwargs['lmc_potential']])]
        else:
            sat_r2 = get_tidal_r(t,raw_w[:,0:],chandra_kwargs['lmc_potential'],[chandra_kwargs['smc_potential']])
            sat_r = [sat_r2]
        
        for i in range(len(sat_r)):
            oldparams = nbody.particle_potentials[i+1]['halo'].parameters
            nbody.particle_potentials[i+1]['halo'] = TruncatedHernquistPotential(m=oldparams['m'],c=oldparams['c'],rmax=sat_r[i]*u.kpc,units=galactic)

    return wdot

def F_many(t, raw_w, nbody, chandra_kwargs):

    w = gd.PhaseSpacePosition.from_w(raw_w, units=nbody.units)
    nbody.w0 = w
    
    wdot = np.zeros((2 * w.ndim, w.shape[0]))
    
    # Compute the mutual N-body acceleration:
    wdot[3:] = nbody._nbody_acceleration()

    ### Dynamical Friction
    xi = chandra_kwargs['xi']
    ngals = np.sum([gp is not None for gp in chandra_kwargs['gal_pots']])
    for i in range(ngals-1):
        for j in range(ngals):
            if (j<=i):
                continue
            wdot[3:,j] += xi*chandrasekhar_acc(t, raw_w[:,[i,j]],mw_potential=chandra_kwargs['gal_pots'][i],Msat=np.array([chandra_kwargs['Msats'][j]]),vmax=chandra_kwargs['vmax'][i],rs=[chandra_kwargs['rs'][j]],LCa=chandra_kwargs['LCa'],mw=not i)

    wdot[:3] = w.v_xyz.decompose(nbody.units).value

    return wdot

def get_default_potentials():
    # (00_isolated/MW/ICs/dm_only/MW_dm_only.hdf5) + (00_isolated/MW/ICs/mw_just_corona_salem_corr_2e10msol_1e6K_medres.hdf5)
    pot_mw = gp.CCompositePotential()
    pot_mw['halo'] = gp.HernquistPotential(m=1.06554e12*u.Msun, c=22, units=galactic)
    pot_mw['cgm'] = gp.HernquistPotential(m=2e10*u.Msun, c=100, units=galactic)

    # (00_isolated/LMC/ICs/less_gas_medres/LMC_less_gas_medres.hdf5) + (00_isolated/LMC/LMC_corona_8.3e9msol_2.4e5K_hot_medres.hdf5)
    pot_lmc = gp.CCompositePotential()
    pot_lmc['disk'] = gp.MiyamotoNagaiPotential(m=3.5e9*u.Msun, a=1.7*u.kpc, b=800*u.pc, units=galactic)
    pot_lmc['halo'] = gp.HernquistPotential(m=1.75e11*u.Msun, c=9, units=galactic)
    pot_lmc['cgm'] = gp.HernquistPotential(m=8.3e9*u.Msun, c=10, units=galactic)

    # 00_isolated/SMC/stability/med_disk_medres/output/snapshot_060.hdf5
    pot_smc = gp.CCompositePotential()
    pot_smc['disk'] = gp.MiyamotoNagaiPotential(m=2.16e9*u.Msun, a=1*u.kpc, b=600*u.pc, units=galactic)
    pot_smc['halo'] = gp.HernquistPotential(m=1.0e10*u.Msun, c=15, units=galactic)

    return pot_mw,pot_lmc,pot_smc

def get_hydro_cpts(pot,name='cgm'):
    if (pot is None):
        return None
    m = [k.startswith(name) for k in pot.keys()]
    potlist = []
    for k in np.array(list(pot.keys()))[m]:
        potlist.append(pot[k])
    return potlist

def run_orbit(lmc_pos,lmc_vel,smc_pos=None,smc_vel=None,t2=-5*u.Gyr,dt=-10*u.Myr,coulombLogXi=1,rampressure=0,tidal=0,massloss=0,pot_mw=None,pot_lmc=None,pot_smc=None,progress=True,hydro_areas=np.array([[np.pi*8**2,np.pi*4**2]]),disk_ram_pressure=0,LCa=[0,1.22,1],return_nbody=False):
    
    global disk_interaction
    disk_interaction = np.zeros((2,2))

    if (massloss & (dt.value<0)):
        print("Warning: Massloss with backwards integration is not supported. Setting massloss = 0.")
        massloss = 0

    ### Set potentials
    # pot1,pot2,pot3 = get_default_potentials()
    # if (pot_mw is None):
    #     pot_mw = pot1
    # if (pot_lmc is None):
    #     pot_lmc = pot2
    # if (pot_smc is None):
    #     pot_smc = pot3

    lmc_mhalo = 0.0; lmc_mdisk = 0.0; smc_mhalo = 0.0; smc_mdisk = 0.0
    if 'halo' in pot_lmc:
        lmc_mhalo = pot_lmc['halo'].parameters['m'].to('Msun').value if 'halo' in pot_lmc else 0.0
    if 'disk' in pot_lmc:
        lmc_mdisk = pot_lmc['disk'].parameters['m'].to('Msun').value if 'disk' in pot_lmc else 0.0
    if pot_smc is not None:
        if 'halo' in pot_smc:
            smc_mhalo = pot_smc['halo'].parameters['m'].to('Msun').value if 'halo' in pot_smc else 0.0
        if 'disk' in pot_smc:
            smc_mdisk = pot_smc['disk'].parameters['m'].to('Msun').value if 'disk' in pot_smc else 0.0

    if pot_mw is not None:
        vmax_mw = np.nanmax(pot_mw.circular_velocity(np.array(list(zip(np.zeros(101),np.zeros(101),np.linspace(0,200,101)))).T))
    else:
        vmax_mw = None
    vmax_lmc = np.nanmax(pot_lmc.circular_velocity(np.array(list(zip(np.zeros(101),np.zeros(101),np.linspace(0,200,101)))).T))
    if pot_smc is not None:
        vmax_smc = np.nanmax(pot_smc.circular_velocity(np.array(list(zip(np.zeros(101),np.zeros(101),np.linspace(0,200,101)))).T))
    else:
        vmax_smc = None

    chandra_kwargs = {
        'Msats': np.array([[lmc_mhalo,smc_mhalo]]) if smc_pos is not None else np.array([[lmc_mhalo]]),
        'mw_potential': pot_mw, 
        'lmc_potential': pot_lmc,
        'smc_potential': pot_smc if smc_pos is not None else None,
        'xi': coulombLogXi,
        'vmax': [vmax_mw,vmax_lmc,vmax_smc],
        'tidal': tidal,
        'hydro_factor': rampressure,
        'hydro_component': [get_hydro_cpts(pot_mw),get_hydro_cpts(pot_lmc)], # pot_mw['cgm'] if 'cgm' in pot_mw else None,
        'hydro_Msat': np.array([[lmc_mdisk,smc_mdisk]]),
        'hydro_area': hydro_areas, # disk surface areas
        'massloss': massloss,
        'disk_ram_pressure':disk_ram_pressure,
        'LCa':LCa,
        'rs':[9.87,16,8] if pot_mw is not None else [16,8]
    }
    
    ### Set initial positions
    w0_mw = gd.PhaseSpacePosition(
            pos=[0, 0, 0]*u.kpc,
            vel=[0, 0, 0]*u.km/u.s
        )
    w0_lmc = gd.PhaseSpacePosition(
            pos=lmc_pos*u.kpc,
            vel=lmc_vel*u.km/u.s
        )
    if (pot_mw is not None):
        if (smc_pos is not None):
            w0_smc = gd.PhaseSpacePosition(
                    pos=smc_pos*u.kpc,
                    vel=smc_vel*u.km/u.s
                )
            w0 = gd.combine((w0_mw, w0_lmc, w0_smc))
            nbody = gd.DirectNBody(w0, [pot_mw, pot_lmc, pot_smc])
        else:
            w0 = gd.combine((w0_mw, w0_lmc))
            nbody = gd.DirectNBody(w0, [pot_mw, pot_lmc])
    else:
        w0_smc = gd.PhaseSpacePosition(
                    pos=smc_pos*u.kpc,
                    vel=smc_vel*u.km/u.s
                )
        w0 = gd.combine((w0_lmc, w0_smc))
        nbody = gd.DirectNBody(w0, [pot_lmc, pot_smc])

    ### Run integrator
    integrator = gi.DOPRI853Integrator(
            F, func_args=(nbody, chandra_kwargs), 
            func_units=nbody.units, 
            progress=progress
        )
    orbits = integrator.run(w0, dt=dt, t1=0, t2=t2)
    
    if (return_nbody):
        return orbits,nbody
    else:
        return orbits


def run_orbit_many(gal_pos,gal_vel,gal_pots,rs,t2=-5*u.Gyr,dt=-10*u.Myr,coulombLogXi=1,progress=True,LCa=[0,1.22,1]):

    ngals = len(gal_pos)
    if (len(gal_vel) != ngals):
        raise Exception("Galaxy velocities ({}) doesn't match positions ({}).".format(len(gal_vel),ngals))
    if (len(gal_pots) != ngals):
        raise Exception("Galaxy potentials ({}) doesn't match positions ({})".format(len(gal_pots),ngals))
    if (len(rs) != ngals):
        raise Exception("Galaxy rs ({}) doesn't match positions ({})".format(len(rs),ngals))

    mhalos = []
    for pot in gal_pots:
        if (pot is not None):
            mhalos.append(pot['halo'].parameters['m'].to('Msun').value if 'halo' in pot else 0.0)
        else:
            mhalos.append(0.0)

    vmaxs = []
    for pot in gal_pots:
        if (pot is not None):
            vmaxs.append(np.nanmax(pot.circular_velocity(np.array(list(zip(np.zeros(101),np.zeros(101),np.linspace(0,200,101)))).T)))
        else:
            vmaxs.append(0)

    chandra_kwargs = {
        'Msats': np.array(mhalos),
        'gal_pots': gal_pots, 
        'xi': coulombLogXi,
        'vmax': vmaxs,
        'LCa':LCa,
        'rs':rs
    }
    
    ### Set initial positions
    w0_arr = []
    for i in range(len(gal_pos)):
        w0_arr.append(gd.PhaseSpacePosition(
            pos=gal_pos[i]*u.kpc,
            vel=gal_vel[i]*u.km/u.s
        ))
    w0 = gd.combine(w0_arr)
    nbody = gd.DirectNBody(w0, gal_pots)

    ### Run integrator
    integrator = gi.DOPRI853Integrator(
            F_many, func_args=(nbody, chandra_kwargs), 
            func_units=nbody.units, 
            progress=progress
        )
    orbits = integrator.run(w0, dt=dt, t1=0, t2=t2)
    
    return orbits