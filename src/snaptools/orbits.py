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
def chandrasekhar_acc(t, w, mw_potential, Msat, vmax, xi, mw=True):
    """
    Eqn. 8.2, 8.7 in Binney & Tremaine (2008)
    """
    x0 = np.ascontiguousarray(w[:3].T)[0] 
    x = np.ascontiguousarray(w[:3].T)[1:]
    v0 = w[3:,0]
    v = w[3:,1:]
    v = np.array([v[i] - v0[i] for i in range(3)])

    dens = mw_potential._density((x - x0),t=np.array([0.]))
    v_norm = np.sqrt(np.sum(v**2, axis=0))
    v_disp = get_vdisp(np.linalg.norm((x-x0),axis=1),vmax)

    X = v_norm / (np.sqrt(2) * v_disp)
    fac = erf(X) - 2*X/np.sqrt(np.pi) * np.exp(-X**2)
    
    # from Patel et al (2016) https://arxiv.org/pdf/1609.04823.pdf
    r = np.array([np.linalg.norm(xi - x0) for xi in x])
    # a = 0.1; rs = 3
    # rs = np.array([3,2])[:len(r)] # Besla+ 2007
    # rs = np.array([23.1,2.5]) # Patel+ 2020?
    rs = np.array([23.1,10])[:len(r)]
    # L = 0.0; C = 1.22; alpha = 1.0
    L = 0.1; C = 0.4; alpha = 1.0
    # ln_Lambda = xi*np.array([[max(L,np.log(ri/C/a)**alpha) for ri in r]])
    # ln_Lambda = xi*np.log(np.array([r])/(1.44*rs))
    if mw:
        ln_Lambda = xi*np.array([max(L,np.log(r[i]/(C*rs[i]))**alpha) for i in range(len(r))])
        dv_dynfric = - 4*np.pi * G_gal**2 * Msat * dens * ln_Lambda / v_norm**3 * fac * v
    else:
        ln_Lambda = np.array([[0.3]*len(r)])
        # ln_Lambda = np.array([[4]*len(r)])
        dv_dynfric = - 0.428 * ln_Lambda * G_gal * Msat / r**2 * v / v_norm
    # dv_dynfric = 0

    # print(dv_dynfric)
    
    return dv_dynfric

### Dynamical friction on MW from LMC ##
def chandrasekhar_acc_mw(t, w, mw_potential, Msat, vmax):
    """
    Eqn. 8.2, 8.7 in Binney & Tremaine (2008)
    """
    x0 = np.ascontiguousarray(w[:3].T)[1] 
    x = np.ascontiguousarray(w[:3].T)[:1]
    v0 = w[3:,1]
    v = w[3:,0] - v0

    dens = mw_potential._density((x - x0),t=np.array([0.]))
    v_norm = np.linalg.norm(v)
    v_disp = get_vdisp(np.linalg.norm((x-x0),axis=1),vmax)

    X = v_norm / (np.sqrt(2) * v_disp)
    fac = erf(X) - 2*X/np.sqrt(np.pi) * np.exp(-X**2)
    
    r = np.array([np.linalg.norm(x - x0)])

    if (r > 0) & (v_norm > 0):
        rs = 9.86
        L = 0.0; C = 0.1; alpha = 1.0
        ln_Lambda = max(L,np.log(r[0]/(C*rs))**alpha)
        dv_dynfric = - 4*np.pi * G_gal**2 * Msat * dens * ln_Lambda / v_norm**3 * fac * v
        # ln_Lambda = 0.3
        # dv_dynfric = - 0.428 * ln_Lambda * G_gal * Msat / r**2 * v / v_norm
    else:
        dv_dynfric = 0
    
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

# def tidal_acc(t, w, host_potential, Msat):
#     x0 = np.ascontiguousarray(w[:3].T)[0]*u.kpc
#     x = np.ascontiguousarray(w[:3].T)[1:]*u.kpc
#     v = w[3:,-1] - w[3:,-2]
#     deltavec = x - x0
#     delta = np.linalg.norm(deltavec)
#     if (delta == 0): return 0
#     mass1 = Msat*u.Msun
#     mass2 = host_potential.mass_enclosed(deltavec[0]*u.kpc)
#     tidal_r = delta*(mass1/(2*mass2))**(1/3.)

#     accel = -G.decompose(galactic)*mass2*soften(delta)*v/np.linalg.norm(v)
#     tidal_F = (accel*mass1*(tidal_r**2 + 2*tidal_r*delta)/(delta + tidal_r)**2).decompose(galactic)

#     return tidal_F.value

# def tidal_approx(t, w, host_potential, Msat):
#     x0 = np.ascontiguousarray(w[:3].T)[0]
#     x = np.ascontiguousarray(w[:3].T)[1:]
#     v = w[3:,-1] - w[3:,-2]
#     deltavec = (x - x0)[0]
#     delta = np.linalg.norm(deltavec)
#     if (delta == 0): return 0
#     mass1 = Msat
#     mass2 = host_potential.mass_enclosed(deltavec*u.kpc).value
    
#     deltaE = 4./3.*G_gal**2*mass1*(mass2/np.linalg.norm(v))**2/delta**4/mass1

#     return deltaE

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


def F(t, raw_w, nbody, chandra_kwargs):

    w = gd.PhaseSpacePosition.from_w(raw_w, units=nbody.units)
    nbody.w0 = w
    
    wdot = np.zeros((2 * w.ndim, w.shape[0]))
    
    # Compute the mutual N-body acceleration:
    wdot[3:] = nbody._nbody_acceleration()
    
    # wdot[3:, 0] += chandrasekhar_acc_mw(t, raw_w[:, 0:], mw_potential=chandra_kwargs['lmc_potential'], 
    #                             Msat=1e12,vmax=chandra_kwargs['vmax'][0])

    # Compute MW DF (only on non-MW particles)
    wdot[3:, 1:] += chandrasekhar_acc(t, raw_w[:, 0:], mw_potential=chandra_kwargs['mw_potential'], 
                                Msat=chandra_kwargs['Msats'],vmax=chandra_kwargs['vmax'][0],xi=chandra_kwargs['xi'],mw=True)
    # Compute LMC DF (only on SMC)
    if (chandra_kwargs['smc_potential'] is not None):
        wdot[3:, 2:] += chandrasekhar_acc(t, raw_w[:, 1:], mw_potential=chandra_kwargs['lmc_potential'],
                                    Msat=chandra_kwargs['Msats'][0][1:],vmax=chandra_kwargs['vmax'][1],
                                    xi=chandra_kwargs['xi'],mw=False)
    # wdot[3:, 2:] += tidal_acc(t, raw_w[:, 1:], chandra_kwargs['lmc_potential'], chandra_kwargs['Msmc'])

    wdot[:3] = w.v_xyz.decompose(nbody.units).value

    ### Ram Pressure ###
    # if ((chandra_kwargs['hydro_component'] is not None) and (chandra_kwargs['hydro_factor'] != 0)):
    #     wdot[3:,1:] += chandra_kwargs['hydro_factor']*rampressure_acc(t, raw_w[:,0:],chandra_kwargs['hydro_component'],chandra_kwargs['hydro_area'],chandra_kwargs['hydro_Msat'])
    if (chandra_kwargs['hydro_factor'] != 0):
        for cpt in chandra_kwargs['hydro_component'][0]:
            wdot[3:,1:] += chandra_kwargs['hydro_factor']*rampressure_acc(t, raw_w[:,0:],cpt,chandra_kwargs['hydro_area'],chandra_kwargs['hydro_Msat'])
        if (chandra_kwargs['smc_potential'] is not None):
            for cpt in chandra_kwargs['hydro_component'][1]:
                wdot[3:,2:] += chandra_kwargs['hydro_factor']*rampressure_acc(t, raw_w[:,1:],cpt,chandra_kwargs['hydro_area'][0][1],chandra_kwargs['hydro_Msat'][0][1])
    
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
        if (chandra_kwargs['smc_potential'] is not None):
            sat_r1 = get_tidal_r(t,raw_w[:,0:3],chandra_kwargs['mw_potential'],[chandra_kwargs['lmc_potential'],chandra_kwargs['smc_potential']])
            sat_r2 = get_tidal_r(t,raw_w[:,1:3],chandra_kwargs['lmc_potential'],[chandra_kwargs['smc_potential']])
            sat_r = [sat_r1[0],min(sat_r1[1],sat_r2[0])]
        else:
            sat_r = [get_tidal_r(t,raw_w,chandra_kwargs['mw_potential'],[chandra_kwargs['lmc_potential']])]
        
        for i in range(len(sat_r)):
            oldparams = nbody.particle_potentials[i+1]['halo'].parameters
            nbody.particle_potentials[i+1]['halo'] = TruncatedHernquistPotential(m=oldparams['m'],c=oldparams['c'],rmax=sat_r[i]*u.kpc,units=galactic)

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
    m = [k.startswith(name) for k in pot.keys()]
    potlist = []
    for k in np.array(list(pot.keys()))[m]:
        potlist.append(pot[k])
    return potlist

def run_orbit(lmc_pos,lmc_vel,smc_pos=None,smc_vel=None,t2=-5*u.Gyr,dt=-10*u.Myr,coulombLogXi=1,rampressure=0,tidal=0,massloss=0,pot_mw=None,pot_lmc=None,pot_smc=None,progress=True,hydro_areas=np.array([[np.pi*8**2,np.pi*4**2]])):
    
    ### Set potentials
    pot1,pot2,pot3 = get_default_potentials()
    if (pot_mw is None):
        pot_mw = pot1
    if (pot_lmc is None):
        pot_lmc = pot2
    if (pot_smc is None):
        pot_smc = pot3

    lmc_mhalo = 0.0; lmc_mdisk = 0.0; smc_mhalo = 0.0; smc_mdisk = 0.0
    if 'halo' in pot_lmc:
        lmc_mhalo = pot_lmc['halo'].parameters['m'].to('Msun').value if 'halo' in pot_lmc else 0.0
    if 'disk' in pot_lmc:
        lmc_mdisk = pot_lmc['disk'].parameters['m'].to('Msun').value if 'disk' in pot_lmc else 0.0
    if 'halo' in pot_smc:
        smc_mhalo = pot_smc['halo'].parameters['m'].to('Msun').value if 'halo' in pot_smc else 0.0
    if 'disk' in pot_smc:
        smc_mdisk = pot_smc['disk'].parameters['m'].to('Msun').value if 'disk' in pot_smc else 0.0

    vmax_mw = np.nanmax(pot_mw.circular_velocity(np.array(list(zip(np.zeros(101),np.zeros(101),np.linspace(0,200,101)))).T))
    vmax_lmc = np.nanmax(pot_lmc.circular_velocity(np.array(list(zip(np.zeros(101),np.zeros(101),np.linspace(0,200,101)))).T))

    chandra_kwargs = {
        'Msats': np.array([[lmc_mhalo,smc_mhalo]]) if smc_pos is not None else np.array([[lmc_mhalo]]),
        'mw_potential': pot_mw, 
        'lmc_potential': pot_lmc,
        'smc_potential': pot_smc if smc_pos is not None else None,
        'xi': coulombLogXi,
        'vmax': [vmax_mw,vmax_lmc],
        'tidal': tidal,
        'hydro_factor': rampressure,
        'hydro_component': [get_hydro_cpts(pot_mw),get_hydro_cpts(pot_lmc)], # pot_mw['cgm'] if 'cgm' in pot_mw else None,
        'hydro_Msat': np.array([[lmc_mdisk,smc_mdisk]]),
        'hydro_area': hydro_areas, # disk surface areas
        'massloss': massloss
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

    ### Run integrator
    integrator = gi.DOPRI853Integrator(
            F, func_args=(nbody, chandra_kwargs), 
            func_units=nbody.units, 
            progress=progress
        )
    orbits = integrator.run(w0, dt=dt, t1=0, t2=t2)
    
    return orbits

def run_orbit_old(lmc_pos,lmc_vel,smc_pos,smc_vel,t2,dt=-10*u.Myr,coulombLogXi=1,rampressure=0,tidal=0,pot_mw=None,pot_lmc=None,pot_smc=None,progress=True):
    global pastperi
    pastperi = False

    backwards = False
    if (t2.value < 0):
        backwards = True
    
    ### Set potentials
    pot1,pot2,pot3 = get_default_potentials()
    if (pot_mw is None):
        pot_mw = pot1
    if (pot_lmc is None):
        pot_lmc = pot2
    if (pot_smc is None):
        pot_smc = pot3

    lmc_mhalo = pot_lmc['halo'].parameters['m'].to('Msun').value if 'halo' in pot_lmc else 0.0
    lmc_mdisk = pot_lmc['disk'].parameters['m'].to('Msun').value if 'disk' in pot_lmc else 0.0
    smc_mhalo = pot_smc['halo'].parameters['m'].to('Msun').value if 'halo' in pot_smc else 0.0
    smc_mdisk = pot_smc['disk'].parameters['m'].to('Msun').value if 'disk' in pot_smc else 0.0

    ### Set dynamical friction parameters
    # chandra_kwargs = {
    #         'Msat': np.array([[lmc_mdisk,smc_mdisk]]), # Msun - set this to be the same as used in pot_lmc above
    #         'mw_potential': pot_mw, 
    #         'ln_Lambda': coulombLog, 
    #         'v_disp': (150 * u.km/u.s).decompose(galactic).value
    #     }
    chandra_kwargs = {
        'Msats': np.array([[lmc_mhalo,smc_mhalo]]),
        'mw_potential': pot_mw, 
        'lmc_potential': pot_lmc,
        'xi': coulombLogXi,
        'tidal': tidal,
        'v_disp': (150 * u.km/u.s).decompose(galactic).value,
        'hydro_factor': rampressure,
        'hydro_component': pot_mw['cgm'] if 'cgm' in pot_mw else None,
        'hydro_Msat': np.array([[lmc_mdisk,smc_mdisk]]),
        'hydro_area': np.array([[np.pi*8**2,np.pi*4**2]]) # disk surface areas
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
    w0_smc = gd.PhaseSpacePosition(
            pos=smc_pos*u.kpc,
            vel=smc_vel*u.km/u.s
        )
    w0 = gd.combine((w0_mw, w0_lmc, w0_smc))
    
    ### Build nbody
    nbody = gd.DirectNBody(w0, [pot_mw, pot_lmc, pot_smc])

    ### Run integrator
    integrator = gi.DOPRI853Integrator(
            F, func_args=(nbody, chandra_kwargs), 
            func_units=nbody.units, 
            progress=progress
        )
    orbits = integrator.run(w0, dt=dt, t1=0, t2=t2)
    
    return orbits

def run_orbit_lmconly(lmc_pos,lmc_vel,t2,dt=-10*u.Myr,coulombLogXi=1,rampressure=0,pot_mw=None,pot_lmc=None,progress=True):

    ### Set potentials
    if (pot_mw is None):
        pot_mw = gp.MilkyWayPotential()
    else:
        pot_mw = pot_mw
    
    if (pot_lmc is None):
        pot_lmc = gp.CCompositePotential()
        pot_lmc['disk'] = gp.MiyamotoNagaiPotential(m=1e10*u.Msun, a=1.4*u.kpc, b=800*u.pc, units=galactic)
        pot_lmc['halo'] = gp.HernquistPotential(m=1.8e11*u.Msun, c=9*u.kpc, units=galactic)
    else:
        pot_lmc = pot_lmc


    lmc_mhalo = pot_lmc['halo'].parameters['m'].to('Msun').value if 'halo' in pot_lmc else 0.0
    lmc_mdisk = pot_lmc['disk'].parameters['m'].to('Msun').value if 'disk' in pot_lmc else 0.0

    ### Set dynamical friction parameters
    # chandra_kwargs = {
    #         'Msat': lmc_mdisk, # Msun - set this to be the same as used in pot_lmc above
    #         'mw_potential': pot_mw, 
    #         'ln_Lambda': coulombLog, 
    #         'v_disp': (150 * u.km/u.s).decompose(galactic).value
    #     }
    chandra_kwargs = {
        'Msats': lmc_mhalo,
        'mw_potential': pot_mw, 
        'lmc_potential': None,
        'xi': coulombLogXi,
        'tidal': 0,
        'v_disp': (150 * u.km/u.s).decompose(galactic).value,
        'hydro_factor': rampressure,
        'hydro_component': pot_mw['cgm'] if 'cgm' in pot_mw else None,
        'hydro_Msat': lmc_mdisk,
        'hydro_area': np.pi*8**2,
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
    w0 = gd.combine((w0_mw, w0_lmc))
    
    ### Build nbody
    nbody = gd.DirectNBody(w0, [pot_mw, pot_lmc])

    ### Run integrator
    integrator = gi.DOPRI853Integrator(
            F, func_args=(nbody, chandra_kwargs), 
            func_units=nbody.units, 
            progress=progress
        )
    orbits = integrator.run(w0, dt=dt, t1=0, t2=t2)
    
    return orbits

def orbit_sense(orbits,mcsepv=None):
    ### Assuming orbit order is MW, LMC, SMC, as above

    if (orbits is not None):
        mcsepv = orbits.pos.T[2] - orbits.pos.T[1]
    elif (mcsepv is None):
        raise Exception("Need to supply either orbits or Cloud separation array.")
    else:
        orbits = gd.orbit.Orbit(mcsepv.T,np.zeros_like(mcsepv.T))
        mcsepv = orbits.pos
    mcsep = np.sqrt(mcsepv.x**2 + mcsepv.y**2 + mcsepv.z**2)
    mins = argrelextrema(mcsep,np.less)[0]

    if (len(mins) >= 2):
        m = mins[1]
        # small m is closer to present day
        phi1 = np.arctan2(mcsepv[m-1].z,mcsepv[m-1].y) # later time
        phi0 = np.arctan2(mcsepv[m].z,mcsepv[m].y) # earlier time
        dphi = (phi1 - phi0).value
        if (np.abs(dphi) > np.pi):
            dphi *= -1
        if (dphi > 0):
            return 1 # CCW
        else:
            return -1 # CW
    
    return 0 # no second interaction