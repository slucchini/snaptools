import numpy as np, matplotlib.pyplot as plt, h5py, arepo, os, pickle
from tqdm.notebook import tqdm
from scipy.spatial.transform import Rotation as R
import astropy.units as u, astropy.coordinates as coords

def recenter(f):
    com = np.mean(f['PartType4']['Coordinates'][:],axis=0)
    cov = np.mean(f['PartType4']['Velocities'][:],axis=0)
    pos = {}; vel = {}
    for pt in f.keys():
        if (pt.startswith("PartType")):
            pos[pt] = f[pt]['Coordinates'][:] - com
            vel[pt] = f[pt]['Velocities'][:] - cov
    
    angmom = np.cross(pos['PartType4'],vel['PartType4'])
    meanL = np.mean(angmom,axis=0)
    meanL /= np.linalg.norm(meanL)

    rotangle = np.arccos(np.dot(meanL,[0,0,1]))
    rotvec = np.cross(meanL,[0,0,1])
    rotvec = rotvec/np.linalg.norm(rotvec)
    r = R.from_rotvec(rotangle*rotvec)
    pos2 = {}; vel2 = {}
    for pt in pos.keys():
        pos2[pt] = r.apply(pos[pt])
        vel2[pt] = r.apply(vel[pt])

    return pos2,vel2

def binsim(xvals,masses,dx,xmax,xmin=5,sigmin=None,addlmask=None,getarea=None,verbose=True):
    xbins = []; sig = []
    if verbose:
        loopv = tqdm(range(int(xmax/dx)))
    else:
        loopv = range(int(xmax/dx))
    for i in loopv:
        xi = i*dx; xi1 = (i+1)*dx
        xbins.append((i+0.5)*dx)
        if (addlmask is not None):
            mask = (xvals > xi) & (xvals < xi1) & addlmask
        else:
            mask = (xvals > xi) & (xvals < xi1)
        mtot = np.sum(masses[mask])
        if (getarea is not None):
            dV = getarea(xi,xi1)
            sig.append(mtot/dV)
        else:
            sig.append(mtot)
        if (sigmin is not None):
            if ((xi > xmin) & (sig[-1]<sigmin)): break
    return np.array(xbins), np.array(sig)

def expfunc(r,s0,h):
    return s0*np.exp(-r/h)

def fitsig(xbins,sig,xmin=5,minfitx=5,verbose=False,weights=None):
    xfitmax = xmin+minfitx
    bestfit = [(0,0),1e10,0]
    xmini = np.digitize(xmin,xbins)
    allres = []
    if (weights is None):
        weights = np.ones_like(xbins)
    for xi in np.array(range(int(xbins[-1]-xfitmax)))+xfitmax:
        xfitmaxi = np.digitize(xi,xbins)
        finitemask = np.isfinite(np.log10(sig[xmini:xfitmaxi]))
        m,b = np.polyfit(xbins[xmini:xfitmaxi][finitemask],np.log10(sig[xmini:xfitmaxi][finitemask]),1)
        finitemask = np.isfinite(np.log10(sig))
        res = np.sum(np.abs(np.log10(sig)[finitemask]-(xbins[finitemask]*m+b)*weights[finitemask]))
        allres.append(res)
        if (res < bestfit[1]):
            bestfit = [(m,b),res,xi]
    m,b = bestfit[0]
    h = -np.log10(np.e)/m
    if (verbose):
        print("Best fit at x={:.1f}".format(bestfit[-1]))
        print("h = {:.2f} kpc".format(h))
    return 10**b,h

def get_scale_height(files,outfolder,save=False,verbose=True):
    if isinstance(files,str):
        files = [files]
    hzs = []
    with np.errstate(divide='ignore'):
        for fi in range(len(files)):
            fname = files[fi]
            if (not fname.endswith('_s99.hdf5')): continue
            if (verbose):
                print(fname+'...')
            with h5py.File(outfolder+'/'+fname,'r+') as f:
                pos,vel = recenter(f)
                gaspos = pos['PartType0']
                gasmass = f['PartType0']['Masses'][:]*1e10
                Rradii = np.linalg.norm(gaspos[:,:2],axis=1)
                
                zbins,zsig = binsim(gaspos[:,2],gasmass,0.1,5,sigmin=None,addlmask=Rradii<20,getarea=None,verbose=verbose)
                s0z,hz = fitsig(zbins,zsig,xmin=0,minfitx=1,verbose=verbose)
                hzs.append(hz)
                if (save):
                    if ('Scale_Height' not in list(f['Header'].attrs)):
                        f['Header'].attrs.create('Scale_Height',hz)
                    else:
                        f['Header'].attrs.modify('Scale_Height',hz)

            if (verbose):
                fig,ax = plt.subplots(1,1,figsize=(5,4))
                ax.plot(zbins,np.log10(zsig))
                ax.plot(zbins,np.log10(expfunc(zbins,s0z,hz)),'k--')
                ax.set_xlabel("z (kpc)")
                plt.show()
                plt.close(fig)
        return hzs

def get_scale_length(files,outfolder,save=False,verbose=True):
    if isinstance(files,str):
        files = [files]
    hs = []
    ptype = 'PartType4'

    with np.errstate(divide='ignore'):
        for fi in range(len(files)):
            fname = files[fi]
            if (not fname.endswith('_s99.hdf5')): continue
            if verbose:
                print(fname+'...')
            with h5py.File(outfolder+'/'+fname,'r+') as f:
                pos,vel = recenter(f)
                ppos = pos[ptype]
                pmass = f[ptype]['Masses'][:]*1e10
                Rradii = np.linalg.norm(ppos[:,:2],axis=1)
                
                getarea = lambda xi,xi1: 4*np.pi*(xi1**2-xi**2)*(u.kpc.to(u.pc))**2
                Rbins,sig = binsim(Rradii,pmass,0.1,30,xmin=5,sigmin=None,addlmask=np.abs(ppos[:,2])<3,getarea=getarea,verbose=verbose)
                s0,h = fitsig(Rbins,sig,verbose=verbose)
                if (save):
                    if ('Scale_Length' not in list(f['Header'].attrs)):
                        f['Header'].attrs.create('Scale_Length',h)
                    else:
                        f['Header'].attrs.modify('Scale_Length',h)

            if verbose:
                fig,ax = plt.subplots(1,1,figsize=(5,4))
                ax.plot(Rbins,np.log10(sig))
                ax.plot(Rbins,np.log10(expfunc(Rbins,s0,h)),'k--')
                ax.set_xlabel("R (kpc)")
                plt.show()
                plt.close(fig)

def get_deviation_vel(p,v,filename,rsun=8,verbose=False):
    vavgfile = '.'.join(filename.split('.')[:-1])+"_vavg.pkl"

    G = 4.3*10**(4) # (km/s)^2 kpc (10^10 solar masses)^-1
    rmax = 300 #kpc
    deltar = 0.1 #kpc
    intmax = int(rmax/deltar)

    if os.path.exists(vavgfile):
        with open(vavgfile,'rb') as f:
            vavg = pickle.load(f)
    else:
        sa = arepo.Snapshot(filename)
        sa.pos[:] = sa.pos - np.median(sa.part4.pos,axis=0)
        massencl = {} # in 10^10 solar masses
        vavg = {} # in km/s
        loop = enumerate(tqdm(sa.groups)) if verbose else enumerate(sa.groups)
        for gi,g in loop:
            if ('pos' in g.data):
                radii = np.linalg.norm(g.pos,axis=1)
                if ('mass' in g.data):
                    massencl[gi] = np.array([np.sum(g.mass[radii < ((i+1)*deltar)]) for i in range(intmax)])
                else:
                    massencl[gi] = np.array([len(g.pos[radii < ((i+1)*deltar)])*sa.masses[gi] for i in range(intmax)])
                vavg[gi] = np.array([np.sqrt(G*m/((i+1)*deltar)) for i,m in enumerate(massencl[gi])])

        vavg["total"] = np.array([np.sqrt(np.sum([vavg[gi][i]**2 for gi in massencl.keys()])) for i in range(intmax)])
        with open(vavgfile,'wb') as f:
            pickle.dump(vavg,f)
    xvals = np.array(range(intmax))*deltar

    sc = coords.SkyCoord(coords.Galactocentric(x=p[:,0]*u.kpc,y=p[:,1]*u.kpc,z=p[:,2]*u.kpc,
                                           v_x=v[:,0]*u.km/u.s,v_y=v[:,1]*u.km/u.s,v_z=v[:,2]*u.km/u.s,
                                           galcen_distance=8*u.kpc,representation_type='cartesian'))
    galc = sc.transform_to(coords.Galactic)
    lsr = sc.transform_to(coords.LSR)
    vlsr = lsr.radial_velocity.value

    rxy = np.linalg.norm(p[:,:2],axis=1)
    ixy = np.digitize(rxy,xvals)
    ixy[ixy >= len(xvals)] = len(xvals)-1
    vsun = -1*vavg['total'][np.digitize(rsun,xvals)]
    vrad = -1*(vavg['total'][ixy]*rsun/rxy - vsun)*np.sin(galc.l.to(u.rad).value)*np.cos(galc.b.to(u.rad).value)
    return np.abs(vlsr - vrad)