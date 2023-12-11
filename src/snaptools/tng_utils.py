import numpy as np, astropy.units as u, matplotlib.pyplot as plt, h5py
from tqdm.notebook import tqdm
from scipy.spatial.transform import Rotation as R

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