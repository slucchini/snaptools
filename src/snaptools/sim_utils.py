import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, os, gc, pickle, pygad as pg
from tqdm.notebook import tqdm
from snaptools import simulation
import gadget.units as u, astropy.units as au, astropy.constants as constants
from scipy.spatial.transform import Rotation as R
import arepo

def make_list(obj):
    if ((type(obj) == list) | (type(obj) == np.ndarray)):
        return obj
    else:
        return [obj]

def get_snap_list(sim,output='output',partition='lfs'):
    if (partition == 'lfs'):
        basedir = '/n/holylfs05/LABS/hernquist_lab/Users/slucchini/Arepo/Cosmo/zooms/'
    elif (partition == 'netscratch'):
        basedir = '/n/netscratch/hernquist_lab/Lab/slucchini/Arepo/Cosmo/zooms/'
    else:
        raise Exception("Partition {} not recognized".format(partition))
    folder_list = []
    sim = make_list(sim); output = make_list(output)
    if (len(sim) == len(output)):
        for i in range(len(sim)):
            folder_list.append(basedir+"{}/{}/".format(sim[i],output[i]))
    elif (len(sim) == 1):
        for i in range(len(output)):
            folder_list.append(basedir+"{}/{}/".format(sim[0],output[i]))
    else:
        raise Exception("Multiple sims with different number of outputs not supported.")

    snap_list = []
    for folder in folder_list:
        filelist = os.listdir(folder)
        filelist = list(np.array(filelist)[[f.startswith('snap_') & f.endswith('.hdf5') for f in filelist]])
        filelist.sort(key=lambda x:int(int(x.split('.')[0].split('_')[1])))
        filelist = [folder+f for f in filelist]
        snap_list.extend(filelist)
    return snap_list

def getL(pos,vel,center=None,rlim=40):
    if (center is None):
        center = np.median(pos,axis=0)
    pos = pos - center
    radii = np.linalg.norm(pos,axis=1)
    
    meanv = np.mean(vel[radii < rlim],axis=0)
    vel = vel - meanv
    
    pos = pos[radii < rlim]
    vel = vel[radii < rlim]

    angmom = np.cross(pos,vel)
    meanL = np.mean(angmom,axis=0)
    meanL /= np.linalg.norm(meanL)
    
    return meanL

def rotate(pos,vel,vector,axis):
    vec = vector/np.linalg.norm(vector); ax = axis/np.linalg.norm(axis)
    rotangle = np.arccos(np.dot(vec,ax))
    rotvec = np.cross(vec,ax)
    rotvec = rotvec/np.linalg.norm(rotvec)
    r = R.from_rotvec(rotangle*rotvec)

    if isinstance(pos,dict):
        for k in pos.keys():
            pos[k] = r.apply(pos[k])
            if (vel is not None):
                vel[k] = r.apply(vel[k])
    else:
        pos = r.apply(pos)
        if (vel is not None):
            vel = r.apply(vel)
    
    if (vel is not None):
        return pos,vel
    else:
        return pos

def get_temp(s):
    xe = s['ne']
    U = np.array(s['InternalEnergy'])*au.km**2/au.s**2

    Xh = 0.76
    gamma = 5./3.

    mu = (1 + Xh /(1-Xh)) / (1 + Xh/(4*(1-Xh)) + xe)*constants.m_p
    # mu = 4./(1 + 3*Xh + 4*Xh*xe)*constants.m_p.cgs.value
    temp = (gamma - 1)*U/constants.k_B*mu

    return temp.to('K').value

def prep_snap(s,center='bh',Lval=None,rlim=40,physical=True):
    if (type(s) == arepo.loader.Snapshot):
        if (physical):
            s.to_physical()
        
        ptypearr = np.zeros(len(s.NumPart_Total))
        ptypearr[0] = 1
        s.addField("vol",ptypearr)
        # s.part0.vol[:] = u.Quantity((s.part0.mass[:]/s.part0.rho[:]).as_unit(u.kpc**3),unit=u.kpc**3)
        s.part0.vol[:] = s.part0.mass[:]/s.part0.rho[:]
        s.addField("temp",ptypearr)
        s.part0.data['temp'][:] = get_temp(s.part0)

        if (center == 'bh'):
            cent = np.copy(s.part5.pos[0])
        elif (center == 'stars'):
            cent = np.median(s.part4.pos,axis=0)
        else:
            raise Exception("Center method not recognized: {}. Choose one of 'bh','stars'.".format(center))
        if (Lval is None):
            Lval = getL(s.part4.pos,s.part4.vel,cent)
        radii = np.linalg.norm(s.part4.pos-cent,axis=1)
        meanv = np.mean(s.part4.vel[radii < rlim],axis=0)
        for group in s.groups:
            if 'pos' in group.data:
                p,v = rotate(group.pos-cent,group.vel-meanv,Lval,[0,0,1])
                group.pos[:] = p
                group.vel[:] = v
        
        return s
    elif (type(s) == pg.snapshot.Snapshot):
        s.gas['hsml'] = pg.UnitArr(np.cbrt(np.array(s.gas['dV'].in_units_of('kpc**3',subs=s))),'kpc')
        if (physical):
            s.to_physical_units()

        if (center == 'bh'):
            cent = s.part5['pos'][0]
        elif (center == 'stars'):
            cent = np.median(s.part4['pos'],axis=0)
        else:
            raise Exception("Center method not recognized: {}. Choose one of 'bh','stars'.".format(center))
        if (Lval is None):
            Lval = getL(s.part4['pos'],s.part4['vel'],cent)
        
        radii = np.linalg.norm(s.part4['pos']-cent,axis=1)
        meanv = np.mean(s.part4['vel'][radii < rlim],axis=0)
        # p,v = rotate(s['pos']-cent,s['vel']-meanv,Lval,[0,0,1])
        ntypes = len(s._N_part)
        for i in range(ntypes):
            sub = getattr(s,'part{}'.format(i))
            if ('pos' in sub):
                p,v = rotate(sub['pos']-cent,sub['vel']-meanv,Lval,[0,0,1])
                posunits = sub['pos'].units
                velunits = sub['vel'].units
                sub['pos'] = pg.UnitArr(p,posunits)
                sub['vel'] = pg.UnitArr(v,velunits)
        return s
    elif (type(s) == scida.customs.arepo.dataset.ArepoSnapshotWithUnitMixinAndCosmologyMixin):
        raise NotImplementedError("Currently don't support scida snapshots.")
        # import dask.array as da
        # ds = s
        # if (center == 'bh'):
        #     cent = ds.data.['PartType5']['Coordinates'].compute()[0]
        # elif (center == 'stars'):
        #     cent = da.median(ds.data['PartType4']['Coordinates'],axis=0).compute()
        # else:
        #     raise Exception("Center method not recognized: {}. Choose one of 'bh','stars'.".format(center))
        # if (Lval is None):
        #     p4p = ds.data['PartType4']['Coordinates'].compute().magnitude; p4v = ds.data['PartType4']['Velocities'].compute().magnitude
        #     Lval = getL(p4p,p4v,cent)
        
        # radii = np.linalg.norm(p4p-cent,axis=1)
        # meanv = np.mean(p4v[radii < rlim],axis=0)
        # ntypes = len(ds.header['NumPart_Total'])
        # for i in range(ntypes):
        #     ptype = 'PartType{}'.format(i)
        #     if ('Coordinates' in ds.data[ptype]):
        #         p,v = rotate(ds.data[ptype]['Coordinates'].compute().magnitude,)
    else:
        raise Exception("Type of snapshot not recognized: {}".format(type(s)))