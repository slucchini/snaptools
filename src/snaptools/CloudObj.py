import numpy as np

class CloudObj():
    def __init__(self,members=[],snapnum=0):
        list(np.unique(members)).sort()
        self.members = members
        self.snapnum = snapnum
    
    def get_snapmask(self,sids,argsort):
        mask = np.zeros(len(sids),dtype=bool)
        idcheck = [m in sids for m in self.members]
        idmask = np.searchsorted(sids,self.members[idcheck])
        pos_to_change = np.arange(0,len(mask))[argsort][idmask]
        mask[pos_to_change] = True
        return pos_to_change
    
    def __repr__(self):
        if ((len(self.members)>0) and hasattr(self,'snapnum')):
            return "CloudObj ({}) {} <len {}>".format(self.snapnum,self.members[0],len(self.members))
        elif (len(self.members)>0):
            return "CloudObj {} <len {}>".format(self.members[0],len(self.members))
        else:
            return "CloudObj -- <len 0>"
    
    def __len__(self):
        return len(self.members)
    
    def __eq__(self,other):
        return self.__hash__() == other.__hash__()
    
    def __hash__(self):
        self.members.sort()
        try:
            return hash(frozenset([self.snapnum]+self.members))
        except:
            return hash(frozenset(self.members))