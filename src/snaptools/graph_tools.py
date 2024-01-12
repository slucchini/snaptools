import networkx as nx, numpy as np
from itertools import permutations
from CloudObj import *

def get_graph_pos(G,xpad=1):
    snapnums = np.array([G.nodes[c]['snapnum'] for c in list(G.nodes)])
    xvals = snapnums - min(snapnums)
    xpadarr = np.ones_like(xvals)*xpad
    xvals *= xpadarr
    yvals = np.zeros_like(xvals)
    for s,sn in enumerate(np.unique(snapnums)):
        mask = np.nonzero(snapnums == sn)[0]
        n = len(snapnums[mask])
        if (n > 1):
            if (s==0):
                yvals[mask] = np.arange(1-n,n,2)
                prev_mask = mask
                continue
            yvs = np.arange(1-n,n,2)
            opts = list(permutations(yvs))
            loss = np.zeros(len(opts))
            for k in range(len(opts)):
                for i in range(n):
                    for j in range(len(snapnums[prev_mask])):
                        edge_test = (list(G.nodes)[prev_mask[j]],list(G.nodes)[mask[i]])
                        if ((edge_test in list(G.edges)) | (edge_test[::-1] in list(G.edges))):
                            loss[k] += (xvals[prev_mask[j]] - xvals[mask[i]])**2 + (yvals[prev_mask[j]] - opts[k][i])**2
            loss[loss==0] = np.nan
            k = np.nanargmin(loss)
            for i in range(n):
                yvals[mask[i]] = opts[k][i]
        prev_mask = mask
    pos = {}
    for i,c in enumerate(list(G.nodes)):
        pos[c] = np.array([xvals[i],yvals[i]])
    return pos

def graph_data(subgraph):
    nodes = list(subgraph.nodes)
    snaplist = np.array([subgraph.nodes[n]['snapnum'] for n in nodes])
    snapedges = [min(snaplist),max(snaplist)]
    return nodes,snaplist,snapedges

def get_largest_connected(subgraph):
    cc = list(nx.connected_components(subgraph))
    cclen = np.array([len(c) for c in cc])
    return subgraph.subgraph(cc[np.argmax(cclen)]).copy()

def prune_stragglers(subgraph):
    nodes,snaplist,snapedges = graph_data(subgraph)
    removelist = []
    for i,n in enumerate(nodes):
        edgemask = np.array([n in e for e in subgraph.edges])
        nedges = np.sum(edgemask)
        ## if it's a dead-end branch
        if (snaplist[i] not in snapedges):
            if (nedges == 1):
                removelist.append(n)
            else:
                cnodes = np.array(subgraph.edges)[edgemask].ravel()
                cnodes = cnodes[cnodes != n]
                csnums = np.array([c.snapnum for c in cnodes])
                if (len(np.unique(csnums)) == 1):
                    removelist.append(n)
    subgraph.remove_nodes_from(removelist)
    subgraph = get_largest_connected(subgraph)
    return subgraph

def prune_fork(subgraph):
    nodes,snaplist,snapedges = graph_data(subgraph)
    removelist = []
    for i,n in enumerate(nodes):
        edgemask = np.array([n in e for e in subgraph.edges])
        nedges = np.sum(edgemask)
        if (nedges > 2):
            cnodes = np.array(subgraph.edges)[edgemask].ravel()
            cnodes = cnodes[cnodes != n]
            csnums = np.array([c.snapnum for c in cnodes])
            for si in np.unique(csnums):
                cmask = csnums == si
                if (np.sum(cmask) > 1):
                    sizes = np.array([len(c) for c in cnodes[cmask]])
                    szmask = sizes != max(sizes)
                    removelist.extend(cnodes[cmask][szmask])
            break
    subgraph.remove_nodes_from(removelist)
    subgraph = get_largest_connected(subgraph)
    return subgraph

def prune_graph(subgraph):
    allgraphs = [subgraph.copy()]
    while True:
        oldgraph = subgraph.copy()
        subgraph = prune_stragglers(subgraph)
        if (len(subgraph) == len(oldgraph)):
            break
    allgraphs.append(subgraph.copy())
    while True:
        oldgraph = subgraph.copy()
        subgraph = prune_fork(subgraph)
        allgraphs.append(subgraph.copy())
        while True:
            oldgraph = subgraph.copy()
            subgraph = prune_stragglers(subgraph)
            if (len(subgraph) == len(oldgraph)):
                break
        allgraphs.append(subgraph.copy())
        if (len(subgraph) == len(oldgraph)):
            break
    return subgraph