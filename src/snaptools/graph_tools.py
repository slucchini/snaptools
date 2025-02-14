import networkx as nx, numpy as np
from itertools import permutations
from .CloudObj import *

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

# def prune_stragglers(subgraph,safeclouds=[]):
#     nodes,snaplist,snapedges = graph_data(subgraph)
#     removelist = []
#     for i,n in enumerate(nodes):
#         edgemask = np.array([n in e for e in subgraph.edges])
#         nedges = np.sum(edgemask)
#         ## if it's a dead-end branch
#         if (snaplist[i] not in snapedges):
#             if (nedges == 1):
#                 removelist.append(n)
#             else:
#                 cnodes = np.array(subgraph.edges)[edgemask].ravel()
#                 cnodes = cnodes[cnodes != n]
#                 csnums = np.array([c.snapnum for c in cnodes])
#                 if (len(np.unique(csnums)) == 1):
#                     removelist.append(n)
#     for sc in safeclouds:
#         while (sc in removelist):
#             removelist.remove(sc)
#     subgraph.remove_nodes_from(removelist)
#     subgraph = get_largest_connected(subgraph)
#     return subgraph
def prune_stragglers(subgraph,safeclouds=[]):
    leaf_nodes = [node for node, degree in subgraph.degree() if degree == 1]
    nodes,snaplist,snapedges = graph_data(subgraph)
    removelist = list(np.array(leaf_nodes)[[n.snapnum not in snapedges for n in leaf_nodes]])
    for sc in safeclouds:
        while (sc in removelist):
            removelist.remove(sc)
    subgraph.remove_nodes_from(removelist)
    subgraph = get_largest_connected(subgraph)
    return subgraph

# def prune_fork(subgraph,safeclouds=[]):
#     nodes,snaplist,snapedges = graph_data(subgraph)
#     removelist = []
#     for i,n in enumerate(nodes):
#         edgemask = np.array([n in e for e in subgraph.edges])
#         nedges = np.sum(edgemask)
#         if ((nedges > 2) | ((nedges == 2) & (snaplist[i] in snapedges))):
#             cnodes = np.array(subgraph.edges)[edgemask].ravel()
#             cnodes = cnodes[cnodes != n]
#             csnums = np.array([c.snapnum for c in cnodes])
#             for si in np.unique(csnums):
#                 cmask = csnums == si
#                 if (np.sum(cmask) > 1):
#                     sizes = np.array([len(c) for c in cnodes[cmask]])
#                     szmask = sizes != max(sizes)
#                     if (np.sum(szmask) == 0):
#                         szmask = np.ones(len(sizes),dtype=bool)
#                         szmask[0] = False
#                     removelist.extend(cnodes[cmask][szmask])
#             break
#     for sc in safeclouds:
#         while (sc in removelist):
#             removelist.remove(sc)
#     subgraph.remove_nodes_from(removelist)
#     subgraph = get_largest_connected(subgraph)
#     return subgraph
def prune_fork(subgraph,safeclouds=[]):
    nodes,snaplist,snapedges = graph_data(subgraph)
    removelist = []
    nodes1 = [node for node, degree in subgraph.degree() if degree > 2]
    nodes2 = [node for node, degree in subgraph.degree() if degree == 2]
    nodes2 = list(np.array(nodes2)[[n.snapnum in snapedges for n in nodes2]])
    for i,n in enumerate(nodes1+nodes2):
        cnodes = np.array(list(nx.neighbors(subgraph,n)))
        csnums = np.array([c.snapnum for c in cnodes])
        for si in np.unique(csnums):
            cmask = csnums == si
            if (np.sum(cmask) > 1):
                sizes = np.array([len(c) for c in cnodes[cmask]])
                szmask = sizes != max(sizes)
                if (np.sum(szmask) == 0):
                    szmask = np.ones(len(sizes),dtype=bool)
                    szmask[0] = False
                removelist.extend(cnodes[cmask][szmask])
        break
    for sc in safeclouds:
        while (sc in removelist):
            removelist.remove(sc)
    subgraph.remove_nodes_from(removelist)
    subgraph = get_largest_connected(subgraph)
    return subgraph

def find_safeclouds(subgraph):
    subgraph_snums = np.array([n.snapnum for n in list(subgraph.nodes)])
    csnum = max(subgraph_snums)
    alledges = np.array(list(subgraph.edges))
    final_clouds = np.array(list(subgraph.nodes))[subgraph_snums == max(subgraph_snums)]
    safeclouds = []
    for cc in final_clouds:
        safeclouds.append([cc])
    doneclouds = np.zeros(len(final_clouds),dtype=bool)
    while csnum > 2900:
        print(csnum)
        for j in range(len(final_clouds)):
            cc = safeclouds[j][-1]
            for i,cl in enumerate(list(subgraph.nodes)):
                if (cl == cc):
                    print(i)
            edgemask = np.array([cc in e for e in alledges])
            edgeclouds = np.ravel(alledges[edgemask])
            edgeclouds = edgeclouds[edgeclouds != cc]
            prevclouds = edgeclouds[[ec.snapnum == csnum-1 for ec in edgeclouds]]
            if (len(prevclouds) > 0):
                prevcloud_len = np.array([len(pc) for pc in prevclouds])
                prevcloud = prevclouds[np.argmax(prevcloud_len)]
                safeclouds[j].append(prevcloud)
                for i,cl in enumerate(list(subgraph.nodes)):
                    if (cl == prevcloud):
                        print(i)
            print("")
        checked_clouds = []
        for j in range(len(final_clouds)):
            if not doneclouds[j]:
                if (safeclouds[j][-1] in checked_clouds):
                    safeclouds[j][-1] = None
                    doneclouds[j] = True
                else:
                    checked_clouds.append(safeclouds[j][-1])
        if (np.sum(doneclouds) == len(doneclouds)-1):
            print("All done")
            break
        csnum -= 1
    # safeclouds = np.ravel(safeclouds)
    scflat = []
    for sc in safeclouds:
        scflat.extend(sc)
    safeclouds = np.array(scflat)
    safeclouds = safeclouds[[sc is not None for sc in safeclouds]]
    return safeclouds

def prune_graph(subgraph,keep_final_clouds=False):
    breakready = 0
    while True:
        oldgraph = subgraph.copy()
        subgraph = prune_stragglers(subgraph)
        if (len(subgraph) == len(oldgraph)):
            break
    if keep_final_clouds:
        safeclouds = find_safeclouds(subgraph)
    else:
        safeclouds = []
    while True:
        oldgraph = subgraph.copy()
        subgraph = prune_fork(subgraph,safeclouds)
        if (len(oldgraph) == len(subgraph)):
            breakready += 1
        else:
            breakready = 0
        while True:
            oldgraph = subgraph.copy()
            subgraph = prune_stragglers(subgraph,safeclouds)
            if (len(subgraph) == len(oldgraph)):
                break
        if (breakready >= 2):
            break
    return subgraph