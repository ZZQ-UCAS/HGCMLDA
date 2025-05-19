import numpy as np
import torch
from sklearn.mixture import GaussianMixture

def hyperedge_concat(*H_list):

    H = None
    for h in H_list:
        if h is not None and len(h) != 0:
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def _construct_edge_list_from_cluster(X, clusters):

    N = X.shape[0]
    gmm = GaussianMixture(n_components=clusters, random_state=0).fit(X)
    assignment = gmm.predict(X)
    H = np.zeros([N, clusters])
   
    for i in range(N):
        H[i, assignment[i]] = 1

    return H

def construct_H_with_GMM(X, clusters, split_diff_scale=False):

    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(clusters) == int:
        clusters = [clusters]

    H = []
    for clusters in clusters:
        H_tmp = _construct_edge_list_from_cluster(X, clusters)

        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H



def _generate_G_from_H(H, variable_weight=False):
  
    H = np.array(H) 

    n_edge = H.shape[1]
    W = np.ones(n_edge)

    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)
 
    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))

  
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        G = torch.Tensor(G)
        return G


def generate_G_from_H(H, variable_weight=False):

    if type(H) != list:   
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:    
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G

