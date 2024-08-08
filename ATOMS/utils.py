import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import KDTree
import pandas as pd

def filter_2D_structured(elements, nodes, r_min, nelx, nely):
    
    element_centroids = np.mean(nodes[elements], axis=1)
    r = np.linalg.norm(element_centroids[0] - element_centroids[1])
    
    filter_rad = r_min/r

    n_neigbours = int(filter_rad) * 2 + 1
    a,b = np.meshgrid(np.arange(n_neigbours),np.arange(n_neigbours))
    kernel_field = np.vstack([a.flatten(),b.flatten()]).T - int(filter_rad)


    a,b = np.meshgrid(np.arange(nely),np.arange(nelx))
    element_ids = np.vstack([b.flatten(),a.flatten()]).T

    relevant_elements = (element_ids[:,np.newaxis,:] + kernel_field[np.newaxis,:,:])


    mid_idx = np.sum(element_ids[nelx//2 * nely + nely//2] * [nely,1])

    relevant_elements_ = relevant_elements[nelx//2 * nely + nely//2]
    relevant_elements_ = relevant_elements_[np.all(relevant_elements_>=0,axis=1)]
    relevant_elements_ = relevant_elements_[relevant_elements_[:,0] < nelx]
    relevant_elements_ = relevant_elements_[relevant_elements_[:,1] < nely]

    idx = relevant_elements_[:,0] * nely + relevant_elements_[:,1]


    rad = r_min

    ws = (
        rad
        - np.linalg.norm(element_centroids[idx] - element_centroids[mid_idx], axis=-1)
    ) / rad

    ws = ws[np.newaxis].repeat(elements.shape[0],axis=0)

    relevant_elements_idx = (relevant_elements * [nely,1]).sum(-1)
    element_idx = element_ids[:,0] * nely + element_ids[:,1]
    element_idx = element_idx[:,np.newaxis].repeat(relevant_elements.shape[1],axis=1)
    filter_kernel_inds = np.concatenate([element_idx[:,:,np.newaxis],relevant_elements_idx[:,:,np.newaxis]],axis=-1)


    mask = np.all(relevant_elements >= 0,axis=-1)
    mask = mask & (relevant_elements[:,:,0] < nelx)
    mask = mask & (relevant_elements[:,:,1] < nely)
    mask = mask & (ws > 0)

    ws /= np.sum(ws * mask,-1,keepdims=True)

    filter_kernel_inds = filter_kernel_inds[mask]
    ws = ws[mask]
    filter_kernel = coo_matrix(
        (ws, (filter_kernel_inds[:, 0], filter_kernel_inds[:, 1])),
        shape=[len(elements), len(elements)],
    ).tocsr()
    
    return filter_kernel

def kernels2D_nf(elements, Ks, node_positions):
    element_centroids = []
    inds_map = []
    k_flat = []
    elements_flat = []
    k_id_full = []
    elements_size = []
    el_ids = []
    el_ids_per_row = []
    el_ids_per_element = []

    for i in range(len(elements)):
        k_flat += Ks[i].flatten().tolist()
        el_ids += [i] * (Ks[i].shape[0] ** 2)
        el_ids_per_row += [i] * Ks[i].shape[0]
        el_ids_per_element += [i] * len(elements[i])
        elements_flat += list(elements[i])
        k_id_full += list(elements[i]) * len(elements[i]) * 2
        elements_size += [len(elements[i]) * 2] * len(elements[i]) * 2

    element_centroids = coo_matrix(
        (2 / np.array(elements_size[::2]), (el_ids_per_element, elements_flat))
    ).dot(node_positions)
    el_ids = np.array(el_ids)
    k_flat = np.array(k_flat)
    el_ids_per_row = np.array(el_ids_per_row)
    elements_flat = np.array(elements_flat)
    elements_size = np.array(elements_size)

    k_id_full = np.array(k_id_full)
    k_id_full = np.vstack((k_id_full * 2, k_id_full * 2 + 1)).T.flatten()
    eqn_series = np.vstack((2 * elements_flat, 2 * elements_flat + 1)).T.flatten()
    grad_map = np.vstack((el_ids_per_row, eqn_series)).T
    eqn_series = eqn_series.repeat(elements_size)
    inds_map = np.vstack((eqn_series, k_id_full)).T
   # k_map, kerned_ids = np.unique(inds_map, axis=0, return_inverse=True)
    mapped_inds_1d = (inds_map*[inds_map.max(),1]).sum(axis=1)
    uniques = pd.unique(mapped_inds_1d)
    uniques = np.sort(uniques)
    sorter = np.argsort(uniques)
    kerned_ids = sorter[np.searchsorted(uniques, mapped_inds_1d, sorter=sorter)]

    sorter = np.argsort(mapped_inds_1d)
    unique_ids = sorter[np.searchsorted(mapped_inds_1d, uniques, sorter=sorter)]

    k_map = inds_map[unique_ids]
    dk_map = np.vstack(
        (np.arange(el_ids_per_row.shape[0]).repeat(elements_size), k_id_full)
    ).T

    K_kernel = coo_matrix((k_flat, (kerned_ids, el_ids))).tocsr()
    dK = coo_matrix((k_flat, (dk_map.T[0], dk_map.T[1]))).tocsr()

    return K_kernel, k_map, dK, grad_map, element_centroids

def filter_3D_structured(elements, nodes, r_min, nelx, nely, nelz):
    
    element_centroids = np.mean(nodes[elements], axis=1)
    r = np.linalg.norm(element_centroids[0] - element_centroids[1])

    filter_rad = r_min/r

    n_neigbours = int(filter_rad) * 2 + 1
    a,b,c = np.meshgrid(np.arange(n_neigbours),np.arange(n_neigbours),np.arange(n_neigbours))
    kernel_field = np.vstack([a.flatten(),b.flatten(),c.flatten()]).T - int(filter_rad)


    a,b,c = np.meshgrid(np.arange(nely),np.arange(nelx),np.arange(nelz))
    element_ids = np.vstack([b.flatten(),a.flatten(),c.flatten()]).T

    relevant_elements = (element_ids[:,np.newaxis,:] + kernel_field[np.newaxis,:,:])


    mid_idx = np.sum(element_ids[nelx//2 * nely * nelz + nely//2 * nelz + nelz//2] * [nely*nelz,nelz,1])

    relevant_elements_ = relevant_elements[nelx//2 * nely * nelz + nely//2 * nelz + nelz//2]
    relevant_elements_ = relevant_elements_[np.all(relevant_elements_>=0,axis=1)]
    relevant_elements_ = relevant_elements_[relevant_elements_[:,0] < nelx]
    relevant_elements_ = relevant_elements_[relevant_elements_[:,1] < nely]
    relevant_elements_ = relevant_elements_[relevant_elements_[:,2] < nelz]

    idx = relevant_elements_[:,0] * nely * nelz + relevant_elements_[:,1] * nelz + relevant_elements_[:,2]

    rad = r_min

    ws = (
        rad
        - np.linalg.norm(element_centroids[idx] - element_centroids[mid_idx], axis=-1)
    ) / rad

    ws = ws[np.newaxis].repeat(elements.shape[0],axis=0)

    relevant_elements_idx = (relevant_elements * [nely*nelz,nelz,1]).sum(-1)
    element_idx = element_ids[:,0] * nely * nelz + element_ids[:,1] * nelz + element_ids[:,2]
    element_idx = element_idx[:,np.newaxis].repeat(relevant_elements.shape[1],axis=1)
    filter_kernel_inds = np.concatenate([element_idx[:,:,np.newaxis],relevant_elements_idx[:,:,np.newaxis]],axis=-1)


    mask = np.all(relevant_elements >= 0,axis=-1)
    mask = mask & (relevant_elements[:,:,0] < nelx)
    mask = mask & (relevant_elements[:,:,1] < nely)
    mask = mask & (relevant_elements[:,:,2] < nelz)
    mask = mask & (ws > 0)

    ws /= np.sum(ws * mask,-1,keepdims=True)

    filter_kernel_inds = filter_kernel_inds[mask]
    ws = ws[mask]
    filter_kernel = coo_matrix(
        (ws, (filter_kernel_inds[:, 0], filter_kernel_inds[:, 1])),
        shape=[len(elements), len(elements)],
    ).tocsr()
    
    return filter_kernel

def kernels3D_nf(elements, Ks, node_positions):
    element_centroids = []
    inds_map = []
    k_flat = []
    elements_flat = []
    k_id_full = []
    elements_size = []
    el_ids = []
    el_ids_per_row = []
    el_ids_per_element = []

    for i in range(len(elements)):
        k_flat += Ks[i].flatten().tolist()
        el_ids += [i] * (Ks[i].shape[0] ** 2)
        el_ids_per_row += [i] * Ks[i].shape[0]
        el_ids_per_element += [i] * len(elements[i])
        elements_flat += list(elements[i])
        k_id_full += list(elements[i]) * len(elements[i]) * 3
        elements_size += [len(elements[i]) * 3] * len(elements[i]) * 3

    element_centroids = coo_matrix(
        (3 / np.array(elements_size[::3]), (el_ids_per_element, elements_flat))
    ).dot(node_positions)
    el_ids = np.array(el_ids)
    k_flat = np.array(k_flat)
    el_ids_per_row = np.array(el_ids_per_row)
    elements_flat = np.array(elements_flat)
    elements_size = np.array(elements_size)

    k_id_full = np.array(k_id_full)
    k_id_full = np.vstack(
        (k_id_full * 3, k_id_full * 3 + 1, k_id_full * 3 + 2)
    ).T.flatten()
    eqn_series = np.vstack(
        (3 * elements_flat, 3 * elements_flat + 1, 3 * elements_flat + 2)
    ).T.flatten()
    grad_map = np.vstack((el_ids_per_row, eqn_series)).T
    eqn_series = eqn_series.repeat(elements_size)
    inds_map = np.vstack((eqn_series, k_id_full)).T
   # k_map, kerned_ids = np.unique(inds_map, axis=0, return_inverse=True)
    mapped_inds_1d = (inds_map*[inds_map.max(),1]).sum(axis=1)
    uniques = pd.unique(mapped_inds_1d)
    uniques = np.sort(uniques)
    sorter = np.argsort(uniques)
    kerned_ids = sorter[np.searchsorted(uniques, mapped_inds_1d, sorter=sorter)]

    sorter = np.argsort(mapped_inds_1d)
    unique_ids = sorter[np.searchsorted(mapped_inds_1d, uniques, sorter=sorter)]

    k_map = inds_map[unique_ids]
    dk_map = np.vstack(
        (np.arange(el_ids_per_row.shape[0]).repeat(elements_size), k_id_full)
    ).T

    K_kernel = coo_matrix((k_flat, (kerned_ids, el_ids))).tocsr()
    dK = coo_matrix((k_flat, (dk_map.T[0], dk_map.T[1]))).tocsr()

    return K_kernel, k_map, dK, grad_map, element_centroids

def kernels2D(elements, Ks, node_positions, r_min):
    element_centroids = []
    inds_map = []
    k_flat = []
    elements_flat = []
    k_id_full = []
    elements_size = []
    el_ids = []
    el_ids_per_row = []
    el_ids_per_element = []

    for i in range(len(elements)):
        k_flat += Ks[i].flatten().tolist()
        el_ids += [i] * (Ks[i].shape[0] ** 2)
        el_ids_per_row += [i] * Ks[i].shape[0]
        el_ids_per_element += [i] * len(elements[i])
        elements_flat += list(elements[i])
        k_id_full += list(elements[i]) * len(elements[i]) * 2
        elements_size += [len(elements[i]) * 2] * len(elements[i]) * 2

    element_centroids = coo_matrix(
        (2 / np.array(elements_size[::2]), (el_ids_per_element, elements_flat))
    ).dot(node_positions)
    el_ids = np.array(el_ids)
    k_flat = np.array(k_flat)
    el_ids_per_row = np.array(el_ids_per_row)
    elements_flat = np.array(elements_flat)
    elements_size = np.array(elements_size)

    k_id_full = np.array(k_id_full)
    k_id_full = np.vstack((k_id_full * 2, k_id_full * 2 + 1)).T.flatten()
    eqn_series = np.vstack((2 * elements_flat, 2 * elements_flat + 1)).T.flatten()
    grad_map = np.vstack((el_ids_per_row, eqn_series)).T
    eqn_series = eqn_series.repeat(elements_size)
    inds_map = np.vstack((eqn_series, k_id_full)).T
   # k_map, kerned_ids = np.unique(inds_map, axis=0, return_inverse=True)
    mapped_inds_1d = (inds_map*[inds_map.max(),1]).sum(axis=1)
    uniques = pd.unique(mapped_inds_1d)
    uniques = np.sort(uniques)
    sorter = np.argsort(uniques)
    kerned_ids = sorter[np.searchsorted(uniques, mapped_inds_1d, sorter=sorter)]

    sorter = np.argsort(mapped_inds_1d)
    unique_ids = sorter[np.searchsorted(mapped_inds_1d, uniques, sorter=sorter)]

    k_map = inds_map[unique_ids]
    dk_map = np.vstack(
        (np.arange(el_ids_per_row.shape[0]).repeat(elements_size), k_id_full)
    ).T

    K_kernel = coo_matrix((k_flat, (kerned_ids, el_ids))).tocsr()
    dK = coo_matrix((k_flat, (dk_map.T[0], dk_map.T[1]))).tocsr()

    search_tree = KDTree(element_centroids)
    Ne = search_tree.query_ball_point(element_centroids, r_min)

    filter_kernel_inds = []
    filter_kernel_vals = []
    for i in range(len(elements)):

        ws = (
            r_min
            - np.linalg.norm(element_centroids[Ne[i]] - element_centroids[i], axis=-1)
        ) / r_min
        ws = ws / ws.sum()
        filter_kernel_inds += np.pad(
            np.array(Ne[i]).reshape(-1, 1), [[0, 0], [1, 0]], constant_values=i
        ).tolist()
        filter_kernel_vals += ws.tolist()

    filter_kernel_inds = np.array(filter_kernel_inds)
    filter_kernel_vals = np.array(filter_kernel_vals)

    filter_kernel = coo_matrix(
        (filter_kernel_vals, (filter_kernel_inds[:, 0], filter_kernel_inds[:, 1])),
        shape=[len(elements), len(elements)],
    ).tocsr()

    return K_kernel, k_map, dK, grad_map, filter_kernel, element_centroids


def kernels3D(elements, Ks, node_positions, r_min):
    element_centroids = []
    inds_map = []
    k_flat = []
    elements_flat = []
    k_id_full = []
    elements_size = []
    el_ids = []
    el_ids_per_row = []
    el_ids_per_element = []

    for i in range(len(elements)):
        k_flat += Ks[i].flatten().tolist()
        el_ids += [i] * (Ks[i].shape[0] ** 2)
        el_ids_per_row += [i] * Ks[i].shape[0]
        el_ids_per_element += [i] * len(elements[i])
        elements_flat += list(elements[i])
        k_id_full += list(elements[i]) * len(elements[i]) * 3
        elements_size += [len(elements[i]) * 3] * len(elements[i]) * 3

    element_centroids = coo_matrix(
        (3 / np.array(elements_size[::3]), (el_ids_per_element, elements_flat))
    ).dot(node_positions)
    el_ids = np.array(el_ids)
    k_flat = np.array(k_flat)
    el_ids_per_row = np.array(el_ids_per_row)
    elements_flat = np.array(elements_flat)
    elements_size = np.array(elements_size)

    k_id_full = np.array(k_id_full)
    k_id_full = np.vstack(
        (k_id_full * 3, k_id_full * 3 + 1, k_id_full * 3 + 2)
    ).T.flatten()
    eqn_series = np.vstack(
        (3 * elements_flat, 3 * elements_flat + 1, 3 * elements_flat + 2)
    ).T.flatten()
    grad_map = np.vstack((el_ids_per_row, eqn_series)).T
    eqn_series = eqn_series.repeat(elements_size)
    inds_map = np.vstack((eqn_series, k_id_full)).T
   # k_map, kerned_ids = np.unique(inds_map, axis=0, return_inverse=True)
    mapped_inds_1d = (inds_map*[inds_map.max(),1]).sum(axis=1)
    uniques = pd.unique(mapped_inds_1d)
    uniques = np.sort(uniques)
    sorter = np.argsort(uniques)
    kerned_ids = sorter[np.searchsorted(uniques, mapped_inds_1d, sorter=sorter)]

    sorter = np.argsort(mapped_inds_1d)
    unique_ids = sorter[np.searchsorted(mapped_inds_1d, uniques, sorter=sorter)]

    k_map = inds_map[unique_ids]
    dk_map = np.vstack(
        (np.arange(el_ids_per_row.shape[0]).repeat(elements_size), k_id_full)
    ).T

    K_kernel = coo_matrix((k_flat, (kerned_ids, el_ids))).tocsr()
    dK = coo_matrix((k_flat, (dk_map.T[0], dk_map.T[1]))).tocsr()

    search_tree = KDTree(element_centroids)
    Ne = search_tree.query_ball_point(element_centroids, r_min)

    filter_kernel_inds = []
    filter_kernel_vals = []
    for i in range(len(elements)):

        ws = (
            r_min
            - np.linalg.norm(element_centroids[Ne[i]] - element_centroids[i], axis=-1)
        ) / r_min
        ws = ws / ws.sum()
        filter_kernel_inds += np.pad(
            np.array(Ne[i]).reshape(-1, 1), [[0, 0], [1, 0]], constant_values=i
        ).tolist()
        filter_kernel_vals += ws.tolist()

    filter_kernel_inds = np.array(filter_kernel_inds)
    filter_kernel_vals = np.array(filter_kernel_vals)

    filter_kernel = coo_matrix(
        (filter_kernel_vals, (filter_kernel_inds[:, 0], filter_kernel_inds[:, 1])),
        shape=[len(elements), len(elements)],
    ).tocsr()

    return K_kernel, k_map, dK, grad_map, filter_kernel, element_centroids
