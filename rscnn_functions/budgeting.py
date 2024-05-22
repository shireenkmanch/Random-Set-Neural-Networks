"""
Code for Budgeting: Selecting k number of focal sets from the random-set space

Returns:
    new_classes: Budgeted focal sets of classes for RS-CNN
"""

from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from scipy import linalg
import numpy as np


def train_embeddings(aux_model, x_train, batch_size):
    train_embeddings = aux_model.predict(x_train, verbose=1, batch_size=batch_size)
    train_embedded_tsne = TSNE(n_components=3, init='random', perplexity=30, n_jobs=-1).fit_transform(train_embeddings)

    return train_embedded_tsne
 

def fit_gmm(classes, train_embedded_tsne, y_train):
    individual_gms = []
    for i in range(len(classes)):
        individual_gms.append(GaussianMixture(n_components=1, random_state=7).fit(train_embedded_tsne[y_train == i]))

    return individual_gms


def ellipse(individual_gms, num_classes):
    means = []
    eigen_vecs = []
    stds = []
    feature_space = 3
    for i_gm in individual_gms:
        means.append(i_gm.means_[0])
        v, w = linalg.eigh(i_gm.covariances_[0])
        v = 2.0 * np.sqrt(7.815) * np.sqrt(v)
        stds.append(v)
        eigen_vecs.append(w)

    means = np.array(means)
    eigen_vecs = np.array(eigen_vecs)
    stds = np.array(stds)

    max_std = np.max(stds)
    max_len = int(max_std)+2

    reg_shape = (max_len,)*feature_space
    center = np.array(reg_shape) // 2
    
    indices=np.indices(reg_shape)
    indices=np.transpose(indices,list(np.arange(1,len(reg_shape)+1))+[0])
    indices=indices.reshape((np.prod(reg_shape), feature_space))
    
    regions = []
    vecs = indices - center
    vec_norms = np.linalg.norm(vecs, axis=-1) + 1e-31
    for i in range(num_classes):
        ellipse = np.sum(vecs[:, None, :] * eigen_vecs[i][None, :, :], axis=-1)
        ellipse = np.abs(ellipse/(vec_norms[:, None] * np.linalg.norm(eigen_vecs[i], axis=-1)[None, :]))
        ellipse = np.linalg.norm(np.sum((ellipse * (stds[i][None, :]/2))[:, :, None] * eigen_vecs[i][None, :, :], axis=1), axis=-1) + 1e-25
        ellipse = (vec_norms <= ellipse).reshape(reg_shape).astype(np.float32)

        regions.append(ellipse)
    
    return regions, means, max_len
    

def overlaps(k, classes, num_clusters, classes_dict, regions, means, max_len):
    clusters = classes
    overlaps = {}
    top_sets = [set([c]) for c in clusters]
    for cardinality in range(2,num_clusters+1):
        for ts in top_sets:
            for clus in clusters:
                s = ts.copy()
                s.add(clus)
                s = sorted(s)
                if len(s) == cardinality and ",".join(s) not in overlaps:
                    region = np.zeros_like(regions[0])
                    smallest_region = np.inf
                    for num, name in enumerate(s):
                        c = classes_dict[name]
                        if num == 0:
                            region += regions[c]
                            reg_cen = means[c]
                        else:
                            top_corner = means[c] - reg_cen
                            if any(top_corner < -max_len) or any(top_corner > max_len):
                                pass
                            else:
                                limits = []
                                start_points = []
                                for val in top_corner:
                                    if val<0:
                                        limits.append((int(abs(val)), max_len))
                                        start_points.append(0)
                                    else:
                                        limits.append((0, max_len - int(val)))
                                        start_points.append(int(val))

                                eval_s = []
                                for n1 in range(len(limits)):
                                    eval_s.append(f"{limits[n1][0]}:{limits[n1][1]}")
                                eval_s = ",".join(eval_s)
                                cutout = eval(f"regions[{c}][{eval_s}]")

                                eval_s = []
                                for n1 in range(len(start_points)):
                                    eval_s.append(f"{start_points[n1]}:{start_points[n1]}+{cutout.shape[n1]}")
                                eval_s = ",".join(eval_s)
                                exec(f"region[{eval_s}] += cutout")

                        if np.sum(regions[c])<smallest_region:
                            smallest_region = np.sum(regions[c])

                    intersection = np.sum([region == len(s)])
                    #op = intersection / smallest_region
                    op = intersection / np.sum(region!=0)
                    # print("s", ",".join(s))
                    overlaps[",".join(s)] = op
        keys = np.array(list(overlaps.keys()))
        values = np.array(list(overlaps.values()))
        arg_sorted = np.argsort(values)[::-1]
        top_sets = [set([num for num in cl.split(",")]) for cl in keys[arg_sorted[:k]] if len(set([num for num in cl.split(",")]))==cardinality]
        
    keys = list(overlaps.keys())
    keys = np.array([set([num for num in cl.split(",")]) for cl in keys])
    values = np.array(list(overlaps.values()))
    arg_sorted = np.argsort(values)[::-1]
    new_k = min(k, np.sum(values[arg_sorted[:k]] != 0))
    new_classes = [set([c]) for c in classes] + list(keys[arg_sorted[:new_k]])# + [set(classes)]
    
    return new_classes

        
