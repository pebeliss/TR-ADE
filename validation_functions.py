import pandas as pd
import numpy as np
from sklearn.utils import resample
from tqdm import tqdm
import datetime

from VADEGAM_pipeline import PREPROCESSOR_WRAPPER, VADEGAM_WRAPPER, set_seed


def train_and_pred(X, y, args, seed=12345):
    """VaDEGam pipeline wrapper for validation & consensus clustering.

    :param X, y: subsampled train data and labels
    :param args: arguments passed to VaDEGam
    :param seed: random seed

    :returns: list (correct order) of indices and list of clusters
    
    """
    preprocessor = PREPROCESSOR_WRAPPER(args, seed=seed)
    preprocessor.build(X)
    train_generator, val_generator, train_data, val_data = preprocessor.data_pipeline(X, y, return_generator=True, split_seed=seed)
    X_all = pd.concat([train_data[0], val_data[0]], axis=0)
    vadegam = VADEGAM_WRAPPER(args, preprocessor.cont_dim, preprocessor.bin_dim, seed=seed, log_interval=150)
    vadegam.build()
    _ = vadegam.model_fit(train_generator, val_generator)
    _, _, clusters, _ = vadegam.model_predict(X_all)
    indices = X_all.index
    return indices, clusters

def resample_runs(X, y, args, nruns=100, subsample_frac=0.5, rseeds = [], mseeds = [], save_results = './results'):
    """Perform nruns number of runs with VaDEGam and subsample_frac*100% subsampling.

    :param X, y: train data and labels
    :param args: arguments passed to VaDEGam
    :param nruns: number of runs
    :param subsample_frac: fraction of subsampled data
    :param mseeds: seed values used to initialise the model on each iteration. If empty, set to rseeds
    :param mseeds: seed values used to initialise the resampling on each iteration
    :param save_results: path where to save intermediate results as .csv files
    
    :returns: list of lists of subsampled indices and list of lists of the cluster assignments
    """

    n_samples = X.shape[0]
    subsample_size = int(n_samples * subsample_frac)

    rseeds = rseeds if len(rseeds) >= nruns else np.random.randint(0, 10**4, size=nruns)
    if not isinstance(mseeds, int) and len(mseeds) < nruns:
        mseeds = mseeds*nruns if len(mseeds) == 1 else rseeds
    
    indices, clusters = [], []

    for i in tqdm(range(nruns)):
        sub_idx = np.sort(resample(X.index, n_samples=subsample_size, replace=False, random_state=rseeds[i]))
        X_subset = X.loc[sub_idx]
        y_subset = y.loc[sub_idx]
        idx, c =  train_and_pred(X_subset, y_subset, args, seed=mseeds[i])
        
        indices.append(idx)
        clusters.append(c)
        pd.DataFrame({'idx': idx, 'clust': c}).to_csv(f'{save_results}/run{i}_{str(datetime.date.today())}.csv')
    
    return indices, clusters

def get_consensus_M(indices, clusters, N, nruns):
    """Construct consensus matrix.
    
    :param indices: list/array of lists/arrays with indexes over nruns runs
    :param clusters: list/array of lists/arrays of cluster assignments for above indices/samples
    :param N: total number of distinct samples
    :param nruns: number of runs

    :returns: Consensus matrix of shape (N, N)

    """
    M0 = np.zeros((N, N))
    I = np.zeros((N, N))

    for i in tqdm(range(nruns)):
        idx = indices[i]
        mask = np.zeros(N, dtype=bool)
        mask[idx] = True
        I += np.outer(mask, mask).astype(int)
        lbls = clusters[i]
        idx_matrix = np.ix_(idx, idx)
        M0[idx_matrix] += np.equal.outer(lbls, lbls)

    M = np.nan_to_num(M0/I)
    return M

def consensus_index(clusters, k, M):
    """Calculate consensus index for given cluster.

    :param clusters: cluster assignment for N samples
    :param k: cluster index
    :param M: Consensus matrix of shape (N, N)

    :returns: float
    
    """
    Ck = np.where(clusters == k)[0]
    if len(Ck) > 1:
        Mk = M.values[np.ix_(Ck, Ck)]
        consensus_sum = np.sum(np.triu(Mk, 1))
        num_pairs = len(Ck) * (len(Ck)-1) / 2
        return consensus_sum / num_pairs
    return 0

def permute_and_pval(clusters, K, M, nperm):
    """Calculate p-values for cluster assignment with permutation test.

    :param clusters: cluster assignment for N samples
    :param K: number of clusters to calculate p-values for
    :param M: Consensus matrix of shape (N, N)
    :param nperm: number of permutations

    :returns: list of p-values
    """
    p_vals = []
    for k in tqdm(range(K)):
        mk_perms = np.array([consensus_index(np.random.permutation(clusters), k, M) for p in range(nperm)])
        p_value = (np.sum(mk_perms >= consensus_index(clusters, k, M)) + 1) / (nperm + 1)
        p_vals.append(p_value)
    return p_vals