import pandas as pd
import numpy as np
from sklearn.utils import resample
from tqdm import tqdm
import datetime

from TR_ADE_pipeline import PREPROCESSOR_WRAPPER, TR_ADE_WRAPPER, set_seed


def train_and_pred(X, y, args, seed=12345, test_data=()):
    """TR_ADE pipeline wrapper for validation & consensus clustering.

    :param X, y: subsampled train data and labels
    :param args: arguments passed to TR_ADE
    :param seed: random seed

    :returns: list (correct order) of indices and list of clusters
    
    """
    results = {}

    assert y is not None and args.classify == True, 'To classify, provide class labels (y)'

    set_seed(seed)
    preprocessor = PREPROCESSOR_WRAPPER(args)
    preprocessor.build(X)
    train_generator, val_generator, train_data, val_data = preprocessor.data_pipeline(X, y, return_generator=True, split_seed=seed)
    X_all = pd.concat([train_data[0], val_data[0]], axis=0)
    tr_ade = TR_ADE_WRAPPER(args, preprocessor.cont_dim, preprocessor.bin_dim, log_interval=150)
    tr_ade.build()
    history = tr_ade.model_fit(train_generator, val_generator)
    results['history'] = history.history
    if args.classify:
        _, _, _, results['pred_label_val'] = tr_ade.model_predict(val_data[0])
        results['val_data_idx'] = val_data[0].index
        _, results['z_mean'], results['clusters'], results['pred_label_tr_val'] = tr_ade.model_predict(X_all)
        results['x_all_idx'] = X_all.index
        if len(test_data) > 0:
            test_data_processed = preprocessor.preprocess(*test_data)
            _, results['z_mean_test'], results['clusters_test'], results['pred_label_test'] = tr_ade.model_predict(test_data_processed[0])
    else:
        results['X_recon'], results['z_mean'], results['clusters'] = tr_ade.model_predict(X_all)
        if len(test_data) > 0:
            test_data_processed = preprocessor.preprocess(test_data)
            X_recon_tst, results['z_mean_test'], results['clusters_test'] = tr_ade.model_predict(test_data_processed[0])
            X_recon_tst_pd = pd.DataFrame(X_recon_tst, columns = test_data_processed[0].columns, index=test_data_processed[0].index)
            X_inverted_tst = preprocessor.preprocessor.inverse_transform(X_recon_tst_pd)
            results['X_recon_test'] = X_recon_tst_pd
            results['X_recon_inv_test'] = X_inverted_tst
    results['indices'] = X_all.index
    return results

def resample_runs(X, y, args, nruns=80, subsample_frac=0.5, rseeds = [], 
                  mseeds = [], save_results = './results', test_data = ()):
    """Perform nruns number of runs with TR_ADE and subsample_frac*100% subsampling.

    :param X, y: train data and labels
    :param args: arguments passed to TR_ADE
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
        y_subset = None
        if y is not None:
            y_subset = y.loc[sub_idx]
        results =  train_and_pred(X_subset, y_subset, args, seed=mseeds[i], test_data=test_data)
        idx, c, z_mean = results['indices'], results['clusters'], results['z_mean']
        if args.classify:
            lbl, lbl_idx = results['pred_label_val'], results['val_data_idx']
            
            labels = pd.DataFrame(lbl, columns = ['RBC_0', 'RBC_1', 'P_0', 'P_1', 'PLT_0', 'PLT_1'])
            labels = pd.concat([labels, pd.DataFrame(lbl_idx, columns=['val_idx'])], axis=1)
            labels.to_csv(f'{save_results}/run{i}_{str(datetime.date.today())}_labels.csv')

            lbl_tr_val, lbl_tr_val_idx = results['pred_label_tr_val'], results['x_all_idx']
            
            labels_tr_val = pd.DataFrame(lbl_tr_val, columns = ['RBC_0', 'RBC_1', 'P_0', 'P_1', 'PLT_0', 'PLT_1'])
            labels_tr_val = pd.concat([labels_tr_val, pd.DataFrame(lbl_tr_val_idx, columns=['tr_val_idx'])], axis=1)
            labels_tr_val.to_csv(f'{save_results}/run{i}_{str(datetime.date.today())}_labels_tr_val.csv')

            if len(test_data) > 0:
                z_mean_tst, c_tst, lbl_tst = results['z_mean_test'], results['clusters_test'], results['pred_label_test']
                test_results = pd.concat([pd.DataFrame(lbl_tst, columns = ['RBC_0', 'RBC_1', 'P_0', 'P_1', 'PLT_0', 'PLT_1']), 
                                          pd.DataFrame(c_tst, columns=['clust']),
                                          pd.DataFrame(z_mean_tst, columns=['z1', 'z2', 'z3'])], axis=1)
                test_results.to_csv(f'{save_results}/run{i}_{str(datetime.date.today())}_test_results.csv')
        else:
            if len(test_data) > 0:
                z_mean_tst, c_tst = results['z_mean_test'], results['clusters_test']
                X_recon_tst, X_recon_inv_tst = results['X_recon_test'], results['X_recon_inv_test']
                test_results = pd.concat([pd.DataFrame(c_tst, columns=['clust']),
                                          pd.DataFrame(z_mean_tst, columns=['z1', 'z2', 'z3'])], axis=1)
                test_results.to_csv(f'{save_results}/run{i}_{str(datetime.date.today())}_test_results.csv')
                X_recon_tst.to_csv(f'{save_results}/run{i}_{str(datetime.date.today())}_test_recon.csv')
                X_recon_inv_tst.to_csv(f'{save_results}/run{i}_{str(datetime.date.today())}_test_recon_inv.csv')
        indices.append(idx)
        clusters.append(c)
        pd.DataFrame({'idx': idx, 'clust': c}).to_csv(f'{save_results}/run{i}_{str(datetime.date.today())}.csv')
        pd.DataFrame(z_mean).to_csv(f'{save_results}/run{i}_{str(datetime.date.today())}_z_mean.csv')
        pd.DataFrame(results['history']).to_csv(f'{save_results}/run{i}_{str(datetime.date.today())}_history.csv')

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