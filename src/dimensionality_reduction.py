from lpproj import LocalityPreservingProjection as LPP
from slmvp import SLMVPTrain, SLMVP_transform
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from lol import LOL
from tqdm import tqdm
import os
import pickle
from math import sqrt, floor
import pandas as pd
import pickle


def apply_slmvp(X_train, X_test, y_train, n, type_kernel, dataset_name, gammas=None, poly_order=None, multilabel=None):
    # Get the principal components
    BAux = SLMVPTrain(X=X_train.T, Y=y_train,
                      rank=n,
                      typeK=type_kernel,
                      gammaX=gammas,
                      gammaY=gammas,
                      polyValue=poly_order,
                      multilabel=multilabel)
    # Pickle model
    save_path = os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))) + f'/models/dimensionality_reduction/{dataset_name}/{str(n)}_dimensions/slmvp_{str(gammas)}.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pickle.dump(BAux, open(save_path, 'wb'))
    # Get the data projected onto the new dimensions
    data_train, data_test = SLMVP_transform(
        BAux.T, X_train.T), SLMVP_transform(BAux.T, X_test.T)

    return data_train, data_test, BAux


def apply_lpp(X_train, X_test, n, dataset_name):
    lpp = LPP(n_components=n)
    lpp.fit(X_train)
    # Pickle model
    save_path = os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))) + f'/models/dimensionality_reduction/{dataset_name}/{str(n)}_dimensions/lpp.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pickle.dump(lpp, open(save_path, 'wb'))
    X_lpp_train = lpp.transform(X_train)
    X_lpp_test = lpp.transform(X_test)

    return X_lpp_train.T, X_lpp_test.T, lpp.projection_


def apply_pca(X_train, X_test, n, dataset_name):
    pca_model = PCA(n_components=n).fit(X_train)
    # Pickle model
    save_path = os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))) + f'/models/dimensionality_reduction/{dataset_name}/{str(n)}_dimensions/pca.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pickle.dump(pca_model, open(save_path, 'wb'))
    X_pca_train = pca_model.transform(X_train)
    X_pca_test = pca_model.transform(X_test)
    # return train, test, eigenvectors, eigenvalues
    return X_pca_train.T, X_pca_test.T, pca_model.components_, pca_model.explained_variance_


def apply_lle(X_train, X_test, n, k, _reg, dataset_name):
    lle = LLE(n_neighbors=k, n_components=n, reg=_reg)
    X_lle_train = lle.fit_transform(X_train)
    save_path = os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))) + f'/models/dimensionality_reduction/{dataset_name}/{str(n)}_dimensions/lle.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pickle.dump(lle, open(save_path, 'wb'))
    X_lle_test = lle.transform(X_test)

    return X_lle_train.T, X_lle_test.T


def apply_kpca(X_train, X_test, n, type_kernel, dataset_name, gamma=None):
    kernel_pca = KernelPCA(
        n_components=n, kernel=type_kernel, fit_inverse_transform=True, gamma=gamma
    )
    X_kpca_train = kernel_pca.fit(X_train).transform(X_train)
    save_path = os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))) + f'/models/dimensionality_reduction/{dataset_name}/{str(n)}_dimensions/kpca.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pickle.dump(kernel_pca, open(save_path, 'wb'))
    X_kpca_test = kernel_pca.transform(X_test)

    return X_kpca_train.T, X_kpca_test.T


def apply_lol(X_train, X_test, y_train, n_components, dataset_name):
    lmao = LOL(n_components=n_components, svd_solver='full')
    lmao.fit(X_train, y_train)
    save_path = os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))) + f'/models/dimensionality_reduction/{dataset_name}/{str(n_components-1)}_dimensions/lol.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pickle.dump(lmao, open(save_path, 'wb'))
    X_lol_train = lmao.transform(X_train)
    X_lol_test = lmao.transform(X_test)

    return X_lol_train.T, X_lol_test.T


def apply_all_dimensionality_reduction(X_train, X_test, y_train, dataset_name,
                                       models_list=['SLMVP', 'PCA',
                                                    'KPCA', 'LOL', 'LPP', 'LLE'],
                                       n_components_list=[1, 2, 5, 10], multilabel=None):
    """Apply dimensionality reduction techniques and save transformed features.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Testing data features.
        y_train (array-like): Training data labels.
        dataset_name (str): Name of the dataset.
        models_list (list, optional): List of dimensionality reduction techniques to be applied. Default is ['SLMVP', 'PCA', 'KPCA', 'LOL', 'LPP', 'LLE'].
        n_components_list (list, optional): List of numbers of components to be obtained with the dimensionality reduction techniques. Default is [1, 2, 5, 10].
        multilabel (bool, optional): Specifies whether the dataset is multilabel.

    Returns:
        dict: A dictionary containing the transformed features obtained by each dimensionality reduction technique.
    """

    if not isinstance(n_components_list, list):
        n_components_list = [n_components_list]

    reduced_data = dict()

    pbar = tqdm(n_components_list)
    for n_components in pbar:

        if 'SLMVP' in models_list:
            # Execution with gammas=0.01
            key = (str(n_components) + 'Dim', 'SLMVP', 'Radial-Gammas=0.01')
            pbar.set_description(str(key))
            reduced_data[key] = apply_slmvp(
                X_train, X_test, y_train, n_components, 'radial', dataset_name=dataset_name, gammas=0.01, multilabel=multilabel)

            # Execution with gammas=0.1
            key = (str(n_components) + 'Dim', 'SLMVP', 'Radial-Gammas=0.1')
            pbar.set_description(str(key))
            reduced_data[key] = apply_slmvp(X_train, X_test, y_train,
                                            n_components, 'radial', dataset_name=dataset_name, gammas=0.1, multilabel=multilabel)

        if 'PCA' in models_list:
            key = (str(n_components) + 'Dim', 'PCA', '')
            pbar.set_description(str(key))
            reduced_data[key] = apply_pca(
                X_train, X_test, n_components, dataset_name=dataset_name)

        if 'KPCA' in models_list:
            key = (str(n_components) + 'Dim', 'KPCA', 'Radial')
            pbar.set_description(str(key))
            reduced_data[key] = apply_kpca(
                X_train, X_test, n_components, 'rbf', dataset_name=dataset_name)

        if 'LOL' in models_list:
            if multilabel is None:
                key = (str(n_components) + 'Dim', 'LOL', '')
                pbar.set_description(str(key))
                reduced_data[key] = apply_lol(
                    X_train, X_test, y_train, dataset_name=dataset_name, n_components=n_components+1)

        if 'LPP' in models_list:
            key = (str(n_components) + 'Dim', 'LPP', '')
            pbar.set_description(str(key))
            reduced_data[key] = apply_lpp(
                X_train, X_test, n_components, dataset_name=dataset_name)

        if 'LLE' in models_list:
            k = floor(sqrt(len(X_train)))
            reg = 0.001
            key = (str(n_components) + 'Dim', 'LLE', 'k=' +
                   str(k) + '-reg=' + str(reg))
            pbar.set_description(str(key))
            reduced_data[key] = apply_lle(
                X_train, X_test, n_components, k, reg, dataset_name=dataset_name)

    # Save as pickle
    save_path = os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))) + f'/data/{dataset_name}/reduced/reduced_X.pkl'

    pickle.dump(reduced_data, open(save_path, 'wb'))

    print('Saved reduced data at path:', save_path)

    return reduced_data
