import numpy as np
import pandas as pd
import re


def get_weights(reduced_X, n_components=5):
    """
    Calculates the variation of the data projected onto the discovered dimensions as a proxy for the eigenvalues.

    Parameters:
        reduced_X (dict): Dictionary containing reduced data along with their key dimensions.
        n_components (int, optional): Number of components to consider. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame containing variations of the data projected onto the discovered dimensions.
    """
    res = pd.DataFrame()
    for key in reduced_X:
        # Calculate variations
        var_dims = [np.var(reduced_X[key][0][i])
                    for i in range(len(reduced_X[key][0]))]
        por_eigenvals = [x/sum(var_dims) for x in var_dims]
        # Make all lists be of length 'n_components'
        por_eigenvals.extend([''] * (n_components - len(por_eigenvals)))
        # res[key+('Var',)] = var_dims
        res[key+('Var %',)] = por_eigenvals

    res.columns = pd.MultiIndex.from_tuples(
        res.columns.to_list())

    return res.droplevel(3, axis=1)


def get_corr_table(reduced_X, X_train, col_names=None, abs=True, weighted=False, weights=None):
    """
    Compute correlations between original data and each principal component.

    Parameters:
        reduced_X (dict): Dictionary containing reduced data along with their key dimensions.
        X_train (array-like): Original training data.
        col_names (list): List of column names for the original data.
        abs (bool): If True, takes the absolute value of correlations. Defaults to True.
        weighted (bool): If True, normalize correlations by the percentage of variation. Defaults to False.
        weights (dict): Dictionary containing weights for each dimensionality reduction technique. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing correlations.
        pd.DataFrame: DataFrame containing weighted average correlations if weighted is True, otherwise None.
    """

    # Load the data into a Pandas df
    if col_names is not None:
        df = pd.DataFrame(X_train, columns=col_names[:-1])
    else:
        df = pd.DataFrame(X_train)

    for key in reduced_X.keys():
        n_components = int(re.match(r'(\d+)Dim', key[0]).group(1))
        for i in range(n_components):
            df[key+(i,)] = reduced_X[key][0][i]

    # Correlations between the original data and each principal component
    df_corrs = df.corr().iloc[:len(X_train[0]), len(X_train[0]):]

    # Make df multi-index
    df_corrs.columns = pd.MultiIndex.from_tuples(
        df_corrs.columns.to_list())

    # Take only the abs value of the correlations
    if abs:
        df_corrs = df_corrs.abs()

    # Normalize by the percentage of variation
    if weighted:
        # Format weights
        weigts_c = []
        for key in reduced_X:
            if weigts_c != '':
                weigts_c = weigts_c + weights[key].tolist()
        weigts_c = [x for x in weigts_c if x != '']

        # Multiply each column by corresponding weigth
        for i, col in enumerate(df_corrs.columns):
            df_corrs[col] *= weigts_c[i]

        # Initialize dictionary to hold lists of tuples
        result_dict = {}

        # Group tuples by their second element
        for tup in df_corrs:

            if tup[1] not in result_dict:
                result_dict[tup[1]] = [(tup)]
            else:
                result_dict[tup[1]].append((tup))

        # Convert dictionary values to lists
        header_groups = [values for values in result_dict.values()]

        # Calculate the weighted average correlations
        df_corrs_avg = pd.DataFrame()
        for header_g in header_groups:
            df_corrs_avg[header_g[0][1]
                         ] = df_corrs[header_g].abs().mean(axis=1)
        return df_corrs, df_corrs_avg

    return df_corrs, None
