import pickle
import os
import pandas as pd
from tqdm import tqdm
from classification import apply_classifiers_original_features


def clf_corr(df_corrs_avg, col_names, X_train, X_test, y_train, y_test):
    # Read processed, split data
    X_train = pickle.load(open(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))) +
        '/data/processed_data/census_income/X_train.pkl', 'rb'))
    X_test = pickle.load(open(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))) +
        '/data/processed_data/census_income/X_test.pkl', 'rb'))
    y_train = pickle.load(open(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))) +
        '/data/processed_data/census_income/y_train.pkl', 'rb'))
    y_test = pickle.load(open(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))) +
        '/data/processed_data/census_income/y_test.pkl', 'rb'))

    results_df = pd.DataFrame(
        columns=['Features', 'AU_ROC'])

    # Use the calculated weighted average of the components-features absolute correlations
    # Order the features by those average correlations in descending order
    features = {}
    for dim_t in df_corrs_avg.keys():
        features[dim_t] = df_corrs_avg[[dim_t]].sort_values(
            by=dim_t, ascending=False).index.to_list()

    for dim_t in features:
        for i in tqdm(range(1, len(col_names))):
            indices = [col_names.index(x) for x in features[dim_t][:i]]
            X_train_subset = X_train[:, indices]
            X_test_subset = X_test[:, indices]

            # Apply ML classifier
            score = apply_classifiers_original_features(
                X_train_subset, y_train, X_test_subset, y_test, save_flag=False)
            # Save results
            results_df = results_df.append({'Dimensionality Technique': dim_t,
                                            'Number of Features': i,
                                            'Features': frozenset(features[dim_t][:i]),
                                            'AU_ROC': score}, ignore_index=True)

    return results_df
