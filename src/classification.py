import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os
import pickle
import pandas as pd
from sklearn.linear_model import SGDClassifier
import random


def train_xgb(X_train, y_train):
    """ Train XGBoost classifier.

    Parameters:
        X_train (array-like): Training data.
        y_train (array-like): Training labels.

        Returns:
        GridSearchCV: Trained XGBoost classifier.
    """
    # Initialize XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(random_state=42)
    params = [{'n_estimators': [5, 10, 20, 50, 100]}]
    gs_xgb = GridSearchCV(xgb_classifier,
                          param_grid=params,
                          scoring='accuracy',
                          cv=3)
    gs_xgb.fit(X_train, y_train)

    return gs_xgb


# def train_svc(X_train, y_train):
#     # Initialize SVC classifier
#     svc_classifier = SVC(random_state=42)
#     params = [{'C': [1], 'gamma': [
#         0.1, 0.01, 1], 'kernel': ['rbf']}]
#     gs_svc = GridSearchCV(svc_classifier,
#                           param_grid=params,
#                           scoring='accuracy',
#                           cv=3)
#     gs_svc.fit(X_train, y_train)

#     return gs_svc

def train_svc(X_train, y_train):
    # Initialize SVC classifier
    svc_classifier = SVC(random_state=42)

    # Train the SVC classifier
    svc_classifier.fit(X_train, y_train)

    # Return the trained classifier
    return svc_classifier


def train_sgd(X_train, y_train):
    # Initialize SGD classifier
    model = SGDClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model


def apply_classifiers_original_features(X_train, y_train, X_test, y_test, dataset_name, classifiers=['XGBOOST'], save_flag=True):
    """
    Run classifiers on original data.

    Parameters:
        X_train (array-like): Training data.
        y_train (array-like): Training labels.
        X_test (array-like): Testing data.
        y_test (array-like): Testing labels.
        classifiers (list, optional): List of classifiers to apply. Defaults to ['XGBOOST'].

    Returns:
        float: Accuracy score.
    """

    if 'XGBOOST' in classifiers:
        # Train XGBoost classifier
        clf = train_xgb(X_train, y_train)
        # Save trained classifier to a file
        if save_flag:
            pickle.dump(
                clf, open(os.path.dirname(os.path.dirname(os.path.abspath(
                    __file__))) + f'/models/classifiers/on_orig_features/{dataset_name}/xgboost.pkl', 'wb'))
        # Predict using trained classifier
        y_pred = clf.predict(X_test)
        # Calculate Accuracy score on test and train data
        score = accuracy_score(y_test, y_pred)
        score_train = accuracy_score(y_train, clf.predict(X_train))

    if 'SVC' in classifiers:  # Train SVC classifier
        clf = train_svc(X_train, y_train)
        # Save trained classifier to a file
        if save_flag:
            pickle.dump(
                clf, open(os.path.dirname(os.path.dirname(os.path.abspath(
                    __file__))) + f'/models/classifiers/on_orig_features/{dataset_name}/svc.pkl', 'wb'))
        # Predict using trained classifier
        y_pred = clf.predict(X_test)

        # Calculate Accuracy score on test and train data
        score = accuracy_score(y_test, y_pred)
        score_train = accuracy_score(y_train, clf.predict(X_train))

    if 'SGD' in classifiers:  # Train SGD classifier
        clf = train_sgd(X_train, y_train)
        # Save trained classifier
        if save_flag:
            pickle.dump(
                clf, open(os.path.dirname(os.path.dirname(os.path.abspath(
                    __file__))) + f'/models/classifiers/on_orig_features/{dataset_name}/sgd.pkl', 'wb'))
        # Predict using trained classifier
        y_pred = clf.predict(X_test)

        # Calculate Accuracy score on test and train data
        score = accuracy_score(y_test, y_pred)
        score_train = accuracy_score(y_train, clf.predict(X_train))

    return score, score_train


def apply_classifiers_reduced_data(reduced_X, y_train, y_test, dataset_name, classifiers=['XGBOOST'], save_flag=True):
    """
    Run classifiers on reduced data.

    Parameters:
        reduced_X (dict): Dictionary containing reduced data along with their key dimensions.
        y_train (array-like): Training labels.
        y_test (array-like): Testing labels.
        dataset_name (str): Name of the dataset.
        classifiers (list, optional): List of classifiers to apply. Defaults to ['XGBOOST'].

    Returns:
        pd.DataFrame: DataFrame containing scores, models, and parameters.
        reduced_X_best: Reduced data for only the best configuration for each dimensionality reduction technique.
    """

    scores = dict()  # Dictionary to store scores of different classifiers

    if 'XGBOOST' in classifiers:

        # Iterate over reduced dimensions
        for key_dim in tqdm(reduced_X, desc='XGBoost'):
            clf = train_xgb(reduced_X[key_dim][0].T, y_train)
            # Save trained classifier to a file
            if save_flag:
                pickle.dump(
                    clf, open(os.path.dirname(os.path.dirname(os.path.abspath(
                        __file__))) + f'/models/classifiers/on_reduced_data/{dataset_name}/xgboost.pkl', 'wb'))
            y_pred = clf.predict(reduced_X[key_dim][1].T)
            score = accuracy_score(y_test, y_pred)
            score_train = accuracy_score(y_train, clf.predict(
                reduced_X[key_dim][0].T))
            # Store model name, score, and best parameters
            scores[('XGBoost', *key_dim)] = ['XGBoost',
                                             score, score_train, clf.best_params_]

    if 'SVC' in classifiers:

        # Iterate over reduced dimensions
        for key_dim in tqdm(reduced_X, desc='SVC'):
            clf = train_svc(reduced_X[key_dim][0].T, y_train)
            # Save trained classifier to a file
            if save_flag:
                pickle.dump(
                    clf, open(os.path.dirname(os.path.dirname(os.path.abspath(
                        __file__))) + f'/models/classifiers/on_reduced_data/{dataset_name}/svc.pkl', 'wb'))
            y_pred = clf.predict(reduced_X[key_dim][1].T)
            score = accuracy_score(y_test, y_pred)
            score_train = accuracy_score(y_train, clf.predict(
                reduced_X[key_dim][0].T))
            # Store model name, score, and best parameters
            scores[('SVC', *key_dim)] = ['SVC', score,
                                         score_train, clf.best_params_]

    if 'SGD' in classifiers:

        # Iterate over reduced dimensions
        for key_dim in tqdm(reduced_X, desc='SGD'):
            clf = train_sgd(reduced_X[key_dim][0].T, y_train)
            # Save trained classifier to a file
            if save_flag:
                pickle.dump(
                    clf, open(os.path.dirname(os.path.dirname(os.path.abspath(
                        __file__))) + f'/models/classifiers/on_reduced_data/{dataset_name}/sgd.pkl', 'wb'))
            y_pred = clf.predict(reduced_X[key_dim][1].T)
            score = accuracy_score(y_test, y_pred)
            score_train = accuracy_score(y_train, clf.predict(
                reduced_X[key_dim][0].T))
            # Store model name, score, and best parameters
            scores[('SGD', *key_dim)] = ['SGD', score,
                                         score_train, clf.get_params()]

    # Format scores in a Pandas df
    scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=[
        'Model', 'Score', 'Score Train', 'Params']).reset_index()
    scores_df[['Model', 'Dimensions', 'Dim. Technique', 'Dim. Params']] = pd.DataFrame(
        scores_df['index'].tolist(), index=scores_df.index)
    scores_df = scores_df.drop('index', axis=1)
    scores_df = scores_df.sort_values('Score', ascending=False)

    # Save scores as CSV
    scores_df.to_csv(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))) + f'/results/scores/{dataset_name}_scores.csv')
    print(f'Scores saved at: /results/scores/{dataset_name}_scores.csv')

    # Filter for the best configuration
    best_config = scores_df.fillna('').loc[scores_df.sort_values(
        'Score', ascending=False).groupby('Dim. Technique')['Score'].idxmax()]
    best_config = [tuple(row) for row in best_config[['Dimensions',
                                                      'Dim. Technique', 'Dim. Params']].to_records(index=False)]
    reduced_X_best = {k: reduced_X[k] for k in best_config}

    # Save as pickle
    save_path = os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))) + f'/data/{dataset_name}/reduced/reduced_X_best.pkl'
    pickle.dump(reduced_X_best, open(save_path, 'wb'))
    print(
        f'Reduced data saved at: /data/{dataset_name}/reduced/reduced_X_best.pkl')

    return scores_df, reduced_X_best


def apply_classifiers_with_random_features(X_train, X_test, y_train, y_test, num_iterations, num_dims, size, dataset_name):
    """
    Apply classifiers using random feature selection and save the results in a DataFrame.

    Args:
        num_dims (list): List of integers representing the number of dimensions to consider.
        size (int): Total number of features in the dataset.
        dataset_name (str): Name of the dataset being used.

    Returns:
        pandas.DataFrame: DataFrame containing the results, including the number of dimensions and classifier scores.

    """

    # Create list of features indices
    indices = list(range(size))

    # Initialize DataFrame to store results
    scores_df = pd.DataFrame(
        columns=['Num Dimensions', 'SGD Score', 'SGD Score Train', 'SVC Score', 'SVC Score Train'])

    for i in range(num_iterations):
        print('Iteration:', i+1)
        # Iterate over different numbers of dimensions
        for num in tqdm(num_dims):
            # Apply classifiers with random features
            random.shuffle(indices)
            selected_indices = indices[:num]

            # Apply SGD classifier and retrieve scores
            score_sgd, score_sgd_train = apply_classifiers_original_features(
                X_train[:, selected_indices], y_train, X_test[:, selected_indices], y_test, None, classifiers=['SGD'], save_flag=False)

            # Apply SVC classifier and retrieve scores
            score_svc, score_svc_train = apply_classifiers_original_features(
                X_train[:, selected_indices], y_train, X_test[:, selected_indices], y_test, None, classifiers=['SVC'], save_flag=False)

            # Append results to DataFrame
            scores_df = scores_df.append({
                'Num Dimensions': num,
                'SGD Score': score_sgd,
                'SGD Score Train': score_sgd_train,
                'SVC Score': score_svc,
                'SVC Score Train': score_svc_train,
            }, ignore_index=True)

    # Calculate mean scores
    scores_df = scores_df.groupby('Num Dimensions').mean().reset_index()

    # Save results as CSV
    scores_df.to_csv(os.path.dirname(os.getcwd(
    )) + '/results/feature_selection/' + dataset_name + '_scores_random.csv', index=False)
    print('Saved results to /results/feature_selection/' +
          dataset_name + '_scores_random.csv')

    return scores_df


def feature_selection_classification(X_train, y_train, X_test, y_test, dataset_name, num_dimensions, df_corrs_avg, most_correlated_pixels):
    """
    Perform feature selection for classification using different techniques and classifiers.

    Parameters:
        X_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training labels.
        X_test (numpy.ndarray): Testing feature matrix.
        y_test (numpy.ndarray): Testing labels.
        dataset_name (str): Name of the dataset.
        num_dimensions (list): List of integers specifying different dimensions to explore.
        df_corrs_avg (pandas.DataFrame): DataFrame containing average correlations between features.
        most_correlated_pixels (dict): Dictionary containing most correlated pixels for each dimension technique.

    Returns:
        None
    """
    scores = dict()  # Dictionary to store scores of different classifiers

    pbar = tqdm(num_dimensions)
    for num_dim in pbar:
        for dim_technique in df_corrs_avg.keys()[1:]:
            pbar.set_description(f'{dim_technique} {num_dim} Dimensions')

            # Classify with SGD
            score_sgd, score_sgd_train = apply_classifiers_original_features(
                X_train[:, most_correlated_pixels[dim_technique][:num_dim]],
                y_train,
                X_test[:, most_correlated_pixels[dim_technique][:num_dim]],
                y_test,
                dataset_name,
                classifiers=['SGD'])

            # Classify with SVC
            score_svc, score_svc_train = apply_classifiers_original_features(
                X_train[:, most_correlated_pixels[dim_technique][:num_dim]],
                y_train,
                X_test[:, most_correlated_pixels[dim_technique][:num_dim]],
                y_test,
                dataset_name,
                classifiers=['SVC'])
            res_svc = 0

            scores[(dim_technique, num_dim)] = [score_sgd,
                                                score_sgd_train, score_svc, score_svc_train]

    scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=[
        'SGD Score', 'SGD Score Train', 'SVC Score', 'SVC Score Train']).reset_index()
    scores_df[['Dim Technique', 'Num Dimensions']
              ] = scores_df['index'].apply(pd.Series)

    scores_df = scores_df[['Dim Technique', 'Num Dimensions',
                           'SGD Score', 'SGD Score Train', 'SVC Score', 'SVC Score Train']]

    # Save results as CSV
    scores_df.to_csv(os.path.dirname(os.getcwd(
    )) + '/results/feature_selection/' + dataset_name + '_scores.csv', index=False)
    print('Saved results to /results/feature_selection/' +
          dataset_name + '_scores.csv')
