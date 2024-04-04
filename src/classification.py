import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os
import pickle
import pandas as pd
from sklearn.linear_model import SGDClassifier


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


def train_svc(X_train, y_train):
    # Initialize SVC classifier
    svc_classifier = SVC(random_state=42)
    params = [{'C': [0.001, 0.0001, 0.00001], 'gamma': [
        0.001, 0.00001, 0.0000001], 'kernel': ['rbf']}]
    gs_svc = GridSearchCV(svc_classifier,
                          param_grid=params,
                          scoring='accuracy',
                          cv=3)
    gs_svc.fit(X_train, y_train)

    return gs_svc


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
        # Calculate Accuracy score
        score = accuracy_score(y_test, y_pred)

    if 'SVC' in classifiers:  # Train SVC classifier
        clf = train_svc(X_train, y_train)
        # Save trained classifier to a file
        if save_flag:
            pickle.dump(
                clf, open(os.path.dirname(os.path.dirname(os.path.abspath(
                    __file__))) + f'/models/classifiers/on_orig_features/{dataset_name}/svc.pkl', 'wb'))
        # Predict using trained classifier
        y_pred = clf.predict(X_test)

        # Calculate Accuracy score
        score = accuracy_score(y_test, y_pred)

    if 'SGD' in classifiers:  # Train SGD classifier
        clf = train_sgd(X_train, y_train)
        # Save trained classifier to a file
        if save_flag:
            pickle.dump(
                clf, open(os.path.dirname(os.path.dirname(os.path.abspath(
                    __file__))) + f'/models/classifiers/on_orig_features/{dataset_name}/sgd.pkl', 'wb'))
        # Predict using trained classifier
        y_pred = clf.predict(X_test)

        # Calculate Accuracy score
        score = accuracy_score(y_test, y_pred)

    return score


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
            # Store model name, score, and best parameters
            scores[('XGBoost', *key_dim)] = ['XGBoost',
                                             score, clf.best_params_]

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
            # Store model name, score, and best parameters
            scores[('SVC', *key_dim)] = ['SVC', score, clf.best_params_]

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
            # Store model name, score, and best parameters
            scores[('SGD', *key_dim)] = ['SGD', score, clf.get_params()]

    # Format scores in a Pandas df
    scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=[
        'Model', 'Score', 'Params']).reset_index()
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
