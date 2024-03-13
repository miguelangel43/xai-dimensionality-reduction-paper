import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import pickle
import pandas as pd


def train_xgb(X_train, y_train):
    # Initialize XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(random_state=42)
    params = [{'n_estimators': [5, 10, 20, 50, 100]}]
    gs_xgb = GridSearchCV(xgb_classifier,
                          param_grid=params,
                          scoring='roc_auc',
                          cv=5)
    gs_xgb.fit(X_train, y_train)

    return gs_xgb


def apply_classifiers_original_features(X_train, y_train, X_test, y_test, classifiers=['XGBOOST']):
    """
    Run classifiers on original data.

    Parameters:
        X_train (array-like): Training data.
        y_train (array-like): Training labels.
        X_test (array-like): Testing data.
        y_test (array-like): Testing labels.
        classifiers (list, optional): List of classifiers to apply. Defaults to ['XGBOOST'].

    Returns:
        float: ROC AUC score.
    """

    if 'XGBOOST' in classifiers:
        # Train XGBoost classifier
        clf = train_xgb(X_train, y_train)
        # Save trained classifier to a file
        pickle.dump(
            clf, open(os.path.dirname(os.path.dirname(os.path.abspath(
                __file__))) + '/models/classifiers/on_orig_features/xgboost.pkl', 'wb'))
        # Predict using trained classifier
        y_pred = clf.predict(X_test)
        # Calculate ROC AUC score
        score = roc_auc_score(y_test, y_pred)
        return score


def apply_classifiers_reduced_data(reduced_X, y_train, y_test, classifiers=['XGBOOST']):
    """
    Run classifiers on reduced data.

    Parameters:
        reduced_X (dict): Dictionary containing reduced data along with their key dimensions.
        y_train (array-like): Training labels.
        y_test (array-like): Testing labels.
        classifiers (list, optional): List of classifiers to apply. Defaults to ['XGBOOST'].

    Returns:
        pd.DataFrame: DataFrame containing scores, models, and parameters.
        reduced_X_best: Reduced data for only the best configuration for each dimensionality reduction technique.
    """

    if 'XGBOOST' in classifiers:

        scores = dict()  # Dictionary to store scores of different classifiers

        # Iterate over reduced dimensions
        for key_dim in tqdm(reduced_X, desc='XGBoost'):
            clf = train_xgb(reduced_X[key_dim][0].T, y_train)
            y_pred = clf.predict(reduced_X[key_dim][1].T)
            score = roc_auc_score(y_test, y_pred)
            # Store model name, score, and best parameters
            scores[('XGBoost', *key_dim)] = ['XGBoost',
                                             score, clf.best_params_]

    # Format scores in a Pandas df
    scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=[
        'Model', 'Score', 'Params']).reset_index()
    scores_df[['Model', 'Dimensions', 'Dim. Technique', 'Dim. Params']] = pd.DataFrame(
        scores_df['index'].tolist(), index=scores_df.index)
    scores_df = scores_df.drop('index', axis=1)
    scores_df = scores_df.sort_values('Score', ascending=False)

    # Filter for the best configuration
    best_config = scores_df.fillna('').loc[scores_df.sort_values(
        'Dimensions').groupby('Dim. Technique')['Score'].idxmax()]
    best_config = [tuple(row) for row in best_config[['Dimensions',
                                                      'Dim. Technique', 'Dim. Params']].to_records(index=False)]
    reduced_X_best = {k: reduced_X[k] for k in best_config}

    return scores_df, reduced_X_best
