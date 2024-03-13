import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pickle import dump
import os


def load_data(file_path, names):
    """Load raw data from file."""
    return pd.read_csv(file_path, names=names).sample(500)


def preprocess_data(data):
    """Preprocess the raw data."""

    # Encode categorical variables
    categorical_cols = ['workclass', 'education', 'marital-status',
                        'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']

    for col in categorical_cols:
        label_encoder = LabelEncoder()
        data[col] = label_encoder.fit_transform(data[col])

    return data


def scale_features(X_train, X_test):
    """Scale numeric features."""

    # define scaler
    scaler = MinMaxScaler()
    # fit scaler on the training dataset
    scaler.fit(X_train)
    # transform both datasets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def split_data(data, target_column):
    """Split the data into features and target, and then into train and test sets."""
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    # Load raw data
    data = load_data(os.getcwd() + '/data/raw_data/census_income/adult.data',
                     names=['age',
                            'workclass',
                            'fnlwgt',
                            'education',
                            'education-num',
                            'marital-status',
                            'occupation',
                            'relationship',
                            'race',
                            'sex',
                            'capital-gain',
                            'capital-loss',
                            'hours-per-week',
                            'native-country',
                            'income'])

    # Preprocess data
    data = preprocess_data(data)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(
        data, target_column='income')

    # Scale numeric features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Save preprocessed data
    dump(X_train_scaled, open(os.getcwd() +
         '/data/processed_data/census_income/X_train.pkl', 'wb'))
    dump(X_test_scaled, open(os.getcwd() +
         '/data/processed_data/census_income/X_test.pkl', 'wb'))
    dump(y_train.to_numpy(), open(os.getcwd() +
         '/data/processed_data/census_income/y_train.pkl', 'wb'))
    dump(y_test.to_numpy(), open(os.getcwd() +
         '/data/processed_data/census_income/y_test.pkl', 'wb'))

    # Save column names
    column_names = list(data.columns)
    with open(os.getcwd() + '/data/processed_data/census_income/column_names.txt', 'w') as file:
        for string in column_names:
            file.write(f"{string}\n")

    print('Data saved in folder: ', '/data/processed_data/census_income/')
