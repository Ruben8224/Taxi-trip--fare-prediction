from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import pathlib

# Load the dataset
df = pd.read_csv(pathlib.Path('data/train.csv'))
df = pd.read_csv(pathlib.Path('data/train_5001.csv'))
df = pd.read_csv(pathlib.Path('data/train_15000.csv'))

try:
    # Define 'y' as the target variable (fare)
    y = df['fare']

    # Keep only the relevant features for training
    selected_features = ['trip_duration', 'distance_traveled', 'num_of_passengers', 'miscellaneous_fees']
    X = df[selected_features]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create a RandomForestRegressor model
    print('Training model..')
    regressor = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)  # Adjust the values here
    regressor.fit(X_train, y_train)

    # Save the trained model
    print('Saving model..')
    dump(regressor, pathlib.Path('model/train-regression-v1.joblib'))

except KeyError as e:
    print(f"Error: The 'fare' column is not found in the dataset. Please check your CSV file and column names.")
